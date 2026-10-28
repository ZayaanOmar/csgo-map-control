"""
extract_supplementary_tables.py
===============================
One pass over each filtered demo (data/filtered_demos/*.json.xz) producing THREE
per-match tables that the existing pipeline does not yet carry:

  outputs/round_economy/{match}.csv   one row per (non-warmup) round
  outputs/kill_events/{match}.csv     one row per kill, with trade/chain flags + nav-node mapping
  outputs/frame_state/{match}.csv     one row per (round, frame_idx): tick, seconds since
                                      freeze end, alive/HP/equipment/utility per side, bomb state

Why these exist
---------------
* round_economy  -> the single biggest confounder for any "does control predict
                    the round" claim (equipment value / buy type / pistol flag).
* kill_events    -> duel-validity analyses, kill density, first-kill timing.
* frame_state    -> a leak-free time axis (seconds since freeze end), alive
                    counts and bomb state *straight from the demo*, so no analysis
                    depends on optional columns in position_maps.

frame_idx convention
--------------------
frame_idx = enumerate() index into rnd["frames"] of the FILTERED demo — the same
convention position_to_node_map.py uses, so (round, frame_idx) joins line up.

Kill relation flags (own definitions, independent of the parser's isTrade)
--------------------------------------------------------------------------
  is_trade_kill : kill i retaliates for kill h  (i.victim == h.attacker AND
                  i.attacker is on h.victim's side AND 0 < dt <= TRADE_WINDOW_S)
  was_traded    : the mirror image — kill h was avenged within TRADE_WINDOW_S
  is_chain_kill : same attacker as an earlier kill within CHAIN_WINDOW_S
  is_mutual     : A kills B and B kills A within MUTUAL_TICKS (outcome undefined)
  parser_is_trade is kept verbatim for cross-checking against is_trade_kill.

Filtering guidance (implemented in analysis_kill_validity.py): drop the *avenging*
kill (is_trade_kill) and chain kills, NOT the first kill of a traded pair —
dropping on was_traded conditions on a future event and biases the sample.

Usage
-----
  python analysis/validation/extract_supplementary_tables.py --inspect
  python analysis/validation/extract_supplementary_tables.py --check-mapper --limit 3
  python analysis/validation/extract_supplementary_tables.py            # full run
"""
from __future__ import annotations

import argparse
import json
import lzma
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402

TRADE_WINDOW_S = 5.0
CHAIN_WINDOW_S = 3.0
MUTUAL_TICK_FRAC = 0.25      # of one second
DEFAULT_TICK_RATE = 128.0

ECON_COLS = ["match_id", "round", "ct_team", "t_team", "ct_score_start", "t_score_start", "is_pistol_round",
             "ct_buy_type", "t_buy_type", "ct_round_start_eq", "t_round_start_eq", "ct_freeze_end_eq",
             "t_freeze_end_eq", "ct_spend", "t_spend", "eq_diff", "log_eq_ratio", "round_len_s"]
KILL_COLS = ["kill_idx", "match_id", "round", "tick", "t_rel", "attacker_name", "attacker_side", "victim_name",
             "victim_side", "attacker_x", "attacker_y", "attacker_z", "victim_x", "victim_y", "victim_z",
             "attacker_area_id", "victim_area_id", "attacker_map_dist", "victim_map_dist", "weapon",
             "weapon_class", "is_headshot", "is_wallbang", "thru_smoke", "no_scope", "attacker_blinded",
             "victim_blinded", "is_first_kill", "distance", "parser_is_trade", "assister_name", "valid_duel",
             "is_mutual", "is_trade_kill", "was_traded", "is_chain_kill"]
FSTATE_COLS = ["match_id", "round", "frame_idx", "tick", "t_rel", "tick_rate", "ct_alive", "t_alive", "ct_hp",
               "t_hp", "ct_eq", "t_eq", "ct_util", "t_util", "bomb_planted", "bomb_site"]

SIDE_MAP = {"CT": "CT", "T": "T", "TERRORIST": "T", "COUNTERTERRORIST": "CT", "COUNTER-TERRORIST": "CT"}


def norm_side(s):
    return SIDE_MAP.get(str(s).upper(), "") if s is not None else ""


def load_demo(path: Path) -> dict:
    with lzma.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _num(v, default=np.nan):
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def flag_relations(df: pd.DataFrame, tick_rate: float) -> pd.DataFrame:
    """Add is_mutual / is_trade_kill / was_traded / is_chain_kill (one round, valid duels only)."""
    n = len(df)
    flags = {k: np.zeros(n, bool) for k in ("is_mutual", "is_trade_kill", "was_traded", "is_chain_kill")}
    if n:
        t = df["tick"].to_numpy(float)
        att, vic = df["attacker_name"].to_numpy(), df["victim_name"].to_numpy()
        asd, vsd = df["attacker_side"].to_numpy(), df["victim_side"].to_numpy()
        valid = df["valid_duel"].to_numpy(bool)
        trade_w, chain_w = TRADE_WINDOW_S * tick_rate, CHAIN_WINDOW_S * tick_rate
        mut_w = MUTUAL_TICK_FRAC * tick_rate
        for i in range(n):
            if not valid[i]:
                continue
            for h in range(n):
                if h == i or not valid[h]:
                    continue
                dt = t[i] - t[h]
                if 0 < dt <= trade_w and vic[i] == att[h] and asd[i] == vsd[h] and att[i] != vic[h]:
                    flags["is_trade_kill"][i] = True
                    flags["was_traded"][h] = True
                if (dt > 0 or (dt == 0 and i > h)) and dt <= chain_w and att[i] == att[h]:
                    flags["is_chain_kill"][i] = True
                if abs(dt) <= mut_w and att[i] == vic[h] and vic[i] == att[h]:
                    flags["is_mutual"][i] = True
    for k, v in flags.items():
        df[k] = v
    return df


def process_demo(demo: dict, ctx, mapper):
    match_id = demo.get("matchId", demo.get("matchID", ""))
    tick_rate = _num(demo.get("tickRate"), np.nan)
    if not np.isfinite(tick_rate) or tick_rate <= 0:
        tick_rate = DEFAULT_TICK_RATE
        vc.log(f"    WARN: demo has no tickRate; assuming {DEFAULT_TICK_RATE}")
    econ, kills, fstate = [], [], []
    kill_idx = 0
    for rnd in demo["gameRounds"]:
        if rnd.get("isWarmup", False):
            continue
        rn = int(rnd["roundNum"])
        fe, et = _num(rnd.get("freezeTimeEndTick")), _num(rnd.get("endTick"))
        # ---------------- economy
        ts, cs = _num(rnd.get("tScore"), 0.0), _num(rnd.get("ctScore"), 0.0)
        tot = ts + cs
        ct_fe, t_fe = _num(rnd.get("ctFreezeTimeEndEqVal")), _num(rnd.get("tFreezeTimeEndEqVal"))
        econ.append({
            "match_id": match_id, "round": rn, "ct_team": rnd.get("ctTeam"), "t_team": rnd.get("tTeam"),
            "ct_score_start": cs, "t_score_start": ts,
            "is_pistol_round": bool(tot in (0, 15)),
            "ct_buy_type": rnd.get("ctBuyType"), "t_buy_type": rnd.get("tBuyType"),
            "ct_round_start_eq": _num(rnd.get("ctRoundStartEqVal")), "t_round_start_eq": _num(rnd.get("tRoundStartEqVal")),
            "ct_freeze_end_eq": ct_fe, "t_freeze_end_eq": t_fe,
            "ct_spend": _num(rnd.get("ctRoundSpendMoney")), "t_spend": _num(rnd.get("tRoundSpendMoney")),
            "eq_diff": ct_fe - t_fe,
            "log_eq_ratio": float(np.log((ct_fe + 1.0) / (t_fe + 1.0))) if np.isfinite(ct_fe) and np.isfinite(t_fe) else np.nan,
            "round_len_s": (et - fe) / tick_rate if np.isfinite(et) and np.isfinite(fe) else np.nan,
        })
        # ---------------- frame state
        for fi, fr in enumerate(rnd.get("frames") or []):
            row = {"match_id": match_id, "round": rn, "frame_idx": fi, "tick": fr["tick"],
                   "t_rel": (fr["tick"] - fe) / tick_rate if np.isfinite(fe) else np.nan, "tick_rate": tick_rate,
                   "bomb_planted": bool(fr.get("bombPlanted", False)), "bomb_site": fr.get("bombsite", "") or ""}
            for side in ("ct", "t"):
                pl = [p for p in ((fr.get(side) or {}).get("players") or []) if p.get("isAlive")]
                row[f"{side}_alive"] = len(pl)
                row[f"{side}_hp"] = float(sum(_num(p.get("hp"), 0.0) for p in pl))
                row[f"{side}_eq"] = float(sum(_num(p.get("equipmentValue"), 0.0) for p in pl))
                row[f"{side}_util"] = float(sum(_num(p.get("totalUtility"), 0.0) for p in pl))
            fstate.append(row)
        # ---------------- kills
        rk = []
        for k in (rnd.get("kills") or []):
            a_side, v_side = norm_side(k.get("attackerSide")), norm_side(k.get("victimSide"))
            an, vn = k.get("attackerName"), k.get("victimName")
            tick = _num(k.get("tick"))
            pos = [_num(k.get(c)) for c in ("attackerX", "attackerY", "attackerZ", "victimX", "victimY", "victimZ")]
            valid = (bool(an) and bool(vn) and a_side != "" and v_side != "" and a_side != v_side
                     and not k.get("isSuicide", False) and not k.get("isTeamkill", False)
                     and np.isfinite(pos).all() and np.isfinite(tick)
                     and (not np.isfinite(fe) or tick >= fe) and (not np.isfinite(et) or tick <= et))
            rk.append({
                "match_id": match_id, "round": rn, "tick": tick,
                "t_rel": (tick - fe) / tick_rate if np.isfinite(fe) else np.nan,
                "attacker_name": an, "attacker_side": a_side, "victim_name": vn, "victim_side": v_side,
                "attacker_x": pos[0], "attacker_y": pos[1], "attacker_z": pos[2],
                "victim_x": pos[3], "victim_y": pos[4], "victim_z": pos[5],
                "weapon": k.get("weapon"), "weapon_class": k.get("weaponClass"),
                "is_headshot": bool(k.get("isHeadshot", False)), "is_wallbang": bool(k.get("isWallbang", False)),
                "thru_smoke": bool(k.get("thruSmoke", False)), "no_scope": bool(k.get("noScope", False)),
                "attacker_blinded": bool(k.get("attackerBlinded", False)), "victim_blinded": bool(k.get("victimBlinded", False)),
                "is_first_kill": bool(k.get("isFirstKill", False)), "distance": _num(k.get("distance")),
                "parser_is_trade": bool(k.get("isTrade", False)), "assister_name": k.get("assisterName"),
                "valid_duel": bool(valid),
            })
        if rk:
            d = pd.DataFrame(rk).sort_values("tick", kind="stable").reset_index(drop=True)
            d = flag_relations(d, tick_rate)
            kills.append(d)
    econ_df = pd.DataFrame(econ, columns=ECON_COLS)
    fs_df = pd.DataFrame(fstate, columns=FSTATE_COLS)
    kdf = pd.concat(kills, ignore_index=True) if kills else pd.DataFrame(columns=KILL_COLS)
    if len(kdf):
        kdf.insert(0, "kill_idx", np.arange(len(kdf)))
        for who in ("attacker", "victim"):
            xyz = kdf[[f"{who}_x", f"{who}_y", f"{who}_z"]].to_numpy(float)
            idx, dist = mapper.map(xyz)
            kdf[f"{who}_area_id"] = np.where(idx >= 0, ctx.area_ids[np.clip(idx, 0, None)], -1)
            kdf[f"{who}_map_dist"] = dist
    else:
        for c in KILL_COLS:
            if c not in kdf.columns:
                kdf[c] = []
    return econ_df, kdf[KILL_COLS], fs_df


def outputs_for(match):
    return {"econ": vc.T_ECON / f"{match}.csv", "kills": vc.T_KILLS / f"{match}.csv", "fstate": vc.T_FSTATE / f"{match}.csv"}


def is_done(match):
    o = outputs_for(match)
    return (vc.schema_ok(o["econ"], ECON_COLS) and vc.schema_ok(o["kills"], KILL_COLS)
            and vc.schema_ok(o["fstate"], FSTATE_COLS))


def cmd_inspect(matches):
    m = matches[0]
    demo = load_demo(vc.FILTERED_DEMOS / f"{m}.json.xz")
    vc.log(f"match: {m}\ntop-level keys: {sorted(demo.keys())}")
    vc.log(f"tickRate={demo.get('tickRate')}  matchId={demo.get('matchId')}  mapName={demo.get('mapName')}")
    rnd = next(r for r in demo["gameRounds"] if not r.get("isWarmup", False))
    vc.log("round-level keys (non-list): " + str(sorted(k for k, v in rnd.items() if not isinstance(v, list))))
    vc.log("round-level list keys: " + str(sorted(k for k, v in rnd.items() if isinstance(v, list))))
    keys_needed = ["ctBuyType", "tBuyType", "ctFreezeTimeEndEqVal", "tFreezeTimeEndEqVal", "ctRoundStartEqVal",
                   "tRoundStartEqVal", "ctRoundSpendMoney", "tRoundSpendMoney", "tScore", "ctScore",
                   "freezeTimeEndTick", "endTick"]
    vc.log("economy keys present: " + str({k: (k in rnd) for k in keys_needed}))
    if rnd.get("kills"):
        vc.log("first kill: " + json.dumps(rnd["kills"][0], indent=1, default=str))
    fr = rnd["frames"][0]
    vc.log("frame keys: " + str(sorted(fr.keys())))
    pl = (fr.get("ct") or {}).get("players") or []
    if pl:
        vc.log("player keys: " + str(sorted(pl[0].keys())))
    bt = {(r.get("ctBuyType"), r.get("tBuyType")) for r in demo["gameRounds"] if not r.get("isWarmup", False)}
    vc.log(f"distinct (ctBuyType, tBuyType) in this demo: {sorted(bt, key=str)}")


def cmd_check_mapper(matches, ctx, mapper):
    for m in matches:
        pp = vc.T_POSITION / f"{m}.csv"
        if not pp.exists():
            vc.log(f"{m}: no position_maps file, skipping")
            continue
        demo = load_demo(vc.FILTERED_DEMOS / f"{m}.json.xz")
        rows = []
        for rnd in [r for r in demo["gameRounds"] if not r.get("isWarmup", False)][:3]:
            for fi, fr in enumerate((rnd.get("frames") or [])[:30]):
                for side in ("ct", "t"):
                    for p in ((fr.get(side) or {}).get("players") or []):
                        if p.get("isAlive"):
                            rows.append((int(rnd["roundNum"]), fi, p["name"], p["x"], p["y"], p["z"]))
        mine = pd.DataFrame(rows, columns=["round", "frame_idx", "player_name", "x", "y", "z"])
        idx, _ = mapper.map(mine[["x", "y", "z"]].to_numpy(float))
        mine["my_area"] = ctx.area_ids[idx]
        pos = vc.read_resolved(pp, vc.POS_SPEC, "position_maps")
        j = mine.merge(pos[["round", "frame_idx", "player_name", "cur_area"]], on=["round", "frame_idx", "player_name"])
        agree = (j["my_area"] == j["cur_area"]).mean() if len(j) else float("nan")
        vc.log(f"{m}: merged {len(j)}/{len(mine)} rows; mapper agreement with position_maps = {agree:.4%} (Z_TOL={vc.Z_TOL})")
        if len(j) and agree < 0.999:
            vc.log("   -> <99.9%: set Z_TOL (env var) to the value used in position_to_node_map.py "
                   "and/or confirm it still uses the same two-tier rule.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    vc.add_limit_arg(ap)
    ap.add_argument("--inspect", action="store_true", help="print the demo schema keys this script relies on, then exit")
    ap.add_argument("--check-mapper", action="store_true", help="verify the KDTree mapper reproduces position_maps area IDs, then exit")
    args = ap.parse_args()

    matches = vc.match_stems(vc.FILTERED_DEMOS, (".json.xz",))
    if args.limit:
        matches = matches[: args.limit]
    if not matches:
        raise SystemExit(f"No filtered demos in {vc.FILTERED_DEMOS}")
    if args.inspect:
        return cmd_inspect(matches)
    ctx = vc.MapContext()
    mapper = vc.NodeMapper(ctx)
    if args.check_mapper:
        return cmd_check_mapper(matches, ctx, mapper)

    for d in (vc.T_ECON, vc.T_KILLS, vc.T_FSTATE):
        vc.ensure_dir(d)
    done = failed = skipped = 0
    for m in matches:
        if is_done(m) and not args.force:
            skipped += 1
            continue
        o = outputs_for(m)
        try:
            demo = load_demo(vc.FILTERED_DEMOS / f"{m}.json.xz")
            econ, kills, fs = process_demo(demo, ctx, mapper)
            econ.to_csv(o["econ"], index=False)
            kills.to_csv(o["kills"], index=False)
            fs.to_csv(o["fstate"], index=False)
            agree = (kills["parser_is_trade"] == kills["is_trade_kill"]).mean() if len(kills) else float("nan")
            vc.log(f"done {m}: {len(econ)} rounds, {len(kills)} kills "
                   f"({int(kills['valid_duel'].sum())} valid duels, parser-vs-own trade agreement {agree:.1%}), {len(fs)} frames")
            done += 1
        except Exception as e:  # noqa: BLE001
            vc.log(f"FAILED {m}: {type(e).__name__}: {e}")
            for p in o.values():
                if p.exists():
                    p.unlink()
            failed += 1
    vc.log(f"finished: {done} written, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
