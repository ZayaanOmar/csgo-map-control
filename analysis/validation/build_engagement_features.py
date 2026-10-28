"""
build_engagement_features.py
============================
For every valid kill, reconstruct what the influence field looked like BEFORE the fight, from
position_maps + the APSP matrix (the exact inf_comp() formula; no node-level parquet is read).

Outputs (one parquet per match):
  outputs/engagement_adv/{match}.parquet     long: kill_idx x lead_s x radius
  outputs/engagement_loc/{match}.parquet     long: kill_idx x lead_s
  outputs/engagement_series/{match}.parquet  long: kill_idx x frame offset (event study)

Definitions
-----------
lead frame     the latest frame with tick <= kill_tick - lead_s * tick_rate   (frames are ~1 s apart,
               so the lead frame is lead_s .. lead_s+1 s before the kill; `frame_dt` stores the exact gap)
anchors        the victim's and attacker's kill-time nav nodes (kill_events.*_area_id)
ball(R)        nodes within geodesic distance R (symmetrised) of EITHER anchor
M_A, M_V       summed per-player influence of the attacker's / victim's team over all nodes,
               from the lead frame's alive players
adv_ctrl_*     mean over ACTIVE ball nodes (M_A+M_V > ACTIVE_THR = exp(-2)) of expit(K*(M_A-M_V)), minus 0.5
adv_share_*    (sum_ball M_A - sum_ball M_V) / (sum_ball M_A + sum_ball M_V)
 *_all         includes the two duelists' own influence
 *_support     EXCLUDES both duelists (teammates' influence only) -> "who else was around";
               if neither side has any support the value is 0.0 (neutral), NaN if a duelist is
               missing from the lead frame
n_att_near / n_vic_near   alive NON-duelist players standing in the ball (local numbers)

Kill-location contest (engagement_loc): control at the VICTIM's kill node, evaluated with the
LEAD frame's field, from the attacker's perspective, vs a matched null = the same quantity at the
nodes where each alive player was standing in that same lead frame (10-bin histogram of
|control-0.5| over [0, 0.5] is stored so pooled null histograms can be rebuilt).
CAVEAT: kills require opposing players to be close, and the field is contested where opposing
players are close, so this comparison is partly mechanical at short leads — it is a supporting
panel, not the headline test (the duel win-rate curve is, because the *direction* of the
advantage is not mechanically tied to who wins).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402

LEADS = [1.0, 2.0, 3.0, 4.0, 5.0]
RADII = [250.0, 400.0, 600.0]
PRIMARY_LEAD = 2.0
R_SERIES = 400.0
SERIES_PRE_S, SERIES_POST_S = 6.0, 6.0
KEY = 100000
NBINS = 10

ADV_COLS = ["kill_idx", "lead_s", "radius", "frame_dt", "adv_ctrl_all", "adv_share_all", "adv_ctrl_support",
            "adv_share_support", "n_att_near", "n_vic_near", "duelists_found"]
TAGS = [vc.mass_tag(m) for m in vc.MASS_THRS]
LOC_COLS = (["kill_idx", "lead_s", "frame_dt", "active_at_kill", "ctrl_att_at_kill", "absdev_at_kill",
             "contested_at_kill", "total_mass_at_kill", "min_mass_at_kill", "null_n", "null_contested_frac",
             "null_absdev_mean", "mm_null_n", "mm_null_contested_frac"]
            + [f"strict_at_kill_{t}" for t in TAGS] + [f"null_strict_frac_{t}" for t in TAGS]
            + [f"null_hist_{i:02d}" for i in range(NBINS)] + ["active_thr"])
ADV_COLS = ADV_COLS + ["active_thr"]
SER_COLS = ["kill_idx", "rel_t", "adv_ctrl_att_all", "att_mu_mean", "vic_mu_mean", "n_att_alive", "n_vic_alive", "active_thr"]


class FrameStore:
    """position_maps indexed by (round, frame_idx) with lazily-computed influence matrices."""

    def __init__(self, pos: pd.DataFrame, ctx: vc.MapContext):
        self.ctx = ctx
        key = pos["round"].to_numpy(np.int64) * KEY + pos["frame_idx"].to_numpy(np.int64)
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.names = pos["player_name"].to_numpy()[order]
        self.is_ct = (pos["side"].to_numpy() == "CT")[order]
        self.cur = ctx.idx_of(pos["cur_area"].to_numpy())[order]
        self.proj = ctx.idx_of(pos["proj_area"].to_numpy())[order]
        uk, start = np.unique(self.key, return_index=True)
        self.span = {int(k): (int(s), int(e)) for k, s, e in zip(uk, start, list(start[1:]) + [len(self.key)])}
        self._cache = {}

    def clear(self):
        self._cache.clear()

    def get(self, rnd: int, fidx: int):
        k = int(rnd) * KEY + int(fidx)
        if k in self._cache:
            return self._cache[k]
        if k not in self.span:
            return None
        lo, hi = self.span[k]
        cur, proj = self.cur[lo:hi], self.proj[lo:hi]
        good = (cur >= 0) & (proj >= 0)
        if not good.any():
            self._cache[k] = None
            return None
        mu = self.ctx.mu_matrix(cur[good], proj[good])
        val = (self.names[lo:hi][good], self.is_ct[lo:hi][good], cur[good], mu)
        self._cache[k] = val
        return val


def adv_pair(MA, MV, ball, neutral_if_empty):
    a, v = MA[ball], MV[ball]
    act = (a + v) > vc.ACTIVE_THR
    ac = float(expit(vc.K_SIGMOID * (a - v))[act].mean() - 0.5) if act.any() else (0.0 if neutral_if_empty else np.nan)
    den = float(a.sum() + v.sum())
    sh = float((a.sum() - v.sum()) / den) if den > 1e-9 else (0.0 if neutral_if_empty else np.nan)
    return ac, sh


def build(match: str, ctx: vc.MapContext):
    kills = pd.read_csv(vc.T_KILLS / f"{match}.csv")
    kills = kills[kills["valid_duel"]].copy()
    fs = vc.read_resolved(vc.T_FSTATE / f"{match}.csv", {k: [k] for k in ("round", "frame_idx", "tick", "tick_rate")}, "frame_state")
    pos = vc.read_resolved(vc.T_POSITION / f"{match}.csv", vc.POS_SPEC, "position_maps")
    store = FrameStore(pos, ctx)
    frames_by_round = {int(r): (g["tick"].to_numpy(float), g["frame_idx"].to_numpy(int))
                       for r, g in fs.sort_values(["round", "frame_idx"]).groupby("round")}
    tick_rate = float(fs["tick_rate"].iloc[0])
    adv_rows, loc_rows, ser_rows = [], [], []
    hist_edges = np.linspace(0.0, 0.5, NBINS + 1)
    ball_cache = {}

    def ball_of(anchors, R):
        k = (anchors, R)
        if k not in ball_cache:
            m = np.zeros(ctx.n, bool)
            for a in anchors:
                m |= ctx.dsym[a] <= R
            ball_cache[k] = m
        return ball_cache[k]

    last_round = None
    for _, k in kills.sort_values(["round", "tick"]).iterrows():
        rnd = int(k["round"])
        if rnd != last_round:
            store.clear()
            last_round = rnd
        if rnd not in frames_by_round:
            continue
        ticks, fidxs = frames_by_round[rnd]
        v_idx = int(ctx.idx_of(np.array([k["victim_area_id"]]))[0])
        a_idx = int(ctx.idx_of(np.array([k["attacker_area_id"]]))[0])
        if v_idx < 0 or a_idx < 0:
            continue
        anchors = (v_idx, a_idx)
        att_is_ct = (k["attacker_side"] == "CT")
        ktick = float(k["tick"])
        # ---------------------------------------------------- lead-frame features
        for lead in LEADS:
            pos_i = np.searchsorted(ticks, ktick - lead * tick_rate, side="right") - 1
            if pos_i < 0:
                continue
            fr = store.get(rnd, fidxs[pos_i])
            if fr is None:
                continue
            names, is_ct, cur, mu = fr
            A = (is_ct == att_is_ct)
            V = ~A
            is_att, is_vic = names == k["attacker_name"], names == k["victim_name"]
            found = bool(is_att.any() and is_vic.any())
            MA, MV = mu[A].sum(0), mu[V].sum(0)
            SA, SV = mu[A & ~is_att].sum(0), mu[V & ~is_vic].sum(0)
            fdt = (ktick - ticks[pos_i]) / tick_rate
            for R in RADII:
                ball = ball_of(anchors, R)
                ac, sh = adv_pair(MA, MV, ball, False)
                if found:
                    sc, ss = adv_pair(SA, SV, ball, True)
                else:
                    sc, ss = np.nan, np.nan
                in_ball = ball[cur]
                adv_rows.append((int(k["kill_idx"]), lead, R, fdt, ac, sh, sc, ss,
                                 int((in_ball & A & ~is_att).sum()), int((in_ball & V & ~is_vic).sum()), found,
                                 vc.ACTIVE_THR))
            # ---- kill-location contest vs matched null (victim's kill node, lead-frame field)
            ma_v, mv_v = float(MA[v_idx]), float(MV[v_idx])
            ctrl_node = float(expit(vc.K_SIGMOID * (ma_v - mv_v)))
            tk_mass = ma_v + mv_v
            cont_node = bool(vc.CONTEST_LO <= ctrl_node <= vc.CONTEST_HI)
            ma_o, mv_o = MA[cur], MV[cur]
            ctrl_occ = expit(vc.K_SIGMOID * (ma_o - mv_o))
            absd_occ = np.abs(ctrl_occ - 0.5)
            cont_occ = (ctrl_occ >= vc.CONTEST_LO) & (ctrl_occ <= vc.CONTEST_HI)
            row = {"kill_idx": int(k["kill_idx"]), "lead_s": lead, "frame_dt": fdt,
                   "active_at_kill": bool(tk_mass > vc.ACTIVE_THR), "ctrl_att_at_kill": ctrl_node,
                   "absdev_at_kill": abs(ctrl_node - 0.5), "contested_at_kill": cont_node,
                   "total_mass_at_kill": tk_mass, "min_mass_at_kill": min(ma_v, mv_v), "null_n": len(cur),
                   "null_contested_frac": float(cont_occ.mean()), "null_absdev_mean": float(absd_occ.mean())}
            for m, tag in zip(vc.MASS_THRS, TAGS):   # ROBUSTLY contested: both teams carry >= m mass
                row[f"strict_at_kill_{tag}"] = bool(cont_node and ma_v >= m and mv_v >= m)
                row[f"null_strict_frac_{tag}"] = float((cont_occ & (ma_o >= m) & (mv_o >= m)).mean())
            # mass-matched null: contested fraction among ALL nodes whose total mass is within x/÷2 of the kill node's
            tot_all = MA + MV
            if tk_mass > vc.ACTIVE_THR:
                selm = (tot_all >= tk_mass / 2) & (tot_all <= tk_mass * 2)
                ctrl_all = expit(vc.K_SIGMOID * (MA - MV))
                cont_all = (ctrl_all >= vc.CONTEST_LO) & (ctrl_all <= vc.CONTEST_HI)
                row["mm_null_n"] = int(selm.sum())
                row["mm_null_contested_frac"] = float(cont_all[selm].mean()) if selm.sum() >= 5 else np.nan
            else:
                row["mm_null_n"], row["mm_null_contested_frac"] = 0, np.nan
            for i, h in enumerate(np.histogram(absd_occ, bins=hist_edges)[0]):
                row[f"null_hist_{i:02d}"] = int(h)
            row["active_thr"] = vc.ACTIVE_THR
            loc_rows.append(row)
        # ---------------------------------------------------- event-study series
        sel = np.where((ticks >= ktick - SERIES_PRE_S * tick_rate) & (ticks <= ktick + SERIES_POST_S * tick_rate))[0]
        ball = ball_of(anchors, R_SERIES)
        for pi in sel:
            fr = store.get(rnd, fidxs[pi])
            if fr is None:
                continue
            names, is_ct, cur, mu = fr
            A = (is_ct == att_is_ct)
            MA, MV = mu[A].sum(0), mu[~A].sum(0)
            ac, _ = adv_pair(MA, MV, ball, False)
            ser_rows.append((int(k["kill_idx"]), (ticks[pi] - ktick) / tick_rate, ac, float(MA[ball].mean()),
                             float(MV[ball].mean()), int(A.sum()), int((~A).sum()), vc.ACTIVE_THR))
    return (pd.DataFrame(adv_rows, columns=ADV_COLS), pd.DataFrame(loc_rows, columns=LOC_COLS),
            pd.DataFrame(ser_rows, columns=SER_COLS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    vc.add_limit_arg(ap)
    args = ap.parse_args()
    matches = [m for m in vc.match_stems(vc.T_KILLS, (".csv",))
               if (vc.T_POSITION / f"{m}.csv").exists() and (vc.T_FSTATE / f"{m}.csv").exists()]
    if args.limit:
        matches = matches[: args.limit]
    if not matches:
        raise SystemExit("No kill_events found — run extract_supplementary_tables.py first.")
    ctx = vc.MapContext()
    for d in (vc.T_ENG_ADV, vc.T_ENG_LOC, vc.T_ENG_SER):
        vc.ensure_dir(d)
    done = skipped = failed = 0
    for m in matches:
        outs = (vc.T_ENG_ADV / f"{m}.parquet", vc.T_ENG_LOC / f"{m}.parquet", vc.T_ENG_SER / f"{m}.parquet")
        if (vc.schema_ok(outs[0], ADV_COLS) and vc.schema_ok(outs[1], LOC_COLS)
                and vc.schema_ok(outs[2], SER_COLS) and vc.stamp_matches(outs[0], "active_thr", vc.ACTIVE_THR)
                and not args.force):
            skipped += 1
            continue
        try:
            adv, loc, ser = build(m, ctx)
            adv.to_parquet(outs[0], index=False)
            loc.to_parquet(outs[1], index=False)
            ser.to_parquet(outs[2], index=False)
            vc.log(f"done {m}: {adv['kill_idx'].nunique()} kills, {len(adv)} adv rows, {len(loc)} loc rows, {len(ser)} series rows")
            done += 1
        except Exception as e:  # noqa: BLE001
            vc.log(f"FAILED {m}: {type(e).__name__}: {e}")
            for p in outs:
                if p.exists():
                    p.unlink()
            failed += 1
    vc.log(f"finished: {done} written, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
