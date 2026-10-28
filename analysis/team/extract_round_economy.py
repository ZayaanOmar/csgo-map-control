#!/usr/bin/env python
"""
Extract per-round economy from the filtered demos, validate it, and report how much data
each eco vs full-buy contrast would have.

Properties of the ESTA economy fields that this script relies on (sections 1-2 of the
validation output below re-check them on every run):
  * ctBuyType / tBuyType are one of Full Eco / Semi Eco / Semi Buy / Full Buy and are a clean
    function of team equipment value at freeze-time end (bands do not overlap).
  * PISTOL ROUNDS (1 and 16) ARE LABELLED "Full Eco": always exclude them from any eco class.

Input : data/filtered_demos/*.json.xz, data/team_aliases.json, outputs/round_metadata/*.csv
        outputs/analysis/round_signatures.parquet   (optional: enables the feasibility section)
Output: outputs/analysis/round_economy.csv   (one row per round)
Usage : python3 extract_round_economy.py
"""
import glob
import json
import lzma
import os

import pandas as pd

DEMO_GLOB = "data/filtered_demos/*.json.xz"
ALIASES_PATH = "data/team_aliases.json"
RM_GLOB = "outputs/round_metadata/*.csv"
SIG_PATH = "outputs/analysis/round_signatures.parquet"
OUT_PATH = "outputs/analysis/round_economy.csv"

PISTOL_ROUNDS = (1, 16)
BUY = {"Full Eco": "eco", "Semi Eco": "semi_eco", "Semi Buy": "semi_buy", "Full Buy": "full"}
ORDER = ["eco", "semi_eco", "semi_buy", "full"]
MIN_ROUND_COUNTS = (5, 8, 12)


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def extract(path):
    with lzma.open(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    name = os.path.basename(path)[: -len(".json.xz")]
    rows = []
    for r in d["gameRounds"]:
        if r.get("isWarmup"):
            continue
        rows.append({
            "match_name": name, "match_id": d["matchId"], "round": r["roundNum"],
            "ct_team_raw": r["ctTeam"], "t_team_raw": r["tTeam"],
            "ct_buy_label": r["ctBuyType"], "t_buy_label": r["tBuyType"],
            "ct_eq": r["ctFreezeTimeEndEqVal"], "t_eq": r["tFreezeTimeEndEqVal"],
            "ct_eq_start": r["ctRoundStartEqVal"], "t_eq_start": r["tRoundStartEqVal"],
            "ct_spend": r["ctRoundSpendMoney"], "t_spend": r["tRoundSpendMoney"],
            "ct_score": r["ctScore"], "t_score": r["tScore"],
            "winning_side": r["winningSide"], "round_end_reason": r["roundEndReason"],
            "bomb_plant_tick": r.get("bombPlantTick"),
        })
    return rows


def build_table():
    files = sorted(glob.glob(DEMO_GLOB))
    print(f"{len(files)} filtered demos")
    rows, failed = [], []
    for i, p in enumerate(files, 1):
        try:
            rows.extend(extract(p))
        except Exception as e:
            failed.append((os.path.basename(p), f"{type(e).__name__}: {e}"))
        if i % 50 == 0 or i == len(files):
            print(f"  extracted {i}/{len(files)}")
    for name, err in failed:
        print(f"  FAILED {name}: {err}")
    assert not failed, "some demos failed; fix before continuing"
    econ = pd.DataFrame(rows)

    unknown = (set(econ["ct_buy_label"]) | set(econ["t_buy_label"])) - set(BUY)
    assert not unknown, f"unexpected buy labels: {unknown}"
    econ["ct_buy"], econ["t_buy"] = econ["ct_buy_label"].map(BUY), econ["t_buy_label"].map(BUY)
    econ["is_pistol"] = econ["round"].isin(PISTOL_ROUNDS)
    econ["is_overtime"] = econ["round"] > 30

    with open(ALIASES_PATH) as f:
        aliases = json.load(f)
    unmapped = (set(econ["ct_team_raw"]) | set(econ["t_team_raw"])) - set(aliases)
    assert not unmapped, f"unmapped team names: {sorted(unmapped)}"
    econ["ct_team"], econ["t_team"] = econ["ct_team_raw"].map(aliases), econ["t_team_raw"].map(aliases)
    assert not econ.duplicated(["match_id", "round"]).any(), "duplicate (match_id, round)"
    return econ, aliases


def validate(econ, aliases):
    section("1. equipment-value bands by label (team total at freeze end)")
    for side in ("ct", "t"):
        g = econ.groupby(f"{side}_buy")[f"{side}_eq"].agg(["count", "min", "median", "max"]).reindex(ORDER)
        print(f"\n{side.upper()}:\n", g.round(0).to_string())
        overlap = [(a, b) for a, b in zip(ORDER, ORDER[1:]) if g.loc[a, "max"] >= g.loc[b, "min"]]
        edges = [(g.loc[a, "max"] + g.loc[b, "min"]) / 2 for a, b in zip(ORDER, ORDER[1:])]
        print(f"overlapping adjacent bands: {overlap or 'none'} | implied thresholds ~ "
              f"{', '.join(f'${e:,.0f}' for e in edges)}")

    section("2. pistol rounds")
    pist = econ[econ["is_pistol"]]
    print(f"{len(pist)} pistol rounds; labels (both sides):\n",
          pd.concat([pist["ct_buy"], pist["t_buy"]]).value_counts().to_string())
    npist = econ[~econ["is_pistol"]]
    print(f"non-pistol rounds: {len(npist)} | overtime rounds (>30): {int(econ['is_overtime'].sum())}")

    section("3. agreement with round_metadata")
    rm = pd.concat((pd.read_csv(p) for p in glob.glob(RM_GLOB)), ignore_index=True)
    rm = rm[["match_id", "round", "winning_side", "ct_team", "t_team"]].rename(
        columns={"winning_side": "ws_rm", "ct_team": "ct_rm", "t_team": "t_rm"})
    rm["ct_rm"], rm["t_rm"] = rm["ct_rm"].map(aliases), rm["t_rm"].map(aliases)
    m = econ.merge(rm, on=["match_id", "round"], how="outer", indicator=True)
    print("row coverage:\n", m["_merge"].value_counts().to_string())
    both = m[m["_merge"] == "both"]
    print(f"winning_side mismatches: {int((both['winning_side'] != both['ws_rm']).sum())} | "
          f"ct_team mismatches: {int((both['ct_team'] != both['ct_rm']).sum())} | "
          f"t_team mismatches: {int((both['t_team'] != both['t_rm']).sum())}")

    section("4. label shares and T win rate by matchup (non-pistol, all matches)")
    print("share of non-pistol team-rounds by buy class:")
    print(pd.DataFrame({"CT": npist["ct_buy"].value_counts(normalize=True),
                        "T": npist["t_buy"].value_counts(normalize=True)}).reindex(ORDER).round(3).to_string())
    print("\nrounds (rows = T buy, cols = CT buy):\n",
          pd.crosstab(npist["t_buy"], npist["ct_buy"]).reindex(index=ORDER, columns=ORDER).to_string())
    win = npist.assign(t_win=(npist["winning_side"] == "T").astype(float)).pivot_table(
        index="t_buy", columns="ct_buy", values="t_win", aggfunc="mean").reindex(index=ORDER, columns=ORDER)
    print("\nT win rate:\n", win.round(2).to_string())


def feasibility(econ):
    section("5. feasibility of the eco vs full-buy contrasts")
    if not os.path.exists(SIG_PATH):
        print(f"{SIG_PATH} not found; skipped")
        return
    sig = pd.read_parquet(SIG_PATH, columns=["match_id", "round", "side", "window", "method"])
    sig = sig[sig["method"] == "field"].drop(columns="method").drop_duplicates()
    parts = []
    for side, own, opp, team in (("T", "t_buy", "ct_buy", "t_team"), ("CT", "ct_buy", "t_buy", "ct_team")):
        d = econ.loc[~econ["is_pistol"], ["match_id", "round", own, opp, team]]
        parts.append(d.rename(columns={own: "own", opp: "opp", team: "team"}).assign(side=side))
    long = pd.concat(parts, ignore_index=True)

    for window in ("pre_first_loss", "early_5_20"):
        have = sig[sig["window"] == window][["match_id", "round", "side"]].assign(usable=True)
        w = long.merge(have, on=["match_id", "round", "side"], how="left")
        w["usable"] = w["usable"].eq(True)
        print(f"\n--- window {window} ---")
        sel = w.groupby("own")["usable"].agg(rounds="size", usable_rate="mean").reindex(ORDER)
        print("selection effect: share of non-pistol team-rounds with a usable window, by OWN buy class:")
        print(sel.round(3).to_string())

        E = w[(w["own"] == "eco") & (w["opp"] == "full")]
        F = w[(w["own"] == "full") & (w["opp"] == "full")]
        A = w[(w["own"] == "full") & (w["opp"] == "eco")]
        for side in ("T", "CT"):
            e, f, a = (x[x["side"] == side] for x in (E, F, A))
            print(f"\n{side}: eco vs full-buy opp {len(e)} rounds (usable {int(e['usable'].sum())}) | "
                  f"full vs full {len(f)} (usable {int(f['usable'].sum())}) | "
                  f"anti-eco (full vs eco opp) {len(a)} (usable {int(a['usable'].sum())})")
            eu = e[e["usable"]].groupby("team").size().rename("eco")
            fu = f[f["usable"]].groupby("team").size().rename("full")
            tc = pd.concat([eu, fu], axis=1).fillna(0)
            print("  teams with >= k usable rounds in BOTH eco and full:",
                  {k: int(((tc["eco"] >= k) & (tc["full"] >= k)).sum()) for k in MIN_ROUND_COUNTS})
            ue = e[e["usable"]].groupby(["team", "match_id"]).size().rename("eco")
            uf = f[f["usable"]].groupby(["team", "match_id"]).size().rename("full")
            u = pd.concat([ue, uf], axis=1).dropna()
            print(f"  team-matches with BOTH conditions (the within-match permutation units): {len(u)} "
                  f"| eco rounds in them {int(u['eco'].sum())}, full rounds {int(u['full'].sum())} "
                  f"| teams contributing {u.index.get_level_values(0).nunique()}")


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    econ, aliases = build_table()
    cols = ["match_name", "match_id", "round", "is_pistol", "is_overtime", "ct_team", "t_team",
            "ct_buy", "t_buy", "ct_buy_label", "t_buy_label", "ct_eq", "t_eq", "ct_eq_start", "t_eq_start",
            "ct_spend", "t_spend", "ct_score", "t_score", "winning_side", "round_end_reason", "bomb_plant_tick"]
    econ[cols].to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}: {len(econ)} rounds from {econ['match_id'].nunique()} matches")
    validate(econ, aliases)
    feasibility(econ)


if __name__ == "__main__":
    main()
