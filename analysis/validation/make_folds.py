"""
make_folds.py   (location-aware predictor, step 2 of 3)
=======================================================
Create the fold assignments ONCE so every model in the grid is scored on identical splits.

Two groupings, each repeated R times with different random seeds:
    fold_match     whole matches are assigned to folds (PRIMARY grouping)
    fold_teampair  all matches between the same two teams go to the same fold (SENSITIVITY: stops a
                   team's habits learned in training matches from helping on its other matches)

Assignment is randomised greedy balancing: groups are visited in random order and each goes to the fold
that currently holds the fewest ROUNDS (ties broken at random), so folds are balanced by rounds while
still differing between repeats. Folds are defined on ALL rounds in window_rounds.csv, so the same table
serves the old (legacy) and the new sample; rows simply inherit their match's fold.

Output: outputs/predictor/folds.csv
    match_name, n_rounds, team_pair, repeat, fold_match, fold_teampair
(team_pair / fold_teampair are blank if the round table has no ct_team/t_team columns)

Usage:  python analysis/validation/make_folds.py [--k 5] [--repeats 5] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402


def assign_groups(sizes: pd.Series, k: int, rng: np.random.Generator) -> dict:
    """sizes: index = group id, values = number of rounds. Returns {group: fold}."""
    load = np.zeros(k)
    out = {}
    for g in rng.permutation(sizes.index.to_numpy()):
        lightest = np.flatnonzero(load == load.min())
        f = int(rng.choice(lightest))
        out[g] = f
        load[f] += sizes[g]
    return out


def team_pairs(rounds: pd.DataFrame) -> pd.Series:
    """match_name -> 'teamA|teamB' (sorted), from ct_team / t_team across the match's rounds.
    Written without groupby.apply so it works on older pandas versions too."""
    long = pd.concat([
        rounds[["match_name", "ct_team"]].rename(columns={"ct_team": "team"}),
        rounds[["match_name", "t_team"]].rename(columns={"t_team": "team"}),
    ]).dropna()
    return long.groupby("match_name")["team"].agg(lambda s: "|".join(sorted(set(s))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", default="outputs/predictor/window_rounds.csv")
    ap.add_argument("--out", default="outputs/predictor/folds.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rounds = pd.read_csv(args.rounds)
    sizes = rounds.groupby("match_name").size().rename("n_rounds")
    base = sizes.reset_index()
    has_team = {"ct_team", "t_team"} <= set(rounds.columns)
    if has_team:
        tp = team_pairs(rounds)
        base["team_pair"] = base["match_name"].map(tp)
        pair_sizes = base.groupby("team_pair")["n_rounds"].sum()
        vc.log(f"{len(base)} matches, {len(pair_sizes)} distinct team pairs "
               f"({(base.groupby('team_pair').size() > 1).sum()} pairs appear in more than one match)")
        bad = base[~base["team_pair"].str.contains(r"\|", regex=True)]
        if len(bad):
            vc.log(f"  WARNING: {len(bad)} matches do not have exactly two teams recorded "
                   f"(e.g. {bad['match_name'].iloc[0]}); they are grouped by whatever teams exist.")
    else:
        base["team_pair"] = ""
        vc.log(f"{len(base)} matches; no ct_team/t_team columns -> fold_teampair left blank")

    rows = []
    for r in range(args.repeats):
        rng = np.random.default_rng(args.seed + r)
        fm = assign_groups(sizes, args.k, rng)
        df = base.copy()
        df["repeat"] = r
        df["fold_match"] = df["match_name"].map(fm)
        if has_team:
            ft = assign_groups(pair_sizes, args.k, rng)
            df["fold_teampair"] = df["team_pair"].map(ft)
        else:
            df["fold_teampair"] = np.nan
        rows.append(df)
    folds = pd.concat(rows, ignore_index=True)

    # ---- checks
    vc.log("\nVERIFICATION")
    for r, d in folds.groupby("repeat"):
        assert d["match_name"].is_unique and d["fold_match"].notna().all()
        load = d.groupby("fold_match")["n_rounds"].sum()
        msg = f"  repeat {r}: rounds per fold (match grouping) min {load.min()} max {load.max()}"
        if has_team:
            assert (d.groupby("team_pair")["fold_teampair"].nunique() == 1).all(), "a team pair spans folds"
            lt = d.groupby("fold_teampair")["n_rounds"].sum()
            msg += f" | (team-pair grouping) min {lt.min()} max {lt.max()}"
        vc.log(msg)
    same = folds.pivot(index="match_name", columns="repeat", values="fold_match").nunique(axis=1)
    vc.log(f"  matches whose fold changes across repeats: {(same > 1).mean():.0%} (should be high)")
    vc.log("  every match appears exactly once per repeat; no match or team pair spans two folds -> OK")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.out, index=False)
    vc.log(f"\nWrote {args.out}  ({len(folds)} rows = {len(base)} matches x {args.repeats} repeats)")


if __name__ == "__main__":
    main()
