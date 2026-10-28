"""
analysis_location_site_team.py   (site choice: does location add beyond TEAM HABIT?)
===================================================================================
Early positions predict the planted site surprisingly well, even in the first 2 s after freeze end. Two readings:
    (1) positions encode THIS round's plan (pre-round positioning during freeze time), or
    (2) positions identify the team / player slot, and teams have habitual site preferences (a confound).
Neither match grouping nor team-pair grouping removes (2): a team recurs in training and test folds against other
opponents. This script controls for it directly.

MODELS (same rows and same folds as analysis_location_site.py)
    M_const    A/B base rate
    M_team     shrunken per-team B rate of the T-side team, from TRAINING rows only
               p = (B_team + a * base) / (n_team + a), a = --prior-strength (default 10); unseen teams -> base rate
    M1         area net control (ridge logistic, nested lambda)
    M_team+M1  location as a ridge correction on top of M_team: M_team's logit is a fixed OFFSET; training rows use a
               leave-one-MATCH-out prior so a row's prior never contains its own label or its match-mates'
    --presence adds M1p and M_team+M1p (net and presence as separate scaled blocks)

TESTS
    PRIMARY   M_team+M1 vs M_team   (does location add beyond team habit?)
    descriptive: M_team vs M_const (how much does team habit alone explain?), M1 vs M_const, M1 vs M_team,
                 M1p vs M1 and M_team+M1p vs M_team+M1 when --presence is set.

TIMING OPTION
    --tc-dir outputs/predictor_timecourse_t0 --cap 2   runs the same comparison on the area features of ONE window
    end from build_site_timecourse_features.py (e.g. the first 2 s after freeze end), on the same fixed rounds.
    Team names are taken from <feat-dir>/window_rounds.csv (column t_team = the team playing T that round).

READING IT
    * If M_team+M1 vs M_team is clearly positive, location carries round-specific information beyond team habit.
    * If it is ~0 while M1 beats M_const, the early-window signal was mostly team habit (positions identify teams).
    * M_team is a weak-ish baseline (shrunken rate, ~11 matches per team on average), so a residual location effect
      can still partly be habit that the prior misses; the 2 s result is the sharpest test.
    * AUC of near-constant models is unreliable (fold-wise intercept artefact); read log-loss.

Outputs: same files as analysis_location_site.py, in outputs/metrics/location_site_team[_cap<c>][_teampair]/

Usage (project root):
    python analysis/validation/analysis_location_site_team.py --presence
    python analysis/validation/analysis_location_site_team.py --tc-dir outputs/predictor_timecourse_t0 --cap 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402
import location_predictor_lib as lib  # noqa: E402
import analysis_location_site as als  # noqa: E402
import analysis_location_site_timecourse as tcs  # noqa: E402

PRIMARY_TEAM = [("M_team+M1", "M_team")]


def build_pairs(presence: bool) -> list:
    pairs = list(PRIMARY_TEAM) + [("M_team", "M_const"), ("M1", "M_const"), ("M1", "M_team")]
    if presence:
        pairs += [("M1p", "M1"), ("M_team+M1p", "M_team+M1"), ("M_team+M1p", "M_team")]
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat-dir", default="outputs/predictor")
    ap.add_argument("--folds", default=None, help="default: <feat-dir>/folds.csv")
    ap.add_argument("--grouping", choices=["match", "teampair"], default="match")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=vc.DEFAULT_BOOT)
    ap.add_argument("--presence", action="store_true")
    ap.add_argument("--bomb-dir", default="outputs/bomb_plants")
    ap.add_argument("--min-lead", type=float, default=10.0)
    ap.add_argument("--prior-strength", type=float, default=10.0, help="shrinkage a of the team prior")
    ap.add_argument("--tc-dir", default=None, help="use the features of ONE window end from the timecourse builder")
    ap.add_argument("--cap", type=float, default=None, help="window end (s) to use with --tc-dir")
    ap.add_argument("--out-dir", default="outputs/metrics/location_site_team")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feat = Path(args.feat_dir)
    suffix = (f"_cap{args.cap:g}" if args.tc_dir else "") + ("" if args.grouping == "match" else "_teampair")
    out = Path(args.out_dir + suffix)
    ref = pd.read_csv(feat / "window_rounds.csv")
    if "t_team" not in ref.columns:
        raise SystemExit("window_rounds.csv has no t_team column; rebuild the features with team names "
                         "(round table needs ct_team/t_team)")

    if args.tc_dir:
        if args.cap is None:
            raise SystemExit("--tc-dir needs --cap")
        tc = Path(args.tc_dir)
        rounds = pd.read_csv(tc / "rounds.csv").merge(ref[["rk", "t_team"]], on="rk", how="left")
        D = tcs.make_data(rounds, tc, args.cap, args.presence)
        vc.log(f"window: [{json.loads((tc / 'timecourse_meta.json').read_text())['t0']:g}, {args.cap:g}] s from {tc}")
    else:
        D = lib.load_data(feat, args.presence, round_filter=als.make_filter(Path(args.bomb_dir), args.min_lead),
                          label_col="y_site_B")
    teams = D.el["t_team"].fillna("unknown").astype(str)
    D.tcodes, tnames = pd.factorize(teams)
    D.prior_strength = args.prior_strength
    rate = D.el.groupby(teams)["y_site_B"].agg(["mean", "size"])
    vc.log(f"{D.n} rounds, {len(D.mnames)} matches, {len(tnames)} T-side teams; share of plants at B {D.y.mean():.3f}; "
           f"team B-rate spread (teams with >= 30 rounds): "
           f"{rate[rate['size'] >= 30]['mean'].min():.2f}-{rate[rate['size'] >= 30]['mean'].max():.2f}; "
           f"prior strength a = {args.prior_strength:g}")

    specs = {
        "M_const": dict(kind="const"),
        "M_team": dict(kind="team"),
        "M1": dict(kind="ridge", loc=["area_net"], offset=False),
        "M_team+M1": dict(kind="ridge", loc=["area_net"], offset="team"),
    }
    if args.presence:
        specs["M1p"] = dict(kind="ridge", loc=["area_net", "area_pres"], offset=False)
        specs["M_team+M1p"] = dict(kind="ridge", loc=["area_net", "area_pres"], offset="team")

    folds = pd.read_csv(args.folds or feat / "folds.csv")
    repeats = args.repeats or int(folds["repeat"].nunique())
    lib.run_grid(D, folds, args.grouping, repeats, out, PRIMARY_TEAM, presence=args.presence, n_boot=args.n_boot,
                 seed=args.seed, y_col="y_site_B", record_prefix="locpred_site_team", specs=specs,
                 pairs=build_pairs(args.presence),
                 meta=dict(target="site", control="team prior", prior_strength=args.prior_strength,
                           tc_dir=args.tc_dir, cap=args.cap, feat_dir=str(feat)))


if __name__ == "__main__":
    main()
