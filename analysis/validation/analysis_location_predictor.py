"""
analysis_location_predictor.py   (location-aware predictor: target = CT WINS THE ROUND)
=====================================================================================
Does WHERE control is held predict the round winner beyond economy?
Rows: one per eligible round (pre-first-kill window, see build_window_features.py). The engine (models, nested CV,
bootstrap, tables) lives in location_predictor_lib.py; this script only decides the rows, the primary tests and
the output directory.

MODELS (same rows, same folds from folds.csv)
    M_const    no features: training-fold CT win rate
    M0         economy only: log_eq_ratio, eq_diff_k, is_pistol   StandardScaler + LogisticRegression(C=1)
    M_agg      one scalar: window-mean active-footprint control (ca_win);   M0+M_agg: both
    M1 / M2    per-AREA / per-NODE net control, ridge logistic                    [location only]
    M0+M1 / M0+M2   economy + location: M0's logit is a fixed OFFSET, only location is penalised
    --presence adds M1p, M2p, M0+M1p, M0+M2p (net and presence as separate scaled blocks)
        net      = window mean of (control_val - 0.5)   directional, bounded +-0.5
        presence = window mean of (ct_mu + t_mu)         non-directional total influence
PRIMARY tests (Holm-adjusted together): M0+M1 vs M0 and M0+M2 vs M0. Everything else is descriptive.

BASELINE REPRODUCTION (on by default): old pre-engagement sample (keep_legacy), sklearn GroupKFold(5) by match;
the old headline AUC was ~0.716.

Subset option (is location just a proxy for economy?):
    --econ-band 0.15 --no-pistol   keeps only non-pistol rounds with similar buys (|log_eq_ratio| <= 0.15);
                                   output goes to <out-dir>_econband0.15_nopistol/

Outputs (outputs/metrics/location_predictor/ by default): model_scores.csv, model_comparisons.csv, lambda_log.csv,
oof_predictions.parquet, area_coefficients.csv, node_coefficients.csv, run_meta.json, figures/delta_logloss.png

Usage (project root):
    python analysis/validation/analysis_location_predictor.py
    python analysis/validation/analysis_location_predictor.py --grouping teampair --presence
    python analysis/validation/analysis_location_predictor.py --econ-band 0.15 --no-pistol --presence
    python analysis/validation/analysis_location_predictor.py --feat-dir outputs/predictor_nocap --out-dir outputs/metrics/location_predictor_nocap
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402
import location_predictor_lib as lib  # noqa: E402

PRIMARY = [("M0+M1", "M0"), ("M0+M2", "M0")]


def reproduce_baseline(D: lib.Data) -> None:
    vc.log("\nBASELINE REPRODUCTION (M0, GroupKFold(5) by match, StandardScaler + LogisticRegression C=1)")
    ra = D.rounds_all
    for label, mask in (("legacy sample (old cap-30 headline)", ra["keep_legacy"].astype(bool)),
                        ("new sample (this analysis)", ra["eligible"].astype(bool))):
        d = ra[mask].dropna(subset=["y_ct_win"] + lib.ECON)
        X, y, g = d[lib.ECON].to_numpy(float), d["y_ct_win"].to_numpy(int), d["match_name"].to_numpy()
        p = np.zeros(len(d))
        for a, b in GroupKFold(5).split(X, y, g):
            p[b] = lib.econ_pipe().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        vc.log(f"  {label:<38} n={len(d):>5}  AUC {roc_auc_score(y, p):.4f}  "
               f"logloss {lib.row_ll(y, p).mean():.4f}  Brier {brier_score_loss(y, p):.4f}")
    vc.log("  (the old pre-engagement headline AUC was ~0.716; a clearly different legacy value means the "
           "baseline spec or sample differs from the earlier run)")


def make_filter(max_abs_log_eq, no_pistol):
    if max_abs_log_eq is None and not no_pistol:
        return None

    def f(el: pd.DataFrame) -> pd.DataFrame:
        if no_pistol:
            el = el[el["is_pistol"] == 0]
        if max_abs_log_eq is not None:
            el = el[el["log_eq_ratio"].abs() <= max_abs_log_eq]
        return el
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat-dir", default="outputs/predictor")
    ap.add_argument("--folds", default=None, help="default: <feat-dir>/folds.csv")
    ap.add_argument("--grouping", choices=["match", "teampair"], default="match")
    ap.add_argument("--repeats", type=int, default=None, help="default: all repeats in folds.csv")
    ap.add_argument("--n-boot", type=int, default=vc.DEFAULT_BOOT)
    ap.add_argument("--presence", action="store_true", help="add the net+presence rows")
    ap.add_argument("--no-reproduce", action="store_true")
    ap.add_argument("--econ-band", type=float, default=None,
                    help="keep only rounds with |log_eq_ratio| <= this (balanced-economy subset)")
    ap.add_argument("--no-pistol", action="store_true", help="drop pistol rounds")
    ap.add_argument("--out-dir", default="outputs/metrics/location_predictor")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feat = Path(args.feat_dir)
    suffix = "" if args.grouping == "match" else "_teampair"
    if args.econ_band is not None or args.no_pistol:
        suffix += "_econband" + (f"{args.econ_band:g}" if args.econ_band is not None else "") + \
                  ("_nopistol" if args.no_pistol else "")
    out = Path(args.out_dir + suffix)
    meta = json.loads((feat / "window_meta.json").read_text())
    vc.log(f"window: t0={meta['t0']}s cap={meta['cap']}s min_len={meta['min_len']}s; "
           f"{meta['n_rounds']} eligible rounds, {meta['n_nodes']} nodes, {len(meta['areas'])} areas")

    D = lib.load_data(feat, args.presence, round_filter=make_filter(args.econ_band, args.no_pistol))
    if args.econ_band is not None or args.no_pistol:
        vc.log(f"SUBSET: |log_eq_ratio| <= {args.econ_band}, no_pistol={args.no_pistol} -> {D.n} rounds "
               f"(M0 should have little left to explain here; that is the point)")
    folds = pd.read_csv(args.folds or feat / "folds.csv")
    repeats = args.repeats or int(folds["repeat"].nunique())
    vc.log(f"{D.n} rounds, {len(D.mnames)} matches; grouping={args.grouping}; {repeats} repeats x "
           f"{folds['fold_match'].nunique()} folds; CT win rate {D.y.mean():.3f}")

    if not args.no_reproduce:
        reproduce_baseline(D)

    lib.run_grid(D, folds, args.grouping, repeats, out, PRIMARY, presence=args.presence, n_boot=args.n_boot,
                 seed=args.seed, y_col="y_ct_win", record_prefix="locpred",
                 meta=dict(target="win", feat_dir=str(feat), econ_band=args.econ_band, no_pistol=args.no_pistol,
                           window=meta))


if __name__ == "__main__":
    main()
