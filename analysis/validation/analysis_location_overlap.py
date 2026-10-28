"""
analysis_location_overlap.py   (is location just a proxy for economy?)
=====================================================================
Location-only models (M1, M2) predict the round winner, yet add nothing once economy is in the model.
Two ways to check whether that is because location mostly encodes the buy state:

PART A  overlap of out-of-fold predictions (needs only the output of analysis_location_predictor.py)
    * correlation of logit(p_M0) with logit(p_location-model)            (how alike the two predictors are)
    * one-parameter STACKING test:   y ~ b0 + b1*logit(p_M0) + c*logit(p_loc)
      c is the information the location model carries BEYOND economy. 95% CI by match-bootstrap, plus the
      log-loss gain over  y ~ b0 + b1*logit(p_M0)  (in-sample, 3 parameters on out-of-fold predictions).
      Much more powerful than comparing a 900-feature model, because only ONE parameter is tested.

PART B  predict economy FROM location (ridge, same match-grouped folds)
    CV R^2 of log_eq_ratio and eq_diff_k from area net / node net. High R^2 = location encodes the buy state.

Outputs (<run-dir>/overlap/): stacking.csv, economy_from_location.csv

Usage (project root):
    python analysis/validation/analysis_location_overlap.py
    python analysis/validation/analysis_location_overlap.py --run-dir outputs/metrics/location_predictor_nocap \\
                                                           --feat-dir outputs/predictor_nocap
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402
import location_predictor_lib as lib  # noqa: E402

LOC_MODELS = ["M1", "M2", "M1p", "M2p"]
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _fit_ll(X, y):
    # C=1e10 ~ unpenalised; avoids the penalty=None keyword, whose spelling changed across sklearn versions
    m = LogisticRegression(C=1e10, max_iter=1000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], EPS, 1 - EPS)
    return m, -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def stacking(oof: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    y = oof["y_ct_win"].to_numpy(int)
    L0 = logit(oof["p_M0"].to_numpy())
    codes, uniq = pd.factorize(oof["match_name"].to_numpy())
    G = len(uniq)
    rows_by_match = [np.flatnonzero(codes == g) for g in range(G)]
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([rows_by_match[g] for g in rng.integers(0, G, G)]) for _ in range(n_boot)]
    out = []
    for name in LOC_MODELS:
        if f"p_{name}" not in oof.columns:
            continue
        Lm = logit(oof[f"p_{name}"].to_numpy())
        X2, X1 = np.column_stack([L0, Lm]), L0[:, None]
        m2, ll2 = _fit_ll(X2, y)
        _, ll1 = _fit_ll(X1, y)
        c_b, g_b = [], []
        for idx in boots:
            mm, l2 = _fit_ll(X2[idx], y[idx])
            _, l1 = _fit_ll(X1[idx], y[idx])
            c_b.append(mm.coef_[0, 1])
            g_b.append(l1 - l2)
        out.append(dict(
            location_model=name,
            corr_logit_with_M0=np.corrcoef(L0, Lm)[0, 1],
            spearman_with_M0=spearmanr(L0, Lm)[0],
            stack_coef_loc=m2.coef_[0, 1], stack_coef_loc_lo=np.percentile(c_b, 2.5), stack_coef_loc_hi=np.percentile(c_b, 97.5),
            stack_coef_M0=m2.coef_[0, 0],
            logloss_gain_over_M0=ll1 - ll2, gain_lo=np.percentile(g_b, 2.5), gain_hi=np.percentile(g_b, 97.5)))
    return pd.DataFrame(out)


def economy_from_location(D, folds: pd.DataFrame, repeats: int) -> pd.DataFrame:
    # Some numpy/BLAS builds emit a spurious "invalid value encountered in matmul" warning on every
    # ridge solve. Real NaNs are caught by the explicit finiteness checks below instead.
    warnings.filterwarnings("ignore", message="invalid value encountered in matmul", category=RuntimeWarning)
    for k in ("area_net", "node_net"):
        assert np.isfinite(D.mats[k]).all(), f"non-finite values in {k}"
    rows = []
    targets = {"log_eq_ratio": D.mats["econ"][:, 0], "eq_diff_k": D.mats["econ"][:, 1]}
    for feat_name, key in (("area_net", "area_net"), ("node_net", "node_net")):
        for tname, t in targets.items():
            r2s = []
            for r in range(repeats):
                fmap = folds[folds["repeat"] == r].set_index("match_name")["fold_match"]
                fold = D.el["match_name"].map(fmap).to_numpy()
                pred = np.zeros(D.n)
                for f in np.unique(fold):
                    te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
                    T = lib.prep_blocks([D.mats[key]], tr)
                    gs = GridSearchCV(Ridge(), {"alpha": np.logspace(-1, 5, 13)},
                                      cv=GroupKFold(3), scoring="neg_mean_squared_error")
                    Xtr_, Xte_ = T(tr), T(te)
                    assert np.isfinite(Xtr_).all() and np.isfinite(Xte_).all(), "non-finite scaled features"
                    gs.fit(Xtr_, t[tr], groups=D.mcodes[tr])
                    pred[te] = gs.predict(Xte_)
                    assert np.isfinite(pred[te]).all(), "non-finite predictions"
                r2s.append(1 - np.sum((t - pred) ** 2) / np.sum((t - t.mean()) ** 2))
            rows.append(dict(features=feat_name, target=tname, cv_r2=np.mean(r2s), cv_r2_sd_across_repeats=np.std(r2s)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="outputs/metrics/location_predictor")
    ap.add_argument("--feat-dir", default="outputs/predictor")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=2, help="fold repeats for Part B (it refits ridge many times)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-part-b", action="store_true")
    args = ap.parse_args()

    run, feat = Path(args.run_dir), Path(args.feat_dir)
    out = run / "overlap"
    vc.ensure_dir(out)

    oof = pd.read_parquet(run / "oof_predictions.parquet")
    vc.log(f"PART A: overlap of out-of-fold predictions ({len(oof)} rounds, {oof['match_name'].nunique()} matches)")
    st = stacking(oof, args.n_boot, args.seed)
    st.to_csv(out / "stacking.csv", index=False)
    for _, r in st.iterrows():
        vc.log(f"  {r['location_model']:<4} corr(logit) with M0 {r['corr_logit_with_M0']:.3f} (Spearman {r['spearman_with_M0']:.3f}) | "
               f"stack coef on location {r['stack_coef_loc']:+.3f} [{r['stack_coef_loc_lo']:+.3f},{r['stack_coef_loc_hi']:+.3f}] "
               f"(coef on M0 {r['stack_coef_M0']:.3f}) | logloss gain over M0 {r['logloss_gain_over_M0']:+.5f} "
               f"[{r['gain_lo']:+.5f},{r['gain_hi']:+.5f}]")
    vc.log("  read: coef ~ 0 with CI around 0 -> the location model adds nothing to economy; "
           "coef clearly > 0 -> independent information")

    if not args.no_part_b:
        vc.log("\nPART B: predict economy FROM location (match-grouped CV ridge)")
        D = lib.load_data(feat, with_presence=False)
        folds = pd.read_csv(feat / "folds.csv")
        reps = min(args.repeats, int(folds["repeat"].nunique()))
        eb = economy_from_location(D, folds, reps)
        eb.to_csv(out / "economy_from_location.csv", index=False)
        for _, r in eb.iterrows():
            vc.log(f"  {r['features']:<9} -> {r['target']:<13} CV R^2 {r['cv_r2']:+.3f} (sd across repeats {r['cv_r2_sd_across_repeats']:.3f})")
        vc.log("  read: a high R^2 means location mostly encodes the buy state, so it has little left to add")
    vc.log(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
