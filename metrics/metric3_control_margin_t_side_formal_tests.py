"""
Metric 3 (T-side) — Formal Comparison of Margin Definitions

Follow-up to metric3_control_margin_t_side.py. That script's
"effect_size_comparison" block in _summary.json puts two Cohen's d point
estimates side by side (control_margin_active_t vs
control_margin_bombsite_active_t) with no sampling distribution attached
to their DIFFERENCE, and treats them as if independent when they are in
fact measured on the same rounds (paired/dependent data). This script
implements two properly formal alternatives:

-------------------------------------------------------------------------
Test 1 — Nested logistic regression + likelihood-ratio test
-------------------------------------------------------------------------
    Model A:  logit(P(T win)) = b0 + b1 * margin_active
    Model B:  logit(P(T win)) = b0 + b1 * margin_active + b2 * margin_bombsite_active

Both fit on the identical set of rounds (required for a valid nested LR
test — a model cannot be validly compared to a "sub-model" fit on a
different sample). The LR statistic is 2*(llf_B - llf_A), chi-square
distributed with df = 1 (one extra parameter), and directly answers "does
bombsite control explain anything beyond whole-active-footprint control".

Model B is additionally refit with cluster-robust ("sandwich") standard
errors, clustering on match_id, because rounds within the same match are
not independent observations (best-of-X map veto pools, team strength,
etc. are shared within a match) — the plain LR test above does not
correct for this, so its p-value should be read as approximate/optimistic.
The cluster-robust fit does not change the LR statistic itself (that
requires the actual likelihood, which cluster-robust SEs do not provide a
corrected version of), but it gives a more honest per-coefficient
significance test. If you need a fully cluster-corrected LR-style test,
the standard next step beyond what is implemented here is a mixed-effects
/ hierarchical logistic model (random intercept per match_id) compared via
a bootstrap or simulation-based LR test — not implemented here to avoid
adding a second heavy dependency (this script only requires statsmodels).

-------------------------------------------------------------------------
Test 2 — DeLong's test for two correlated ROC curves
-------------------------------------------------------------------------
Treats control_margin_active_t and control_margin_bombsite_active_t as
two competing classifier scores for the same binary outcome (T win vs T
loss) on the same rounds, computes each one's AUC, and tests whether the
difference between the two AUCs is distinguishable from zero while
correctly accounting for the correlation between the two scores (since
they come from the same rounds). This is the standard tool (DeLong,
DeLong & Clarke-Pearson, 1988) for "is classifier A significantly better
than classifier B on the same sample" and needs no modelling assumptions
beyond the AUC's rank-based definition.

The implementation here is the direct (non-fast) DeLong estimator: for
n_win positive cases (T win) and n_loss negative cases (T loss), it is
O(n_win * n_loss) via numpy broadcasting, which is fine at this dataset's
scale (a few thousand rounds -> a few tens of millions of comparisons,
well within memory/time budget for a single run).

-------------------------------------------------------------------------
Inputs
-------------------------------------------------------------------------
    outputs/metrics/control_margin_t_side/{match_name}.csv
        (produced by metric3_control_margin_t_side.py)

-------------------------------------------------------------------------
Outputs
-------------------------------------------------------------------------
    outputs/metrics/control_margin_t_side/_formal_comparison.json
        Full results of both tests, plus the input row counts used by
        each (they can differ slightly from _summary.json's counts,
        since both tests here require non-null values in BOTH margin
        columns simultaneously, not just the one column each margin's
        own Welch/Cohen's-d test needs).

-------------------------------------------------------------------------
Dependency note
-------------------------------------------------------------------------
This script requires statsmodels, which is NOT part of the project's
existing core stack (awpy, networkx, scipy, numpy, pandas, pyarrow,
matplotlib, imageio). Install with:

    pip install statsmodels --break-system-packages
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "This script requires statsmodels, which is not part of the "
        "project's existing dependencies. Install it with:\n"
        "    pip install statsmodels --break-system-packages"
    ) from e

# --/ Path consts — must match metric3_control_margin_t_side.py
METRIC_DIR = "outputs/metrics/control_margin_t_side"
FORMAL_COMPARISON_PATH = os.path.join(METRIC_DIR, "_formal_comparison.json")

MARGIN_A_COL = "control_margin_active_t"
MARGIN_B_COL = "control_margin_bombsite_active_t"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_all_rounds() -> pd.DataFrame:
    """Load every per-match round-level CSV, skipping the underscore-
    prefixed dataset-level files. Mirrors the loader in
    visualize_metric3_control_margin_t_side.py.
    """
    per_match_files = sorted(
        p
        for p in glob.glob(os.path.join(METRIC_DIR, "*.csv"))
        if not os.path.basename(p).startswith("_")
    )
    if not per_match_files:
        raise FileNotFoundError(
            f"No per-match round CSVs found in {METRIC_DIR} — "
            f"run metric3_control_margin_t_side.py first"
        )
    df = pd.concat((pd.read_csv(p) for p in per_match_files), ignore_index=True)
    df["winning_side"] = df["winning_side"].astype(str).str.upper()
    df = df[df["winning_side"].isin(["CT", "T"])]
    return df


def _prepare_paired_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rows valid for BOTH tests: non-null match_id, both margin columns,
    and a valid outcome. Both tests require identical rows across their
    two predictors/classifiers to be a fair paired comparison.
    """
    required = ["match_id", "winning_side", MARGIN_A_COL, MARGIN_B_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Required column(s) missing from the loaded data: {missing}. "
            f"Was metric3_control_margin_t_side.py run against control_fields "
            f"CSVs that included ct_mu/t_mu? (control_margin_bombsite_active_t "
            f"is only produced when those columns are present.)"
        )
    out = df.dropna(subset=required).copy()
    out["y_t_win"] = (out["winning_side"] == "T").astype(int)
    return out


# ---------------------------------------------------------------------------
# Test 1 — nested logistic regression + LR test
# ---------------------------------------------------------------------------
def run_nested_logistic_test(data: pd.DataFrame) -> dict:
    """Model A: y ~ margin_active
    Model B: y ~ margin_active + margin_bombsite_active
    Both z-scored (using Model B's fitting sample) so coefficients are on
    a comparable, interpretable scale; z-scoring does not change the LR
    statistic, only coefficient interpretation.
    """
    y = data["y_t_win"].to_numpy()

    margin_a = data[MARGIN_A_COL].to_numpy()
    margin_b = data[MARGIN_B_COL].to_numpy()

    margin_a_z = (margin_a - margin_a.mean()) / margin_a.std(ddof=0)
    margin_b_z = (margin_b - margin_b.mean()) / margin_b.std(ddof=0)

    X_a = sm.add_constant(margin_a_z)
    X_b = sm.add_constant(np.column_stack([margin_a_z, margin_b_z]))

    model_a = sm.Logit(y, X_a).fit(disp=False)
    model_b = sm.Logit(y, X_b).fit(disp=False)

    lr_stat = 2 * (model_b.llf - model_a.llf)
    df_diff = int(model_b.df_model - model_a.df_model)
    lr_p_value = float(stats.chi2.sf(lr_stat, df_diff))

    # Cluster-robust refit of Model B — clustered on match_id, to account
    # for rounds within a match not being independent. Coefficients are
    # identical to the non-robust fit; only the standard errors / p-values
    # on individual coefficients change. Does NOT correct the LR test
    # above (that would require a full mixed-effects model).
    groups = data["match_id"].to_numpy()
    model_b_robust = sm.Logit(y, X_b).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": groups}
    )

    def _coef_block(fitted_model, names):
        return {
            name: {
                "coef": float(fitted_model.params[i]),
                "std_err": float(fitted_model.bse[i]),
                "z": float(fitted_model.tvalues[i]),
                "p_value": float(fitted_model.pvalues[i]),
            }
            for i, name in enumerate(names)
        }

    return {
        "n_rounds": int(len(data)),
        "n_clusters_matches": int(data["match_id"].nunique()),
        "model_a": {
            "formula": "y_t_win ~ margin_active_z",
            "coefficients": _coef_block(model_a, ["const", "margin_active_z"]),
            "log_likelihood": float(model_a.llf),
            "aic": float(model_a.aic),
            "bic": float(model_a.bic),
        },
        "model_b": {
            "formula": "y_t_win ~ margin_active_z + margin_bombsite_active_z",
            "coefficients": _coef_block(model_b, ["const", "margin_active_z", "margin_bombsite_active_z"]),
            "log_likelihood": float(model_b.llf),
            "aic": float(model_b.aic),
            "bic": float(model_b.bic),
        },
        "model_b_cluster_robust_by_match": {
            "note": (
                "Same point estimates as model_b — only std_err/z/p_value "
                "differ, using sandwich standard errors clustered on "
                "match_id to account for non-independence of rounds "
                "within a match."
            ),
            "coefficients": _coef_block(
                model_b_robust, ["const", "margin_active_z", "margin_bombsite_active_z"]
            ),
        },
        "likelihood_ratio_test": {
            "description": (
                "Tests whether adding margin_bombsite_active to a model "
                "that already has margin_active significantly improves "
                "fit -- i.e. whether bombsite control carries information "
                "about round outcome beyond whole-active-footprint "
                "control."
            ),
            "lr_statistic": float(lr_stat),
            "df": df_diff,
            "p_value": lr_p_value,
            "bombsite_adds_significant_information": bool(lr_p_value < 0.05),
        },
    }


# ---------------------------------------------------------------------------
# Test 2 — DeLong's test for two correlated AUCs
# ---------------------------------------------------------------------------
def _delong_structural_components(pos_scores: np.ndarray, neg_scores: np.ndarray):
    """psi(x, y) = 1 if x>y, 0.5 if x==y, 0 if x<y, applied to every
    (positive, negative) pair via broadcasting.

    Returns:
        v10: per-positive-case average of psi over all negatives (len m)
        v01: per-negative-case average of psi over all positives (len n)
        auc: overall AUC (mean of psi over all pairs)
    """
    diff = pos_scores[:, None] - neg_scores[None, :]
    psi = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
    v10 = psi.mean(axis=1)
    v01 = psi.mean(axis=0)
    auc = float(psi.mean())
    return v10, v01, auc


def delong_test(y_true: np.ndarray, score1: np.ndarray, score2: np.ndarray) -> dict:
    """DeLong et al. (1988) test comparing the AUCs of two classifier
    scores evaluated on the SAME set of labelled cases (i.e. a paired /
    correlated comparison, not two independent-sample AUCs).

    y_true, score1, score2 must be aligned (same row order, same length).
    y_true is truthy for the positive class (here: T win).
    """
    y_true = np.asarray(y_true).astype(bool)
    score1 = np.asarray(score1, dtype=float)
    score2 = np.asarray(score2, dtype=float)

    pos1, neg1 = score1[y_true], score1[~y_true]
    pos2, neg2 = score2[y_true], score2[~y_true]

    m = len(pos1)  # number of positive cases (T win)
    n = len(neg1)  # number of negative cases (T loss)
    if m < 2 or n < 2:
        raise ValueError("Need at least 2 positive and 2 negative cases for DeLong's test")

    v10_1, v01_1, auc1 = _delong_structural_components(pos1, neg1)
    v10_2, v01_2, auc2 = _delong_structural_components(pos2, neg2)

    s10 = np.cov(np.vstack([v10_1, v10_2]))  # 2x2 covariance across positive cases
    s01 = np.cov(np.vstack([v01_1, v01_2]))  # 2x2 covariance across negative cases
    s = s10 / m + s01 / n

    auc_diff = auc1 - auc2
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]

    if var_diff <= 0:
        z_stat, p_value = None, None
    else:
        z_stat = float(auc_diff / np.sqrt(var_diff))
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    return {
        "n_positive_t_win": int(m),
        "n_negative_t_loss": int(n),
        "auc_margin_active": float(auc1),
        "auc_margin_bombsite_active": float(auc2),
        "auc_difference": float(auc_diff),
        "var_auc_difference": float(var_diff),
        "z_statistic": z_stat,
        "p_value": p_value,
        "margin_active_significantly_better": bool(
            p_value is not None and p_value < 0.05 and auc_diff > 0
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading per-match round-level T-side control margin data...")
    df = load_all_rounds()
    data = _prepare_paired_data(df)
    print(
        f"  {len(data):,} rounds usable for both tests "
        f"(non-null in both {MARGIN_A_COL} and {MARGIN_B_COL}), "
        f"across {data['match_id'].nunique()} matches\n"
    )

    print("=== Test 1: Nested logistic regression + likelihood-ratio test ===")
    lr_result = run_nested_logistic_test(data)
    lrt = lr_result["likelihood_ratio_test"]
    print(f"  Model A log-likelihood: {lr_result['model_a']['log_likelihood']:.2f}  "
          f"(AIC={lr_result['model_a']['aic']:.2f})")
    print(f"  Model B log-likelihood: {lr_result['model_b']['log_likelihood']:.2f}  "
          f"(AIC={lr_result['model_b']['aic']:.2f})")
    print(f"  LR statistic = {lrt['lr_statistic']:.4f}, df={lrt['df']}, "
          f"p-value = {lrt['p_value']:.6g}")
    print(f"  Bombsite term adds significant information beyond active "
          f"footprint: {lrt['bombsite_adds_significant_information']}")
    b_coefs = lr_result["model_b"]["coefficients"]
    b_robust_coefs = lr_result["model_b_cluster_robust_by_match"]["coefficients"]
    print("\n  Model B standardized coefficients (naive SE | cluster-robust SE):")
    for name in ["margin_active_z", "margin_bombsite_active_z"]:
        naive = b_coefs[name]
        robust = b_robust_coefs[name]
        print(
            f"    {name:<28} coef={naive['coef']:+.4f}  "
            f"p_naive={naive['p_value']:.4g}  p_cluster_robust={robust['p_value']:.4g}"
        )

    print("\n=== Test 2: DeLong's test for correlated AUCs ===")
    delong_result = delong_test(
        data["y_t_win"].to_numpy(),
        data[MARGIN_A_COL].to_numpy(),
        data[MARGIN_B_COL].to_numpy(),
    )
    print(f"  AUC(margin_active)          = {delong_result['auc_margin_active']:.4f}")
    print(f"  AUC(margin_bombsite_active) = {delong_result['auc_margin_bombsite_active']:.4f}")
    print(f"  AUC difference = {delong_result['auc_difference']:+.4f}, "
          f"z = {delong_result['z_statistic']:.4f}, p = {delong_result['p_value']:.6g}")
    print(f"  margin_active significantly better classifier: "
          f"{delong_result['margin_active_significantly_better']}")

    output = {
        "margin_a_column": MARGIN_A_COL,
        "margin_b_column": MARGIN_B_COL,
        "nested_logistic_regression_test": lr_result,
        "delong_auc_test": delong_result,
    }
    with open(FORMAL_COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full results to {FORMAL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
