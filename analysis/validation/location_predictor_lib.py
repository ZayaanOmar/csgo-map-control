"""
location_predictor_lib.py
=========================
Shared engine for the location-aware predictor analyses. It contains NO command-line interface and no
analysis-specific choices: each question gets its own thin script that decides the rows, the label,
the primary tests and the output directory, then calls run_grid().

    analysis_location_predictor.py   target: CT wins the round         (adds: baseline reproduction, economy subsets)
    analysis_location_site.py        target: planted at B (vs A)       (adds: plant-site loader, lead guard)
    analysis_location_overlap.py     diagnostic: is location a proxy for economy?   (uses load_data, prep_blocks)

WHAT LIVES HERE
    load_data       reads window_rounds.csv + area/node feature tables for the ELIGIBLE rounds; a caller-supplied
                    round_filter(df) may subset rows and must leave the label column in df
    models          M_const, M0, M_agg, M0+M_agg (sklearn, C=1); M1/M2 and offset models M0+M1/M0+M2 (ridge
                    logistic with inner-CV lambda; optional presence rows M1p, M2p, M0+M1p, M0+M2p)
    run_cv          shared fold assignments (folds.csv), nested lambda tuning, out-of-fold predictions
    bootstrap       paired cluster bootstrap over MATCHES; score_table / compare produce the tables
    run_grid        the standard sequence: CV -> lambda log -> bootstrap -> scores/comparisons -> files -> printout

DESIGN NOTES (unchanged from the original single script)
    * features centred on the training mean, each block divided by ONE common scalar, near-constant columns dropped,
      all inside each training fold
    * lambda chosen by inner 3-fold match-grouped CV on log-loss; lambda_log.csv records every choice and whether it
      sat on the grid edge ('upper' = strongest penalty = features shrunk to ~nothing)
    * improvement = loss_ref - loss_candidate (positive = candidate better); AUC difference = AUC_cand - AUC_ref
    * mde_80 = 2.8 x bootstrap SE of the log-loss difference
    * CAUTION: a ridge model shrunk to ~intercept-only has pooled out-of-fold AUC typically below 0.5 and, as an
      offset model, differs from M0 by ~1e-5 with a spuriously tight CI. Judge such models by the SIZE of the
      log-loss difference, not AUC or p alone. M_const's AUC is meaningless for the same reason.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402

ECON = ["log_eq_ratio", "eq_diff_k", "is_pistol"]
LAM_GRID = np.logspace(-1, 6, 15)          # sum-NLL + 0.5*lam*|w|^2 (lam ~ 1/C); wide on purpose
EPS = 1e-6


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class Data:
    pass


def load_data(feat_dir: Path, with_presence: bool, round_filter=None, label_col: str = "y_ct_win") -> Data:
    """Load the eligible rounds and their features.

    round_filter: optional callable(df) -> df applied to the eligible rows of window_rounds.csv. It may drop
                  rows and/or add columns; label_col must exist in what it returns (default y_ct_win).
    """
    D = Data()
    allr = pd.read_csv(feat_dir / "window_rounds.csv")
    D.rounds_all = allr
    el = allr[allr["eligible"]].copy()
    if round_filter is not None:
        el = round_filter(el)
    el = el.reset_index(drop=True)
    D.el = el
    D.label_col = label_col
    D.y = el[label_col].to_numpy(int)
    D.match = el["match_name"].to_numpy()
    D.mcodes, D.mnames = pd.factorize(D.match)
    D.n = len(el)

    def getcsv(name):
        d = pd.read_csv(feat_dir / name).set_index("rk")
        return d.loc[el["rk"]]

    def getpq(name):
        d = pd.read_parquet(feat_dir / name).set_index("rk")
        return d.loc[el["rk"]]

    area_net, node_net = getcsv("window_area_net.csv"), getpq("window_node_net.parquet")
    mats = {"econ": el[ECON].to_numpy(float), "ca": el[["ca_win"]].to_numpy(float),
            "area_net": area_net.to_numpy(float), "node_net": node_net.to_numpy(float)}
    D.names = {"area_net": list(area_net.columns),
               "node_net": [int(str(c)[1:]) for c in node_net.columns]}
    if with_presence:
        mats["area_pres"] = getcsv("window_area_presence.csv").to_numpy(float)
        mats["node_pres"] = getpq("window_node_presence.parquet").to_numpy(float)
    for k, M in mats.items():
        bad = ~np.isfinite(M)
        if bad.any():
            vc.log(f"  WARNING: {int(bad.sum())} non-finite values in {k}; filled with column mean/0")
            M[bad] = 0.0
    D.mats = mats
    return D


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def econ_pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def prep_blocks(blocks: list, tr: np.ndarray):
    """Fit centring/scaling/column-dropping on the training rows; return transform(idx)."""
    params = []
    for B in blocks:
        mu, sd = B[tr].mean(0), B[tr].std(0)
        keep = sd > 1e-6
        s = float(np.sqrt(np.mean(sd[keep] ** 2))) if keep.any() else 1.0
        params.append((mu, keep, s))

    def transform(idx):
        return np.hstack([((B[idx] - mu) / s)[:, keep] for B, (mu, keep, s) in zip(blocks, params)])

    transform.params = params
    return transform


def fit_ridge_logit(X, y, off, lam, theta0=None, max_iter=300):
    """min  sum[ log(1+exp(z)) - y z ] + 0.5*lam*|w|^2,   z = off + b0 + X w   (intercept unpenalised)."""
    n, p = X.shape

    def f(theta):
        z = off + theta[0] + X @ theta[1:]
        nll = np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * (theta[1:] @ theta[1:])
        r = expit(z) - y
        g = np.empty(p + 1)
        g[0] = r.sum()
        g[1:] = X.T @ r + lam * theta[1:]
        return nll, g

    t0 = np.zeros(p + 1) if theta0 is None else theta0
    res = minimize(f, t0, jac=True, method="L-BFGS-B", options=dict(maxiter=max_iter, gtol=1e-6, ftol=1e-12))
    return res.x


def team_prior(D, tr: np.ndarray, te: np.ndarray):
    """Shrunken per-team positive-class rate from TRAINING rows only (needs D.tcodes, D.prior_strength).

    Test rows use all training rows of their team. Training rows use a leave-one-MATCH-out version (their own
    match's rows of that team are removed), so a row's prior never contains its own label or its match-mates'.
    Unseen teams fall back to the training base rate. Returns (p_train_rows, p_test_rows)."""
    a = getattr(D, "prior_strength", 10.0)
    T = int(D.tcodes.max()) + 1
    base = D.y[tr].mean()
    n_t = np.bincount(D.tcodes[tr], minlength=T).astype(float)
    k_t = np.bincount(D.tcodes[tr], weights=D.y[tr], minlength=T)
    p_te = (k_t[D.tcodes[te]] + a * base) / (n_t[D.tcodes[te]] + a)
    key = D.mcodes.astype(np.int64) * T + D.tcodes
    n_mt = np.bincount(key[tr], minlength=int(key.max()) + 1).astype(float)
    k_mt = np.bincount(key[tr], weights=D.y[tr], minlength=int(key.max()) + 1)
    n_loo = n_t[D.tcodes[tr]] - n_mt[key[tr]]
    k_loo = k_t[D.tcodes[tr]] - k_mt[key[tr]]
    p_tr = (k_loo + a * base) / (n_loo + a)
    return p_tr, p_te


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _offsets(D, itr, ite, use_offset):
    """use_offset: False (none) | True (economy model M0) | 'team' (shrunken team prior)."""
    if not use_offset:
        return np.zeros(len(itr)), np.zeros(len(ite))
    if use_offset == "team":
        p_tr, p_te = team_prior(D, itr, ite)
        return _logit(p_tr), _logit(p_te)
    m0 = econ_pipe().fit(D.mats["econ"][itr], D.y[itr])
    return m0.decision_function(D.mats["econ"][itr]), m0.decision_function(D.mats["econ"][ite])


def _ll(y, z):
    p = np.clip(expit(z), EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def tune_lambda(D, blocks, tr, use_offset, k_inner=3):
    lams = np.sort(LAM_GRID)[::-1]                  # strong -> weak penalty, warm-started
    loss = np.zeros(len(lams))
    splits = list(GroupKFold(n_splits=k_inner).split(tr, groups=D.mcodes[tr]))
    for a, b in splits:
        itr, ite = tr[a], tr[b]
        T = prep_blocks(blocks, itr)
        Xa, Xb = T(itr), T(ite)
        oa, ob = _offsets(D, itr, ite, use_offset)
        theta = None
        for j, lam in enumerate(lams):
            theta = fit_ridge_logit(Xa, D.y[itr], oa, lam, theta)
            loss[j] += _ll(D.y[ite], ob + theta[0] + Xb @ theta[1:]) * len(ite)
    j = int(np.argmin(loss))
    edge = "upper" if j == 0 else "lower" if j == len(lams) - 1 else ""   # upper = strongest penalty
    return float(lams[j]), edge


def predict_model(spec: dict, D: Data, tr: np.ndarray, te: np.ndarray):
    if spec["kind"] == "const":                       # training-fold CT win rate, no features
        return np.full(len(te), D.y[tr].mean()), None, ""
    if spec["kind"] == "team":                        # shrunken team prior only (needs D.tcodes)
        return team_prior(D, tr, te)[1], None, ""
    if spec["kind"] == "glm":
        X = np.hstack([D.mats[k] for k in spec["feats"]])
        m = econ_pipe().fit(X[tr], D.y[tr])
        return m.predict_proba(X[te])[:, 1], None, ""
    blocks = [D.mats[k] for k in spec["loc"]]
    lam, edge = tune_lambda(D, blocks, tr, spec["offset"])
    T = prep_blocks(blocks, tr)
    Xtr, Xte = T(tr), T(te)
    otr, ote = _offsets(D, tr, te, spec["offset"])
    th = fit_ridge_logit(Xtr, D.y[tr], otr, lam)
    return expit(ote + th[0] + Xte @ th[1:]), lam, edge


def build_specs(presence: bool) -> dict:
    s = {
        "M_const": dict(kind="const"),
        "M0": dict(kind="glm", feats=["econ"]),
        "M_agg": dict(kind="glm", feats=["ca"]),
        "M0+M_agg": dict(kind="glm", feats=["econ", "ca"]),
        "M1": dict(kind="ridge", loc=["area_net"], offset=False),
        "M2": dict(kind="ridge", loc=["node_net"], offset=False),
        "M0+M1": dict(kind="ridge", loc=["area_net"], offset=True),
        "M0+M2": dict(kind="ridge", loc=["node_net"], offset=True),
    }
    if presence:
        s.update({
            "M1p": dict(kind="ridge", loc=["area_net", "area_pres"], offset=False),
            "M2p": dict(kind="ridge", loc=["node_net", "node_pres"], offset=False),
            "M0+M1p": dict(kind="ridge", loc=["area_net", "area_pres"], offset=True),
            "M0+M2p": dict(kind="ridge", loc=["node_net", "node_pres"], offset=True),
        })
    return s


# --------------------------------------------------------------------------- #
# Cross-validation over the shared folds
# --------------------------------------------------------------------------- #
def run_cv(D: Data, specs: dict, folds: pd.DataFrame, col: str, repeats: int):
    P = {m: np.full((repeats, D.n), np.nan) for m in specs}
    lam_rows = []
    for r in range(repeats):
        fmap = folds[folds["repeat"] == r].set_index("match_name")[col]
        fold = D.el["match_name"].map(fmap).to_numpy()
        if np.isnan(fold.astype(float)).any():
            raise SystemExit(f"repeat {r}: some matches have no fold in folds.csv ({col})")
        for f in sorted(np.unique(fold)):
            te = np.flatnonzero(fold == f)
            tr = np.flatnonzero(fold != f)
            for name, spec in specs.items():
                p, lam, edge = predict_model(spec, D, tr, te)
                P[name][r, te] = p
                if lam is not None:
                    lam_rows.append(dict(model=name, repeat=r, fold=int(f), lam=lam, grid_edge=edge))
        vc.log(f"  repeat {r + 1}/{repeats} done")
    return P, pd.DataFrame(lam_rows)


# --------------------------------------------------------------------------- #
# Metrics + paired cluster bootstrap
# --------------------------------------------------------------------------- #
def row_ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


class AUCW:
    """Weighted AUC with a precomputed ordering (probabilities are continuous; ties ignored)."""
    def __init__(self, y, s):
        self.o = np.argsort(s, kind="mergesort")
        self.y = y[self.o].astype(float)

    def __call__(self, w):
        w = w[self.o]
        wp, wn = w * self.y, w * (1 - self.y)
        below = np.cumsum(wn) - wn
        return float((wp * below).sum() / (wp.sum() * wn.sum()))


def bootstrap(D: Data, P: dict, B: int, seed: int):
    y, G = D.y, len(D.mnames)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G)[D.mcodes] for _ in range(B)]).astype(np.float32)
    Wsum = W.sum(1)
    ll = {m: row_ll(y[None, :], p).mean(0) for m, p in P.items()}
    br = {m: ((p - y[None, :]) ** 2).mean(0) for m, p in P.items()}
    pm = {m: p.mean(0) for m, p in P.items()}
    auc = {m: AUCW(y, pm[m]) for m in P}
    return dict(W=W, Wsum=Wsum, ll=ll, br=br, pm=pm, auc=auc, B=B)


def score_table(D, P, bs):
    rows = []
    ones = np.ones(D.n)
    for m, p in P.items():
        per_rep = [(row_ll(D.y, p[r]).mean(), roc_auc_score(D.y, p[r]), brier_score_loss(D.y, p[r]))
                   for r in range(p.shape[0])]
        per_rep = np.array(per_rep)
        bll = (bs["W"] @ bs["ll"][m]) / bs["Wsum"]
        bbr = (bs["W"] @ bs["br"][m]) / bs["Wsum"]
        bau = np.array([bs["auc"][m](w) for w in bs["W"]])
        row = dict(model=m, n_rounds=D.n,
                   logloss=per_rep[:, 0].mean(), logloss_lo=np.percentile(bll, 2.5), logloss_hi=np.percentile(bll, 97.5),
                   auc=per_rep[:, 1].mean(), auc_lo=np.percentile(bau, 2.5), auc_hi=np.percentile(bau, 97.5),
                   brier=per_rep[:, 2].mean(), brier_lo=np.percentile(bbr, 2.5), brier_hi=np.percentile(bbr, 97.5),
                   logloss_sd_across_repeats=per_rep[:, 0].std(), auc_sd_across_repeats=per_rep[:, 1].std())
        rows.append(row)
    return pd.DataFrame(rows)


def pval(b):
    B = len(b)
    p = 2 * min((np.sum(b <= 0) + 1) / (B + 1), (np.sum(b >= 0) + 1) / (B + 1))
    return min(p, 1.0)


def compare(D, bs, pairs, primary):
    rows = []
    for cand, ref in pairs:
        d_ll = bs["ll"][ref] - bs["ll"][cand]
        d_br = bs["br"][ref] - bs["br"][cand]
        est_ll = d_ll.mean()
        est_br = d_br.mean()
        b_ll = (bs["W"] @ d_ll) / bs["Wsum"]
        b_br = (bs["W"] @ d_br) / bs["Wsum"]
        est_auc = bs["auc"][cand](np.ones(D.n)) - bs["auc"][ref](np.ones(D.n))
        b_auc = np.array([bs["auc"][cand](w) - bs["auc"][ref](w) for w in bs["W"]])
        rows.append(dict(candidate=cand, reference=ref, primary=(cand, ref) in primary,
                         dlogloss=est_ll, dlogloss_lo=np.percentile(b_ll, 2.5), dlogloss_hi=np.percentile(b_ll, 97.5),
                         dlogloss_pct=100 * est_ll / bs["ll"][ref].mean(), p_logloss=pval(b_ll),
                         mde80_logloss=2.8 * b_ll.std(),
                         dauc=est_auc, dauc_lo=np.percentile(b_auc, 2.5), dauc_hi=np.percentile(b_auc, 97.5),
                         p_auc=pval(b_auc),
                         dbrier=est_br, dbrier_lo=np.percentile(b_br, 2.5), dbrier_hi=np.percentile(b_br, 97.5)))
    df = pd.DataFrame(rows)
    prim = df[df["primary"]].sort_values("p_logloss")
    k = len(prim)
    holm, running = {}, 0.0
    for i, (idx, r) in enumerate(prim.iterrows()):
        running = max(running, min(1.0, (k - i) * r["p_logloss"]))
        holm[idx] = running
    df["holm_p_logloss"] = pd.Series(holm)
    return df


def export_coefficients(D: Data, specs: dict, lam_log: pd.DataFrame, out: Path):
    """Descriptive full-data fits of M1 and M2 (no offset) at the median CV-chosen lambda."""
    for model, key, fname in (("M1", "area_net", "area_coefficients.csv"), ("M2", "node_net", "node_coefficients.csv")):
        L = lam_log[lam_log["model"] == model]["lam"]
        if L.empty:
            continue
        lam = float(np.exp(np.median(np.log(L))))
        blocks = [D.mats[key]]
        idx = np.arange(D.n)
        T = prep_blocks(blocks, idx)
        mu, keep, s = T.params[0]
        th = fit_ridge_logit(T(idx), D.y, np.zeros(D.n), lam)
        names = np.array(D.names[key])[keep]
        df = pd.DataFrame({("area" if key == "area_net" else "area_id"): names, "coef_scaled": th[1:],
                           "odds_ratio_per_0.1_net": np.exp(th[1:] * 0.1 / s)})
        df["lambda_used"] = lam
        df.sort_values("coef_scaled", key=np.abs, ascending=False).to_csv(out / fname, index=False)


def forest_plot(comp: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    c = comp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(c) + 1.5))
    y = np.arange(len(c))
    ax.errorbar(c["dlogloss"], y, xerr=[c["dlogloss"] - c["dlogloss_lo"], c["dlogloss_hi"] - c["dlogloss"]],
                fmt="o", capsize=3, color="tab:blue")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} vs {b}" + ("  [primary]" if p else "") for a, b, p in
                        zip(c["candidate"], c["reference"], c["primary"])], fontsize=8)
    ax.set_xlabel("log-loss improvement of candidate over reference (positive = candidate better), 95% match-bootstrap CI")
    vc.savefig(fig, path)


# --------------------------------------------------------------------------- #
# Standard sequence
# --------------------------------------------------------------------------- #
def default_pairs(primary: list, presence: bool) -> list:
    """Comparison list: primary tests first, then the descriptive ones (candidate, reference)."""
    base = [("M0", "M_const"), ("M_agg", "M_const"), ("M1", "M_const"), ("M2", "M_const"),
            ("M_agg", "M0"), ("M0+M_agg", "M0"), ("M1", "M0"), ("M2", "M0"),
            ("M0+M1", "M0+M_agg"), ("M0+M2", "M0+M_agg"), ("M2", "M1"), ("M0+M2", "M0+M1"),
            ("M0+M1", "M0"), ("M0+M2", "M0")]
    pairs = list(primary) + [p for p in base if p not in primary]
    if presence:
        pairs += [("M1p", "M1"), ("M2p", "M2"), ("M0+M1p", "M0+M1"), ("M0+M2p", "M0+M2")]
    return pairs


def run_grid(D: Data, folds: pd.DataFrame, grouping: str, repeats: int, out: Path, primary: list,
             presence: bool = False, n_boot: int = None, seed: int = 0, y_col: str = "y_ct_win",
             record_prefix: str = "locpred", meta: dict = None, specs: dict = None, pairs: list = None) -> tuple:
    """Cross-validate the model grid, bootstrap, write all tables/figures, print the summary."""
    n_boot = vc.DEFAULT_BOOT if n_boot is None else n_boot
    col = "fold_match" if grouping == "match" else "fold_teampair"
    if folds[col].isna().all():
        raise SystemExit(f"{col} is empty in folds.csv (no team columns when the folds were built)")
    vc.ensure_dir(out / "figures")

    specs = specs if specs is not None else build_specs(presence)
    vc.log(f"\nCROSS-VALIDATION ({len(specs)} models)")
    P, lam_log = run_cv(D, specs, folds, col, repeats)
    if len(lam_log):
        lam_log["_up"] = lam_log["grid_edge"] == "upper"
        lam_log["_lo"] = lam_log["grid_edge"] == "lower"
        s = lam_log.groupby("model").agg(median_lambda=("lam", "median"), pct_upper_edge=("_up", "mean"),
                                         pct_lower_edge=("_lo", "mean"))
        vc.log("\nlambda chosen by inner CV:\n" + s.to_string(float_format=lambda v: f"{v:,.3g}"))
        if (s["pct_lower_edge"] > 0.2).any():
            vc.log("  WARNING: lambda hit the WEAK-penalty edge for some models; the grid may need extending downwards")
        if (s["pct_upper_edge"] > 0.5).any():
            vc.log("  NOTE: for models with pct_upper_edge > 0.5 the CV chose the STRONGEST penalty, i.e. it shrank the "
                   "location features to ~nothing (no usable signal); their predictions are ~intercept-only. "
                   "Read such models by log-loss, not AUC (pooled out-of-fold AUC of a near-constant model is "
                   "typically BELOW 0.5, an artefact of fold-wise intercepts).")
    lam_log.drop(columns=[c for c in ("_up", "_lo") if c in lam_log.columns]).to_csv(out / "lambda_log.csv", index=False)

    vc.log(f"\nBOOTSTRAP ({n_boot} match-level resamples)")
    bs = bootstrap(D, P, n_boot, seed)
    scores = score_table(D, P, bs)
    scores.to_csv(out / "model_scores.csv", index=False)
    comp = compare(D, bs, pairs if pairs is not None else default_pairs(primary, presence), primary)
    comp.to_csv(out / "model_comparisons.csv", index=False)

    pd.DataFrame({"rk": D.el["rk"], "match_name": D.match, y_col: D.y,
                  **{f"p_{m}": P[m].mean(0) for m in P}}).to_parquet(out / "oof_predictions.parquet", index=False)
    export_coefficients(D, specs, lam_log, out)
    forest_plot(comp, out / "figures" / "delta_logloss.png")

    for _, r in comp[comp["primary"]].iterrows():
        vc.record_result(f"{record_prefix}.{r['candidate']}_vs_{r['reference']}.dlogloss", r["dlogloss"], r["dlogloss_lo"],
                         r["dlogloss_hi"], D.n, note=f"log-loss improvement over {r['reference']} ({grouping} folds)")
        vc.record_result(f"{record_prefix}.{r['candidate']}_vs_{r['reference']}.dauc", r["dauc"], r["dauc_lo"],
                         r["dauc_hi"], D.n, note=f"AUC difference vs {r['reference']} ({grouping} folds)")

    vc.log("\nMODEL SCORES (pooled out-of-fold; 95% match-bootstrap CI)")
    for _, r in scores.iterrows():
        vc.log(f"  {r['model']:<9} logloss {r['logloss']:.4f} [{r['logloss_lo']:.4f},{r['logloss_hi']:.4f}]  "
               f"AUC {r['auc']:.4f} [{r['auc_lo']:.4f},{r['auc_hi']:.4f}]  Brier {r['brier']:.4f}")
    vc.log("\nCOMPARISONS (improvement of candidate over reference; positive = better)")
    for _, r in comp.iterrows():
        tag = "  [PRIMARY, Holm p=%.3f]" % r["holm_p_logloss"] if r["primary"] else ""
        vc.log(f"  {r['candidate']:<9} vs {r['reference']:<9} dLL {r['dlogloss']:+.5f} [{r['dlogloss_lo']:+.5f},{r['dlogloss_hi']:+.5f}] "
               f"({r['dlogloss_pct']:+.2f}%) p={r['p_logloss']:.3f} | dAUC {r['dauc']:+.4f} "
               f"[{r['dauc_lo']:+.4f},{r['dauc_hi']:+.4f}] | MDE80(LL) {r['mde80_logloss']:.5f}{tag}")
    vc.log(f"\n(two-sided p-values cannot go below ~{2 / (n_boot + 1):.4f} with {n_boot} resamples - the bootstrap floor)")

    (out / "run_meta.json").write_text(json.dumps(dict(
        grouping=grouping, repeats=repeats, n_boot=n_boot, presence=presence, n_rounds=int(D.n),
        lam_grid=[float(x) for x in LAM_GRID], primary=[list(p) for p in primary], **(meta or {})), indent=2, default=str))
    vc.log(f"\nWrote {out}/")
    return scores, comp
