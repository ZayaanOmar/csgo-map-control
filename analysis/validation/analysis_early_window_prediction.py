"""
analysis_early_window_prediction.py   (Parts 2a, 2b, 2c)
========================================================
Does positional control EARLY in a round carry information about who wins it — beyond economy,
beyond the head-count, and beyond the Voronoi baseline?

Why this exists: Metric 3 averages control over the whole round, and late-round control is
partly a CONSEQUENCE of winning (reverse causation). Here the predictors are built only from
frames inside a window that closes before the outcome can leak in.

WINDOW DEFINITIONS (2a)
  pre-engagement (HEADLINE) : frames from freeze end until min(cap, first kill of the round).
        Alive count is 5v5 throughout by construction, so any signal is pure positioning.
        Rounds whose first kill comes before --min-pre seconds are dropped.
  fixed T seconds           : frames with t_rel <= T; rounds shorter than T are dropped (default)
        because including their end-of-round frames would leak the outcome. (--allow-partial relaxes this.)
  pct of the round timer    : T = pct * --timer-s (default 115 s). NOT a pct of the round's actual
        length (round length is itself outcome-correlated).

MODELS (2b): logistic regression on standardised features, evaluated with GROUPED (by match)
  K-fold cross-validation, so no match is split across train and test. CIs: cluster bootstrap over
  matches of the out-of-fold predictions; deltas are PAIRED (same resamples).
  base_rate | econ | alive | vor | ctrl_full | ctrl_active | econ+alive | econ+vor |
  econ+ctrl_active | econ+alive+ctrl_active | econ+ctrl_active+contested | econ+ctrl_active_dm
  (_dm = within-match demeaned feature; uses no outcome information).
  econ = [log equipment ratio, equipment difference (k), pistol flag].
  The key row is the PAIRED delta AUC: (econ + ctrl_active) - (econ).

READING THE HEAD-COUNT-CONTROLLED DELTAS (important): the rows (econ+alive+ctrl) - (econ+alive) exist only for
  windows that contain kills. Window-mean control is a NONLINEAR TIME-AVERAGE of the head-count (influence sums scale
  with the number of players, then pass through a sigmoid), so a gain over a head-count baseline there cannot be
  attributed to positioning: it can be head-count history the baseline failed to capture. The baseline is given
  end/mean/min-style history features (ALIVE) and still cannot rule this out. Treat those rows as an UPPER BOUND on
  what positioning adds. The clean tests are (1) the pre-engagement windows (5v5 throughout), (2) the "still 5v5 at t"
  panel of the sweep, and (3) analysis_metric3_stratified.py (equal numbers frame by frame).

AUC-vs-TIME SWEEP (2c): for t = 5, 10, ..., 60 s, AUC of econ / alive-count / instantaneous control /
  expanding-window control / Voronoi / econ+control, on (a) all rounds still running at t and
  (b) rounds still 5v5 at t (no kill yet). Panel (b) is the cleanest answer to "is control just
  tracking the score?".

Outputs: outputs/metrics/early_window/{window_models.csv, window_deltas.csv, sweep_auc.csv,
         headline_coefficients.csv, figures/*.png}
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = vc.METRICS / "early_window"
FIGDIR = OUTDIR / "figures"
MIN_FRAMES = 3
ECON = ["log_eq_ratio", "eq_diff_k", "is_pistol"]
# HEAD-COUNT baseline. End-of-window difference alone is NOT enough: window-mean control is a time-average that
# mechanically encodes WHEN kills happened, so a model with (end diff + mean control) can reconstruct head-count
# history and look like "positioning" gains. The baseline therefore also gets the window-mean difference and the
# separate end counts (5v4 and 2v1 are the same difference but very different rounds).
ALIVE = ["alive_diff_end", "alive_diff_mean", "ct_alive_end", "t_alive_end", "ct_alive_mean", "t_alive_mean",
         "frac_5v5", "alive_diff_end_sq", "alive_diff_mean_sq", "alive_prod_mean"]
MODELS = {
    "base_rate": [],
    "econ": ECON,
    "alive": ALIVE,
    "vor": ["vor"],
    "ctrl_full": ["cf"],
    "ctrl_active": ["ca"],
    "econ+alive": ECON + ALIVE,
    "econ+vor": ECON + ["vor"],
    "econ+ctrl_full": ECON + ["cf"],
    "econ+alive+vor": ECON + ALIVE + ["vor"],
    "econ+ctrl_active": ECON + ["ca"],
    "econ+alive+ctrl_active": ECON + ALIVE + ["ca"],
    "econ+ctrl_active+contested": ECON + ["ca", "cont"],
    "econ+ctrl_active_dm": ECON + ["ca_dm"],
}
PAIRS = [("econ+ctrl_active", "econ"), ("econ+vor", "econ"), ("econ+ctrl_active", "econ+vor"),
         ("econ+ctrl_full", "econ"), ("ctrl_active", "vor"), ("econ+ctrl_active_dm", "econ"),
         # THE head-count-controlled tests (only exist for windows that contain kills):
         ("econ+alive+ctrl_active", "econ+alive"), ("econ+alive+vor", "econ+alive"),
         ("econ+alive+ctrl_active", "econ+alive+vor")]


# ----------------------------------------------------------------- data assembly
def load_data():
    rt = vc.load_round_table()
    rt["rk"] = rt["match_name"] + ":" + rt["round"].astype(str)
    for c in ("log_eq_ratio", "eq_diff", "is_pistol_round"):
        if c not in rt.columns:
            raise SystemExit(f"round table lacks '{c}' — run extract_supplementary_tables.py (round_economy) first")
    rt["eq_diff_k"] = rt["eq_diff"] / 1000.0
    rt["is_pistol"] = rt["is_pistol_round"].astype(float)
    fr = vc.load_frame_summaries(columns=["round", "frame_idx", "t_rel", "ct_alive", "t_alive", "ctrl_full",
                                          "ctrl_active", "contested_active", "vor_margin"])
    fr["rk"] = fr["match_name"] + ":" + fr["round"].astype(str)
    fr = fr[fr["t_rel"] >= 0].copy()
    last_t = fr.groupby("rk")["t_rel"].max()
    rt["round_len"] = rt["round_len_s"].fillna(rt["rk"].map(last_t))
    fr["cf"] = fr["ctrl_full"] - 0.5
    fr["ca"] = fr["ctrl_active"] - 0.5
    fr["cont"] = fr["contested_active"]
    fr["vor"] = fr["vor_margin"]
    fr["alive_diff"] = fr["ct_alive"] - fr["t_alive"]
    fr["is55"] = ((fr["ct_alive"] == 5) & (fr["t_alive"] == 5)).astype(float)
    fr["alive_prod"] = fr["ct_alive"] * fr["t_alive"]
    fr = fr.sort_values(["rk", "t_rel"])
    return rt, fr


def window_features(fr, rt, end_t: pd.Series, require_end: pd.Series | None):
    """Round-level features from frames with t_rel <= end_t[rk]."""
    lim = fr["rk"].map(end_t)
    w = fr[fr["t_rel"] <= lim]
    g = w.groupby("rk")
    f = g[["cf", "ca", "cont", "vor"]].mean()
    f["alive_diff_end"] = g["alive_diff"].last()
    f["alive_diff_mean"] = g["alive_diff"].mean()
    f["ct_alive_end"] = g["ct_alive"].last()
    f["t_alive_end"] = g["t_alive"].last()
    f["ct_alive_mean"] = g["ct_alive"].mean()
    f["t_alive_mean"] = g["t_alive"].mean()
    f["frac_5v5"] = g["is55"].mean()
    f["alive_diff_end_sq"] = f["alive_diff_end"] ** 2
    f["alive_diff_mean_sq"] = f["alive_diff_mean"] ** 2
    f["alive_prod_mean"] = g["alive_prod"].mean()
    f["n_frames"] = g.size()
    f["last_t"] = g["t_rel"].last()
    f = f.reset_index().merge(rt[["rk", "match_name", "y_ct_win", "round_len", "first_kill_t"] + ECON], on="rk", how="inner")
    f = f[f["n_frames"] >= MIN_FRAMES]
    f["ca_dm"] = f["ca"] - f.groupby("match_name")["ca"].transform("mean")
    return f


# ----------------------------------------------------------------- evaluation
def make_model(cols):
    if not cols:
        return DummyClassifier(strategy="prior")
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))


def oof_predict(df, cols, n_splits):
    y, groups = df["y_ct_win"].to_numpy(), df["match_name"].to_numpy()
    X = df[cols].to_numpy(float) if cols else np.zeros((len(df), 1))
    p = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        p[te] = make_model(cols).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


def evaluate(df, models: dict, n_boot=200, seed=0, pairs=PAIRS):
    """Grouped-CV OOF predictions + paired cluster bootstrap. Returns (metrics_df, deltas_df)."""
    cols_needed = sorted({c for cs in models.values() for c in cs})
    d = df.dropna(subset=cols_needed + ["y_ct_win"]).reset_index(drop=True)
    if len(d) < 60 or d["y_ct_win"].nunique() < 2 or d["match_name"].nunique() < 3:
        return pd.DataFrame(), pd.DataFrame()
    ns = int(min(5, d["match_name"].nunique()))
    y = d["y_ct_win"].to_numpy()
    P = {n: oof_predict(d, cs, ns) for n, cs in models.items()}
    gi = vc.group_rows(d["match_name"].to_numpy())
    rng = np.random.default_rng(seed)
    boots = {n: [] for n in models}
    for _ in range(n_boot):
        rows = np.concatenate([gi[i] for i in rng.integers(0, len(gi), len(gi))])
        yb = y[rows]
        for n in models:
            boots[n].append(roc_auc_score(yb, P[n][rows]) if len(np.unique(yb)) == 2 else np.nan)
    boots = {n: np.array(v) for n, v in boots.items()}
    rows_m = []
    for n in models:
        lo, hi = vc.pctl_ci(boots[n])
        rows_m.append({"model": n, "n_rounds": len(d), "n_matches": d["match_name"].nunique(), "auc": roc_auc_score(y, P[n]),
                       "auc_lo": float(lo), "auc_hi": float(hi), "brier": brier_score_loss(y, P[n]), "logloss": log_loss(y, P[n])})
    rows_d = []
    for a, b in pairs:
        if a in models and b in models:
            diff = boots[a] - boots[b]
            lo, hi = vc.pctl_ci(diff)
            rows_d.append({"a": a, "b": b, "delta_auc": roc_auc_score(y, P[a]) - roc_auc_score(y, P[b]),
                           "lo": float(lo), "hi": float(hi), "p_boot": vc.boot_pvalue(diff)})
    return pd.DataFrame(rows_m), pd.DataFrame(rows_d), d, P


def safe_eval(*a, **k):
    out = evaluate(*a, **k)
    return out if len(out) == 4 else (pd.DataFrame(), pd.DataFrame(), None, None)


# ----------------------------------------------------------------- main analyses
def run_windows(fr, rt, args):
    vc.log("2a/2b: window models")
    first_kill = rt.set_index("rk")["first_kill_t"]
    round_len = rt.set_index("rk")["round_len"]
    defs = {}
    for cap in args.caps:
        end = np.minimum(first_kill.fillna(np.inf), cap)
        e = pd.Series(end, index=first_kill.index)
        keep = (first_kill.fillna(np.inf) >= args.min_pre) & (round_len >= np.minimum(first_kill.fillna(np.inf), cap) - 1e-9)
        defs[f"pre_engagement_cap{int(cap)}"] = (e, keep)
    for T in args.fixed:
        e = pd.Series(float(T), index=first_kill.index)
        keep = (round_len >= T) if not args.allow_partial else pd.Series(True, index=first_kill.index)
        defs[f"fixed_{int(T)}s"] = (e, keep)
    for p in args.pct:
        T = p * args.timer_s
        e = pd.Series(float(T), index=first_kill.index)
        keep = (round_len >= T) if not args.allow_partial else pd.Series(True, index=first_kill.index)
        defs[f"pct{int(p * 100)}_of_{int(args.timer_s)}s"] = (e, keep)
    all_m, all_d, headline = [], [], None
    for name, (end_t, keep) in defs.items():
        f = window_features(fr, rt, end_t, keep)
        f = f[f["rk"].map(keep).fillna(False).astype(bool)]
        models = dict(MODELS)
        if name.startswith("pre_engagement"):
            for mname in ("alive", "econ+alive", "econ+alive+ctrl_active", "econ+alive+vor"):
                models.pop(mname)
        pairs = [p for p in PAIRS if p[0] in models and p[1] in models]
        m, d, dd, P = safe_eval(f, models, args.n_boot, pairs=pairs)
        if m.empty:
            vc.log(f"  {name}: too little data ({len(f)} rounds)")
            continue
        m.insert(0, "window", name); d.insert(0, "window", name)
        all_m.append(m); all_d.append(d)
        r = d[(d["a"] == "econ+ctrl_active") & (d["b"] == "econ")]
        vc.log(f"  {name:28s} n={len(dd):5d}  AUC econ={m.set_index('model').loc['econ','auc']:.3f}  "
               f"econ+ctrl={m.set_index('model').loc['econ+ctrl_active','auc']:.3f}  dAUC={r['delta_auc'].iloc[0]:+.4f} [{r['lo'].iloc[0]:+.4f},{r['hi'].iloc[0]:+.4f}]")
        dshow = d.drop(columns=["window"]).copy()
        dshow["pair"] = dshow["a"] + "  -  " + dshow["b"]
        vc.log("      paired AUC differences (a minus b), 95% match-clustered CI:\n" +
               "\n".join(f"        {r_['pair']:52s} {r_['delta_auc']:+.4f} [{r_['lo']:+.4f}, {r_['hi']:+.4f}]  p={r_['p_boot']:.3f}" for _, r_ in dshow.iterrows()))
        if name == f"pre_engagement_cap{int(args.headline_cap)}":
            headline = (dd, P, f)
            ca = m.set_index("model").loc["ctrl_active"]
            vc.record_result("early.auc_ctrl_active_pre_engagement", ca["auc"], ca["auc_lo"], ca["auc_hi"], ca["n_rounds"], name)
            vc.record_result("early.delta_auc_econ_plus_ctrl_vs_econ_pre_engagement", r["delta_auc"].iloc[0], r["lo"].iloc[0], r["hi"].iloc[0], len(dd), name)
            r2 = d[(d["a"] == "econ+ctrl_active") & (d["b"] == "econ+vor")]
            if len(r2):
                vc.record_result("early.delta_auc_influence_vs_voronoi_pre_engagement", r2["delta_auc"].iloc[0], r2["lo"].iloc[0], r2["hi"].iloc[0], len(dd), name)
    if all_m:
        pd.concat(all_m).to_csv(OUTDIR / "window_models.csv", index=False)
        pd.concat(all_d).to_csv(OUTDIR / "window_deltas.csv", index=False)
    return headline


def headline_figures(headline, args):
    if headline is None:
        vc.log("  (headline window unavailable; skipping calibration/coefficients)")
        return
    dd, P, f = headline
    y = dd["y_ct_win"].to_numpy()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    for name in ("econ", "econ+vor", "econ+ctrl_active"):
        p = P[name]
        q = pd.qcut(p, 8, duplicates="drop")
        t = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True).agg(p=("p", "mean"), y=("y", "mean"))
        ax.plot(t["p"], t["y"], "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("predicted P(CT win)"); ax.set_ylabel("observed CT win rate"); ax.set_title("Calibration (out-of-fold), pre-engagement window"); ax.legend(fontsize=8)
    ax = axs[1]
    ax.scatter(dd["ca"], dd["y_ct_win"] + np.random.default_rng(0).normal(0, .02, len(dd)), s=6, alpha=.3)
    bins = pd.qcut(dd["ca"], 10, duplicates="drop")
    t = dd.groupby(bins, observed=True).agg(x=("ca", "mean"), y=("y_ct_win", "mean"))
    ax.plot(t["x"], t["y"], "r-o")
    ax.set_xlabel("mean CT control (active footprint) - 0.5, pre-engagement window"); ax.set_ylabel("CT round win rate (decile means in red)")
    ax.set_title("Raw relationship")
    vc.savefig(fig, FIGDIR / "headline_calibration_and_relationship.png")
    cols = ECON + ["ca"]
    d2 = dd.dropna(subset=cols)
    X = StandardScaler().fit_transform(d2[cols].to_numpy(float))
    y2 = d2["y_ct_win"].to_numpy()
    est = LogisticRegression(C=1.0, max_iter=500).fit(X, y2).coef_[0]
    boot = vc.cluster_bootstrap(d2["match_name"].to_numpy(), lambda r: LogisticRegression(C=1.0, max_iter=500).fit(X[r], y2[r]).coef_[0], 200, seed=11)
    lo, hi = vc.pctl_ci(boot)
    pd.DataFrame({"feature": cols, "std_coef": est, "lo": lo, "hi": hi}).to_csv(OUTDIR / "headline_coefficients.csv", index=False)


def run_sweep(fr, rt, args):
    vc.log("2c: AUC-vs-time sweep")
    ts = list(range(args.sweep_start, args.sweep_stop + 1, args.sweep_step))
    fk = rt.set_index("rk")["first_kill_t"].fillna(np.inf)
    rl = rt.set_index("rk")["round_len"]
    sweep_models = {"econ": ECON, "alive_count": ALIVE, "ctrl_instant": ["ca_inst"], "ctrl_expanding": ["ca"],
                    "voronoi_expanding": ["vor"], "econ+alive_count": ECON + ALIVE, "econ+ctrl_expanding": ECON + ["ca"],
                    "econ+voronoi_expanding": ECON + ["vor"], "econ+alive+ctrl_expanding": ECON + ALIVE + ["ca"],
                    "econ+alive+voronoi_expanding": ECON + ALIVE + ["vor"]}
    pairs_all = [("econ+ctrl_expanding", "econ"), ("econ+alive+ctrl_expanding", "econ+alive_count"),
                 ("econ+alive+voronoi_expanding", "econ+alive_count"), ("econ+alive+ctrl_expanding", "econ+alive+voronoi_expanding")]
    pairs_5v5 = [("econ+ctrl_expanding", "econ"), ("econ+voronoi_expanding", "econ"), ("econ+ctrl_expanding", "econ+voronoi_expanding")]
    alive_models = ("alive_count", "econ+alive_count", "econ+alive+ctrl_expanding", "econ+alive+voronoi_expanding")
    rows, drows = [], []
    for t in ts:
        end = pd.Series(float(t), index=rt["rk"])
        f = window_features(fr, rt, end, None)
        f = f[f["rk"].map(rl) >= t]
        inst = fr[fr["t_rel"] <= t].groupby("rk")["ca"].last().rename("ca_inst")
        f = f.merge(inst, on="rk", how="left")
        for pop, sub in (("all rounds running at t", f), ("still 5v5 at t", f[f["rk"].map(fk) > t])):
            models = dict(sweep_models)
            if pop == "still 5v5 at t":          # head-count is constant -> alive-based models are meaningless
                for mname in alive_models:
                    models.pop(mname)
            pairs = [p for p in (pairs_5v5 if pop == "still 5v5 at t" else pairs_all) if p[0] in models and p[1] in models]
            m, d, dd, _ = safe_eval(sub, models, args.sweep_boot, pairs=pairs)
            if m.empty:
                continue
            m.insert(0, "t", t); m.insert(1, "population", pop)
            rows.append(m)
            if len(d):
                d.insert(0, "t", t); d.insert(1, "population", pop)
                drows.append(d)
    if not rows:
        vc.log("  sweep: no usable populations")
        return
    sw = pd.concat(rows)
    sw.to_csv(OUTDIR / "sweep_auc.csv", index=False)
    sd = pd.concat(drows) if drows else pd.DataFrame()
    if len(sd):
        sd.to_csv(OUTDIR / "sweep_deltas.csv", index=False)
    names = sorted(sw["model"].unique())
    colors = {n: f"C{i % 10}" for i, n in enumerate(names)}     # ONE colour per model across every panel
    styles = {n: "--" if n.startswith("voronoi") or "voronoi" in n else "-" for n in names}
    fig, axs = plt.subplots(2, 3, figsize=(18, 9))
    for ax, pop in zip(axs[0, :2], ("all rounds running at t", "still 5v5 at t")):
        g = sw[sw["population"] == pop]
        for name, grp in g.groupby("model"):
            grp = grp.sort_values("t")
            ax.plot(grp["t"], grp["auc"], marker="o", ms=3, color=colors[name], ls=styles[name], label=name)
            ax.fill_between(grp["t"], grp["auc_lo"], grp["auc_hi"], color=colors[name], alpha=.10)
        ax.axhline(0.5, color="k", ls=":", lw=.8)
        ax.set_xlabel("seconds since freeze end"); ax.set_ylabel("grouped-CV AUC (predict CT win)"); ax.set_title(pop)
        ax.legend(fontsize=7, ncol=2)
    n = sw.drop_duplicates(["t", "population"])
    for pop, grp in n.groupby("population"):
        axs[0, 2].plot(grp["t"], grp["n_rounds"], marker="o", label=pop)
    axs[0, 2].set_xlabel("seconds since freeze end"); axs[0, 2].set_ylabel("rounds in sample")
    axs[0, 2].set_title("Sample size n(t) (survivorship)"); axs[0, 2].legend(fontsize=8)
    for ax, pop in zip(axs[1, :2], ("all rounds running at t", "still 5v5 at t")):
        g = sd[sd["population"] == pop] if len(sd) else pd.DataFrame()
        if len(g):
            for (a, b), grp in g.groupby(["a", "b"]):
                grp = grp.sort_values("t")
                ax.plot(grp["t"], grp["delta_auc"], marker="o", ms=3, label=f"{a} - {b}")
                ax.fill_between(grp["t"], grp["lo"], grp["hi"], alpha=.12)
        ax.axhline(0, color="k", ls="--", lw=.8)
        ax.set_xlabel("seconds since freeze end"); ax.set_ylabel("paired delta AUC (95% match-clustered CI)")
        ax.set_title(f"Does control add information? ({pop})"); ax.legend(fontsize=7)
    axs[1, 2].axis("off")
    axs[1, 2].text(0.0, 0.95, "Bottom row = paired AUC differences.\n\nLeft: control added on top of economy + head-count\n"
                   "(the test of whether control is more than the score).\nRight: rounds still 5v5, so head-count is constant;\n"
                   "any gain is positioning alone.\n\nVoronoi rows are the like-for-like baseline.", va="top", fontsize=10)
    vc.savefig(fig, FIGDIR / "auc_vs_time_sweep.png")
    if len(sd):
        k = sd[(sd["population"] == "all rounds running at t") & (sd["a"] == "econ+alive+ctrl_expanding") & (sd["b"] == "econ+alive_count")]
        if len(k):
            vc.log("  control beyond head-count + economy (all running rounds), delta AUC by t:\n" +
                   "\n".join(f"    t={r_['t']:>3.0f}s  {r_['delta_auc']:+.4f} [{r_['lo']:+.4f}, {r_['hi']:+.4f}]" for _, r_ in k.iterrows()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caps", type=float, nargs="*", default=[20, 30, 40], help="pre-engagement caps (s)")
    ap.add_argument("--headline-cap", type=float, default=30)
    ap.add_argument("--min-pre", type=float, default=5.0, help="drop rounds whose first kill is earlier than this (s)")
    ap.add_argument("--fixed", type=float, nargs="*", default=[15, 30, 45, 55])
    ap.add_argument("--pct", type=float, nargs="*", default=[0.25, 0.5])
    ap.add_argument("--timer-s", type=float, default=115.0)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--sweep-start", type=int, default=5)
    ap.add_argument("--sweep-stop", type=int, default=60)
    ap.add_argument("--sweep-step", type=int, default=5)
    ap.add_argument("--sweep-boot", type=int, default=200)
    ap.add_argument("--skip", nargs="*", default=[], choices=["windows", "sweep"])
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    vc.ensure_dir(FIGDIR)
    rt, fr = load_data()
    vc.log(f"{len(rt)} rounds, {rt['match_name'].nunique()} matches, {len(fr)} frames")
    if "windows" not in args.skip:
        headline_figures(run_windows(fr, rt, args), args)
    if "sweep" not in args.skip:
        run_sweep(fr, rt, args)


if __name__ == "__main__":
    main()
