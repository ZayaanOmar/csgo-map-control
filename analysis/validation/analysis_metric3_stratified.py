"""
analysis_metric3_stratified.py   (Part 2e)
==========================================
Re-runs Metric 3 (per-round mean CT control, split by round winner, Welch t + Cohen's d) while
HOLDING THE HEAD-COUNT FIXED, to test whether the win/loss control gap is just "the winning team
has more players alive".

Strata (frame filters, then the usual two-stage frame -> round averaging):
  all frames            reproduces Metric 3 (check against your reported d: 1.62 full-map, 1.84 active)
  5v5, 4v4, 3v3, 2v2    only frames with exactly equal alive counts k
  equal numbers pooled  frames with ct_alive == t_alive
  |diff| <= 1           frames within one player of equal
  first 30 s            all frames with t_rel <= 30 (early-round only)
Rounds need >= MIN_FRAMES frames in a stratum to contribute.

Variants: ctrl_full (all nodes); ctrl_active (nodes with ct_mu+t_mu > ACTIVE_THR = exp(-2), i.e. 2 sigma);
and ctrl_active_legacy (the project's ORIGINAL 3-sigma threshold, exp(-4.5)) — the latter exists ONLY so the
reproduction check compares like with like against your reported Metric 3 numbers.
CIs: cluster bootstrap over matches for the mean difference and Cohen's d.
If the gap shrinks sharply at equal numbers, the original Metric 3 was partly a head-count effect;
if it survives, the early/equal-numbers evidence for control is substantially strengthened.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = vc.METRICS / "metric3_stratified"
FIGDIR = OUTDIR / "figures"
MIN_FRAMES = 3
N_BOOT = vc.DEFAULT_BOOT
REPORTED_D = {"ctrl_full": 1.62, "ctrl_active_legacy": 1.84}   # from project notes (Metric 3, all frames)
VARIANTS = ("ctrl_full", "ctrl_active", "ctrl_active_legacy")


def _d(win, loss):
    sp = np.sqrt(((len(win) - 1) * win.var(ddof=1) + (len(loss) - 1) * loss.var(ddof=1)) / (len(win) + len(loss) - 2))
    return (win.mean() - loss.mean()) / sp if sp > 0 else np.nan


def main():
    vc.ensure_dir(FIGDIR)
    rt = vc.load_round_table()
    rt["rk"] = rt["match_name"] + ":" + rt["round"].astype(str)
    fr = vc.load_frame_summaries(columns=["round", "t_rel", "ct_alive", "t_alive", "ctrl_full", "ctrl_active", "ctrl_active_legacy"])
    fr["rk"] = fr["match_name"] + ":" + fr["round"].astype(str)
    fr = fr[fr["t_rel"] >= 0].merge(rt[["rk", "y_ct_win"]], on="rk", how="inner")
    strata = {"all frames": np.ones(len(fr), bool)}
    for k in (5, 4, 3, 2):
        strata[f"{k}v{k}"] = ((fr["ct_alive"] == k) & (fr["t_alive"] == k)).to_numpy()
    strata["equal numbers pooled"] = (fr["ct_alive"] == fr["t_alive"]).to_numpy()
    strata["|diff| <= 1"] = ((fr["ct_alive"] - fr["t_alive"]).abs() <= 1).to_numpy()
    strata["first 30 s"] = (fr["t_rel"] <= 30).to_numpy()
    rows = []
    for sname, mask in strata.items():
        sub = fr[mask]
        for var in VARIANTS:
            g = sub.groupby("rk").agg(v=(var, "mean"), n=(var, "size"), y=("y_ct_win", "first"), match_name=("match_name", "first"))
            g = g[(g["n"] >= MIN_FRAMES) & g["v"].notna()]
            win, loss = g[g["y"] == 1]["v"].to_numpy(), g[g["y"] == 0]["v"].to_numpy()
            if len(win) < 5 or len(loss) < 5:
                rows.append({"stratum": sname, "variant": var, "n_ct_win": len(win), "n_t_win": len(loss)})
                continue
            t, p = stats.ttest_ind(win, loss, equal_var=False)
            v, y, mn = g["v"].to_numpy(), g["y"].to_numpy(), g["match_name"].to_numpy()
            def stat(r):
                a, b = v[r][y[r] == 1], v[r][y[r] == 0]
                return np.array([a.mean() - b.mean(), _d(a, b)]) if len(a) > 2 and len(b) > 2 else np.array([np.nan, np.nan])
            boot = vc.cluster_bootstrap(mn, stat, N_BOOT, seed=9)
            lo, hi = vc.pctl_ci(boot)
            rows.append({"stratum": sname, "variant": var, "n_ct_win": len(win), "n_t_win": len(loss), "mean_ct_win": win.mean(),
                         "mean_t_win": loss.mean(), "diff": win.mean() - loss.mean(), "diff_lo": lo[0], "diff_hi": hi[0],
                         "cohens_d": _d(win, loss), "d_lo": lo[1], "d_hi": hi[1], "welch_t": t, "p": p})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUTDIR / "metric3_stratified.csv", index=False)
    vc.log(tbl.round(4).to_string(index=False))
    for var, rep_d in REPORTED_D.items():
        r = tbl[(tbl["stratum"] == "all frames") & (tbl["variant"] == var)]
        if len(r) and "cohens_d" in r:
            vc.log(f"reproduction check [{var}]: this script d={r['cohens_d'].iloc[0]:.2f} vs reported {rep_d:.2f}  "
                   f"{'OK' if abs(r['cohens_d'].iloc[0] - rep_d) < 0.15 else '<-- DIFFERS: check Metric 3 definition/filters (e.g. warmup, window, sign convention)'}")
    for var in VARIANTS:
        r = tbl[(tbl["stratum"] == "equal numbers pooled") & (tbl["variant"] == var)]
        if len(r) and "cohens_d" in r:
            vc.record_result(f"metric3.cohens_d_equal_numbers.{var}", r["cohens_d"].iloc[0], r["d_lo"].iloc[0], r["d_hi"].iloc[0], r["n_ct_win"].iloc[0] + r["n_t_win"].iloc[0])
    a2, a3 = tbl[(tbl["stratum"] == "all frames") & (tbl["variant"] == "ctrl_active")], tbl[(tbl["stratum"] == "all frames") & (tbl["variant"] == "ctrl_active_legacy")]
    if len(a2) and len(a3) and "cohens_d" in a2:
        vc.log(f"threshold effect (all frames): ctrl_active d={a2['cohens_d'].iloc[0]:.2f} at 2 sigma vs {a3['cohens_d'].iloc[0]:.2f} at the original 3 sigma")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for off, var in zip((-0.22, 0.0, 0.22), VARIANTS):
        t = tbl[(tbl["variant"] == var)].dropna(subset=["cohens_d"])
        x = np.arange(len(t)) + off
        ax.errorbar(x, t["cohens_d"], yerr=[t["cohens_d"] - t["d_lo"], t["d_hi"] - t["cohens_d"]], fmt="o", capsize=3, label=var)
        ax.set_xticks(np.arange(len(t))); ax.set_xticklabels(t["stratum"], rotation=30, ha="right")
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("Cohen's d (CT-win vs T-win rounds)"); ax.set_title("Metric 3 effect size when the head-count is held fixed"); ax.legend()
    vc.savefig(fig, FIGDIR / "metric3_effect_by_stratum.png")


if __name__ == "__main__":
    main()
