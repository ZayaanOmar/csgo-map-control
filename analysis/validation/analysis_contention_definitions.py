"""
analysis_contention_definitions.py   (Metric 1 re-run with a weak-mass-robust definition)
=========================================================================================
Metric 1 (contention) counts nodes whose control_val lies in [0.4, 0.6]. Because control_val is a sigmoid of
(ct_mu - t_mu), a node where BOTH teams are merely faint also scores ~0.5 — it is "contested" only by symmetry,
nobody is actually fighting for it. This script recomputes contention under several definitions side by side:

  all nodes in band            contested_all           (Metric 1 as I understand it: [0.4,0.6] over ALL nodes)
  active nodes in band         contested_active        (denominator = active footprint, ct_mu+t_mu > exp(-2))
  robust, m = 0.25 / 0.5 / 1   contested_strict_<m>    band AND both teams >= m influence mass, over ACTIVE nodes
                               contested_strict_all_<m>   ... over ALL nodes
(m = 1.0 means each team has at least one full-strength player's worth of influence at the node.)

CHECK against your own Metric 1: the 'all nodes in band' row should reproduce your reported Metric 1 value if my
reading of its definition (fraction of ALL nodes per frame, averaged over frames) is right. If it does not, tell me
how Metric 1 is actually aggregated and I will align this script.

Outputs: outputs/metrics/contention/{contention_definitions.csv, contention_timeseries.csv,
         contention_vs_outcome.csv, figures/*.png}; registry keys metric1.*
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = vc.METRICS / "contention"
FIGDIR = OUTDIR / "figures"
N_BOOT = vc.DEFAULT_BOOT
TMAX = 100
TAGS = [vc.mass_tag(m) for m in vc.MASS_THRS]


def main():
    vc.ensure_dir(FIGDIR)
    defs = {"all nodes in band (Metric 1)": "contested_all", "active nodes in band": "contested_active"}
    for t in TAGS:
        defs[f"robust m={t}, over active nodes"] = f"contested_strict_{t}"
        defs[f"robust m={t}, over ALL nodes"] = f"contested_strict_all_{t}"
    cols = ["round", "t_rel", "ct_alive", "t_alive", "active_frac"] + list(defs.values())
    fr = vc.load_frame_summaries(columns=cols)
    fr = fr[fr["t_rel"] >= 0].copy()
    fr["rk"] = fr["match_name"] + ":" + fr["round"].astype(str)
    mn = fr["match_name"].to_numpy()
    zero = np.zeros(len(fr), int)
    vc.log(f"{len(fr)} frames from {fr['match_name'].nunique()} matches (ACTIVE_THR={vc.ACTIVE_THR:.4f})")

    rows = []
    for lab, col in defs.items():
        for scope, mask in (("all frames", np.ones(len(fr), bool)), ("first 30 s", (fr["t_rel"] <= 30).to_numpy()),
                            ("5v5 frames", ((fr["ct_alive"] == 5) & (fr["t_alive"] == 5)).to_numpy())):
            v = fr[col].to_numpy(float)
            m, lo, hi, c = vc.boot_binned_mean(mn[mask], zero[mask], v[mask], 1, N_BOOT, seed=1)
            rows.append({"definition": lab, "column": col, "scope": scope, "mean": m[0], "lo": lo[0], "hi": hi[0], "n_frames": int(c[0])})
            if scope == "all frames":
                vc.record_result(f"metric1.{col}", m[0], lo[0], hi[0], int(c[0]), "mean per-frame fraction of nodes (see column definition)")
    t = pd.DataFrame(rows)
    t.to_csv(OUTDIR / "contention_definitions.csv", index=False)
    piv = t.pivot_table(index=["definition"], columns="scope", values="mean", sort=False).round(4)
    vc.log(piv.to_string())
    base = t[(t["column"] == "contested_all") & (t["scope"] == "all frames")]["mean"].iloc[0]
    for tag in TAGS:
        r = t[(t["column"] == f"contested_strict_all_{tag}") & (t["scope"] == "all frames")]["mean"].iloc[0]
        vc.log(f"  share of Metric-1 contention that survives the robust filter (m={tag}): {r / base:.1%}   "
               f"=> {1 - r / base:.1%} of 'contested' nodes carry less than {tag} mass on at least one side")
        vc.record_result(f"metric1.robust_share_{tag}", r / base, note="contested_strict_all / contested_all, mean over frames")

    # ------------------------------------------------ time profile
    fr["bin"] = np.clip(np.rint(fr["t_rel"]).astype(int), 0, TMAX)
    ts, rows = {}, []
    for lab, col in (("all nodes in band (Metric 1)", "contested_all"), ("active nodes in band", "contested_active"),
                     (f"robust m={vc.mass_tag(vc.STRICT_MIN_MASS)}, over ALL nodes", f"contested_strict_all_{vc.mass_tag(vc.STRICT_MIN_MASS)}")):
        m, lo, hi, c = vc.boot_binned_mean(mn, fr["bin"].to_numpy(), fr[col].to_numpy(float), TMAX + 1, 300, seed=2)
        ts[lab] = (m, lo, hi)
        rows.append(pd.DataFrame({"definition": lab, "t": np.arange(TMAX + 1), "mean": m, "lo": lo, "hi": hi, "n_frames": c}))
    pd.concat(rows).to_csv(OUTDIR / "contention_timeseries.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3))
    x = np.arange(TMAX + 1)
    for lab, (m, lo, hi) in ts.items():
        axs[0].plot(x, m, label=lab); axs[0].fill_between(x, lo, hi, alpha=.15)
    axs[0].set_xlabel("seconds since freeze end"); axs[0].set_ylabel("fraction of nodes"); axs[0].set_title("Contention under three definitions"); axs[0].legend(fontsize=8)
    keys = list(ts)
    ratio = ts[keys[2]][0] / ts[keys[0]][0]
    axs[1].plot(x, ratio)
    axs[1].set_ylim(0, 1); axs[1].set_xlabel("seconds since freeze end"); axs[1].set_ylabel("robust / original contention")
    axs[1].set_title("Share of Metric-1 contention that is genuinely two-sided")
    vc.savefig(fig, FIGDIR / "contention_definitions.png")

    # ------------------------------------------------ relation to the round outcome (per-round means)
    rt = vc.load_round_table()
    rt["rk"] = rt["match_name"] + ":" + rt["round"].astype(str)
    fr = fr.merge(rt[["rk", "y_ct_win"]], on="rk", how="inner")
    rows = []
    for scope, mask in (("first 30 s", fr["t_rel"] <= 30), ("all frames", fr["t_rel"] >= 0)):
        sub = fr[mask]
        for lab, col in defs.items():
            g = sub.groupby("rk").agg(v=(col, "mean"), y=("y_ct_win", "first"), mn=("match_name", "first")).dropna()
            a, b = g[g["y"] == 1]["v"].to_numpy(), g[g["y"] == 0]["v"].to_numpy()
            if len(a) < 5 or len(b) < 5:
                continue
            v, y, m_ = g["v"].to_numpy(), g["y"].to_numpy(), g["mn"].to_numpy()
            def stat(r):
                x1, x0 = v[r][y[r] == 1], v[r][y[r] == 0]
                return vc.cohens_d(x1, x0) if len(x1) > 2 and len(x0) > 2 else np.nan
            boot = vc.cluster_bootstrap(m_, stat, min(N_BOOT, 500), seed=3)
            lo, hi = vc.pctl_ci(boot)
            rows.append({"scope": scope, "definition": lab, "mean_ct_win": a.mean(), "mean_t_win": b.mean(), "cohens_d": vc.cohens_d(a, b), "d_lo": float(lo), "d_hi": float(hi), "n_rounds": len(g)})
    o = pd.DataFrame(rows)
    o.to_csv(OUTDIR / "contention_vs_outcome.csv", index=False)
    vc.log("Does contention differ between CT-win and T-win rounds? (Cohen's d, CT-win minus T-win)\n" + o.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
