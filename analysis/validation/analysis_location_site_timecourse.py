"""
analysis_location_site_timecourse.py   (site-choice timing analysis, step 2 of 2)
================================================================================
How early is the planted site readable from positioning?  For each window end c (from
build_site_timecourse_features.py) the area-level model is cross-validated on the SAME rounds and the SAME folds:

    M_const   A/B base rate
    M1        area net control            (ridge logistic, nested lambda; as in analysis_location_site.py)
    M1p       area net + presence         (with --presence)

REPORTED PER WINDOW END
    log-loss and AUC of M1 (and M1p) with 95% match-bootstrap CIs
    improvement over M_const (log-loss) with CI
    "gap to the final window": how much worse than the LAST window end each earlier one is (log-loss increase and
    AUC drop, paired bootstrap on the same match resamples), i.e. how much of the final readability is already
    there. The earliest window end whose gap CI includes 0 is the time by which the site is, statistically,
    as readable as it ever gets here.

Interpretation notes
    * Windows are cumulative from t0 = 5 s, so "readable by c" means from everything seen up to c.
    * Rounds whose first kill / plant comes before a cap use their shorter window, so late caps are a mixture of
      "time" and "round pace". The sample itself never changes.
    * The sample is the slow-execute subset of the main site analysis, so this describes how quickly such plans
      become readable, not when teams decide.
    * AUC is unreliable when a model is shrunk to ~intercept-only (typically at the earliest windows, where it
      can sit below 0.5): M_const's AUC is meaningless (fold-wise constants), and a shrunk M1 inherits the same
      artefact. Read the early windows by the log-loss improvement over M_const.

Outputs (outputs/metrics/location_site_timecourse/):
    timecourse_scores.csv   one row per (window end, model)
    timecourse_gaps.csv     gap to the final window per window end (M1, and M1p if present)
    figures/timecourse.png  AUC and log-loss improvement vs window end

Usage (project root):
    python analysis/validation/analysis_location_site_timecourse.py --presence
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


def fmt(c: float) -> str:
    return f"{c:g}"


def make_data(rounds: pd.DataFrame, tc: Path, cap: float, presence: bool) -> lib.Data:
    D = lib.Data()
    D.el = rounds
    D.label_col = "y_site_B"
    D.y = rounds["y_site_B"].to_numpy(int)
    D.match = rounds["match_name"].to_numpy()
    D.mcodes, D.mnames = pd.factorize(D.match)
    D.n = len(rounds)
    net = pd.read_csv(tc / f"area_net_cap{fmt(cap)}.csv").set_index("rk").loc[rounds["rk"]]
    D.mats = {"area_net": net.to_numpy(float)}
    D.names = {"area_net": list(net.columns)}
    if presence:
        D.mats["area_pres"] = pd.read_csv(tc / f"area_pres_cap{fmt(cap)}.csv").set_index("rk").loc[rounds["rk"], net.columns].to_numpy(float)
    for k, M in D.mats.items():
        if not np.isfinite(M).all():
            raise SystemExit(f"non-finite values in {k} at cap {cap}")
    return D


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tc-dir", default="outputs/predictor_timecourse")
    ap.add_argument("--folds", default="outputs/predictor/folds.csv")
    ap.add_argument("--grouping", choices=["match", "teampair"], default="match")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=vc.DEFAULT_BOOT)
    ap.add_argument("--presence", action="store_true")
    ap.add_argument("--out-dir", default="outputs/metrics/location_site_timecourse")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tc = Path(args.tc_dir)
    out = Path(args.out_dir + ("" if args.grouping == "match" else "_teampair"))
    vc.ensure_dir(out / "figures")
    meta = json.loads((tc / "timecourse_meta.json").read_text())
    caps = meta["caps"]
    rounds = pd.read_csv(tc / "rounds.csv")
    folds = pd.read_csv(args.folds)
    col = "fold_match" if args.grouping == "match" else "fold_teampair"
    repeats = args.repeats or int(folds["repeat"].nunique())
    vc.log(f"{len(rounds)} rounds, {rounds['match_name'].nunique()} matches; share at B {rounds['y_site_B'].mean():.3f}; "
           f"window ends {', '.join(fmt(c) for c in caps)} s; grouping={args.grouping}, {repeats} repeats")

    specs = {"M_const": dict(kind="const"), "M1": dict(kind="ridge", loc=["area_net"], offset=False)}
    if args.presence:
        specs["M1p"] = dict(kind="ridge", loc=["area_net", "area_pres"], offset=False)
    models = [m for m in specs if m != "M_const"]

    per_cap = {}
    rows = []
    for c in caps:
        vc.log(f"\n=== window end {fmt(c)} s ===")
        D = make_data(rounds, tc, c, args.presence)
        P, lam_log = lib.run_cv(D, specs, folds, col, repeats)
        bs = lib.bootstrap(D, P, args.n_boot, args.seed)        # same seed + same rows -> same match resamples every cap
        scores = lib.score_table(D, P, bs).set_index("model")
        comp = lib.compare(D, bs, [(m, "M_const") for m in models], []).set_index("candidate")
        per_cap[c] = bs
        for m in specs:
            r = dict(window_end=c, model=m, mean_frames=rounds[f"n_frames_cap{fmt(c)}"].mean(),
                     logloss=scores.loc[m, "logloss"], logloss_lo=scores.loc[m, "logloss_lo"], logloss_hi=scores.loc[m, "logloss_hi"],
                     auc=scores.loc[m, "auc"], auc_lo=scores.loc[m, "auc_lo"], auc_hi=scores.loc[m, "auc_hi"])
            if m != "M_const":
                r.update(dll_vs_const=comp.loc[m, "dlogloss"], dll_lo=comp.loc[m, "dlogloss_lo"], dll_hi=comp.loc[m, "dlogloss_hi"],
                         dll_p=comp.loc[m, "p_logloss"], median_lambda=float(lam_log[lam_log["model"] == m]["lam"].median()))
            rows.append(r)
        vc.log("  " + "  ".join(f"{m}: LL {scores.loc[m, 'logloss']:.4f} AUC {scores.loc[m, 'auc']:.3f} "
                                 f"dLL vs const {comp.loc[m, 'dlogloss']:+.4f} [{comp.loc[m, 'dlogloss_lo']:+.4f},{comp.loc[m, 'dlogloss_hi']:+.4f}]"
                                 for m in models))
    sc = pd.DataFrame(rows)
    sc.to_csv(out / "timecourse_scores.csv", index=False)

    # ---- gap to the final window (paired: same rows, same match resamples)
    last = caps[-1]
    W, Wsum = per_cap[last]["W"], per_cap[last]["Wsum"]
    gaps = []
    for c in caps[:-1]:
        for m in models:
            d_ll = per_cap[c]["ll"][m] - per_cap[last]["ll"][m]          # >0: earlier window is worse
            b_ll = (W @ d_ll) / Wsum
            a_c, a_l = per_cap[c]["auc"][m], per_cap[last]["auc"][m]
            est_auc = a_l(np.ones(len(d_ll))) - a_c(np.ones(len(d_ll)))
            b_auc = np.array([a_l(w) - a_c(w) for w in W])
            gaps.append(dict(window_end=c, model=m, final_window_end=last,
                             logloss_gap=d_ll.mean(), gap_lo=np.percentile(b_ll, 2.5), gap_hi=np.percentile(b_ll, 97.5),
                             auc_gap=est_auc, auc_gap_lo=np.percentile(b_auc, 2.5), auc_gap_hi=np.percentile(b_auc, 97.5)))
    gp = pd.DataFrame(gaps)
    gp.to_csv(out / "timecourse_gaps.csv", index=False)

    # ---- printout
    vc.log("\nTIMECOURSE (95% match-bootstrap CIs)")
    for m in models:
        vc.log(f"\n  {m}:  window end | mean frames | log-loss | AUC [95% CI] | improvement over constant [CI]")
        for _, r in sc[sc["model"] == m].iterrows():
            vc.log(f"    {r['window_end']:>5g} s | {r['mean_frames']:>5.0f} | {r['logloss']:.4f} | {r['auc']:.3f} "
                   f"[{r['auc_lo']:.3f},{r['auc_hi']:.3f}] | {r['dll_vs_const']:+.4f} [{r['dll_lo']:+.4f},{r['dll_hi']:+.4f}]")
        vc.log(f"  gap to the {last:g} s window ({m}): log-loss increase / AUC drop")
        for _, r in gp[gp["model"] == m].iterrows():
            flag = "  <- indistinguishable from final" if (r["gap_lo"] <= 0 <= r["gap_hi"]) else ""
            vc.log(f"    {r['window_end']:>5g} s: log-loss increase {r['logloss_gap']:+.4f} [{r['gap_lo']:+.4f},{r['gap_hi']:+.4f}]  "
                   f"AUC drop {r['auc_gap']:+.3f} [{r['auc_gap_lo']:+.3f},{r['auc_gap_hi']:+.3f}]{flag}")

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for m, colr in zip(models, ("tab:blue", "tab:orange")):
        d = sc[sc["model"] == m]
        axs[0].plot(d["window_end"], d["auc"], "o-", color=colr, label=m)
        axs[0].fill_between(d["window_end"], d["auc_lo"], d["auc_hi"], color=colr, alpha=0.2)
        axs[1].plot(d["window_end"], d["dll_vs_const"], "o-", color=colr, label=m)
        axs[1].fill_between(d["window_end"], d["dll_lo"], d["dll_hi"], color=colr, alpha=0.2)
    axs[0].axhline(0.5, color="k", ls="--", lw=1)
    axs[0].set_ylabel("AUC (A vs B plant)")
    axs[1].axhline(0, color="k", ls="--", lw=1)
    axs[1].set_ylabel("log-loss improvement over constant")
    for a in axs:
        a.set_xlabel("window end (s after freeze end; window starts at 5 s)")
        a.legend()
    fig.suptitle(f"How early is the site readable?  ({len(rounds)} rounds, fixed sample)", fontsize=10)
    vc.savefig(fig, out / "figures" / "timecourse.png")
    vc.log(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
