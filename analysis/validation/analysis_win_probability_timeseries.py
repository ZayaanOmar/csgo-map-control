"""
analysis_win_probability_timeseries.py   (Part 2d)
==================================================
Win probability and control margin over time — without circularity.

BASELINE WP model  : features that never touch the nav mesh — seconds since freeze end, alive
                     counts/difference, HP difference, alive-equipment difference, bomb planted,
                     time since plant, round-level equipment ratio, pistol flag.
AUGMENTED WP model : baseline + current-frame control features (ctrl_active, ctrl_full,
                     contested_active, Voronoi margin).
Both are gradient-boosted classifiers (sklearn HistGradientBoosting) producing OUT-OF-FOLD
predictions with grouped (by match) K-fold, so every frame's WP comes from a model that never saw
that match. Because the baseline has never seen a nav-mesh feature, plotting it against control
margin is independent evidence; the augmented-vs-baseline log-loss difference is the
quantitative "control adds information" claim.

Outputs (outputs/metrics/win_probability/):
  wp_oof_predictions.parquet     per-frame OOF WP_base / WP_aug (+ keys)  [used by case studies]
  wp_logloss_by_time.csv         base vs augmented log-loss by 5 s bin, paired CI of the improvement
  wp_crosscorr.csv               cross-correlation of changes: d(control) vs d(logit WP_base) at lags
  figures/ aggregate_forward.png, aggregate_round_end_aligned.png, wp_improvement_by_time.png,
           cross_correlation.png, case_study_*.png

Lead-lag note: positive lag L means control changes at frame t are correlated with WP changes at
frame t+L (control LEADS). Frames are ~1 s apart; kills move both series at once, so lag 0 is
inflated by that shared jump — read the SHAPE of the lag curve, not just lag 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = vc.METRICS / "win_probability"
FIGDIR = OUTDIR / "figures"
BASE = ["t_rel", "ct_alive", "t_alive", "alive_diff", "hp_diff", "eq_diff_alive_k", "bomb_planted", "since_plant",
        "log_eq_ratio", "is_pistol"]
CTRL = ["ca", "cf", "cont", "vor"]
LAGS = list(range(-8, 9))


def load(args):
    rt = vc.load_round_table()
    rt["rk"] = rt["match_name"] + ":" + rt["round"].astype(str)
    fr = vc.load_frame_summaries(columns=["round", "frame_idx", "t_rel", "ct_alive", "t_alive", "ct_hp", "t_hp", "ct_eq",
                                          "t_eq", "bomb_planted", "ctrl_full", "ctrl_active", "contested_active", "vor_margin"])
    fr["rk"] = fr["match_name"] + ":" + fr["round"].astype(str)
    fr = fr[fr["t_rel"] >= 0].copy()
    r = rt[["rk", "y_ct_win", "log_eq_ratio", "is_pistol_round", "plant_t"]].copy()
    fr = fr.merge(r, on="rk", how="inner")
    fr["is_pistol"] = fr["is_pistol_round"].astype(float)
    fr["alive_diff"] = fr["ct_alive"] - fr["t_alive"]
    fr["hp_diff"] = fr["ct_hp"] - fr["t_hp"]
    fr["eq_diff_alive_k"] = (fr["ct_eq"] - fr["t_eq"]) / 1000.0
    fr["bomb_planted"] = fr["bomb_planted"].astype(float)
    fr["since_plant"] = np.where(fr["bomb_planted"] > 0, np.clip(fr["t_rel"] - fr["plant_t"].fillna(fr["t_rel"]), 0, None), 0.0)
    fr["ca"], fr["cf"] = fr["ctrl_active"] - 0.5, fr["ctrl_full"] - 0.5
    fr["cont"], fr["vor"] = fr["contested_active"], fr["vor_margin"]
    fr = fr.sort_values(["rk", "t_rel"]).reset_index(drop=True)
    fr["end_t"] = fr.groupby("rk")["t_rel"].transform("max")
    return rt, fr


def oof_wp(fr, cols, n_splits, seed=0):
    X, y, g = fr[cols].to_numpy(float), fr["y_ct_win"].to_numpy(), fr["match_name"].to_numpy()
    p = np.zeros(len(fr))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=4, random_state=seed)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return np.clip(p, 1e-5, 1 - 1e-5)


def row_loss(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def curves(fr, col, x_col, lo_x, hi_x, ax, label_prefix=""):
    for outcome, lab in ((1, "CT wins"), (0, "T wins")):
        g = fr[(fr["y_ct_win"] == outcome) & (fr[x_col] >= lo_x) & (fr[x_col] <= hi_x)]
        b = np.rint(g[x_col].to_numpy() - lo_x).astype(int)
        m, lo, hi, c = vc.boot_binned_mean(g["match_name"].to_numpy(), b, g[col].to_numpy(), int(hi_x - lo_x) + 1, vc.DEFAULT_BOOT, seed=1)
        x = np.arange(lo_x, hi_x + 1)
        ax.plot(x, m, label=f"{label_prefix}{lab}")
        ax.fill_between(x, lo, hi, alpha=.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame-stride", type=int, default=1, help="train/evaluate on every k-th frame (speed)")
    ap.add_argument("--max-t", type=int, default=100)
    ap.add_argument("--n-cases", type=int, default=4)
    args = ap.parse_args()
    vc.ensure_dir(FIGDIR)
    rt, fr = load(args)
    if args.frame_stride > 1:
        fr = fr[fr["frame_idx"] % args.frame_stride == 0].reset_index(drop=True)
    vc.log(f"{fr['rk'].nunique()} rounds, {len(fr)} frames, {fr['match_name'].nunique()} matches")
    ns = int(min(5, fr["match_name"].nunique()))
    fr["wp_base"] = oof_wp(fr, BASE, ns)
    fr["wp_aug"] = oof_wp(fr, BASE + CTRL, ns)
    y = fr["y_ct_win"].to_numpy()
    fr[["match_name", "round", "frame_idx", "t_rel", "y_ct_win", "wp_base", "wp_aug"]].to_parquet(OUTDIR / "wp_oof_predictions.parquet", index=False)
    vc.log(f"  overall AUC base={roc_auc_score(y, fr['wp_base']):.4f} aug={roc_auc_score(y, fr['wp_aug']):.4f}; "
           f"logloss base={row_loss(y, fr['wp_base']).mean():.4f} aug={row_loss(y, fr['wp_aug']).mean():.4f}")

    # --------------------------------------- improvement by time bin (paired cluster bootstrap)
    fr["tbin"] = np.clip((fr["t_rel"] // 5).astype(int), 0, args.max_t // 5 - 1)
    gain = row_loss(y, fr["wp_base"].to_numpy()) - row_loss(y, fr["wp_aug"].to_numpy())
    nb = args.max_t // 5
    m, lo, hi, c = vc.boot_binned_mean(fr["match_name"].to_numpy(), fr["tbin"].to_numpy(), gain, nb, vc.DEFAULT_BOOT, seed=2)
    pd.DataFrame({"t_start": np.arange(nb) * 5, "logloss_gain_aug_over_base": m, "lo": lo, "hi": hi, "n_frames": c}).to_csv(OUTDIR / "wp_logloss_by_time.csv", index=False)
    mm, ll, hh, _ = vc.boot_binned_mean(fr["match_name"].to_numpy(), np.zeros(len(fr), int), gain, 1, vc.DEFAULT_BOOT, seed=3)
    vc.record_result("wp.logloss_gain_aug_over_base", mm[0], ll[0], hh[0], len(fr), "mean per-frame log-loss improvement from adding control features")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(np.arange(nb) * 5 + 2.5, m, yerr=[m - lo, hi - m], fmt="o-", capsize=2)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("seconds since freeze end"); ax.set_ylabel("log-loss improvement (augmented over baseline)")
    ax.set_title("Where do control features add information to win probability?")
    vc.savefig(fig, FIGDIR / "wp_improvement_by_time.png")

    # --------------------------------------- aggregate plots
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    curves(fr, "ca", "t_rel", 0, args.max_t, axs[0]); axs[0].set_ylabel("CT control margin (active footprint) - 0.5")
    curves(fr, "wp_base", "t_rel", 0, args.max_t, axs[1]); axs[1].set_ylabel("baseline P(CT win)  (no control features)")
    cnt = fr.groupby(np.rint(fr["t_rel"]).astype(int))["rk"].nunique()
    axs[2].plot(cnt.index, cnt.values); axs[2].set_ylabel("rounds still running"); axs[2].set_xlim(0, args.max_t)
    for a in axs:
        a.set_xlabel("seconds since freeze end")
    axs[0].legend(); axs[1].legend()
    vc.savefig(fig, FIGDIR / "aggregate_forward.png")
    fr["to_end"] = fr["t_rel"] - fr["end_t"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    curves(fr, "ca", "to_end", -60, 0, axs[0]); axs[0].set_ylabel("CT control margin - 0.5")
    curves(fr, "wp_base", "to_end", -60, 0, axs[1]); axs[1].set_ylabel("baseline P(CT win)")
    for a in axs:
        a.set_xlabel("seconds before the round ended"); a.legend()
    vc.savefig(fig, FIGDIR / "aggregate_round_end_aligned.png")

    # --------------------------------------- cross-correlation of CHANGES (lead-lag)
    g = fr.groupby("rk", sort=False)
    fr["d_ca"] = g["ca"].diff()
    fr["lw"] = np.log(fr["wp_base"] / (1 - fr["wp_base"]))
    fr["d_w"] = fr.groupby("rk", sort=False)["lw"].diff()
    codes, uniq = pd.factorize(fr["match_name"].to_numpy())
    G = len(uniq)
    stats_ = np.zeros((G, len(LAGS), 6))
    for li, L in enumerate(LAGS):
        dw = fr.groupby("rk", sort=False)["d_w"].shift(-L).to_numpy()
        x = fr["d_ca"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(dw)
        xs, ys, cs = x[ok], dw[ok], codes[ok]
        for k, v in enumerate((np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys)):
            np.add.at(stats_[:, li, k], cs, v)
    def corr(S):
        n, sx, sy, sxx, syy, sxy = S.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return (sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))
    est = corr(stats_.sum(0))
    rng = np.random.default_rng(5)
    boot = np.array([corr(stats_[rng.integers(0, G, G)].sum(0)) for _ in range(vc.DEFAULT_BOOT)])
    lo, hi = vc.pctl_ci(boot)
    pd.DataFrame({"lag_frames": LAGS, "corr": est, "lo": lo, "hi": hi}).to_csv(OUTDIR / "wp_crosscorr.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(LAGS, est, yerr=[est - lo, hi - est], fmt="o-", capsize=2)
    ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="grey", ls="--", lw=.8)
    ax.set_xlabel("lag L (frames, ~1 s): corr( d control[t], d logit WP_base[t+L] )"); ax.set_ylabel("correlation")
    ax.set_title("Do control changes lead win-probability changes?")
    vc.savefig(fig, FIGDIR / "cross_correlation.png")
    pk = int(np.nanargmax(est))
    vc.record_result("wp.xcorr_lag0", est[LAGS.index(0)], lo[LAGS.index(0)], hi[LAGS.index(0)], len(fr), "corr of changes, control vs baseline WP")
    vc.record_result("wp.xcorr_peak_lag_frames", LAGS[pk], n=len(fr), note="lag of maximum cross-correlation (positive => control leads)")
    vc.log(f"  cross-correlation: lag0={est[LAGS.index(0)]:.3f}; peak at lag {LAGS[pk]} ({est[pk]:.3f})")

    # --------------------------------------- case studies
    kills = vc.concat_tables(vc.T_KILLS, ext=".csv", columns=["round", "t_rel", "attacker_side", "valid_duel"])
    kills = kills[kills["valid_duel"]]
    kills["rk"] = kills["match_name"] + ":" + kills["round"].astype(str)
    rng = np.random.default_rng(args.n_cases)
    cand = rt.merge(fr.groupby("rk").size().rename("nf").reset_index(), on="rk")
    cand = cand[cand["nf"] >= 25]
    picks = []
    for outcome in (1, 0):
        c = cand[cand["y_ct_win"] == outcome]
        picks += list(c["rk"].to_numpy()[rng.permutation(len(c))[: args.n_cases // 2]])
    for i, rk in enumerate(picks):
        d = fr[fr["rk"] == rk]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(d["t_rel"], d["ca"], color="tab:green", label="CT control margin (left)")
        ax.axhline(0, color="grey", lw=.6)
        ax.set_ylabel("control margin - 0.5"); ax.set_xlabel("seconds since freeze end")
        ax2 = ax.twinx()
        ax2.plot(d["t_rel"], d["wp_base"], color="k", label="WP baseline (right)")
        ax2.plot(d["t_rel"], d["wp_aug"], color="purple", ls="--", label="WP + control (right)")
        ax2.set_ylim(0, 1); ax2.set_ylabel("P(CT win)")
        for _, k in kills[kills["rk"] == rk].iterrows():
            ax.axvline(k["t_rel"], color="tab:blue" if k["attacker_side"] == "CT" else "tab:orange", alpha=.5, lw=1)
        pt = rt.loc[rt["rk"] == rk, "plant_t"].iloc[0]
        if np.isfinite(pt):
            ax.axvline(pt, color="red", ls=":", lw=2)
        ax.set_title(f"{rk[:40]}…  winner: {'CT' if d['y_ct_win'].iloc[0] else 'T'}   (blue/orange = CT/T kill, red dotted = plant)", fontsize=8)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        vc.savefig(fig, FIGDIR / f"case_study_{i + 1}.png")


if __name__ == "__main__":
    main()
