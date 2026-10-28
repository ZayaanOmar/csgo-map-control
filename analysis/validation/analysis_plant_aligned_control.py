"""
analysis_plant_aligned_control.py   (Part 1E, with head-count conditioning)
===========================================================================
Generalises Metric 2 (a single snapshot at the frame before the plant) into curves: the planted
site's T-perspective control (T control = 1 - control_val) from -30 s to +15 s around every plant,
split by site, by round outcome, and now by how many CTs were ALIVE at the plant.

WHY THE CONDITIONING MATTERS: the unadjusted T-win vs CT-win gap in site control at the plant is
confounded by the head-count (fewer surviving CTs => less CT influence near the site AND an easier
round for T). The conditioned analyses ask whether control still discriminates outcomes among plants
with the SAME number of CTs alive:
  1  per-stratum table by CT-alive-at-plant (k = 1..5): means, difference, Cohen's d, cluster-bootstrap CIs
  2  d after residualising control on (ct_alive, t_alive) fixed effects (single pooled number)
  3  adjusted logistic model  P(T wins | plant) ~ control + ct_alive + t_alive + site + economy
       -> standardised coefficient of control (cluster-bootstrap CI), and grouped-CV delta AUC of
          (alive+site+economy+control) over (alive+site+economy)
Caveat for the write-up: only rounds that reach a plant are included, so everything is conditional on a
plant happening.

Inputs : frame_summaries, round_metadata, bomb_plants, round_economy
Outputs: outputs/metrics/plant_aligned/{plant_aligned_curves.csv, plant_snapshot.csv,
         plant_snapshot_summary.csv, plant_by_ct_alive.csv, plant_adjusted_models.csv, figures/*.png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = vc.METRICS / "plant_aligned"
FIGDIR = OUTDIR / "figures"
PRE, POST = 30, 15
NB = PRE + POST + 1
N_BOOT = vc.DEFAULT_BOOT
MIN_PER_GROUP = 15
ECON = ["log_eq_ratio", "eq_diff_k", "is_pistol"]


def _d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def main():
    vc.ensure_dir(FIGDIR)
    rt = vc.load_round_table()
    rt = rt[rt["plant_tick"].notna() & rt["plant_site"].isin(["A", "B"])].copy()
    rt["eq_diff_k"] = rt["eq_diff"] / 1000.0 if "eq_diff" in rt else np.nan
    rt["is_pistol"] = rt["is_pistol_round"].astype(float) if "is_pistol_round" in rt else np.nan
    vc.log(f"{len(rt)} rounds with a plant ({(rt['plant_site'] == 'A').sum()} A / {(rt['plant_site'] == 'B').sum()} B)")
    if len(rt) < 50:
        raise SystemExit("too few plants to analyse")
    fr = vc.load_frame_summaries(columns=["round", "frame_idx", "tick", "tick_rate", "ct_alive", "t_alive", "site_ctrl_A", "site_ctrl_B"])
    d = fr.merge(rt[["match_name", "round", "plant_tick", "plant_site", "y_ct_win"]], on=["match_name", "round"], how="inner")
    d["rel"] = (d["tick"] - d["plant_tick"]) / d["tick_rate"]
    d["round_key"] = d["match_name"] + ":" + d["round"].astype(str)
    is_a = d["plant_site"] == "A"
    d["t_ctrl_planted"] = 1 - np.where(is_a, d["site_ctrl_A"], d["site_ctrl_B"])
    d["t_ctrl_other"] = 1 - np.where(is_a, d["site_ctrl_B"], d["site_ctrl_A"])

    # ------------------------------------------------ snapshot: last frame strictly before the plant
    pre = d[d["rel"] < 0].sort_values("rel").groupby("round_key", as_index=False).tail(1)
    snap = pre[["match_name", "round", "round_key", "plant_site", "y_ct_win", "rel", "ct_alive", "t_alive", "t_ctrl_planted"]].rename(columns={"rel": "snapshot_rel_s"})
    snap = snap.merge(rt[["match_name", "round"] + ECON], on=["match_name", "round"], how="left")
    snap["t_win"] = 1 - snap["y_ct_win"]
    snap.to_csv(OUTDIR / "plant_snapshot.csv", index=False)
    d = d.merge(snap[["round_key", "ct_alive"]].rename(columns={"ct_alive": "ct_alive_at_plant"}), on="round_key", how="left")
    d = d[(d["rel"] >= -PRE - 0.5) & (d["rel"] <= POST + 0.5)].copy()
    d["bin"] = np.rint(d["rel"]).astype(int) + PRE
    x = np.arange(-PRE, POST + 1)

    # ------------------------------------------------ aligned curves (all plants, then by CT alive at plant)
    rows, curves = [], {}
    groups = {"all plants": d, "site A": d[d["plant_site"] == "A"], "site B": d[d["plant_site"] == "B"],
              "T wins round": d[d["y_ct_win"] == 0], "CT wins round (retake/defuse)": d[d["y_ct_win"] == 1]}
    for lab, g in groups.items():
        for col in ("t_ctrl_planted", "t_ctrl_other"):
            m, lo, hi, c = vc.boot_binned_mean(g["match_name"].to_numpy(), g["bin"].to_numpy(), g[col].to_numpy(), NB, min(N_BOOT, 500), seed=7)
            curves[(lab, col)] = (m, lo, hi)
            rows += [{"group": lab, "series": col, "rel_s": i - PRE, "mean": m[i], "lo": lo[i], "hi": hi[i], "n_obs": c[i]} for i in range(NB)]
    strata_defs = {"<=2 CT alive at plant": d["ct_alive_at_plant"] <= 2, "3 CT alive": d["ct_alive_at_plant"] == 3, ">=4 CT alive": d["ct_alive_at_plant"] >= 4}
    for sl, smask in strata_defs.items():
        for oc, lab in ((0, "T wins"), (1, "CT wins")):
            g = d[smask & (d["y_ct_win"] == oc)]
            if g["round_key"].nunique() < 20:
                continue
            m, lo, hi, c = vc.boot_binned_mean(g["match_name"].to_numpy(), g["bin"].to_numpy(), g["t_ctrl_planted"].to_numpy(), NB, min(N_BOOT, 500), seed=8)
            curves[(sl, lab)] = (m, lo, hi, g["round_key"].nunique())
            rows += [{"group": f"{sl} | {lab}", "series": "t_ctrl_planted", "rel_s": i - PRE, "mean": m[i], "lo": lo[i], "hi": hi[i], "n_obs": c[i]} for i in range(NB)]
    pd.DataFrame(rows).to_csv(OUTDIR / "plant_aligned_curves.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for lab in ("site A", "site B"):
        m, lo, hi = curves[(lab, "t_ctrl_planted")]
        axs[0].plot(x, m, label=lab); axs[0].fill_between(x, lo, hi, alpha=.2)
    axs[0].set_title("T control of the PLANTED site, by site")
    for lab in ("T wins round", "CT wins round (retake/defuse)"):
        m, lo, hi = curves[(lab, "t_ctrl_planted")]
        axs[1].plot(x, m, label=lab); axs[1].fill_between(x, lo, hi, alpha=.2)
    axs[1].set_title("... by round outcome (NOT head-count adjusted)")
    m, lo, hi = curves[("all plants", "t_ctrl_planted")]
    axs[2].plot(x, m, label="planted site"); axs[2].fill_between(x, lo, hi, alpha=.2)
    m, lo, hi = curves[("all plants", "t_ctrl_other")]
    axs[2].plot(x, m, "--", label="other site"); axs[2].fill_between(x, lo, hi, alpha=.2)
    axs[2].set_title("planted vs other site (all plants)")
    for ax in axs:
        ax.axvline(0, color="k", ls="--", lw=1); ax.axhline(0.5, color="grey", lw=.7)
        ax.set_xlabel("seconds relative to the plant"); ax.legend(fontsize=8)
    axs[0].set_ylabel("T-perspective control (1 - control_val)")
    vc.savefig(fig, FIGDIR / "plant_aligned_site_control.png")

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, sl in zip(axs, strata_defs):
        for lab in ("T wins", "CT wins"):
            if (sl, lab) in curves:
                m, lo, hi, n = curves[(sl, lab)]
                ax.plot(x, m, label=f"{lab} (n={n})"); ax.fill_between(x, lo, hi, alpha=.2)
        ax.axvline(0, color="k", ls="--", lw=1); ax.axhline(0.5, color="grey", lw=.7)
        ax.set_title(sl); ax.set_xlabel("seconds relative to the plant"); ax.legend(fontsize=8)
    axs[0].set_ylabel("T control of the planted site")
    vc.savefig(fig, FIGDIR / "plant_aligned_by_ct_alive.png")

    # ------------------------------------------------ 0: unadjusted snapshot summaries (Metric-2 style)
    out = []
    for lab, g in {"site A": snap[snap["plant_site"] == "A"], "site B": snap[snap["plant_site"] == "B"],
                   "T wins": snap[snap["y_ct_win"] == 0], "CT wins": snap[snap["y_ct_win"] == 1]}.items():
        v = g["t_ctrl_planted"].to_numpy()
        boot = vc.cluster_bootstrap(g["match_name"].to_numpy(), lambda r: v[r].mean(), N_BOOT, seed=9)
        lo, hi = vc.pctl_ci(boot)
        out.append({"group": lab, "n": len(g), "mean_T_site_control_at_plant": v.mean(), "lo": float(lo), "hi": float(hi)})
        vc.record_result(f"plant.t_site_control.{lab.replace(' ', '_')}", v.mean(), lo, hi, len(g), "T control of planted site, frame before plant (Metric 2 style)")
    tbl = pd.DataFrame(out)
    tbl.to_csv(OUTDIR / "plant_snapshot_summary.csv", index=False)
    vc.log(tbl.round(4).to_string(index=False))
    tw, cw = snap[snap["t_win"] == 1]["t_ctrl_planted"], snap[snap["t_win"] == 0]["t_ctrl_planted"]
    t, p = stats.ttest_ind(tw, cw, equal_var=False)
    vc.log(f"UNADJUSTED Welch (T-win vs CT-win plants): t={t:.2f} p={p:.2g} d={_d(tw, cw):.3f}")
    vc.record_result("plant.t_site_control.cohens_d_twin_vs_ctwin", _d(tw, cw), n=len(snap), note="UNADJUSTED for head-count")

    # ------------------------------------------------ 1: by CT alive at the plant
    rows = []
    for k in sorted(snap["ct_alive"].unique()):
        g = snap[snap["ct_alive"] == k]
        a, b = g[g["t_win"] == 1], g[g["t_win"] == 0]
        res = {"ct_alive_at_plant": int(k), "n_plants": len(g), "n_t_win": len(a), "n_ct_win": len(b), "t_win_rate": len(a) / len(g)}
        if len(a) >= MIN_PER_GROUP and len(b) >= MIN_PER_GROUP:
            v, w, mn = g["t_ctrl_planted"].to_numpy(), g["t_win"].to_numpy(), g["match_name"].to_numpy()
            def stat(r):
                x1, x0 = v[r][w[r] == 1], v[r][w[r] == 0]
                return np.array([x1.mean() - x0.mean(), _d(x1, x0)]) if len(x1) > 2 and len(x0) > 2 else np.array([np.nan, np.nan])
            boot = vc.cluster_bootstrap(mn, stat, min(N_BOOT, 500), seed=10)
            lo, hi = vc.pctl_ci(boot)
            res.update(mean_t_win=a["t_ctrl_planted"].mean(), mean_ct_win=b["t_ctrl_planted"].mean(),
                       diff=a["t_ctrl_planted"].mean() - b["t_ctrl_planted"].mean(), diff_lo=lo[0], diff_hi=hi[0],
                       cohens_d=_d(a["t_ctrl_planted"], b["t_ctrl_planted"]), d_lo=lo[1], d_hi=hi[1])
        rows.append(res)
    bk = pd.DataFrame(rows)
    for c in ("mean_t_win", "mean_ct_win", "diff", "diff_lo", "diff_hi", "cohens_d", "d_lo", "d_hi"):
        if c not in bk.columns:          # no stratum had enough plants in both outcome groups (small subsets)
            bk[c] = np.nan
    bk.to_csv(OUTDIR / "plant_by_ct_alive.csv", index=False)
    vc.log("BY CT-ALIVE-AT-PLANT:\n" + bk.round(3).to_string(index=False))

    # ------------------------------------------------ 2: d after residualising on head-count fixed effects
    s2 = snap.dropna(subset=["t_ctrl_planted"]).copy()
    fe = pd.get_dummies(s2["ct_alive"].astype(int).astype(str) + "v" + s2["t_alive"].astype(int).astype(str)).to_numpy(float)
    def resid_d(rows_):
        F, v, w = fe[rows_], s2["t_ctrl_planted"].to_numpy()[rows_], s2["t_win"].to_numpy()[rows_]
        beta = np.linalg.lstsq(F, v, rcond=None)[0]
        r = v - F @ beta
        return _d(r[w == 1], r[w == 0])
    est = resid_d(np.arange(len(s2)))
    boot = vc.cluster_bootstrap(s2["match_name"].to_numpy(), resid_d, min(N_BOOT, 500), seed=11)
    lo, hi = vc.pctl_ci(boot)
    vc.log(f"d after removing (ct_alive x t_alive) fixed effects: {est:.3f} [{lo:.3f}, {hi:.3f}]  (unadjusted {_d(tw, cw):.3f})")
    vc.record_result("plant.cohens_d_within_headcount_cells", est, lo, hi, len(s2), "T-win vs CT-win plants after residualising on ct_alive x t_alive")

    # ------------------------------------------------ 3: adjusted logistic model + grouped-CV delta AUC
    cols_base = ["ct_alive", "t_alive", "is_A"] + ECON
    s3 = s2.copy()
    s3["is_A"] = (s3["plant_site"] == "A").astype(float)
    s3 = s3.dropna(subset=cols_base + ["t_ctrl_planted"]).reset_index(drop=True)
    if len(s3) >= 100 and s3["t_win"].nunique() == 2:
        y, grp = s3["t_win"].to_numpy(), s3["match_name"].to_numpy()
        def zs(a):
            a = np.asarray(a, float)
            return (a - a.mean()) / (a.std() or 1.0)
        Xb = np.column_stack([zs(s3[c]) for c in cols_base])
        Xc = np.column_stack([zs(s3["t_ctrl_planted"]), Xb])
        est = vc.logit_newton(Xc, y)[0][0]
        boot = vc.cluster_bootstrap(grp, lambda r: vc.logit_newton(Xc[r], y[r])[0][0], N_BOOT, seed=12)
        lo, hi = vc.pctl_ci(boot)
        pv = vc.boot_pvalue(boot)
        vc.log(f"adjusted logit coef of T control (per SD), given head-count + site + economy: {est:.3f} [{lo:.3f}, {hi:.3f}] p~{pv:.3g}")
        vc.record_result("plant.adj_logit_coef_t_ctrl_per_sd", est, lo, hi, len(s3), "P(T wins | plant) ~ control + ct_alive + t_alive + site + economy")
        def oof(X):
            p = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=int(min(5, len(np.unique(grp))))).split(X, y, grp):
                p[te] = LogisticRegression(C=1.0, max_iter=500).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            return p
        pb, pc = oof(Xb), oof(Xc)
        gi = vc.group_rows(grp)
        rng = np.random.default_rng(13)
        diffs = []
        for _ in range(min(N_BOOT, 500)):
            rr = np.concatenate([gi[i] for i in rng.integers(0, len(gi), len(gi))])
            if len(np.unique(y[rr])) == 2:
                diffs.append(roc_auc_score(y[rr], pc[rr]) - roc_auc_score(y[rr], pb[rr]))
        dlo, dhi = vc.pctl_ci(np.array(diffs))
        da = roc_auc_score(y, pc) - roc_auc_score(y, pb)
        vc.log(f"AUC base (head-count+site+economy)={roc_auc_score(y, pb):.4f}; + control={roc_auc_score(y, pc):.4f}; delta={da:+.4f} [{dlo:+.4f}, {dhi:+.4f}]")
        vc.record_result("plant.delta_auc_control_given_headcount_site_econ", da, dlo, dhi, len(s3), "grouped-CV, outcome = T wins given a plant")
        pd.DataFrame([{"model": "alive+site+economy", "auc": roc_auc_score(y, pb)}, {"model": "+ T site control", "auc": roc_auc_score(y, pc),
                                                                                       "delta_auc": da, "delta_lo": dlo, "delta_hi": dhi, "ctrl_coef_per_sd": est, "coef_lo": lo, "coef_hi": hi,
                                                                                       "n_plants": len(s3)}]).to_csv(OUTDIR / "plant_adjusted_models.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    b = bk.dropna(subset=["cohens_d"])
    if len(b):
        axs[0].errorbar(b["ct_alive_at_plant"], b["cohens_d"], yerr=[b["cohens_d"] - b["d_lo"], b["d_hi"] - b["cohens_d"]], fmt="o-", capsize=3)
        axs[0].axhline(0, color="k", lw=.8); axs[0].axhline(_d(tw, cw), color="r", ls="--", lw=1, label=f"unadjusted d = {_d(tw, cw):.2f}")
        axs[0].set_xlabel("CT players alive at the plant"); axs[0].set_ylabel("Cohen's d: T-win vs CT-win plants"); axs[0].legend()
        axs[0].set_title("Does control discriminate outcomes at a fixed CT head-count?")
    axs[1].bar(bk["ct_alive_at_plant"], bk["t_win_rate"]); axs[1].set_ylim(0, 1)
    for _, r in bk.iterrows():
        axs[1].text(r["ct_alive_at_plant"], r["t_win_rate"] + .02, f"n={int(r['n_plants'])}", ha="center", fontsize=8)
    axs[1].set_xlabel("CT players alive at the plant"); axs[1].set_ylabel("P(T wins | plant)"); axs[1].set_title("Head-count drives the outcome")
    vc.savefig(fig, FIGDIR / "plant_conditioned_on_ct_alive.png")


if __name__ == "__main__":
    main()
