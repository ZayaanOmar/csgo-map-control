"""
analysis_location_importance.py   (Part 3c)
===========================================
"Which locations matter?" has no single definition, so six are computed per named area (areaName,
blank names excluded from rankings) and then compared — disagreement between them is itself a result.

 1 OUTCOME ASSOCIATION   per-area logistic regression  y(CT win) ~ std(area control) + economy, on the
                         PRE-ENGAGEMENT window (no kills yet, so the head-count cannot explain it).
                         Match-clustered bootstrap CI/p, Benjamini-Hochberg FDR across areas.
                         PER-SIDE version uses player OCCUPANCY: y ~ std(CT occupancy) + std(T occupancy) + economy
                         -> CT importance = coef(CT occupancy);  T importance = -coef(T occupancy).
 2 PERMUTATION IMPORTANCE  multivariate model (all area controls + economy), grouped-CV out-of-fold AUC
                         drop when one area's feature is shuffled in the held-out fold (collinear
                         neighbours share credit, so this complements 1).
 3 CONTEST FREQUENCY     mean fraction of an area's nodes that are active AND contested (Metric 1, per area).
 4 KILL DENSITY          valid kills per round whose VICTIM died in the area, and per 1000 occupied
                         player-frames (danger given that someone is there).
 5 OCCUPANCY / TRAFFIC   mean alive players in the area per frame.
 6 STRUCTURAL CENTRALITY weighted betweenness on the patched nav graph, reported as the SUM over the area's
                         nodes and PER NODE. The sum scales with how many tiles the area was split into, so
                         comparisons use the per-node value (and occupancy per node). No gameplay data at all —
                         a structural prior, not truth.
 3b ROBUST CONTEST       like 3 but a node counts only if BOTH teams carry >= STRICT_MIN_MASS influence, so
                         faint-both-sides nodes (control ~0.5 by symmetry) are not "contested".

Output: outputs/metrics/location_importance/{area_importance.csv, area_importance_correlations.csv, figures/*.png}
and registry keys area_rank.<Area>.<measure> (1 = most important), area_corr.*.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = vc.METRICS / "location_importance"
FIGDIR = OUTDIR / "figures"
ECON = ["log_eq_ratio", "eq_diff_k", "is_pistol"]


def z(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / s if s > 0 else None


def coef_with_boot(X, y, boot_rows, j=0):
    """Point estimate and bootstrap coefficients (fast Newton logistic; intercept dropped)."""
    est = vc.logit_newton(X, y)[0]
    boots = np.array([vc.logit_newton(X[r], y[r])[0] for r in boot_rows])
    return est, boots


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=float, default=30.0, help="pre-engagement window cap (s)")
    ap.add_argument("--min-pre", type=float, default=5.0)
    ap.add_argument("--n-boot", type=int, default=1500, help="bootstrap resamples; the smallest resolvable p is 1/(n+1)")
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--min-nodes", type=int, default=2, help="ignore areas with fewer nav nodes")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    vc.ensure_dir(FIGDIR)
    ctx = vc.MapContext()
    areas = [a for a, idx in ctx.name_groups.items() if a != "UNNAMED" and len(idx) >= args.min_nodes]
    vc.log(f"{len(areas)} named areas analysed")

    rt = vc.load_round_table()
    rt["rk"] = rt["match_name"] + ":" + rt["round"].astype(str)
    rt["eq_diff_k"] = rt["eq_diff"] / 1000.0
    rt["is_pistol"] = rt["is_pistol_round"].astype(float)
    cols = ["round", "t_rel"] + [f"{p}__{a}" for a in areas for p in ("area_ctrl", "area_cont", "area_contS", "occ_ct", "occ_t")]
    fr = vc.load_frame_summaries(columns=cols)
    fr["rk"] = fr["match_name"] + ":" + fr["round"].astype(str)
    fr = fr[fr["t_rel"] >= 0]
    n_rounds = rt["rk"].nunique()

    # ------------------------------------------------ 3, 5 : whole-round descriptive measures
    desc = []
    for a in areas:
        c = fr[f"area_cont__{a}"].to_numpy(float)
        o = (fr[f"occ_ct__{a}"] + fr[f"occ_t__{a}"]).to_numpy(float)
        cs = fr[f"area_contS__{a}"].to_numpy(float)
        zero = np.zeros(len(fr), int)
        mc = vc.boot_binned_mean(fr["match_name"].to_numpy(), zero, c, 1, 300, seed=1)
        mcs = vc.boot_binned_mean(fr["match_name"].to_numpy(), zero, cs, 1, 300, seed=3)
        mo = vc.boot_binned_mean(fr["match_name"].to_numpy(), zero, o, 1, 300, seed=2)
        nn = len(ctx.name_groups[a])
        desc.append({"area": a, "n_nodes": nn, "contest_freq": mc[0][0], "contest_lo": mc[1][0], "contest_hi": mc[2][0],
                     "contest_strict_freq": mcs[0][0], "contest_strict_lo": mcs[1][0], "contest_strict_hi": mcs[2][0],
                     "occupancy": mo[0][0], "occupancy_lo": mo[1][0], "occupancy_hi": mo[2][0], "occupancy_per_node": mo[0][0] / nn,
                     "occ_frames": o.sum()})
    out = pd.DataFrame(desc).set_index("area")

    # ------------------------------------------------ 4 : kill density
    kills = vc.concat_tables(vc.T_KILLS, ext=".csv", columns=["valid_duel", "victim_area_id"])
    kills = kills[kills["valid_duel"]]
    vi = ctx.idx_of(kills["victim_area_id"].to_numpy())
    kn = pd.Series(np.where(vi >= 0, ctx.names[np.clip(vi, 0, None)], "UNNAMED"))
    kc = kn.value_counts()
    out["kills"] = [int(kc.get(a, 0)) for a in areas]
    out["kills_per_round"] = out["kills"] / n_rounds
    out["kills_per_1000_occ_frames"] = out["kills"] / out["occ_frames"].replace(0, np.nan) * 1000

    # ------------------------------------------------ 6 : structural centrality
    vc.log("betweenness centrality (nav graph)…")
    bc = nx.betweenness_centrality(ctx.G, weight="weight")
    bnode = np.array([bc.get(int(a), 0.0) for a in ctx.area_ids])
    out["betweenness_sum"] = [bnode[ctx.name_groups[a]].sum() for a in areas]
    out["betweenness_per_node"] = out["betweenness_sum"] / out["n_nodes"]

    # ------------------------------------------------ window features for 1, 2
    fk = rt.set_index("rk")["first_kill_t"].fillna(np.inf)
    end = np.minimum(fk, args.cap)
    rl = rt.set_index("rk")["round_len"] if "round_len" in rt else rt.set_index("rk")["round_len_s"]
    keep = (fk >= args.min_pre) & (rl >= end - 1e-9)
    w = fr[fr["t_rel"] <= fr["rk"].map(end)]
    agg_cols = [c for c in cols if "__" in c and not c.startswith("area_cont")]   # contest columns are not window features
    f = w.groupby("rk")[agg_cols].mean().reset_index()
    f = f.merge(rt[["rk", "match_name", "y_ct_win"] + ECON], on="rk")
    f = f[f["rk"].map(keep).fillna(False).astype(bool)].dropna(subset=ECON).reset_index(drop=True)
    vc.log(f"pre-engagement window rounds: {len(f)} from {f['match_name'].nunique()} matches")
    y = f["y_ct_win"].to_numpy()
    if len(f) < 60 or len(np.unique(y)) < 2:
        vc.log("too few pre-engagement rounds for outcome-based measures; writing descriptive measures only")
    else:
        E = np.column_stack([z(f[c]) if z(f[c]) is not None else np.zeros(len(f)) for c in ECON])
        gi = vc.group_rows(f["match_name"].to_numpy())
        rng = np.random.default_rng(0)
        boot_rows = [np.concatenate([gi[i] for i in rng.integers(0, len(gi), len(gi))]) for _ in range(args.n_boot)]
        res = {"outcome": [], "ct_occ": [], "t_occ": []}
        for a in areas:
            zc = z(f[f"area_ctrl__{a}"] - 0.5)
            zo_ct, zo_t = z(f[f"occ_ct__{a}"]), z(f[f"occ_t__{a}"])
            r = {"area": a, "outcome_coef": np.nan, "outcome_lo": np.nan, "outcome_hi": np.nan, "outcome_p": np.nan,
                 "ct_occ_coef": np.nan, "ct_occ_p": np.nan, "t_importance": np.nan, "t_occ_p": np.nan}
            if zc is not None:
                est, b = coef_with_boot(np.column_stack([zc, E]), y, boot_rows)
                lo, hi = vc.pctl_ci(b[:, 0])
                r.update(outcome_coef=est[0], outcome_lo=float(lo), outcome_hi=float(hi), outcome_p=vc.boot_pvalue(b[:, 0]))
            if zo_ct is not None and zo_t is not None:
                est, b = coef_with_boot(np.column_stack([zo_ct, zo_t, E]), y, boot_rows)
                r.update(ct_occ_coef=est[0], ct_occ_p=vc.boot_pvalue(b[:, 0]), t_importance=-est[1], t_occ_p=vc.boot_pvalue(b[:, 1]))
            res["outcome"].append(r)
        o1 = pd.DataFrame(res["outcome"]).set_index("area")
        o1["outcome_q"] = vc.bh_fdr(o1["outcome_p"].to_numpy())
        o1["ct_occ_q"] = vc.bh_fdr(o1["ct_occ_p"].to_numpy())
        o1["t_occ_q"] = vc.bh_fdr(o1["t_occ_p"].to_numpy())
        out = out.join(o1)
        # permutation importance
        Xa = np.column_stack([f[f"area_ctrl__{a}"].to_numpy(float) - 0.5 for a in areas] + [f[c].to_numpy(float) for c in ECON])
        grp = f["match_name"].to_numpy()
        drops = np.zeros(len(areas))
        nf = 0
        prng = np.random.default_rng(1)
        for tr, te in GroupKFold(n_splits=int(min(5, len(np.unique(grp))))).split(Xa, y, grp):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)).fit(Xa[tr], y[tr])
            base = roc_auc_score(y[te], mdl.predict_proba(Xa[te])[:, 1])
            for j in range(len(areas)):
                ds = []
                for _ in range(args.n_perm):
                    Xp = Xa[te].copy()
                    Xp[:, j] = prng.permutation(Xp[:, j])
                    ds.append(base - roc_auc_score(y[te], mdl.predict_proba(Xp)[:, 1]))
                drops[j] += np.mean(ds)
            nf += 1
        out["perm_auc_drop"] = drops / max(nf, 1)

    out = out.reset_index()
    out.to_csv(OUTDIR / "area_importance.csv", index=False)
    vc.log(out.round(4).to_string(index=False))

    # ------------------------------------------------ ranks, correlations, registry
    measures = {"outcome_assoc": out.get("outcome_coef", pd.Series(dtype=float)).abs() if "outcome_coef" in out else None,
                "permutation": out.get("perm_auc_drop"), "contest": out["contest_freq"], "kills_per_round": out["kills_per_round"],
                "kills_per_occupied_frame": out["kills_per_1000_occ_frames"], "occupancy": out["occupancy"],
                "contest_strict": out["contest_strict_freq"], "betweenness_per_node": out["betweenness_per_node"]}
    measures = {k: v for k, v in measures.items() if v is not None}
    ranks = pd.DataFrame({k: v.rank(ascending=False, method="min") for k, v in measures.items()})
    ranks.insert(0, "area", out["area"])
    ranks.to_csv(OUTDIR / "area_ranks.csv", index=False)
    for _, r in ranks.iterrows():
        for k in measures:
            if np.isfinite(r[k]):
                vc.record_result(f"area_rank.{r['area']}.{k}", r[k], note="1 = most important by this measure")
    for _, r in out.iterrows():
        if "outcome_coef" in out and np.isfinite(r["outcome_coef"]):
            vc.record_result(f"area_value.{r['area']}.outcome_coef", r["outcome_coef"], r["outcome_lo"], r["outcome_hi"], len(f), "std. logit coef of CT control on CT win")
        if "t_importance" in out and np.isfinite(r.get("t_importance", np.nan)):
            vc.record_result(f"area_value.{r['area']}.ct_occupancy_coef", r["ct_occ_coef"], note="CT presence -> CT win (pre-engagement)")
            vc.record_result(f"area_value.{r['area']}.t_occupancy_importance", r["t_importance"], note="T presence -> T win (pre-engagement)")
    crow = []
    for struct in ("betweenness_per_node", "betweenness_sum"):
        for k, v in list(measures.items()) + [("occupancy_per_node", out["occupancy_per_node"])]:
            if k == "betweenness_per_node":
                continue
            ok = v.notna() & out[struct].notna()
            if ok.sum() >= 4:
                rho, p = spearmanr(out.loc[ok, struct], v[ok])
                crow.append({"structural": struct, "vs": k, "spearman_rho": rho, "p": p, "n_areas": int(ok.sum())})
                vc.record_result(f"area_corr.{struct}_vs_{k}", rho, n=int(ok.sum()), note=f"Spearman, p={p:.3g}")
    pd.DataFrame(crow).to_csv(OUTDIR / "area_importance_correlations.csv", index=False)
    vc.log(pd.DataFrame(crow).round(3).to_string(index=False))

    # ------------------------------------------------ figures
    ms = list(measures.items())
    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (k, v) in zip(axs.ravel(), ms):
        s = pd.Series(v.to_numpy(), index=out["area"]).dropna().sort_values()
        ax.barh(s.index, s.values); ax.set_title(k, fontsize=10)
    for ax in axs.ravel()[len(ms):]:
        ax.axis("off")
    vc.savefig(fig, FIGDIR / "area_importance_by_measure.png")
    if "outcome_coef" in out and out["outcome_coef"].notna().any():
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        o = out.dropna(subset=["outcome_coef"]).sort_values("outcome_coef")
        axs[0].errorbar(o["outcome_coef"], o["area"], xerr=[o["outcome_coef"] - o["outcome_lo"], o["outcome_hi"] - o["outcome_coef"]], fmt="o", capsize=2)
        axs[0].axvline(0, color="k", lw=.8); axs[0].set_xlabel("std. logit coef: CT control of area -> CT round win (pre-engagement, econ-adjusted)")
        for i, (nm, q) in enumerate(zip(o["area"], o["outcome_q"])):
            if q < 0.05:
                axs[0].text(axs[0].get_xlim()[1], i, " *FDR<.05", va="center", fontsize=7)
        s = out.dropna(subset=["ct_occ_coef"]).set_index("area")
        idx = np.arange(len(s))
        axs[1].barh(idx - .2, s["ct_occ_coef"], .4, label="CT presence → CT win"); axs[1].barh(idx + .2, s["t_importance"], .4, label="T presence → T win")
        axs[1].set_yticks(idx); axs[1].set_yticklabels(s.index); axs[1].axvline(0, color="k", lw=.8); axs[1].legend()
        axs[1].set_xlabel("std. logit coefficient (side-specific importance)")
        vc.savefig(fig, FIGDIR / "area_outcome_association.png")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(out["betweenness_per_node"], out["outcome_coef"].abs())
        for _, r in out.iterrows():
            ax.annotate(r["area"], (r["betweenness_per_node"], abs(r["outcome_coef"])), fontsize=7)
        ax.set_xlabel("structural betweenness PER NODE"); ax.set_ylabel("|outcome association|")
        ax.set_title("Structural vs behavioural importance")
        vc.savefig(fig, FIGDIR / "structural_vs_behavioural.png")


if __name__ == "__main__":
    main()
