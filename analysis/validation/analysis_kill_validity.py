"""
analysis_kill_validity.py
=========================
Three kill-based validity analyses of the influence field, built on the tables produced by
extract_supplementary_tables.py and build_engagement_features.py.

 S1  DUEL WIN-RATE vs LOCAL ADVANTAGE   (headline test)
     For every kill take the attacker-team minus victim-team local advantage measured in a frame
     BEFORE the fight. Fold it: the "favoured side" is whichever team had the higher local
     advantage; report P(favoured side takes the kill) overall, binned by advantage magnitude, and
     via a logistic slope with randomised orientation. Cluster bootstrap over MATCHES.

     KILL FILTER PRESETS (they bracket a selection bias — report them side by side)
       all            every valid, non-mutual kill
       fresh          all minus AVENGING kills and same-killer chain kills. Biased DOWN: avenging kills
                      are wins by a team that was locally ahead, and deleting them removes favoured wins.
       strict         fresh minus kills that were later avenged. Biased UP: it conditions on a future
                      event, and un-avenged kills are mostly ones where the victim had no nearby help.
       opener_cluster the FIRST kill of each engagement cluster (kills chained within CLUSTER_GAP_S seconds
                      AND CLUSTER_DIST units of the previous kill). Defined only from the past, so the
                      state before it is clean (no fresh death in the neighbourhood). DEFAULT.
       opener_round   the first kill of the ROUND (any earlier death, even a non-duel one, disqualifies it).
                      Always 5v5, so head-count cannot matter at all. Smallest sample, cleanest test.

 S2  KILL-LOCATION CONTEST (supporting; see the caveats printed with the results)
     Is control at the victim's kill node (lead-frame field) closer to 0.5 than at comparison nodes?
     Three nulls, because the naive one is unfair:
       (a) occupied nodes in the same frame (naive: occupied nodes always carry a full player's mass,
           kill nodes often carry little, and weak-mass nodes score ~0.5 by symmetry)
       (b) ROBUSTLY contested: both teams must carry >= m influence mass at the node (m = 0.25/0.5/1.0)
       (c) MASS-MATCHED: all nodes in that frame with total mass within x/÷2 of the kill node's
     If (a) shows an excess but (b)/(c) do not, the excess was a weak-mass artefact.

 S3  KILL EVENT STUDY
     Local control around the kill site from -6 s to +6 s, split into fresh-not-avenged / fresh-then-
     avenged / avenging kills. The step at t = 0 is partly mechanical (the victim leaves the field);
     read the pre-kill trend and the recovery.

Usage:  python analysis/validation/analysis_kill_validity.py [--preset opener_cluster] [--lead 2] [--radius 400]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = vc.METRICS / "kill_validity"
FIGDIR = OUTDIR / "figures"
PRESETS = ["all", "fresh", "strict", "opener_cluster", "opener_round"]
CLUSTER_GAP_S = 6.0
CLUSTER_DIST = 800.0
N_BOOT = vc.DEFAULT_BOOT


# ------------------------------------------------------------------ kill presets
def add_opener_flags(k: pd.DataFrame) -> pd.DataFrame:
    """is_round_opener / is_cluster_opener, computed from the PAST only (no future information)."""
    k = k.sort_values(["match_name", "round", "tick", "kill_idx"], kind="stable").reset_index(drop=True)
    k["round_kill_rank"] = k.groupby(["match_name", "round"]).cumcount()       # over ALL kill rows
    k["is_round_opener"] = (k["round_kill_rank"] == 0) & k["valid_duel"] & ~k["is_mutual"]
    v = k[k["valid_duel"] & ~k["is_mutual"]]
    same = ((v["match_name"] == v["match_name"].shift()) & (v["round"] == v["round"].shift())).to_numpy()
    dt = (v["t_rel"] - v["t_rel"].shift()).to_numpy()
    p = v[["victim_x", "victim_y", "victim_z"]].to_numpy(float)
    dist = np.r_[np.inf, np.linalg.norm(p[1:] - p[:-1], axis=1)]
    new_cluster = ~same | (dt > CLUSTER_GAP_S) | (dist > CLUSTER_DIST)
    k["is_cluster_opener"] = False
    k.loc[v.index, "is_cluster_opener"] = new_cluster
    return k


def preset_mask(k: pd.DataFrame, preset: str) -> pd.Series:
    base = k["valid_duel"] & ~k["is_mutual"]
    fresh = base & ~k["is_trade_kill"] & ~k["is_chain_kill"]
    if preset == "all":
        return base
    if preset == "fresh":
        return fresh
    if preset == "strict":
        return fresh & ~k["was_traded"]
    if preset == "opener_cluster":
        return base & k["is_cluster_opener"]
    if preset == "opener_round":
        return k["is_round_opener"]
    raise ValueError(preset)


def duel_frame(kills, adv, preset, lead, radius, feature, max_distance=None):
    k = kills[preset_mask(kills, preset)]
    if max_distance:
        k = k[k["distance"] <= max_distance]
    a = adv[(adv["lead_s"] == lead) & (adv["radius"] == radius)]
    d = k.merge(a, on=["match_name", "kill_idx"], how="inner")
    d = d[np.isfinite(d[feature]) & (d[feature] != 0)].copy()
    d["x"] = d[feature]
    d["fav_wins"] = (d["x"] > 0).astype(int)
    return d.reset_index(drop=True)


def fav_rate_ci(d, n_boot=None):
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    fw = d["fav_wins"].to_numpy()
    boot = vc.cluster_bootstrap(d["match_name"].to_numpy(), lambda r: fw[r].mean(), n_boot or N_BOOT, seed=1)
    lo, hi = vc.pctl_ci(boot)
    return float(fw.mean()), float(lo), float(hi)


def logistic_slope(d, seed=0, n_boot=None):
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], len(d))
    X, y = (s * d["x"].to_numpy())[:, None], (s > 0).astype(int)
    sd = X.std() or 1.0
    def fit(rows):
        return float(vc.logit_newton(X[rows] / sd, y[rows])[0][0])
    est = fit(np.arange(len(d)))
    boot = vc.cluster_bootstrap(d["match_name"].to_numpy(), fit, n_boot or N_BOOT, seed=2)
    lo, hi = vc.pctl_ci(boot)
    return est, float(lo), float(hi), vc.boot_pvalue(boot)


def binned_curve(d, n_bins=6):
    ax = d["x"].abs().to_numpy()
    edges = np.unique(np.quantile(ax, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, ax, side="right") - 1, 0, len(edges) - 2)
    m, lo, hi, c = vc.boot_binned_mean(d["match_name"].to_numpy(), b, d["fav_wins"].to_numpy(float), len(edges) - 1, N_BOOT, seed=3)
    centers = np.array([ax[b == i].mean() if (b == i).any() else np.nan for i in range(len(edges) - 1)])
    return pd.DataFrame({"abs_adv_center": centers, "fav_win_rate": m, "lo": lo, "hi": hi, "n": c})


def fmt_p(p, n_boot):
    return f"p<{1.0 / (n_boot + 1):.4f}" if p <= 1.0 / (n_boot + 1) + 1e-12 else f"p={p:.4f}"


# ------------------------------------------------------------------------ S1
def section_duel(kills, adv, args):
    vc.log("S1: duel win-rate vs local advantage")
    feat = args.feature
    d = duel_frame(kills, adv, args.preset, args.lead, args.radius, feat, args.max_distance)
    vc.log(f"  kills analysed: {len(d)} (preset={args.preset}, lead={args.lead}s, R={args.radius}, feature={feat})")
    # ---- every preset side by side (this table IS the selection-bias bracket)
    prow = []
    for preset in PRESETS:
        g = duel_frame(kills, adv, preset, args.lead, args.radius, feat, args.max_distance)
        if len(g) < 30:
            prow.append({"preset": preset, "n": len(g)})
            continue
        e, lo, hi = fav_rate_ci(g)
        sl, slo, shi, sp = logistic_slope(g)
        prow.append({"preset": preset, "n": len(g), "fav_win_rate": e, "lo": lo, "hi": hi, "slope_per_sd": sl,
                     "slope_lo": slo, "slope_hi": shi, "slope_p": sp})
        vc.record_result(f"duel.fav_win_rate.{preset}", e, lo, hi, len(g), f"{feat}/lead{args.lead}/R{args.radius}")
        vc.record_result(f"duel.logit_slope_per_sd.{preset}", sl, slo, shi, len(g), "randomised-orientation logistic, per SD of advantage")
    pt = pd.DataFrame(prow)
    pt.to_csv(OUTDIR / "duel_by_preset.csv", index=False)
    vc.log(pt.round(4).to_string(index=False))
    if len(d) < 30:
        vc.log("  too few kills for the chosen preset; skipping the detailed S1 outputs")
        return
    est, lo, hi = fav_rate_ci(d)
    sl, slo, shi, sp = logistic_slope(d)
    curve = binned_curve(d)
    curve.to_csv(OUTDIR / "duel_curve.csv", index=False)
    vc.record_result("duel.fav_win_rate", est, lo, hi, len(d), f"preset={args.preset}/{feat}/lead{args.lead}/R{args.radius}")
    vc.record_result("duel.logit_slope_per_sd", sl, slo, shi, len(d), f"preset={args.preset}")
    vc.log(f"  [{args.preset}] P(favoured side takes the kill) = {est:.3f} [{lo:.3f}, {hi:.3f}]; "
           f"slope/SD = {sl:.3f} [{slo:.3f}, {shi:.3f}] {fmt_p(sp, N_BOOT)}")

    rows = []
    sup = duel_frame(kills, adv, args.preset, args.lead, args.radius, "adv_ctrl_support", args.max_distance)
    for name, sub in [("all kills", d),
                      ("equal local numbers (|n_att-n_vic|=0)", d[d["n_att_near"] == d["n_vic_near"]]),
                      ("unequal local numbers", d[d["n_att_near"] != d["n_vic_near"]]),
                      ("support-only advantage (duelists excluded)", sup)]:
        e, l, h = fav_rate_ci(sub)
        rows.append({"subset": name, "n": len(sub), "fav_win_rate": e, "lo": l, "hi": h})
    dist = d.dropna(subset=["distance"])
    if len(dist) > 60:
        q = np.quantile(dist["distance"], [0, 1 / 3, 2 / 3, 1])
        for i, lab in enumerate(["short range", "mid range", "long range"]):
            sub = dist[(dist["distance"] >= q[i]) & (dist["distance"] <= q[i + 1])]
            e, l, h = fav_rate_ci(sub)
            rows.append({"subset": f"{lab} (<= {q[i + 1]:.0f} u)", "n": len(sub), "fav_win_rate": e, "lo": l, "hi": h})
    strat = pd.DataFrame(rows)
    strat.to_csv(OUTDIR / "duel_stratified.csv", index=False)
    vc.log(strat.round(4).to_string(index=False))
    r = strat[strat["subset"].str.startswith("equal")]
    if len(r):
        vc.record_result("duel.fav_win_rate_equal_local_numbers", r["fav_win_rate"].iloc[0], r["lo"].iloc[0], r["hi"].iloc[0], r["n"].iloc[0])

    grid = []
    for preset in PRESETS:
        for lead in sorted(adv["lead_s"].unique()):
            for R in sorted(adv["radius"].unique()):
                g = duel_frame(kills, adv, preset, lead, R, feat, args.max_distance)
                grid.append({"preset": preset, "lead_s": lead, "radius": R, "n": len(g), "fav_win_rate": g["fav_wins"].mean() if len(g) else np.nan})
    grid = pd.DataFrame(grid)
    grid.to_csv(OUTDIR / "duel_robustness_grid.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axs[0]
    ax.errorbar(curve["abs_adv_center"], curve["fav_win_rate"], yerr=[curve["fav_win_rate"] - curve["lo"], curve["hi"] - curve["fav_win_rate"]],
                fmt="o-", capsize=3)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel(f"|local advantage| ({feat}) before the fight")
    ax.set_ylabel("P(locally favoured side takes the kill)")
    ax.set_title(f"Duel win-rate vs local advantage ({args.preset}, n={len(d)})")
    ax = axs[1]
    for preset in PRESETS:
        g = grid[(grid["preset"] == preset) & (grid["radius"] == args.radius)]
        ax.plot(g["lead_s"], g["fav_win_rate"], marker="o", ls=":" if preset == "strict" else "-", label=preset)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("lead time before the kill (s)")
    ax.set_ylabel("P(favoured side wins)")
    ax.set_title("Selection-bias bracket: kill filter x lead time")
    ax.legend(fontsize=8)
    vc.savefig(fig, FIGDIR / "duel_winrate_vs_local_advantage.png")


# ------------------------------------------------------------------------ S2
def paired_diff(df, a, b, seed):
    diff = (df[a].astype(float) - df[b].astype(float)).to_numpy()
    ok = np.isfinite(diff)
    diff, grp = diff[ok], df["match_name"].to_numpy()[ok]
    if len(diff) < 30:
        return np.nan, np.nan, np.nan, 0
    boot = vc.cluster_bootstrap(grp, lambda r: diff[r].mean(), N_BOOT, seed=seed)
    lo, hi = vc.pctl_ci(boot)
    return float(diff.mean()), float(lo), float(hi), len(diff)


def section_location(kills, loc, args):
    vc.log("S2: kill-location contest vs three nulls")
    k = kills[preset_mask(kills, args.preset)]
    d = k.merge(loc, on=["match_name", "kill_idx"], how="inner")
    if len(d) < 30:
        vc.log("  too few kills; skipping S2")
        return
    tags = [vc.mass_tag(m) for m in vc.MASS_THRS]
    rows = []
    for lead, g in d.groupby("lead_s"):
        act = g[g["active_at_kill"]]
        res = {"lead_s": lead, "n_kills": len(g), "frac_kill_node_inactive": 1 - g["active_at_kill"].mean(),
               "median_total_mass_at_kill": g["total_mass_at_kill"].median(),
               "median_min_mass_at_kill": g["min_mass_at_kill"].median()}
        e, lo, hi, n = paired_diff(act, "contested_at_kill", "null_contested_frac", 4)
        res.update(naive_kill=act["contested_at_kill"].astype(float).mean(), naive_null=act["null_contested_frac"].mean(),
                   naive_diff=e, naive_lo=lo, naive_hi=hi, n_active=n)
        e, lo, hi, n = paired_diff(act, "absdev_at_kill", "null_absdev_mean", 5)
        res.update(absdev_diff=e, absdev_lo=lo, absdev_hi=hi)
        for m, tag in zip(vc.MASS_THRS, tags):
            e, lo, hi, n = paired_diff(g, f"strict_at_kill_{tag}", f"null_strict_frac_{tag}", 6)
            res.update({f"strict{tag}_kill": g[f"strict_at_kill_{tag}"].astype(float).mean(), f"strict{tag}_diff": e,
                        f"strict{tag}_lo": lo, f"strict{tag}_hi": hi})
        mm = act[act["mm_null_contested_frac"].notna()]
        e, lo, hi, n = paired_diff(mm, "contested_at_kill", "mm_null_contested_frac", 7)
        res.update(mm_kill=mm["contested_at_kill"].astype(float).mean() if len(mm) else np.nan,
                   mm_null=mm["mm_null_contested_frac"].mean() if len(mm) else np.nan, mm_diff=e, mm_lo=lo, mm_hi=hi, n_mm=n)
        rows.append(res)
    t = pd.DataFrame(rows)
    t.to_csv(OUTDIR / "kill_location_contest.csv", index=False)
    show = ["lead_s", "n_kills", "frac_kill_node_inactive", "median_total_mass_at_kill", "naive_kill", "naive_null", "naive_diff",
            "strict0p50_kill", "strict0p50_diff", "mm_kill", "mm_null", "mm_diff"]
    vc.log(t[[c for c in show if c in t.columns]].round(4).to_string(index=False))
    p = t[t["lead_s"] == 2.0]
    if len(p):
        vc.record_result("killloc.contested_rate_diff_lead2_naive", p["naive_diff"].iloc[0], p["naive_lo"].iloc[0], p["naive_hi"].iloc[0], p["n_active"].iloc[0],
                         "P(kill node contested) - P(occupied node contested)")
        vc.record_result("killloc.contested_rate_diff_lead2_massmatched", p["mm_diff"].iloc[0], p["mm_lo"].iloc[0], p["mm_hi"].iloc[0], p["n_mm"].iloc[0],
                         "kill node vs nodes with similar total mass")
        tg = vc.mass_tag(vc.STRICT_MIN_MASS)
        vc.record_result("killloc.contested_rate_diff_lead2_strict", p[f"strict{tg}_diff"].iloc[0], p[f"strict{tg}_lo"].iloc[0], p[f"strict{tg}_hi"].iloc[0], len(d[d["lead_s"] == 2.0]),
                         f"robustly contested (both teams >= {vc.STRICT_MIN_MASS} mass)")
    prim = d[(d["lead_s"] == 2.0) & d["active_at_kill"]]
    if len(prim):
        hcols = [c for c in d.columns if c.startswith("null_hist_")]
        null = prim[hcols].sum().to_numpy().astype(float)
        edges = np.linspace(0, 0.5, len(hcols) + 1)
        kill_h = np.histogram(prim["absdev_at_kill"], bins=edges)[0].astype(float)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
        w = edges[1] - edges[0]
        axs[0].bar(edges[:-1], kill_h / kill_h.sum(), width=w * .45, align="edge", label="kill node")
        axs[0].bar(edges[:-1] + w * .5, null / null.sum(), width=w * .45, align="edge", label="occupied nodes (naive null)")
        axs[0].set_xlabel("|control - 0.5| at the node (lead = 2 s)")
        axs[0].set_ylabel("share")
        axs[0].set_title("Where do kills sit on the control scale?")
        axs[0].legend()
        ax = axs[1]
        for pre, lab in (("naive", "naive null: occupied nodes"), (f"strict{vc.mass_tag(vc.STRICT_MIN_MASS)}", f"robustly contested (both teams >= {vc.STRICT_MIN_MASS} mass)"),
                         ("mm", "mass-matched null")):
            ax.errorbar(t["lead_s"], t[f"{pre}_diff"], yerr=[t[f"{pre}_diff"] - t[f"{pre}_lo"], t[f"{pre}_hi"] - t[f"{pre}_diff"]], fmt="o-", capsize=3, label=lab)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("lead time before the kill (s)")
        ax.set_ylabel("excess P(contested) at the kill node")
        ax.set_title("Is the excess real? Three nulls")
        ax.legend(fontsize=8)
        vc.savefig(fig, FIGDIR / "kill_location_contest.png")


# ------------------------------------------------------------------------ S3
def section_event_study(kills, ser, args):
    vc.log("S3: kill event study")
    k = kills[kills["valid_duel"] & ~kills["is_mutual"]].copy()
    k["etype"] = np.select([k["is_trade_kill"], k["is_chain_kill"], k["was_traded"]],
                           ["trade kill (avenging)", "chain kill", "fresh, then avenged"], "fresh, not avenged")
    d = ser.merge(k[["match_name", "kill_idx", "etype"]], on=["match_name", "kill_idx"], how="inner")
    d = d[d["rel_t"].between(-5.5, 5.5)].copy()
    d["bin"] = (np.rint(d["rel_t"]).astype(int) + 5)
    rel = np.arange(-5, 6)
    metrics = [("adv_ctrl_att_all", "Local control advantage of the killer's team"),
               ("att_mu_mean", "Killer team: mean local influence"),
               ("vic_mu_mean", "Victim team: mean local influence")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    out_rows = []
    for ax, (col, title) in zip(axs, metrics):
        for et, g in d.groupby("etype"):
            if len(g) < 40 or et == "chain kill":
                continue
            m, lo, hi, c = vc.boot_binned_mean(g["match_name"].to_numpy(), g["bin"].to_numpy(), g[col].to_numpy(), 11, N_BOOT, seed=5)
            ax.plot(rel, m, marker="o", label=f"{et} (n={g.groupby(['match_name', 'kill_idx']).ngroups})")
            ax.fill_between(rel, lo, hi, alpha=.2)
            for r_, mm, l_, h_, cc in zip(rel, m, lo, hi, c):
                out_rows.append({"etype": et, "metric": col, "rel_s": r_, "mean": mm, "lo": l_, "hi": h_, "n_obs": cc})
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("seconds relative to the kill (bin 0 mixes pre- and post-kill frames)")
        ax.set_title(title, fontsize=10)
    axs[0].legend(fontsize=8)
    vc.savefig(fig, FIGDIR / "kill_event_study.png")
    pd.DataFrame(out_rows).to_csv(OUTDIR / "kill_event_study.csv", index=False)
    g = d[(d["etype"] == "fresh, not avenged") & (d["rel_t"] >= -5) & (d["rel_t"] < 0) & d["adv_ctrl_att_all"].notna()]
    if len(g) > 50:
        x, y = g["rel_t"].to_numpy(), g["adv_ctrl_att_all"].to_numpy()
        slope = lambda r: np.polyfit(x[r], y[r], 1)[0]
        est = slope(np.arange(len(g)))
        boot = vc.cluster_bootstrap(g["match_name"].to_numpy(), slope, min(N_BOOT, 500), seed=6)
        lo, hi = vc.pctl_ci(boot)
        vc.record_result("eventstudy.pre_kill_adv_slope_per_s", est, lo, hi, len(g), "killer-team local advantage trend, -5..0 s, fresh non-avenged kills")
        vc.log(f"  pre-kill trend of the killer team's local advantage: {est:+.4f}/s [{lo:+.4f}, {hi:+.4f}]  (>0 => already building before the kill)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="opener_cluster", choices=PRESETS)
    ap.add_argument("--lead", type=float, default=2.0)
    ap.add_argument("--radius", type=float, default=400.0)
    ap.add_argument("--feature", default="adv_ctrl_all",
                    choices=["adv_ctrl_all", "adv_share_all", "adv_ctrl_support", "adv_share_support"])
    ap.add_argument("--max-distance", type=float, default=None, help="drop kills with attacker-victim distance above this (game units)")
    ap.add_argument("--skip", nargs="*", default=[], choices=["duel", "location", "event"])
    args = ap.parse_args()
    vc.ensure_dir(FIGDIR)
    kills = vc.concat_tables(vc.T_KILLS, ext=".csv")
    for c in ("valid_duel", "is_mutual", "is_trade_kill", "was_traded", "is_chain_kill"):
        kills[c] = kills[c].astype(bool)
    kills = add_opener_flags(kills)
    vc.log(f"{len(kills)} kills across {kills['match_name'].nunique()} matches; valid duels: {int(kills['valid_duel'].sum())}")
    vc.log("  kills per preset: " + ", ".join(f"{p}={int(preset_mask(kills, p).sum())}" for p in PRESETS))
    if "duel" not in args.skip:
        section_duel(kills, vc.concat_tables(vc.T_ENG_ADV, ext=".parquet"), args)
    if "location" not in args.skip:
        section_location(kills, vc.concat_tables(vc.T_ENG_LOC, ext=".parquet"), args)
    if "event" not in args.skip:
        section_event_study(kills, vc.concat_tables(vc.T_ENG_SER, ext=".parquet"), args)


if __name__ == "__main__":
    main()
