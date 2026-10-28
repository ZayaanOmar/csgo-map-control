"""
analysis_side_site_favourability.py   (Parts 3a and 3b)
=======================================================
Quantities describing de_mirage's CT/T balance and site balance, comparable with common claims
about the map. Every headline estimate is also written to the results registry
(validation_common.record_result).

3a  OVERALL SIDE FAVOURABILITY
    * CT round win rate: all rounds, excluding pistols, by half, pistol rounds only
      (Wilson CI and match-clustered bootstrap CI).
    * Control-based layer: CT-perspective control margin and Voronoi margin through the round,
      plus territory share (fraction of active nodes held by each side). The point to show is
      that territory != win probability (T must take space to win; CT can hold less territory
      and still win more), so a divergence is a finding, not an error.
3b  SITE FAVOURABILITY (conditional on a plant, so a T-side choice is built into the sample)
    * plant share A vs B, plant rate, post-plant T win rate by site, retake success,
      the A-minus-B difference with a clustered CI, plant timing by site.
    * Structural, demo-free layer from the APSP matrix: directed distance from T spawn and CT
      spawn to each site, and site-to-site. Edge weights are centroid distances, NOT run times
      (no stairs/jump/peek modelling) — "structural distance", not "seconds to rotate".

Outputs: outputs/metrics/side_site/{side_win_rates.csv, side_control_timeseries.csv,
         site_summary.csv, structural_distances.csv, figures/*.png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = vc.METRICS / "side_site"
FIGDIR = OUTDIR / "figures"
N_BOOT = vc.DEFAULT_BOOT
TMAX = 100


def rate_row(label, y, groups, key_prefix=None):
    y = np.asarray(y, float)
    n = len(y)
    if n == 0:
        return {"subset": label, "n": 0}
    k = int(y.sum())
    p, wlo, whi = vc.wilson(k, n)
    boot = vc.cluster_bootstrap(groups, lambda r: y[r].mean(), N_BOOT, seed=1)
    blo, bhi = vc.pctl_ci(boot)
    if key_prefix:
        vc.record_result(key_prefix, p, blo, bhi, n, label)
    return {"subset": label, "n": n, "rate": p, "wilson_lo": wlo, "wilson_hi": whi, "boot_lo": float(blo), "boot_hi": float(bhi)}


def main():
    vc.ensure_dir(FIGDIR)
    rt = vc.load_round_table()
    rt["rk"] = rt["match_name"] + ":" + rt["round"].astype(str)
    have_econ = "is_pistol_round" in rt.columns
    tot = (rt["ct_score_start"] + rt["t_score_start"]) if have_econ else pd.Series(np.nan, index=rt.index)
    rt["half"] = np.select([tot < 15, tot < 30], ["first half", "second half"], "overtime") if have_econ else "n/a"
    pist = rt["is_pistol_round"].astype(bool) if have_econ else pd.Series(False, index=rt.index)

    # ----------------------------------------------------------------------- 3a
    vc.log("3a: side favourability")
    rows = []
    subsets = [("all rounds", rt.index == rt.index, "side.ct_win_rate_all"),
               ("non-pistol rounds", ~pist, "side.ct_win_rate_nonpistol"),
               ("pistol rounds", pist, "side.ct_win_rate_pistol")]
    if have_econ:
        subsets += [("first half (non-pistol)", (rt["half"] == "first half") & ~pist, "side.ct_win_rate_first_half"),
                    ("second half (non-pistol)", (rt["half"] == "second half") & ~pist, "side.ct_win_rate_second_half")]
    for lab, mask, key in subsets:
        s = rt[np.asarray(mask)]
        rows.append(rate_row(lab, s["y_ct_win"], s["match_name"].to_numpy(), key))
    wr = pd.DataFrame(rows)
    wr.to_csv(OUTDIR / "side_win_rates.csv", index=False)
    vc.log(wr.round(4).to_string(index=False))

    fr = vc.load_frame_summaries(columns=["round", "t_rel", "ctrl_active", "ctrl_full", "ct_held_frac_active",
                                          "t_held_frac_active", "vor_margin"])
    fr["rk"] = fr["match_name"] + ":" + fr["round"].astype(str)
    fr = fr[(fr["t_rel"] >= 0) & (fr["t_rel"] <= TMAX)]
    if have_econ:
        fr = fr[~fr["rk"].isin(rt.loc[pist, "rk"])]
    fr["bin"] = np.rint(fr["t_rel"]).astype(int)
    fr["terr_margin"] = fr["ct_held_frac_active"] - fr["t_held_frac_active"]
    fr["ca"] = fr["ctrl_active"] - 0.5
    series = {}
    out = []
    for col in ("ca", "vor_margin", "terr_margin"):
        m, lo, hi, c = vc.boot_binned_mean(fr["match_name"].to_numpy(), fr["bin"].to_numpy(), fr[col].to_numpy(), TMAX + 1, min(N_BOOT, 500), seed=2)
        series[col] = (m, lo, hi)
        out.append(pd.DataFrame({"series": col, "t": np.arange(TMAX + 1), "mean": m, "lo": lo, "hi": hi, "n_obs": c}))
    pd.concat(out).to_csv(OUTDIR / "side_control_timeseries.csv", index=False)
    per_round = fr.groupby(["match_name", "rk"])[["ca", "vor_margin", "terr_margin"]].mean().reset_index()
    for col, key in (("ca", "side.mean_ctrl_active_margin"), ("vor_margin", "side.mean_vor_margin"), ("terr_margin", "side.mean_territory_margin")):
        v = per_round[col].to_numpy()
        ok = np.isfinite(v)
        boot = vc.cluster_bootstrap(per_round["match_name"].to_numpy()[ok], lambda r, v=v[ok]: v[r].mean(), N_BOOT, seed=3)
        lo, hi = vc.pctl_ci(boot)
        vc.record_result(key, v[ok].mean(), lo, hi, ok.sum(), "mean over non-pistol rounds, positive => CT-held")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axs[0]
    ax.bar(range(len(wr)), wr["rate"], yerr=[wr["rate"] - wr["boot_lo"], wr["boot_hi"] - wr["rate"]], capsize=4)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(wr))); ax.set_xticklabels(wr["subset"], rotation=25, ha="right")
    ax.set_ylabel("CT round win rate"); ax.set_title("Round outcomes by side (95% match-clustered CI)")
    ax = axs[1]
    x = np.arange(TMAX + 1)
    for col, lab in (("ca", "influence-field control margin (active)"), ("vor_margin", "Voronoi margin"), ("terr_margin", "territory share (CT - T held)")):
        m, lo, hi = series[col]
        ax.plot(x, m, label=lab); ax.fill_between(x, lo, hi, alpha=.15)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("seconds since freeze end"); ax.set_ylabel("positive = CT holds more map")
    ax.set_title("Territory is not the same as winning (non-pistol rounds)"); ax.legend(fontsize=8)
    vc.savefig(fig, FIGDIR / "side_favourability.png")

    # ----------------------------------------------------------------------- 3b
    vc.log("3b: site favourability")
    pl = rt[rt["plant_site"].isin(["A", "B"])].copy()
    pl["t_win"] = 1 - pl["y_ct_win"]
    rows = []
    n_rounds = len(rt)
    p, lo, hi = vc.wilson(len(pl), n_rounds)
    vc.record_result("site.plant_rate", p, lo, hi, n_rounds, "fraction of rounds with a plant")
    rows.append({"quantity": "plant rate (all rounds)", "site": "both", "n": n_rounds, "estimate": p, "lo": lo, "hi": hi})
    for s in ("A", "B"):
        g = pl[pl["plant_site"] == s]
        sh, slo, shi = vc.wilson(len(g), len(pl))
        vc.record_result(f"site.{s}.plant_share", sh, slo, shi, len(pl), "share of plants at this site")
        rows.append({"quantity": "plant share", "site": s, "n": len(pl), "estimate": sh, "lo": slo, "hi": shi})
        r = rate_row(f"post-plant T win rate site {s}", g["t_win"], g["match_name"].to_numpy(), f"site.{s}.post_plant_t_win")
        rows.append({"quantity": "post-plant T win rate", "site": s, "n": r["n"], "estimate": r.get("rate"), "lo": r.get("boot_lo"), "hi": r.get("boot_hi")})
        vc.record_result(f"site.{s}.retake_success", 1 - r["rate"], 1 - r["boot_hi"], 1 - r["boot_lo"], r["n"], "CT win rate given a plant at this site")
        pt = g["plant_t"].dropna().to_numpy()
        if len(pt):
            boot = vc.cluster_bootstrap(g.loc[g["plant_t"].notna(), "match_name"].to_numpy(), lambda rr: np.median(pt[rr]), N_BOOT, seed=4)
            plo, phi = vc.pctl_ci(boot)
            vc.record_result(f"site.{s}.median_plant_time_s", np.median(pt), plo, phi, len(pt), "seconds since freeze end")
            rows.append({"quantity": "median plant time (s)", "site": s, "n": len(pt), "estimate": float(np.median(pt)), "lo": float(plo), "hi": float(phi)})
    a, b = pl[pl["plant_site"] == "A"], pl[pl["plant_site"] == "B"]
    if len(a) > 10 and len(b) > 10:
        tw, mn, site = pl["t_win"].to_numpy(float), pl["match_name"].to_numpy(), (pl["plant_site"] == "A").to_numpy()
        def diff(r):
            x, s_ = tw[r], site[r]
            return x[s_].mean() - x[~s_].mean() if s_.any() and (~s_).any() else np.nan
        est = diff(np.arange(len(pl)))
        boot = vc.cluster_bootstrap(mn, diff, N_BOOT, seed=5)
        lo, hi = vc.pctl_ci(boot)
        vc.record_result("site.post_plant_t_win_diff_A_minus_B", est, lo, hi, len(pl), "positive => planted A is better for T than planted B")
        vc.record_result("site.retake_success_diff_A_minus_B", -est, -hi, -lo, len(pl), "positive => A is easier for CT to retake than B")
        rows.append({"quantity": "post-plant T win rate: A minus B", "site": "A-B", "n": len(pl), "estimate": est, "lo": float(lo), "hi": float(hi)})
        sh = (pl["plant_site"] == "A").to_numpy(float)
        boot = vc.cluster_bootstrap(mn, lambda r: sh[r].mean() - 0.5, N_BOOT, seed=6)
        lo, hi = vc.pctl_ci(boot)
        vc.record_result("site.plant_share_A_minus_half", sh.mean() - 0.5, lo, hi, len(pl), "positive => T plants A more than B")
    snap_p = vc.METRICS / "plant_aligned" / "plant_snapshot_summary.csv"
    if snap_p.exists():
        for _, r in pd.read_csv(snap_p).iterrows():
            if r["group"] in ("site A", "site B"):
                rows.append({"quantity": "T site control at plant (Metric 2 style)", "site": r["group"][-1], "n": r["n"],
                             "estimate": r["mean_T_site_control_at_plant"], "lo": r["lo"], "hi": r["hi"]})
    else:
        vc.log("  (run analysis_plant_aligned_control.py first to include site control at plant time)")
    pd.DataFrame(rows).to_csv(OUTDIR / "site_summary.csv", index=False)
    vc.log(pd.DataFrame(rows).round(4).to_string(index=False))

    # structural distances (no demo data)
    ctx = vc.MapContext()
    groups = {"TSpawn": ctx.name_groups.get("TSpawn"), "CTSpawn": ctx.name_groups.get("CTSpawn"),
              "SiteA": ctx.sites.get("A"), "SiteB": ctx.sites.get("B")}
    missing = [k for k, v in groups.items() if v is None or len(v) == 0]
    if missing:
        vc.log(f"  structural distances: groups not found in graph names: {missing}")
    else:
        srows = []
        for src, dst in (("TSpawn", "SiteA"), ("TSpawn", "SiteB"), ("CTSpawn", "SiteA"), ("CTSpawn", "SiteB"),
                         ("SiteA", "SiteB"), ("SiteB", "SiteA")):
            D = ctx.dist[np.ix_(groups[src], groups[dst])]
            fin = D[np.isfinite(D)]
            srows.append({"from": src, "to": dst, "min": fin.min(), "median": np.median(fin), "n_pairs": fin.size})
        st = pd.DataFrame(srows)
        st.to_csv(OUTDIR / "structural_distances.csv", index=False)
        vc.log(st.round(1).to_string(index=False))
        g = lambda f, t, c: float(st[(st["from"] == f) & (st["to"] == t)][c].iloc[0])
        for s in ("A", "B"):
            for c in ("min", "median"):
                vc.record_result(f"struct.{s}.T_minus_CT_{c}", g("TSpawn", f"Site{s}", c) - g("CTSpawn", f"Site{s}", c),
                                 note="positive => site is structurally nearer CT spawn than T spawn (CT-favourable)")
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.35
        for i, c in enumerate(("min", "median")):
            vals = [g("TSpawn", "SiteA", c), g("CTSpawn", "SiteA", c), g("TSpawn", "SiteB", c), g("CTSpawn", "SiteB", c)]
            ax.bar(np.arange(4) + i * w, vals, w, label=c)
        ax.set_xticks(np.arange(4) + w / 2); ax.set_xticklabels(["T→A", "CT→A", "T→B", "CT→B"])
        ax.set_ylabel("directed graph distance (game units)"); ax.set_title("Structural distance to each site"); ax.legend()
        vc.savefig(fig, FIGDIR / "structural_distances.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ss = pd.DataFrame(rows)
    t = ss[ss["quantity"] == "post-plant T win rate"]
    if len(t):
        ax.bar(t["site"], t["estimate"], yerr=[t["estimate"] - t["lo"], t["hi"] - t["estimate"]], capsize=4)
        ax.axhline(0.5, color="k", ls="--", lw=1); ax.set_ylabel("T win rate given a plant"); ax.set_title("Post-plant outcome by site")
        vc.savefig(fig, FIGDIR / "post_plant_by_site.png")


if __name__ == "__main__":
    main()
