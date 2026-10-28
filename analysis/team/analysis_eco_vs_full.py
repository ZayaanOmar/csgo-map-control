#!/usr/bin/env python
"""
Eco vs full-buy positioning (T and CT side), field and occupancy.

Conditions (non-pistol, non-overtime rounds), from the point of view of one team and side:
  E : own team on eco       vs a full-buy opponent      (primary contrast: E vs F)
  F : own team on full-buy  vs a full-buy opponent      (baseline)
  A : own team on full-buy  vs an eco opponent          (secondary contrast: A vs F)

Design
  * Unit = one team's rounds on one side in one match. Only units containing BOTH conditions
    are used, so every comparison is within team and within match: team style, opponent,
    tournament and era cancel by construction.
  * Pooled signatures: unit means (round-weighted within a unit) combined with weights equal
    to the harmonic mean of the unit's two round counts.
  * Omnibus test: D = Jensen-Shannon distance between the pooled signatures of the two
    conditions. Null = condition labels shuffled among the rounds of the SAME unit (counts per
    unit preserved), N_PERM times.
  * Area shifts: delta_k = pooled share (condition) - pooled share (F). 95% CI from a
    bootstrap over units (units are independent within a side: one unit per match).
    Adjusted p-values use the max-T permutation method (controls the family-wise error rate
    over all areas).
  * Primary cells = E vs F, window PRIMARY_WINDOW, both sides, both methods (4 cells, Holm).
    Everything else is a sensitivity analysis.
  * Diagnostic variant "excl_early_half" drops rounds 2-4 of each half (post-pistol rounds are
    where eco rounds cluster).

Inputs : outputs/analysis/round_signatures.parquet, outputs/analysis/round_economy.csv
Outputs: outputs/analysis/eco_vs_full_omnibus.csv, eco_vs_full_area_shifts.csv,
         outputs/analysis/figures/eco_vs_full_shifts.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIG_PATH = "outputs/analysis/round_signatures.parquet"
ECON_PATH = "outputs/analysis/round_economy.csv"
OUT_DIR = "outputs/analysis"
PRIMARY_WINDOW = "early_5_20"
WINDOWS = ("early_5_20", "mid_20_40", "pre_first_loss")
SIDES = ("T", "CT")
METHODS = ("field", "occupancy")
CONTRASTS = ("E", "A")
MIN_UNITS = 10
N_PERM = 2000
N_BOOT = 500
SEED = 42
EARLY_HALF_MAX = 4      # variant "excl_early_half" drops rounds 1..4 of each 15-round half


# ------------------------------------------------------------------ helpers
def js_pair(p, q):
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.where(p > 0, p * np.log2(p / m), 0.0).sum()
        kq = np.where(q > 0, q * np.log2(q / m), 0.0).sum()
    return float(np.sqrt(max(0.5 * (kp + kq), 0.0)))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def prepare(sig, econ):
    e = econ.loc[~econ["is_pistol"] & ~econ["is_overtime"], ["match_id", "round", "ct_buy", "t_buy"]]
    s = sig.merge(e, on=["match_id", "round"], how="inner")
    t_side = s["side"] == "T"
    s["own"] = np.where(t_side, s["t_buy"], s["ct_buy"])
    s["opp_buy"] = np.where(t_side, s["ct_buy"], s["t_buy"])
    s["cond"] = np.select(
        [(s["own"] == "eco") & (s["opp_buy"] == "full"),
         (s["own"] == "full") & (s["opp_buy"] == "full"),
         (s["own"] == "full") & (s["opp_buy"] == "eco")],
        ["E", "F", "A"], default="other")
    s["rih"] = (s["round"] - 1) % 15 + 1          # round within its half (1..15)
    return s


# ------------------------------------------------------------------ one cell
def analyse_cell(d, acols, cx, rng):
    """d: signature rows of one (window, side, method[, variant]). Returns None if too few units."""
    d = d[d["cond"].isin([cx, "F"])]
    if d.empty:
        return None
    uid0 = d.groupby(["team", "match_id"]).ngroup().to_numpy()
    isx0 = (d["cond"] == cx).to_numpy()
    U0 = uid0.max() + 1
    nx0 = np.bincount(uid0[isx0], minlength=U0)
    nf0 = np.bincount(uid0[~isx0], minlength=U0)
    keep = (nx0 > 0) & (nf0 > 0)
    if keep.sum() < MIN_UNITS:
        return None
    rows = keep[uid0]
    uid = (np.cumsum(keep) - 1)[uid0[rows]]
    isx = isx0[rows]
    dd = d[rows]
    R = dd[acols].to_numpy(float)
    U, K = int(keep.sum()), R.shape[1]
    nx, nf = nx0[keep].astype(float), nf0[keep].astype(float)
    nxi, nfi = nx0[keep], nf0[keep]
    team_of_unit = np.empty(U, dtype=object)
    team_of_unit[uid] = dd["team"].to_numpy()
    w = 2 * nx * nf / (nx + nf)

    def unit_means(order):
        u_sorted = uid[order]
        first = np.searchsorted(u_sorted, np.arange(U))
        cs = np.vstack([np.zeros((1, K)), np.cumsum(R[order], axis=0)])
        sum_x = cs[first + nxi] - cs[first]
        sum_f = cs[first + nxi + nfi] - cs[first + nxi]
        return sum_x / nx[:, None], sum_f / nf[:, None]

    def pooled(mx, mf):
        return (w[:, None] * mx).sum(0) / w.sum(), (w[:, None] * mf).sum(0) / w.sum()

    mx, mf = unit_means(np.lexsort((~isx, uid)))              # observed: X rows first in each unit
    px, pf = pooled(mx, mf)
    delta, D = px - pf, js_pair(px, pf)

    d_null, delta_null = np.empty(N_PERM), np.empty((N_PERM, K))
    for b in range(N_PERM):
        mxp, mfp = unit_means(np.lexsort((rng.random(len(uid)), uid)))   # shuffle labels within units
        pxp, pfp = pooled(mxp, mfp)
        delta_null[b], d_null[b] = pxp - pfp, js_pair(pxp, pfp)

    sd = delta_null.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.inf)
    max_null = np.abs(delta_null / sd).max(axis=1)
    z_obs = np.abs(delta / sd)
    p_raw = (1 + (np.abs(delta_null) >= np.abs(delta)).sum(axis=0)) / (1 + N_PERM)
    p_maxt = (1 + (max_null[None, :] >= z_obs[:, None]).sum(axis=1)) / (1 + N_PERM)

    c = rng.multinomial(U, np.full(U, 1.0 / U), size=N_BOOT) * w
    bx = (c @ mx) / c.sum(1, keepdims=True)
    bf = (c @ mf) / c.sum(1, keepdims=True)
    lo, hi = np.percentile(bx - bf, [2.5, 97.5], axis=0)

    teams = np.unique(team_of_unit)
    signs, n_t = [], 0
    for t in teams:
        sel = team_of_unit == t
        if sel.sum() >= 2:
            dt = (w[sel, None] * (mx[sel] - mf[sel])).sum(0) / w[sel].sum()
            signs.append(np.sign(dt))
            n_t += 1
    frac_same = (np.mean(np.array(signs) == np.sign(delta), axis=0) if n_t else np.full(K, np.nan))

    return {
        "n_units": U, "n_teams": len(teams), "n_x": int(nx.sum()), "n_f": int(nf.sum()),
        "D": D, "D_null_mean": float(d_null.mean()), "D_null_p95": float(np.percentile(d_null, 95)),
        "p_D": float((1 + (d_null >= D).sum()) / (1 + N_PERM)),
        "px": px, "pf": pf, "delta": delta, "lo": lo, "hi": hi, "p_raw": p_raw, "p_maxt": p_maxt,
        "frac_same": frac_same, "n_teams_sign": n_t,
    }


# ------------------------------------------------------------------ plotting
def shifts_figure(area_df, areas, path):
    sub = area_df[(area_df["contrast"] == "E") & (area_df["window"] == PRIMARY_WINDOW)
                  & (area_df["variant"] == "all")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=False)
    for i, side in enumerate(SIDES):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            g = sub[(sub["side"] == side) & (sub["method"] == method)].sort_values("delta")
            if g.empty:
                ax.set_axis_off()
                continue
            y = np.arange(len(g))
            sig = (g["p_maxt"] < 0.05).to_numpy()
            ax.barh(y, g["delta"], color=np.where(sig, "tab:red", "0.7"))
            ax.errorbar(g["delta"], y, xerr=[g["delta"] - g["ci_lo"], g["ci_hi"] - g["delta"]],
                        fmt="none", ecolor="k", lw=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(g["area"], fontsize=7)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{side} side, {method}", fontsize=10)
            ax.set_xlabel("share on eco - share on full-buy (95% CI)", fontsize=8)
    fig.suptitle(f"Eco vs full-buy (both vs a full-buy opponent), {PRIMARY_WINDOW}: pooled within-match "
                 f"shift per area. Red = max-T adjusted p < 0.05", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)
    sig = pd.read_parquet(SIG_PATH)
    econ = pd.read_csv(ECON_PATH, dtype={"match_id": str})
    sig["match_id"] = sig["match_id"].astype(str)
    acols = [c for c in sig.columns if c.startswith("a_")]
    areas = [c[2:] for c in acols]
    s = prepare(sig, econ)

    print("=== design diagnostics (non-pistol, non-overtime rounds) ===")
    first = s[(s["window"] == PRIMARY_WINDOW) & (s["method"] == "field")]
    print("rounds per condition and side (primary window):")
    print(first.groupby(["side", "cond"]).size().unstack("cond").reindex(columns=["E", "F", "A", "other"]).to_string())
    print(f"\nshare of rounds that are rounds 2-{EARLY_HALF_MAX} of their half (post-pistol), by condition:")
    first = first.assign(early_half=first["rih"] <= EARLY_HALF_MAX)
    print(first.groupby(["side", "cond"])["early_half"].mean().unstack("cond")
          .reindex(columns=["E", "F", "A"]).round(3).to_string())
    pre = s[(s["window"] == "pre_first_loss") & (s["method"] == "field")]
    pre = pre.assign(win_len=pre["win_end"] - pre["win_start"])
    print("\nmean pre_first_loss window length (s) by condition (why it is not the primary window):")
    print(pre.groupby(["side", "cond"])["win_len"].mean().unstack("cond")
          .reindex(columns=["E", "F", "A"]).round(1).to_string())

    omni, shifts = [], []
    for contrast in CONTRASTS:
        for window in WINDOWS:
            for side in SIDES:
                for method in METHODS:
                    base = s[(s["window"] == window) & (s["side"] == side) & (s["method"] == method)]
                    for variant in ("all", "excl_early_half"):
                        if variant == "excl_early_half" and window != PRIMARY_WINDOW:
                            continue
                        d = base if variant == "all" else base[base["rih"] > EARLY_HALF_MAX]
                        r = analyse_cell(d, acols, contrast, rng)
                        if r is None:
                            print(f"skip: {contrast}/{window}/{side}/{method}/{variant} (too few paired units)")
                            continue
                        meta = {"contrast": contrast, "window": window, "side": side,
                                "method": method, "variant": variant}
                        omni.append({**meta, **{k: r[k] for k in (
                            "n_units", "n_teams", "n_x", "n_f", "D", "D_null_mean", "D_null_p95", "p_D")},
                            "primary": contrast == "E" and window == PRIMARY_WINDOW and variant == "all"})
                        for k, a in enumerate(areas):
                            shifts.append({**meta, "area": a, "pooled_x": r["px"][k], "pooled_f": r["pf"][k],
                                           "delta": r["delta"][k], "ci_lo": r["lo"][k], "ci_hi": r["hi"][k],
                                           "p_raw": r["p_raw"][k], "p_maxt": r["p_maxt"][k],
                                           "frac_teams_same_sign": r["frac_same"][k],
                                           "n_teams_sign": r["n_teams_sign"]})
    omni, shifts = pd.DataFrame(omni), pd.DataFrame(shifts)
    omni["p_D_holm"] = np.nan
    prim = omni["primary"]
    if prim.any():
        omni.loc[prim, "p_D_holm"] = holm(omni.loc[prim, "p_D"])
    omni.to_csv(f"{OUT_DIR}/eco_vs_full_omnibus.csv", index=False)
    shifts.to_csv(f"{OUT_DIR}/eco_vs_full_area_shifts.csv", index=False)

    cols = ["contrast", "window", "side", "method", "variant", "n_units", "n_teams", "n_x", "n_f",
            "D", "D_null_mean", "D_null_p95", "p_D", "p_D_holm"]
    show = omni[cols].copy()
    for c in ("D", "D_null_mean", "D_null_p95"):
        show[c] = show[c].round(4)
    for c in ("p_D", "p_D_holm"):
        show[c] = show[c].round(4)
    print(f"\n=== Omnibus: JS distance between pooled eco-side and full-buy signatures; "
          f"null = labels shuffled within unit ({N_PERM} perms; smallest p = {1 / (1 + N_PERM):.4f}) ===")
    print("(contrast E = eco vs F; A = anti-eco vs F; n_x / n_f = rounds in the paired units)")
    print(show.to_string(index=False))
    print("\np_D_holm: Holm over the 4 primary cells (E vs F, primary window, all rounds). Everything else is exploratory.")

    if not shifts.empty:
        top = shifts[(shifts["contrast"] == "E") & (shifts["window"] == PRIMARY_WINDOW) & (shifts["variant"] == "all")]
        for (side, method), g in top.groupby(["side", "method"]):
            g = g.reindex(g["delta"].abs().sort_values(ascending=False).index).head(6)
            print(f"\n--- {side}/{method}: six largest area shifts (E vs F, {PRIMARY_WINDOW}) ---")
            print(g[["area", "pooled_x", "pooled_f", "delta", "ci_lo", "ci_hi", "p_raw", "p_maxt",
                     "frac_teams_same_sign"]].round(4).to_string(index=False))
        print("\n=== field vs occupancy: correlation of per-area shifts across the 27 areas (E vs F, primary) ===")
        for side in SIDES:
            f = top[(top["side"] == side) & (top["method"] == "field")].set_index("area")["delta"]
            o = top[(top["side"] == side) & (top["method"] == "occupancy")].set_index("area")["delta"]
            if len(f) and len(o):
                print(f"{side}: Pearson r = {np.corrcoef(f, o.reindex(f.index))[0, 1]:.3f} "
                      f"| largest |shift| field {f.abs().max():.4f} vs occupancy {o.abs().max():.4f}")
        shifts_figure(shifts, areas, f"{OUT_DIR}/figures/eco_vs_full_shifts.png")
    print(f"\nWrote {OUT_DIR}/eco_vs_full_omnibus.csv, eco_vs_full_area_shifts.csv, figures/eco_vs_full_shifts.png")


if __name__ == "__main__":
    main()
