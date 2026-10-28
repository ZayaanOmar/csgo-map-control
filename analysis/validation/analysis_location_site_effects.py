"""
analysis_location_site_effects.py   (which areas drive the site prediction, with uncertainty)
=============================================================================================
For the site-choice target (y = 1 if the plant is at B, 0 if at A) this estimates, for each of the 27 areas,
how much the window-mean control there shifts the odds of a B plant.

SIGN CONVENTION (read this first)
    effect > 0  : T holding MORE of the area points to a B plant
    effect < 0  : T holding more of the area points to an A plant
    "T holding more" = net control going down (net = CT-held minus T-held), so net is flipped and standardised:
    z = -(net - mean) / SD. All main effects are log-odds of B per +1 SD of z (OR = exp(effect)).

TWO KINDS OF EFFECT (they answer different questions)
    marginal  one area at a time (logistic regression on z alone). "Does T control HERE, by itself, point to a site?"
              Neighbouring areas share information, so many areas can look informative at once.
    joint     all 27 areas together (the M1 ridge model at its CV-chosen lambda). "What does this area add
              GIVEN the others?" Ridge shrinks and spreads weight across correlated neighbours, so single joint
              coefficients are unstable; treat them as supporting evidence, not as separate causes.
    With --presence the marginal analysis is repeated for PRESENCE (window mean of ct_mu + t_mu, total activity,
    non-directional), per +1 SD.

UNCERTAINTY: 95% percentile intervals from a bootstrap over MATCHES (n-boot resamples, default 1000). Marginal
    p-values are Wald-style two-sided p-values from the bootstrap standard error (no resample-count floor), with
    Benjamini-Hochberg q-values across the 27 areas. The joint intervals refit the ridge at a FIXED lambda (median of the M1 lambdas in lambda_log.csv),
    so they reflect sampling variability of a penalised fit, not exact coverage of an unpenalised effect.

Outputs (<site-run-dir>/effects/ by default):
    area_effects.csv         one row per area: marginal/joint effects, CIs, OR, p, q (+ presence columns)
    figures/forest_net.png   marginal and joint effects, sorted; filled = q < 0.05 (marginal) / CI excludes 0 (joint)
    figures/effect_map.png   radar map coloured by the marginal effect (faded where q >= 0.05)   [needs awpy; --no-map to skip]

Usage (project root):
    python analysis/validation/analysis_location_site_effects.py --presence
    python analysis/validation/analysis_location_site_effects.py --no-map
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "pipeline"))
import validation_common as vc  # noqa: E402
import location_predictor_lib as lib  # noqa: E402
import analysis_location_site as als  # noqa: E402


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def uni_logit(z: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of y ~ z (essentially unpenalised)."""
    return float(LogisticRegression(C=1e10, max_iter=500).fit(z[:, None], y).coef_[0, 0])


def boot_p(est: float, b: np.ndarray) -> float:
    """Two-sided p from the bootstrap standard error (Wald-style): 2*P(Z > |est|/SE_boot).
    The percentile-count p-value cannot go below ~2/(B+1), which would cap the power of the
    Benjamini-Hochberg step across areas; the SE-based p-value has no such floor."""
    se = float(np.std(b, ddof=1))
    if not np.isfinite(se) or se <= 0:
        return 1.0
    return float(2 * norm.sf(abs(est) / se))


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    run = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        run = min(run, p[i] * n / rank)
        q[i] = run
    return q


def match_resamples(codes: np.ndarray, G: int, B: int, seed: int) -> list:
    rows = [np.flatnonzero(codes == g) for g in range(G)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([rows[g] for g in rng.integers(0, G, G)]) for _ in range(B)]


def marginal_effects(X: np.ndarray, y: np.ndarray, resamples: list, flip: bool) -> pd.DataFrame:
    """Per-column effect per +1 SD (flip=True: oriented so that 'more T control' is positive)."""
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / np.where(sd > 1e-12, sd, 1.0)
    if flip:
        Z = -Z
    rows = []
    for j in range(X.shape[1]):
        if sd[j] < 1e-9:
            rows.append((0.0, 0.0, 0.0, 1.0))
            continue
        est = uni_logit(Z[:, j], y)
        b = np.array([uni_logit(Z[idx, j], y[idx]) for idx in resamples])
        rows.append((est, np.percentile(b, 2.5), np.percentile(b, 97.5), boot_p(est, b)))
    out = pd.DataFrame(rows, columns=["effect", "lo", "hi", "p"])
    out["q"] = bh(out["p"].to_numpy())
    return out


def joint_effects(D: lib.Data, lam: float, resamples: list):
    """M1 ridge refit (net block only) per resample. Effect per +1 SD of 'T holds more' for each area."""
    X, y = D.mats["area_net"], D.y
    idx_all = np.arange(D.n)

    def fit(idx):
        T = lib.prep_blocks([X], idx)
        mu, keep, s = T.params[0]
        th = lib.fit_ridge_logit(T(idx), y[idx], np.zeros(len(idx)), lam)
        w = np.zeros(X.shape[1])
        w[keep] = th[1:]
        sd = X[idx].std(0)
        return -(w * sd / s)                     # flipped: positive = T holding more -> B

    est = fit(idx_all)
    b = np.array([fit(idx) for idx in resamples])
    return est, np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def forest_plot(eff: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    d = eff.sort_values("marginal_effect")
    fig, axs = plt.subplots(1, 2, figsize=(11, 0.32 * len(d) + 1.8), sharey=True)
    y = np.arange(len(d))
    for ax, pre, title, sigcol in ((axs[0], "marginal", "Marginal (one area at a time)", "marginal_q"),
                                   (axs[1], "joint", "Joint (all areas, ridge)", None)):
        e, lo, hi = d[f"{pre}_effect"].to_numpy(), d[f"{pre}_lo"].to_numpy(), d[f"{pre}_hi"].to_numpy()
        sig = (d[sigcol].to_numpy() < 0.05) if sigcol else ((lo > 0) | (hi < 0))
        for i in range(len(d)):
            ax.errorbar(e[i], y[i], xerr=[[e[i] - lo[i]], [hi[i] - e[i]]], fmt="o", capsize=2,
                        color="tab:red" if e[i] > 0 else "tab:blue", mfc=("none" if not sig[i] else None))
        ax.axvline(0, color="k", lw=1, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("log-odds of B per +1 SD", fontsize=9)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(d["area"], fontsize=8)
    fig.suptitle("effect of 'T holds more of the area' on the odds of a B plant; > 0 points to B, < 0 to A. "
                 "Bars: 95% match-bootstrap CI; filled = significant", fontsize=8)
    vc.savefig(fig, path)


def draw_effect_map(ax, tiles: list, area_effect: dict, area_sig: dict, vmax: float, cmap="RdBu_r",
                    label_top: int = 8):
    """Draw tile rectangles coloured by their area's effect. tiles: dicts with x,y,w,h,name (radar coordinates)."""
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cm = plt.get_cmap(cmap)
    for t in tiles:
        n = t["name"]
        if n not in area_effect:
            continue
        ax.add_patch(patches.Rectangle((t["x"], t["y"]), t["w"], t["h"], facecolor=cm(norm(area_effect[n])),
                                       edgecolor="black", linewidth=0.2,
                                       alpha=0.9 if area_sig.get(n, False) else 0.25))
    top = sorted([n for n in area_effect if area_sig.get(n, False)], key=lambda n: -abs(area_effect[n]))[:label_top]
    for n in top:
        m = [t for t in tiles if t["name"] == n]
        tot = sum(t["w"] * t["h"] for t in m) or 1.0
        cx = sum((t["x"] + t["w"] / 2) * t["w"] * t["h"] for t in m) / tot
        cy = sum((t["y"] + t["h"] / 2) * t["w"] * t["h"] for t in m) / tot
        ax.text(cx, cy, f"{n}\n{area_effect[n]:+.2f}", fontsize=6, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, linewidth=0))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    return sm


def effect_map(eff: pd.DataFrame, labels: dict, path: Path, dark: bool) -> None:
    import matplotlib.pyplot as plt
    from awpy.visualization.plot import plot_map
    from plot_area_names import collect_areas
    tiles, _ = collect_areas(labels)
    fig, ax = plot_map("de_mirage", map_type="simpleradar", dark=dark)
    fig.set_size_inches(14, 10)
    area_effect = dict(zip(eff["area"], eff["marginal_effect"]))
    area_sig = dict(zip(eff["area"], eff["marginal_q"] < 0.05))
    vmax = float(np.abs(eff["marginal_effect"]).max()) or 1.0
    sm = draw_effect_map(ax, tiles, area_effect, area_sig, vmax)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("log-odds of a B plant per +1 SD of 'T holds more'   (red -> B, blue -> A; faded = q >= 0.05)")
    ax.set_title("Where does early T control point to a B (vs A) plant? (marginal effects)")
    ax.axis("off")
    vc.savefig(fig, path)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat-dir", default="outputs/predictor")
    ap.add_argument("--site-run-dir", default="outputs/metrics/location_site")
    ap.add_argument("--bomb-dir", default="outputs/bomb_plants")
    ap.add_argument("--min-lead", type=float, default=10.0)
    ap.add_argument("--assignment", default="data/de_mirage_area_assignment.csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--presence", action="store_true", help="also estimate marginal PRESENCE effects")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--light", action="store_true", help="light radar for the map (default: dark)")
    ap.add_argument("--out-dir", default=None, help="default: <site-run-dir>/effects")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir or Path(args.site_run_dir) / "effects")
    vc.ensure_dir(out / "figures")
    D = lib.load_data(Path(args.feat_dir), args.presence,
                      round_filter=als.make_filter(Path(args.bomb_dir), args.min_lead), label_col="y_site_B")
    vc.log(f"{D.n} rounds, {len(D.mnames)} matches; share of plants at B {D.y.mean():.3f}")

    lam_log = pd.read_csv(Path(args.site_run_dir) / "lambda_log.csv")
    L = lam_log[lam_log["model"] == "M1"]["lam"]
    lam = float(np.exp(np.median(np.log(L))))
    vc.log(f"joint model: M1 ridge at lambda = {lam:g} (median of {len(L)} CV choices)")

    resamples = match_resamples(D.mcodes, len(D.mnames), args.n_boot, args.seed)
    vc.log(f"\nmarginal effects ({D.mats['area_net'].shape[1]} areas x {args.n_boot} resamples) ...")
    mnet = marginal_effects(D.mats["area_net"], D.y, resamples, flip=True)
    vc.log("joint effects ...")
    jest, jlo, jhi = joint_effects(D, lam, resamples)

    eff = pd.DataFrame({"area": D.names["area_net"],
                        "marginal_effect": mnet["effect"], "marginal_lo": mnet["lo"], "marginal_hi": mnet["hi"],
                        "marginal_or": np.exp(mnet["effect"]), "marginal_p": mnet["p"], "marginal_q": mnet["q"],
                        "joint_effect": jest, "joint_lo": jlo, "joint_hi": jhi, "joint_or": np.exp(jest)})
    if args.presence:
        mp = marginal_effects(D.mats["area_pres"], D.y, resamples, flip=False)
        eff["presence_effect"], eff["presence_lo"], eff["presence_hi"] = mp["effect"], mp["lo"], mp["hi"]
        eff["presence_p"], eff["presence_q"] = mp["p"], mp["q"]
    eff = eff.sort_values("marginal_effect", ascending=False).reset_index(drop=True)
    eff.to_csv(out / "area_effects.csv", index=False)

    vc.log("\nMARGINAL effect of 'T holds more' per +1 SD (log-odds of B; sorted; q = BH across areas)")
    for _, r in eff.iterrows():
        star = "*" if r["marginal_q"] < 0.05 else " "
        vc.log(f"  {star} {r['area']:<15} {r['marginal_effect']:+.3f} [{r['marginal_lo']:+.3f},{r['marginal_hi']:+.3f}] "
               f"OR {r['marginal_or']:.2f} q={r['marginal_q']:.3f} | joint {r['joint_effect']:+.3f} "
               f"[{r['joint_lo']:+.3f},{r['joint_hi']:+.3f}]")
    vc.log(f"  ({int((eff['marginal_q'] < 0.05).sum())} of {len(eff)} areas significant marginally; "
           f"{int(((eff['joint_lo'] > 0) | (eff['joint_hi'] < 0)).sum())} jointly)")
    if args.presence:
        vc.log("\nPRESENCE (total activity, per +1 SD): significant areas (q < 0.05)")
        for _, r in eff[eff["presence_q"] < 0.05].sort_values("presence_effect").iterrows():
            vc.log(f"    {r['area']:<15} {r['presence_effect']:+.3f} [{r['presence_lo']:+.3f},{r['presence_hi']:+.3f}] "
                   f"q={r['presence_q']:.3f}")

    forest_plot(eff, out / "figures" / "forest_net.png")
    if not args.no_map:
        labels, _ = lib_load_assignment(args.assignment)
        effect_map(eff, labels, out / "figures" / "effect_map.png", dark=not args.light)
    vc.log(f"\nWrote {out}/")


def lib_load_assignment(path):
    from area_assignment import load_assignment
    return load_assignment(path)


if __name__ == "__main__":
    main()
