"""
Metric 2 — Site Control Score: Visualizations

Reads the output of metric2_site_control_score.py and produces three static
PNG figures to support interpretation of RQ2 ("does field control at bomb
plants predict outcomes / show T-side control at the moment of planting?").

Inputs:
    outputs/metrics/site_control_score/{match_name}.csv — match_id, round, bomb_site,
        plant_tick, frame_idx, site_control_score, note
    outputs/metrics/site_control_score/_summary.json    — C_bar, per_site_breakdown

Outputs (all under outputs/metrics/site_control_score/figures/):
    01_score_distribution.png        — histogram of site_control_score with a 0.5 reference line
    02_score_by_site_violin.png      — violin/box plot split by bomb_site (A vs B)
    03_score_vs_round_scatter.png    — site_control_score vs round number, colored by site (QA check)
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --/ Path consts
METRIC_DIR = "outputs/metrics/site_control_score"
SUMMARY_PATH = os.path.join(METRIC_DIR, "_summary.json")
FIGURES_DIR = os.path.join(METRIC_DIR, "figures")

SITE_COLORS = {"A": "#1f77b4", "B": "#ff7f0e"}
DEFAULT_COLOR = "#7f7f7f"


def load_all_scores() -> pd.DataFrame:
    """Load and concatenate every per-match plant-event CSV.

    Only scored events (site_control_score not null) are kept for plotting
    — unscorable events (note != "") were already excluded from C_bar in
    the metric script itself and are excluded here for consistency.
    """
    per_match_files = sorted(glob.glob(os.path.join(METRIC_DIR, "*.csv")))
    if not per_match_files:
        raise FileNotFoundError(
            f"No per-match CSVs found in {METRIC_DIR} — run metric2_site_control_score.py first"
        )

    frames = []
    for path in per_match_files:
        df = pd.read_csv(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined[combined["site_control_score"].notna()].copy()


def plot_score_distribution(df: pd.DataFrame, c_bar: float, out_path: str) -> None:
    """Histogram of site_control_score with a vertical reference line at 0.5.

    This is the primary chart for RQ2's validity check: visually, is the
    mass of the distribution to the left of 0.5 (T-side control at the
    moment of planting, as it must be for a successful plant to occur)?
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["site_control_score"], bins=40, color="#2ca02c", alpha=0.8, edgecolor="white")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Neutral (0.5)")
    if c_bar is not None:
        ax.axvline(c_bar, color="#d62728", linestyle="-", linewidth=2,
                    label=f"C̄ = {c_bar:.4f}")
    ax.set_xlabel("Site Control Score  C_e  (0 = full T control, 1 = full CT control)")
    ax.set_ylabel("Number of plant events")
    ax.set_title("Distribution of Site Control Score at Plant Events")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_by_site(df: pd.DataFrame, out_path: str) -> None:
    """Box + strip plot of site_control_score split by bomb_site (A vs B).

    Checks whether the RQ2 validity result holds independently for both
    sites rather than being driven by one — A and B have very different
    chokepoint geometry on de_mirage, so this is worth inspecting
    separately rather than trusting only the pooled result.
    """
    sites = sorted(s for s in df["bomb_site"].unique() if s in ("A", "B"))
    if not sites:
        print("  No valid A/B site labels found — skipping site-split plot")
        return

    data = [df.loc[df["bomb_site"] == s, "site_control_score"].values for s in sites]
    colors = [SITE_COLORS.get(s, DEFAULT_COLOR) for s in sites]

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot(data, labels=sites, patch_artist=True, showfliers=False, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # jittered strip overlay so individual plant events remain visible
    # alongside the box summary, rather than being hidden by it
    rng = np.random.default_rng(0)
    for i, (site, vals) in enumerate(zip(sites, data), start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=colors[i - 1],
                   alpha=0.5, s=14, edgecolor="none", zorder=3)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Neutral (0.5)")
    ax.set_ylabel("Site Control Score  C_e")
    ax.set_title("Site Control Score by Bomb Site")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_vs_round(df: pd.DataFrame, out_path: str) -> None:
    """Scatter of site_control_score vs round number, colored by site.

    Primarily a data-quality check: does the score drift systematically
    as a match wears on (e.g. late-match eco rounds with fewer players
    alive to contribute to the field)? A flat scatter with no visible
    trend is the expected/healthy result.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for site in sorted(s for s in df["bomb_site"].unique() if s in ("A", "B")):
        sub = df[df["bomb_site"] == site]
        ax.scatter(sub["round"], sub["site_control_score"], label=f"Site {site}",
                   color=SITE_COLORS.get(site, DEFAULT_COLOR), alpha=0.6, s=20)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Round number")
    ax.set_ylabel("Site Control Score  C_e")
    ax.set_title("Site Control Score vs Round Number (data-quality check)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found — run metric2_site_control_score.py first"
        )
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
    c_bar = summary.get("mean_site_control_score")

    print("Loading per-match site control score data...")
    df = load_all_scores()
    print(f"  Loaded {len(df):,} scored plant events\n")

    if df.empty:
        print("No scored plant events available — nothing to plot")
        return

    print("Plotting 01_score_distribution.png")
    plot_score_distribution(df, c_bar, os.path.join(FIGURES_DIR, "01_score_distribution.png"))

    print("Plotting 02_score_by_site_violin.png")
    plot_score_by_site(df, os.path.join(FIGURES_DIR, "02_score_by_site_violin.png"))

    print("Plotting 03_score_vs_round_scatter.png")
    plot_score_vs_round(df, os.path.join(FIGURES_DIR, "03_score_vs_round_scatter.png"))

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
