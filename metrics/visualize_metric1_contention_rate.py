"""
Metric 1 — Contention Rate: Visualizations

Reads the output of metric1_contention_rate.py and produces PNG figures to
support interpretation of RQ1 ("does the continuous influence field capture
contested space better than a Voronoi baseline?"). Figures 01, 02, and 04
are drawn as TWO-PANEL comparisons — full-map (left) vs active-footprint
(right) — for the same reason as visualize_metric3_control_margin.py: the
active-footprint restriction specifically corrects a known problem with the
full-map version (here, that a node with nobody nearby sits at c=0.5, which
falls INSIDE the [0.4, 0.6] contested band and is therefore miscounted as
"contested" rather than "empty" — see metric1_contention_rate.py's module
docstring). Putting the two versions side by side makes the correction's
direction and magnitude visible directly, rather than requiring the reader
to hold two separate figures' y-axes in their head.

Figure 03 (continuous vs Voronoi) is the one exception to the two-panel
layout — it is already a direct comparison chart, so the active-footprint
field is added as a third bar rather than a second panel.

If contention_rate_active / _summary.json's "active_footprint" block are
still all-null (i.e. team_agg.py has not yet been patched to persist
ct_mu/t_mu — see metric3_control_margin.py), every figure degrades to a
single full-map panel with an annotation explaining why, rather than
erroring out.

Inputs:
    outputs/metrics/contention_rate/{match_name}.csv  — round, frame_idx,
        num_nodes, num_contested, contention_rate, num_active_nodes,
        num_active_contested, contention_rate_active, active_node_fraction
    outputs/metrics/contention_rate/_summary.json      — nested "full_map"
        and "active_footprint" blocks, each with rho_bar_contention_rate
        and hypothesis_check; top-level mean_active_node_fraction

Outputs (all under outputs/metrics/contention_rate/figures/):
    01_contention_over_round_progress.png — mean contention_rate vs
        normalised round-fraction, with IQR band, full-map vs active panels
    02_contention_rate_distribution.png   — histogram of per-frame
        contention_rate across the dataset, full-map vs active panels
    03_continuous_vs_voronoi.png          — three-bar comparison: full-map
        field, active-footprint field, and the analytic Voronoi value (0.0)
    04_per_match_boxplot.png              — per-match box plot of per-frame
        contention_rate (outlier/QA check), full-map vs active panels
    05_active_node_fraction.png           — NEW: distribution of
        active_node_fraction per frame — how much of the 903-node map is
        actually "in play" on average; also a sanity check on EPSILON's
        calibration. Skipped if active_node_fraction is unavailable.
    06_contention_by_area.png             — NEW: per-area breakdown (how often
        each named area is in play, and how contested it is when it is), read
        from by_area/_area_summary.csv. Skipped if that file is absent.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --/ Path consts
METRIC_DIR = "outputs/metrics/contention_rate"
SUMMARY_PATH = os.path.join(METRIC_DIR, "_summary.json")
FIGURES_DIR = os.path.join(METRIC_DIR, "figures")

N_PROGRESS_BINS = 20  # number of round-progress buckets for the round-progress line plot


def load_all_frame_stats() -> pd.DataFrame:
    """Load and concatenate every per-match frame-level CSV, tagging each row with its match_name.

    These files are small (rounds x frames rows, not rounds x frames x 903
    nodes), so loading the whole dataset into memory is safe here — this is
    a different scale than the control_fields/influence_fields data itself.
    """
    per_match_files = sorted(glob.glob(os.path.join(METRIC_DIR, "*.csv")))
    if not per_match_files:
        raise FileNotFoundError(
            f"No per-match CSVs found in {METRIC_DIR} — run metric1_contention_rate.py first"
        )

    frames = []
    for path in per_match_files:
        mname = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        df["match_name"] = mname
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def add_round_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Add a round_progress column in [0, 1] — frame_idx normalised by the
    max frame_idx seen in that specific (match_name, round) group.

    This lets frames from rounds of very different lengths be pooled onto
    a common x-axis ("what fraction of the way through the round is this"),
    which is necessary before averaging contention_rate across rounds —
    otherwise a round-15 late-round frame and a round-40 late-round frame
    wouldn't line up on frame_idx alone.
    """
    df = df.copy()
    max_frame = df.groupby(["match_name", "round"])["frame_idx"].transform("max")
    # guard against a single-frame round (max_frame == 0) causing division by zero
    df["round_progress"] = np.where(max_frame > 0, df["frame_idx"] / max_frame, 0.0)
    return df


def active_data_available(df: pd.DataFrame) -> bool:
    return "contention_rate_active" in df.columns and df["contention_rate_active"].notna().any()


def _annotate_unavailable(ax, reason: str) -> None:
    """Used in place of a real panel when the active-footprint data isn't
    there yet, so the figure still renders and clearly explains why the
    panel is empty, rather than the script skipping the whole plot.
    """
    ax.text(
        0.5, 0.5, reason,
        ha="center", va="center", wrap=True, fontsize=9, color="dimgray",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])


ACTIVE_UNAVAILABLE_MSG = (
    "Active-footprint data not available yet.\n\n"
    "control_fields/{match}.csv lacks ct_mu/t_mu —\n"
    "team_agg.py needs to persist these alongside\n"
    "control_val before this panel can be computed.\n"
    "See metric1_contention_rate.py docstring."
)


def _round_progress_panel(ax, df: pd.DataFrame, column: str, title: str) -> None:
    """Draws the binned median + IQR band for one contention_rate column
    onto a given axis. Factored out so both the full-map and
    active-footprint versions share identical binning/plotting logic.
    """
    df = add_round_progress(df)
    bins = np.linspace(0.0, 1.0, N_PROGRESS_BINS + 1)
    df["progress_bin"] = pd.cut(df["round_progress"], bins=bins, include_lowest=True)

    grouped = df.groupby("progress_bin", observed=True)[column]
    bin_centers = [interval.mid for interval in grouped.groups.keys()]
    medians = grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)

    order = np.argsort(bin_centers)
    bin_centers = np.array(bin_centers)[order]
    medians = medians.values[order]
    q25 = q25.values[order]
    q75 = q75.values[order]

    ax.plot(bin_centers, medians, color="#1f77b4", linewidth=2, label="Median contention rate")
    ax.fill_between(bin_centers, q25, q75, color="#1f77b4", alpha=0.2, label="IQR (25th-75th pct)")
    ax.set_xlabel("Round progress (0 = round start, 1 = round end)")
    ax.set_ylabel("Contention rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_contention_over_round_progress(df: pd.DataFrame, out_path: str) -> None:
    """Mean contention_rate vs normalised round-fraction, with an IQR band.

    Bins round_progress into N_PROGRESS_BINS equal-width buckets and plots
    the median with a shaded 25th-75th percentile band, rather than a
    raw per-frame scatter, since round-progress-aligned frames from
    different rounds/matches are not the same underlying observation and
    a smoothed summary is more honest than a dense scatter would be.
    Full-map and active-footprint panels are drawn side by side so any
    difference in trend shape (not just level) is directly visible.
    """
    has_active = active_data_available(df)
    fig, axes = plt.subplots(1, 2 if has_active else 1, figsize=(16 if has_active else 9, 5))
    axes = np.atleast_1d(axes)

    _round_progress_panel(axes[0], df, "contention_rate", "Full Map")

    if has_active:
        active_df = df.dropna(subset=["contention_rate_active"])
        _round_progress_panel(axes[1], active_df, "contention_rate_active", "Active Footprint")
    elif len(axes) > 1:
        _annotate_unavailable(axes[1], ACTIVE_UNAVAILABLE_MSG)

    fig.suptitle("Contention Rate over Round Progress", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _distribution_panel(ax, values: np.ndarray, rho_bar, title: str) -> None:
    ax.hist(values, bins=50, color="#1f77b4", alpha=0.8, edgecolor="white")
    if rho_bar is not None:
        ax.axvline(rho_bar, color="#d62728", linestyle="--", linewidth=2,
                    label=f"ρ̄ = {rho_bar:.4f}")
        ax.legend()
    ax.set_xlabel("Contention rate")
    ax.set_ylabel("Number of frames")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_contention_distribution(df: pd.DataFrame, summary: dict, out_path: str) -> None:
    """Histogram of per-frame contention_rate across the whole dataset, with
    rho_bar marked. Full-map and active-footprint panels are drawn side by
    side; comparing the two shapes directly shows how much of the full-map
    distribution's mass near 0.5 was coming from empty-map frames rather
    than genuinely contested ones.
    """
    has_active = active_data_available(df)
    fig, axes = plt.subplots(1, 2 if has_active else 1, figsize=(14 if has_active else 8, 5))
    axes = np.atleast_1d(axes)

    full_rho_bar = summary.get("full_map", {}).get("rho_bar_contention_rate")
    _distribution_panel(axes[0], df["contention_rate"].values, full_rho_bar, "Full Map")

    if has_active:
        active_rho_bar = summary.get("active_footprint", {}).get("rho_bar_contention_rate")
        active_vals = df["contention_rate_active"].dropna().values
        _distribution_panel(axes[1], active_vals, active_rho_bar, "Active Footprint")
    elif len(axes) > 1:
        _annotate_unavailable(axes[1], ACTIVE_UNAVAILABLE_MSG)

    fig.suptitle("Distribution of Per-Frame Contention Rate", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_continuous_vs_voronoi(summary: dict, out_path: str) -> None:
    """Three-bar comparison: full-map field's rho_bar, active-footprint
    field's rho_bar, and the analytic Voronoi value (0.0).

    The Voronoi bar is exactly 0 by construction under BOTH restrictions —
    hard nearest-player assignment can never place a node in the contested
    band, regardless of which subset of nodes is considered — so no
    separate Voronoi computation is required for either bar. This figure
    stays a single three-bar chart rather than two panels, since a
    field-vs-baseline comparison is already the figure's whole purpose;
    adding the active-footprint field as a third bar is more direct than
    duplicating the comparison across two panels.
    """
    full_rho_bar = summary.get("full_map", {}).get("rho_bar_contention_rate")
    active_rho_bar = summary.get("active_footprint", {}).get("rho_bar_contention_rate")

    labels = ["Full-map\nfield"]
    values = [full_rho_bar if full_rho_bar is not None else 0.0]
    colors = ["#1f77b4"]

    if active_rho_bar is not None:
        labels.append("Active-footprint\nfield")
        values.append(active_rho_bar)
        colors.append("#2ca02c")

    labels.append("Voronoi\nbaseline")
    values.append(0.0)
    colors.append("#7f7f7f")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Mean contention rate  ρ̄")
    ax.set_title("RQ1: Continuous Field vs Voronoi Baseline")
    ax.set_ylim(0, max(values + [0.05]) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _per_match_boxplot_panel(ax, df: pd.DataFrame, column: str, match_order: list, title: str) -> None:
    data = [df.loc[df["match_name"] == m, column].dropna().values for m in match_order]
    ax.boxplot(data, labels=match_order, showfliers=False)
    ax.set_ylabel("Contention rate")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.3)


def plot_per_match_boxplot(df: pd.DataFrame, out_path: str) -> None:
    """Per-match box plot of per-frame contention_rate — a QA/outlier check.

    Flags whether any single match is behaving very differently from the
    rest of the dataset (e.g. from a bad z_tol mapping or a corrupted
    parse) before the pooled rho_bar is trusted for the write-up. Full-map
    and active-footprint panels are drawn side by side so an outlier match
    can be checked against both versions at once.
    """
    match_order = sorted(df["match_name"].unique())
    has_active = active_data_available(df)

    fig_width = max(8, 0.4 * len(match_order)) * (2 if has_active else 1)
    fig, axes = plt.subplots(1, 2 if has_active else 1, figsize=(fig_width, 6))
    axes = np.atleast_1d(axes)

    _per_match_boxplot_panel(axes[0], df, "contention_rate", match_order, "Full Map")

    if has_active:
        _per_match_boxplot_panel(axes[1], df, "contention_rate_active", match_order, "Active Footprint")
    elif len(axes) > 1:
        _annotate_unavailable(axes[1], ACTIVE_UNAVAILABLE_MSG)

    fig.suptitle("Per-Match Contention Rate Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_active_node_fraction(df: pd.DataFrame, out_path: str) -> None:
    """Distribution of active_node_fraction per frame — what fraction of
    the 903 nodes actually carried non-negligible influence, on average,
    at any given frame. New information the active-footprint extension
    makes available; also doubles as a sanity check on EPSILON's
    calibration — a fraction near 0 (EPSILON too strict) or near 1
    (EPSILON too permissive, defeating the point of the filter) would
    signal the constant needs revisiting. Matches
    visualize_metric3_control_margin.py's equivalent figure.
    """
    if "active_node_fraction" not in df.columns or df["active_node_fraction"].isna().all():
        print("  Skipping 05_active_node_fraction.png — active_node_fraction not available")
        return

    vals = df["active_node_fraction"].dropna().values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals, bins=30, color="#2ca02c", alpha=0.7, edgecolor="white")
    mean_val = float(np.mean(vals))
    ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.5,
               label=f"mean = {mean_val:.3f}")
    ax.set_xlabel("Active node fraction per frame  (|V_active(f)| / |V|)")
    ax.set_ylabel("Number of frames")
    ax.set_title("How Much of the Map is 'In Play'? (active-footprint EPSILON diagnostic)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


AREA_SUMMARY_PATH = os.path.join(METRIC_DIR, "by_area", "_area_summary.csv")


def plot_contention_by_area(out_path: str) -> None:
    """Where on the map contention happens, by assigned area (see
    metric1_contention_rate.py, PER-AREA BREAKDOWN). Areas are ordered by
    their share of all active contested tiles, largest at the top.

    Left panel: how often the area is "in play" at all (mean active fraction).
    Right panel: when it is in play, how often a tile is balanced (bars =
    active-footprint rate), with the full-map rate for the same area as a
    hollow marker. The gap between marker and bar is the inflation from
    empty tiles sitting at exactly 0.5 inside the contested band; an area
    that is rarely occupied can have a full-map rate near 1.0 purely for
    that reason.

    Area names carry the tile count, and '*' marks areas that contain tiles
    which had a blank awpy areaName and were assigned by the area-assignment
    step. Rates for areas with only 1-2 tiles are noisy (few distinct values).
    Falls back to a single full-map panel if the active-footprint data is
    not available.
    """
    if not os.path.exists(AREA_SUMMARY_PATH):
        print(f"  Skipping 06_contention_by_area.png — {AREA_SUMMARY_PATH} not found "
              f"(run metric1_contention_rate.py with data/de_mirage_area_assignment.csv present)")
        return

    area = pd.read_csv(AREA_SUMMARY_PATH, index_col=0)
    area = area.iloc[::-1]  # barh draws the first row at the bottom; reverse so the largest is on top
    labels = [
        f"{name}{'*' if row['n_formerly_blank_tiles'] > 0 else ''}  (n={int(row['n_tiles'])})"
        for name, row in area.iterrows()
    ]
    y = np.arange(len(area))
    has_active = area["mean_contention_rate_active"].notna().any()

    fig_height = max(5, 0.32 * len(area) + 1.5)
    if has_active:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, fig_height), sharey=True)

        ax_l.barh(y, area["mean_active_fraction"], color="#2ca02c", alpha=0.75)
        ax_l.set_xlabel("Mean active fraction  (share of the area's tiles in play)")
        ax_l.set_title("How often the area is in play")
        ax_l.set_xlim(0, 1)
        ax_l.set_yticks(y)
        ax_l.set_yticklabels(labels, fontsize=8)
        ax_l.grid(axis="x", alpha=0.3)

        ax_r.barh(y, area["mean_contention_rate_active"].fillna(0.0), color="#1f77b4", alpha=0.8,
                  label="Active footprint")
        ax_r.scatter(area["mean_contention_rate_full"], y, facecolors="none", edgecolors="black",
                     s=36, zorder=3, label="Full map (same area)")
        ax_r.set_xlabel("Mean contention rate")
        ax_r.set_title("How often a tile is contested, when in play")
        ax_r.set_xlim(0, 1)
        ax_r.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
        ax_r.grid(axis="x", alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.barh(y, area["mean_contention_rate_full"], color="#1f77b4", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Mean contention rate (full map)")
        ax.set_title("Active-footprint data not available — full-map rate only")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Contention Rate by Map Area  (* = includes reassigned, formerly blank-named tiles)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found — run metric1_contention_rate.py first"
        )
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print("Loading per-match frame-level contention rate data...")
    df = load_all_frame_stats()
    print(f"  Loaded {len(df):,} frame rows across {df['match_name'].nunique()} matches")

    has_active = active_data_available(df)
    print(f"  Active-footprint data available: {has_active}\n")

    print("Plotting 01_contention_over_round_progress.png")
    plot_contention_over_round_progress(
        df, os.path.join(FIGURES_DIR, "01_contention_over_round_progress.png")
    )

    print("Plotting 02_contention_rate_distribution.png")
    plot_contention_distribution(
        df, summary, os.path.join(FIGURES_DIR, "02_contention_rate_distribution.png")
    )

    print("Plotting 03_continuous_vs_voronoi.png")
    plot_continuous_vs_voronoi(
        summary, os.path.join(FIGURES_DIR, "03_continuous_vs_voronoi.png")
    )

    print("Plotting 04_per_match_boxplot.png")
    plot_per_match_boxplot(
        df, os.path.join(FIGURES_DIR, "04_per_match_boxplot.png")
    )

    print("Plotting 05_active_node_fraction.png")
    plot_active_node_fraction(
        df, os.path.join(FIGURES_DIR, "05_active_node_fraction.png")
    )

    print("Plotting 06_contention_by_area.png")
    plot_contention_by_area(os.path.join(FIGURES_DIR, "06_contention_by_area.png"))

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
