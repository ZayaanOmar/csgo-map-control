"""
Metric 3 — Control Margin: Visualizations

Reads the output of metric3_control_margin.py and produces PNG figures to
support interpretation of RQ3 ("does control margin predict round
outcome?"). Every headline figure is drawn as a TWO-PANEL comparison —
full-map (left) vs active-footprint (right) — rather than one panel per
version, because the point of the active-footprint metric (see
metric3_control_margin.py's module docstring for the full derivation) is
specifically that it de-dilutes the full-map result, and putting the two
side by side makes that visible at a glance rather than requiring the
reader to hold two separate figures' y-axes in their head.

If control_margin_active / _summary.json's "active_footprint" block are
still all-null (i.e. team_agg.py has not yet been patched to persist
ct_mu/t_mu — see metric3_control_margin.py), every figure degrades to a
single full-map panel with an annotation explaining why, rather than
erroring out.

Inputs:
    outputs/metrics/control_margin/{match_name}.csv — match_id, round,
        control_margin_full, control_margin_active, active_node_fraction,
        winning_side
    outputs/metrics/control_margin/_summary.json     — nested "full_map"
        and "active_footprint" blocks, each with mu_win, mu_loss,
        welch_t_stat, welch_p_value, cohens_d

Outputs (all under outputs/metrics/control_margin/figures/):
    01_win_loss_boxplot.png        — control margin distribution split by
                                      winning_side, full-map vs active-
                                      footprint panels, primary RQ3 chart
    02_win_loss_histograms.png     — overlaid histograms of the two groups,
                                      same two-panel layout
    03_mean_bar_with_ci.png        — mu_win vs mu_loss bar chart with 95%
                                      CI error bars, same two-panel layout
    04_margin_vs_round_scatter.png — control margin vs round number,
                                      colored by outcome (QA/trend check),
                                      same two-panel layout
    05_active_node_fraction.png    — NEW: distribution of active_node_fraction
                                      per round — how much of the 903-node
                                      map is actually "in play" on average;
                                      informative on its own and a sanity
                                      check on EPSILON's calibration (a
                                      fraction near 0 or near 1 would signal
                                      EPSILON is mis-set). Skipped if
                                      active_node_fraction is unavailable.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --/ Path consts
METRIC_DIR = "outputs/metrics/control_margin"
SUMMARY_PATH = os.path.join(METRIC_DIR, "_summary.json")
FIGURES_DIR = os.path.join(METRIC_DIR, "figures")

OUTCOME_COLORS = {"CT": "#1f77b4", "T": "#d62728"}

NEUTRAL_VALUE = 0.5  # sigmoid-normalised control is centred here regardless of full/active


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_rounds() -> pd.DataFrame:
    """Load and concatenate every per-match round-level CSV.

    Rows are dropped only if winning_side or control_margin_full is
    missing — control_margin_active / active_node_fraction are allowed to
    be NaN (that is the expected pre-team_agg.py-patch state) and each
    plotting function below checks for that itself rather than this
    function silently dropping rows that would still be useful for the
    full-map panel.
    """
    per_match_files = sorted(glob.glob(os.path.join(METRIC_DIR, "*.csv")))
    if not per_match_files:
        raise FileNotFoundError(
            f"No per-match CSVs found in {METRIC_DIR} — run metric3_control_margin.py first"
        )

    frames = [pd.read_csv(path) for path in per_match_files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["winning_side", "control_margin_full"]).copy()
    combined["winning_side"] = combined["winning_side"].str.upper()
    return combined[combined["winning_side"].isin(["CT", "T"])].copy()


def load_summary() -> dict:
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found — run metric3_control_margin.py first"
        )
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def active_data_available(df: pd.DataFrame) -> bool:
    return "control_margin_active" in df.columns and df["control_margin_active"].notna().any()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _win_loss_arrays(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    win_vals = df.loc[df["winning_side"] == "CT", column].dropna().values
    loss_vals = df.loc[df["winning_side"] == "T", column].dropna().values
    return win_vals, loss_vals


def _mean_and_ci(vals: np.ndarray) -> tuple[float, float]:
    """Normal-approximation 95% CI (mean +/- 1.96 * SEM) — standard
    practice for the round counts (tens-to-hundreds) seen in this dataset.
    """
    if len(vals) < 2:
        return (float(vals.mean()) if len(vals) else 0.0, 0.0)
    mean = vals.mean()
    ci = 1.96 * stats.sem(vals)
    return float(mean), float(ci)


def _annotate_unavailable(ax, reason: str) -> None:
    """Used in place of a real panel when the active-footprint data isn't
    there yet, so the figure still renders (rather than the script
    skipping the whole plot) and clearly explains why the panel is empty.
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
    "See metric3_control_margin.py docstring."
)


# ---------------------------------------------------------------------------
# Figure 1 — win/loss boxplot (primary RQ3 chart)
# ---------------------------------------------------------------------------
def _boxplot_panel(ax, df: pd.DataFrame, column: str, mu_win, mu_loss, title: str) -> None:
    win_vals, loss_vals = _win_loss_arrays(df, column)

    bp = ax.boxplot(
        [win_vals, loss_vals], labels=["CT win", "CT loss"],
        patch_artist=True, showfliers=False, widths=0.5,
    )
    for patch, color in zip(bp["boxes"], [OUTCOME_COLORS["CT"], OUTCOME_COLORS["T"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    rng = np.random.default_rng(0)
    for i, vals in enumerate([win_vals, loss_vals], start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.35, s=14,
                   color="black", edgecolor="none", zorder=3)

    if mu_win is not None:
        ax.axhline(mu_win, color=OUTCOME_COLORS["CT"], linestyle="--", linewidth=1.5,
                    label=f"μ_win = {mu_win:.4f}")
    if mu_loss is not None:
        ax.axhline(mu_loss, color=OUTCOME_COLORS["T"], linestyle="--", linewidth=1.5,
                    label=f"μ_loss = {mu_loss:.4f}")

    ax.axhline(NEUTRAL_VALUE, color="black", linestyle=":", linewidth=1, alpha=0.5, label="Neutral (0.5)")
    ax.set_ylabel("Control Margin")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_win_loss_boxplot(df: pd.DataFrame, summary: dict, out_path: str) -> None:
    """Box plot of control margin split by winning_side — the primary RQ3
    chart. Marks mu_win / mu_loss explicitly as horizontal reference lines
    rather than relying on the box median alone, since the summary
    statistic reported in the text (the mean) and the box plot's median
    can differ for a skewed distribution.
    """
    has_active = active_data_available(df)
    fig, axes = plt.subplots(1, 2 if has_active else 1, figsize=(12 if has_active else 6, 6))
    axes = np.atleast_1d(axes)

    full_stats = summary.get("full_map", summary)  # fall back to flat schema if ever re-run against old json
    _boxplot_panel(
        axes[0], df, "control_margin_full",
        full_stats.get("mu_win"), full_stats.get("mu_loss"),
        "Full Map (all 903 nodes)",
    )

    if has_active:
        active_stats = summary.get("active_footprint", {})
        _boxplot_panel(
            axes[1], df, "control_margin_active",
            active_stats.get("mu_win"), active_stats.get("mu_loss"),
            "Active Footprint only",
        )
    elif len(axes) > 1:
        _annotate_unavailable(axes[1], ACTIVE_UNAVAILABLE_MSG)

    fig.suptitle("Control Margin by Round Outcome (CT perspective)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — win/loss histograms
# ---------------------------------------------------------------------------
def _histogram_panel(ax, df: pd.DataFrame, column: str, title: str) -> None:
    win_vals, loss_vals = _win_loss_arrays(df, column)

    bins = np.linspace(
        min(win_vals.min(initial=0), loss_vals.min(initial=0)),
        max(win_vals.max(initial=1), loss_vals.max(initial=1)),
        30,
    )
    ax.hist(win_vals, bins=bins, alpha=0.55, color=OUTCOME_COLORS["CT"], label="CT win", edgecolor="white")
    ax.hist(loss_vals, bins=bins, alpha=0.55, color=OUTCOME_COLORS["T"], label="CT loss", edgecolor="white")
    ax.axvline(NEUTRAL_VALUE, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Control Margin")
    ax.set_ylabel("Number of rounds")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


def plot_win_loss_histograms(df: pd.DataFrame, out_path: str) -> None:
    """Overlaid histograms of control margin for CT-win vs CT-loss rounds.

    Shows distribution shape and overlap directly, which is useful
    alongside Cohen's d — a large d with heavily overlapping histograms
    reads very differently to a reader than a large d with clean
    separation, even though the summary statistic is the same. Comparing
    the full-map panel's overlap against the active-footprint panel's
    overlap is exactly the visual evidence for the dilution story.
    """
    has_active = active_data_available(df)
    fig, axes = plt.subplots(1, 2 if has_active else 1, figsize=(14 if has_active else 8, 5))
    axes = np.atleast_1d(axes)

    _histogram_panel(axes[0], df, "control_margin_full", "Full Map")

    if has_active:
        _histogram_panel(axes[1], df, "control_margin_active", "Active Footprint")
    elif len(axes) > 1:
        _annotate_unavailable(axes[1], ACTIVE_UNAVAILABLE_MSG)

    fig.suptitle("Control Margin Distribution by Round Outcome", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — mean bar chart with 95% CI
# ---------------------------------------------------------------------------
def _bar_panel(ax, df: pd.DataFrame, column: str, title: str) -> None:
    win_vals, loss_vals = _win_loss_arrays(df, column)
    win_mean, win_ci = _mean_and_ci(win_vals)
    loss_mean, loss_ci = _mean_and_ci(loss_vals)

    labels = [f"CT win\n(n={len(win_vals)})", f"CT loss\n(n={len(loss_vals)})"]
    means = [win_mean, loss_mean]
    cis = [win_ci, loss_ci]
    colors = [OUTCOME_COLORS["CT"], OUTCOME_COLORS["T"]]

    bars = ax.bar(labels, means, yerr=cis, capsize=6, color=colors, alpha=0.75, width=0.5)
    max_ci = max(cis) if max(cis) > 0 else 0.01
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_ci * 0.15 + 0.005,
                 f"{mean:.4f}", ha="center", va="bottom", fontsize=11)

    ax.axhline(NEUTRAL_VALUE, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_ylabel("Mean Control Margin")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def plot_mean_bar_with_ci(df: pd.DataFrame, out_path: str) -> None:
    """Bar chart of mu_win vs mu_loss with 95% CI error bars — the
    "headline" summary chart. CIs are computed directly from the loaded
    per-round data (rather than trusting only the scalar means in
    _summary.json), same approach as the original single-panel version.
    """
    has_active = active_data_available(df)
    fig, axes = plt.subplots(1, 2 if has_active else 1, figsize=(10 if has_active else 5, 5))
    axes = np.atleast_1d(axes)

    _bar_panel(axes[0], df, "control_margin_full", "Full Map")

    if has_active:
        _bar_panel(axes[1], df, "control_margin_active", "Active Footprint")
    elif len(axes) > 1:
        _annotate_unavailable(axes[1], ACTIVE_UNAVAILABLE_MSG)

    fig.suptitle("Mean Control Margin by Outcome (95% CI)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — margin vs round number (QA/trend check)
# ---------------------------------------------------------------------------
def _scatter_panel(ax, df: pd.DataFrame, column: str, title: str) -> None:
    for outcome, color in OUTCOME_COLORS.items():
        sub = df[df["winning_side"] == outcome]
        label = "CT win" if outcome == "CT" else "CT loss"
        ax.scatter(sub["round"], sub[column], label=label, color=color, alpha=0.5, s=18)

    ax.axhline(NEUTRAL_VALUE, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Round number")
    ax.set_ylabel("Control Margin")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


def plot_margin_vs_round(df: pd.DataFrame, out_path: str) -> None:
    """Scatter of control margin vs round number, colored by outcome.

    Primarily a data-quality / trend check: does the margin trend
    systematically with round number within a match (e.g. drifting due to
    economy state or side switch at the halfway point)? A roughly flat
    scatter with the two outcome colors interspersed is the expected,
    healthy result; a strong trend would be worth investigating before
    trusting the pooled win/loss comparison. Checking both panels matters
    because the active-footprint version could in principle show a trend
    the full-map version masks (or vice versa), since it's driven by a
    different, more concentrated subset of nodes.
    """
    has_active = active_data_available(df)
    fig, axes = plt.subplots(1, 2 if has_active else 1, figsize=(16 if has_active else 9, 5))
    axes = np.atleast_1d(axes)

    _scatter_panel(axes[0], df, "control_margin_full", "Full Map")

    if has_active:
        _scatter_panel(axes[1], df, "control_margin_active", "Active Footprint")
    elif len(axes) > 1:
        _annotate_unavailable(axes[1], ACTIVE_UNAVAILABLE_MSG)

    fig.suptitle("Control Margin vs Round Number (data-quality check)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 (new) — active_node_fraction distribution
# ---------------------------------------------------------------------------
def plot_active_node_fraction(df: pd.DataFrame, out_path: str) -> None:
    """Distribution of active_node_fraction per round — what fraction of
    the 903 nodes actually carried non-negligible influence, on average,
    in a given round. This is new information the active-footprint
    extension makes available and is worth a figure on its own: it
    quantifies exactly how much dead weight the full-map version was
    averaging in (e.g. a mean active fraction of 0.15 means the original
    Metric 3 was diluting real signal with ~85% structurally-neutral map
    background every frame). It also doubles as a sanity check on
    EPSILON's calibration — a fraction near 0 (EPSILON too strict) or
    near 1 (EPSILON too permissive, defeating the point of the filter)
    would be a signal to revisit the constant.
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
    ax.set_xlabel("Active node fraction per round  (|V_active| / |V|)")
    ax.set_ylabel("Number of rounds")
    ax.set_title("How Much of the Map is 'In Play'? (active-footprint EPSILON diagnostic)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    summary = load_summary()

    print("Loading per-match round-level control margin data...")
    df = load_all_rounds()
    print(f"  Loaded {len(df):,} rounds "
          f"({(df['winning_side'] == 'CT').sum()} CT win, {(df['winning_side'] == 'T').sum()} CT loss)")

    has_active = active_data_available(df)
    print(f"  Active-footprint data available: {has_active}\n")

    if df.empty:
        print("No rounds with a valid outcome available — nothing to plot")
        return

    print("Plotting 01_win_loss_boxplot.png")
    plot_win_loss_boxplot(df, summary, os.path.join(FIGURES_DIR, "01_win_loss_boxplot.png"))

    print("Plotting 02_win_loss_histograms.png")
    plot_win_loss_histograms(df, os.path.join(FIGURES_DIR, "02_win_loss_histograms.png"))

    print("Plotting 03_mean_bar_with_ci.png")
    plot_mean_bar_with_ci(df, os.path.join(FIGURES_DIR, "03_mean_bar_with_ci.png"))

    print("Plotting 04_margin_vs_round_scatter.png")
    plot_margin_vs_round(df, os.path.join(FIGURES_DIR, "04_margin_vs_round_scatter.png"))

    print("Plotting 05_active_node_fraction.png")
    plot_active_node_fraction(df, os.path.join(FIGURES_DIR, "05_active_node_fraction.png"))

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
