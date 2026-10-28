"""
Metric 3 (T-side) — Control Margin Visualization

Companion to metric3_control_margin_t_side.py. Reads its outputs
(outputs/metrics/control_margin_t_side/*.csv, _summary.json,
_area_breakdown.csv) and produces figures under
outputs/metrics/control_margin_t_side/figures/.

Panels are built dynamically: the "Full Map" panel is always present; the
"Active Footprint" and "Bombsite (active-filtered)" panels are included
only if their columns actually have data (i.e. the upstream control_fields
CSVs carried ct_mu/t_mu — see metric3_control_margin_t_side.py's
docstring for the degrade-gracefully policy this mirrors).

Throughout, "T win" means winning_side == "T" and "T loss" means
winning_side == "CT" (T lost that round) — the opposite grouping
direction from the CT-side visualization script.

Figures produced:
    01_win_loss_boxplot.png        — boxplot, T win vs T loss, per panel
    02_win_loss_histograms.png     — overlaid histograms, per panel
    03_mean_bar_with_ci.png        — mean +/- 95% CI bar, per panel
    04_margin_vs_round_scatter.png — margin vs round number (QA), per panel
    05_active_node_fraction.png    — whole-map vs bombsite-only active
                                      fraction distributions
    06_effect_size_comparison.png  — Cohen's d across ALL margin
                                      definitions in one bar chart — the
                                      direct hypothesis-comparison figure
    07_area_breakdown.png          — top named areas by T-win-minus-loss
                                      control differential, bombsites
                                      highlighted
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --/ Path consts — must match metric3_control_margin_t_side.py
METRIC_DIR = "outputs/metrics/control_margin_t_side"
SUMMARY_PATH = os.path.join(METRIC_DIR, "_summary.json")
AREA_BREAKDOWN_PATH = os.path.join(METRIC_DIR, "_area_breakdown.csv")
FIGURES_DIR = os.path.join(METRIC_DIR, "figures")

NEUTRAL_VALUE = 0.5
# CT is always blue, T is always red across this project's figures,
# regardless of which side's script is being plotted, so CT-side and
# T-side figures stay visually consistent with each other.
OUTCOME_COLORS = {"CT": "#1f77b4", "T": "#d62728"}

COLUMN_TO_SUMMARY_KEY = {
    "control_margin_full_t": "full_map",
    "control_margin_active_t": "active_footprint",
    "control_margin_bombsite_active_t": "bombsite_active",
}
COLUMN_TO_TITLE = {
    "control_margin_full_t": "Full Map",
    "control_margin_active_t": "Active Footprint",
    "control_margin_bombsite_active_t": "Bombsite (active-filtered)",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_all_rounds() -> pd.DataFrame:
    """Load every per-match round-level CSV, skipping the underscore-
    prefixed dataset-level files (_summary.json is not a csv, but
    _area_breakdown.csv lives in this same directory and must NOT be
    swept into this glob).
    """
    per_match_files = sorted(
        p
        for p in glob.glob(os.path.join(METRIC_DIR, "*.csv"))
        if not os.path.basename(p).startswith("_")
    )
    if not per_match_files:
        raise FileNotFoundError(
            f"No per-match round CSVs found in {METRIC_DIR} — "
            f"run metric3_control_margin_t_side.py first"
        )
    df = pd.concat((pd.read_csv(p) for p in per_match_files), ignore_index=True)
    df = df.dropna(subset=["winning_side", "control_margin_full_t"]).copy()
    df["winning_side"] = df["winning_side"].str.upper()
    df = df[df["winning_side"].isin(["CT", "T"])]
    return df


def load_summary() -> dict:
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found — run metric3_control_margin_t_side.py first"
        )
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_area_breakdown() -> pd.DataFrame | None:
    if not os.path.exists(AREA_BREAKDOWN_PATH):
        print(f"  NOTE: {AREA_BREAKDOWN_PATH} not found — skipping area-breakdown figure")
        return None
    return pd.read_csv(AREA_BREAKDOWN_PATH)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _panel_specs(df: pd.DataFrame) -> list:
    """Which panels to draw, in order, based on what actually has data."""
    specs = [("control_margin_full_t", COLUMN_TO_TITLE["control_margin_full_t"])]
    if "control_margin_active_t" in df.columns and df["control_margin_active_t"].notna().any():
        specs.append(("control_margin_active_t", COLUMN_TO_TITLE["control_margin_active_t"]))
    if (
        "control_margin_bombsite_active_t" in df.columns
        and df["control_margin_bombsite_active_t"].notna().any()
    ):
        specs.append(
            ("control_margin_bombsite_active_t", COLUMN_TO_TITLE["control_margin_bombsite_active_t"])
        )
    return specs


def _win_loss_arrays(df: pd.DataFrame, column: str) -> tuple:
    win_vals = df.loc[df["winning_side"] == "T", column].dropna().to_numpy()
    loss_vals = df.loc[df["winning_side"] == "CT", column].dropna().to_numpy()
    return win_vals, loss_vals


def _mean_and_ci(values: np.ndarray, confidence: float = 0.95) -> tuple:
    """Mean and half-width of a normal-approximation CI (SEM * z), 0 if
    fewer than 2 observations.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float(values.mean()) if len(values) else float("nan")), 0.0
    mean = float(values.mean())
    sem = stats.sem(values)
    z = stats.norm.ppf(0.5 + confidence / 2)
    return mean, float(sem * z)


# ---------------------------------------------------------------------------
# Figure 1 — boxplot
# ---------------------------------------------------------------------------
def _boxplot_panel(ax, df: pd.DataFrame, column: str, mu_win, mu_loss, title: str) -> None:
    win_vals, loss_vals = _win_loss_arrays(df, column)
    bp = ax.boxplot(
        [win_vals, loss_vals],
        tick_labels=["T win", "T loss"],
        patch_artist=True,
        showfliers=False,
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], [OUTCOME_COLORS["T"], OUTCOME_COLORS["CT"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    rng = np.random.default_rng(0)
    for i, vals in enumerate([win_vals, loss_vals], start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals, alpha=0.35, s=14,
            color="black", edgecolor="none", zorder=3,
        )

    if mu_win is not None:
        ax.axhline(mu_win, color=OUTCOME_COLORS["T"], linestyle="--", linewidth=1.5,
                    label=f"\u03bc_win = {mu_win:.4f}")
    if mu_loss is not None:
        ax.axhline(mu_loss, color=OUTCOME_COLORS["CT"], linestyle="--", linewidth=1.5,
                    label=f"\u03bc_loss = {mu_loss:.4f}")
    ax.axhline(NEUTRAL_VALUE, color="black", linestyle=":", linewidth=1, alpha=0.5, label="Neutral (0.5)")

    ax.set_ylabel("T Control Margin")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_win_loss_boxplot(df: pd.DataFrame, summary: dict, out_path: str) -> None:
    specs = _panel_specs(df)
    fig, axes = plt.subplots(1, len(specs), figsize=(6 * len(specs), 6))
    axes = np.atleast_1d(axes)
    for ax, (column, title) in zip(axes, specs):
        block = summary.get(COLUMN_TO_SUMMARY_KEY.get(column, ""), {})
        _boxplot_panel(ax, df, column, block.get("mu_win"), block.get("mu_loss"), title)
    fig.suptitle("T Control Margin by Round Outcome", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — histograms
# ---------------------------------------------------------------------------
def _histogram_panel(ax, df: pd.DataFrame, column: str, title: str) -> None:
    win_vals, loss_vals = _win_loss_arrays(df, column)
    lo = min(win_vals.min(initial=0.0), loss_vals.min(initial=0.0))
    hi = max(win_vals.max(initial=1.0), loss_vals.max(initial=1.0))
    bins = np.linspace(lo, hi, 30)

    ax.hist(win_vals, bins=bins, alpha=0.55, color=OUTCOME_COLORS["T"], label="T win", edgecolor="white")
    ax.hist(loss_vals, bins=bins, alpha=0.55, color=OUTCOME_COLORS["CT"], label="T loss", edgecolor="white")
    ax.axvline(NEUTRAL_VALUE, color="black", linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xlabel("T Control Margin")
    ax.set_ylabel("Number of rounds")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


def plot_win_loss_histograms(df: pd.DataFrame, out_path: str) -> None:
    specs = _panel_specs(df)
    fig, axes = plt.subplots(1, len(specs), figsize=(5.5 * len(specs), 5))
    axes = np.atleast_1d(axes)
    for ax, (column, title) in zip(axes, specs):
        _histogram_panel(ax, df, column, title)
    fig.suptitle("T Control Margin Distribution by Round Outcome", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — mean bar with CI
# ---------------------------------------------------------------------------
def _bar_panel(ax, df: pd.DataFrame, column: str, title: str) -> None:
    win_vals, loss_vals = _win_loss_arrays(df, column)
    win_mean, win_ci = _mean_and_ci(win_vals)
    loss_mean, loss_ci = _mean_and_ci(loss_vals)

    labels = [f"T win\n(n={len(win_vals)})", f"T loss\n(n={len(loss_vals)})"]
    means = [win_mean, loss_mean]
    cis = [win_ci, loss_ci]
    colors = [OUTCOME_COLORS["T"], OUTCOME_COLORS["CT"]]

    bars = ax.bar(labels, means, yerr=cis, capsize=6, color=colors, alpha=0.75, width=0.5)
    max_ci = max(cis) if max(cis) > 0 else 0.01
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max_ci * 0.15 + 0.005,
            f"{mean:.4f}", ha="center", va="bottom", fontsize=11,
        )
    ax.axhline(NEUTRAL_VALUE, color="black", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_ylabel("Mean T Control Margin")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def plot_mean_bar_with_ci(df: pd.DataFrame, out_path: str) -> None:
    specs = _panel_specs(df)
    fig, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 5))
    axes = np.atleast_1d(axes)
    for ax, (column, title) in zip(axes, specs):
        _bar_panel(ax, df, column, title)
    fig.suptitle("Mean T Control Margin by Outcome (95% CI)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — margin vs round (QA scatter)
# ---------------------------------------------------------------------------
def _scatter_panel(ax, df: pd.DataFrame, column: str, title: str) -> None:
    for outcome, color in OUTCOME_COLORS.items():
        sub = df[df["winning_side"] == outcome]
        label = "T win" if outcome == "T" else "T loss"
        ax.scatter(sub["round"], sub[column], label=label, color=color, alpha=0.5, s=18)
    ax.axhline(NEUTRAL_VALUE, color="black", linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xlabel("Round number")
    ax.set_ylabel("T Control Margin")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


def plot_margin_vs_round(df: pd.DataFrame, out_path: str) -> None:
    specs = _panel_specs(df)
    fig, axes = plt.subplots(1, len(specs), figsize=(6 * len(specs), 5))
    axes = np.atleast_1d(axes)
    for ax, (column, title) in zip(axes, specs):
        _scatter_panel(ax, df, column, title)
    fig.suptitle("T Control Margin vs Round Number (data-quality check)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — active node fraction (whole map vs bombsite-only)
# ---------------------------------------------------------------------------
def plot_active_node_fraction(df: pd.DataFrame, out_path: str) -> None:
    has_map = "active_node_fraction" in df.columns and df["active_node_fraction"].notna().any()
    has_site = (
        "bombsite_active_node_fraction" in df.columns
        and df["bombsite_active_node_fraction"].notna().any()
    )

    if not has_map and not has_site:
        print("  Skipping 05_active_node_fraction.png — no active-fraction data available")
        return

    n_panels = int(has_map) + int(has_site)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    axes = np.atleast_1d(axes)
    idx = 0

    if has_map:
        vals = df["active_node_fraction"].dropna().to_numpy()
        ax = axes[idx]
        idx += 1
        ax.hist(vals, bins=30, color="#2ca02c", alpha=0.7, edgecolor="white")
        mean_val = float(vals.mean())
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.5, label=f"mean = {mean_val:.3f}")
        ax.set_xlabel("Active node fraction per round (whole map)")
        ax.set_ylabel("Number of rounds")
        ax.set_title("Whole-Map Active Footprint")
        ax.legend()
        ax.grid(alpha=0.3)

    if has_site:
        vals = df["bombsite_active_node_fraction"].dropna().to_numpy()
        ax = axes[idx]
        idx += 1
        ax.hist(vals, bins=30, color="#ff7f0e", alpha=0.7, edgecolor="white")
        mean_val = float(vals.mean())
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.5, label=f"mean = {mean_val:.3f}")
        ax.set_xlabel("Active node fraction per round (BombsiteA+B nodes only)")
        ax.set_ylabel("Number of rounds")
        ax.set_title("Bombsite-Only Active Footprint")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("How Much of the Map (or Bombsite) Is 'In Play'?", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — effect size comparison (the direct hypothesis figure)
# ---------------------------------------------------------------------------
def plot_effect_size_comparison(summary: dict, out_path: str) -> None:
    specs = [
        ("full_map", "Full Map"),
        ("active_footprint", "Active\nFootprint"),
        ("bombsite_all", "Bombsite\n(unfiltered)"),
        ("bombsite_active", "Bombsite\n(active)"),
        ("bombsiteA_active", "Site A\n(active)"),
        ("bombsiteB_active", "Site B\n(active)"),
    ]

    labels, values, is_bombsite = [], [], []
    for key, label in specs:
        d = summary.get(key, {}).get("cohens_d")
        if d is not None:
            labels.append(label)
            values.append(d)
            is_bombsite.append(key != "full_map" and key != "active_footprint")

    if not values:
        print("  Skipping 06_effect_size_comparison.png — no Cohen's d values available")
        return

    colors = ["#d4af37" if b else "#4c72b0" for b in is_bombsite]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_ylabel("Cohen's d (T win vs T loss)")
    ax.set_title(
        "Effect Size by Margin Definition\n"
        "(blue = whole-map metrics, gold = bombsite-restricted metrics)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 — area-name breakdown
# ---------------------------------------------------------------------------
def plot_area_breakdown(area_df: pd.DataFrame, out_path: str, top_n: int = 20) -> None:
    df = area_df.dropna(subset=["diff_win_minus_loss"]).copy()
    if df.empty:
        print("  Skipping 07_area_breakdown.png — no area breakdown rows available")
        return

    df = df.sort_values("diff_win_minus_loss", ascending=False).reset_index(drop=True)
    top = df.head(top_n)

    # Bombsite rows are always shown, even if they fall outside the top_n
    bombsite_rows = df[df["is_bombsite"]]
    missing_bombsite = bombsite_rows[~bombsite_rows["area_name"].isin(top["area_name"])]
    plotted = pd.concat([top, missing_bombsite], ignore_index=True)
    plotted = plotted.sort_values("diff_win_minus_loss", ascending=False).reset_index(drop=True)

    colors = ["#d4af37" if is_b else "#4c72b0" for is_b in plotted["is_bombsite"]]

    # Areas come from data/de_mirage_area_assignment.csv (blank awpy names
    # reassigned). Label each area with [formerly-blank nodes / total nodes]
    # where any were reassigned, so a heavily reassigned area is visible.
    has_assignment_cols = {"n_formerly_blank_nodes", "n_nodes_in_area"}.issubset(plotted.columns)
    if has_assignment_cols:
        display_names = [
            f"{name} [{int(nb)}/{int(nn)}]" if nb > 0 else name
            for name, nb, nn in zip(
                plotted["area_name"], plotted["n_formerly_blank_nodes"], plotted["n_nodes_in_area"]
            )
        ]
    else:
        print(
            "  NOTE: _area_breakdown.csv has no n_formerly_blank_nodes column — it was "
            "probably produced before the area assignment was applied; rerun "
            "metric3_control_margin_t_side.py"
        )
        display_names = plotted["area_name"].replace("", "<unnamed>")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plotted))))
    y_pos = np.arange(len(plotted))
    ax.barh(y_pos, plotted["diff_win_minus_loss"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_xlabel("Mean T control, T-win rounds minus T-loss rounds")
    ax.set_title(
        f"Where T's Control Differs Most Between Wins and Losses\n"
        f"(gold = BombsiteA / BombsiteB; top {top_n} shown, "
        f"plus any bombsite row outside the top {top_n};\n"
        f"[k/n] = formerly-blank nodes reassigned / total nodes in area)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    summary = load_summary()

    print("Loading per-match round-level T-side control margin data...")
    df = load_all_rounds()
    n_t_win = int((df["winning_side"] == "T").sum())
    n_t_loss = int((df["winning_side"] == "CT").sum())
    print(f"  Loaded {len(df):,} rounds ({n_t_win} T win, {n_t_loss} T loss)")

    specs = _panel_specs(df)
    print(f"  Panels available: {[title for _, title in specs]}\n")

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

    print("Plotting 06_effect_size_comparison.png")
    plot_effect_size_comparison(summary, os.path.join(FIGURES_DIR, "06_effect_size_comparison.png"))

    print("Plotting 07_area_breakdown.png")
    area_df = load_area_breakdown()
    if area_df is not None:
        plot_area_breakdown(area_df, os.path.join(FIGURES_DIR, "07_area_breakdown.png"))

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
