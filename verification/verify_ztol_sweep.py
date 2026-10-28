import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from awpy.data import NAV, NAV_GRAPHS
from awpy.visualization.plot import position_transform, plot_map

MAP_NAME = "de_mirage"
FCA_CSV = "outputs/player_df_all_rounds_fca_baseline.csv"
SWEEP_DIR = "outputs/ztol_sweep"
PLOTS_DIR = "outputs/ztol_sweep/plots"

MAX_PLOTS_VS_FCA = 30
MAX_PLOTS_PAIRS = 30

DPI = 150

def area_name(area_id: int) -> str:
    name = NAV[MAP_NAME].get(area_id, {}).get("areaName", "")
    return name if name else "(unnamed)"

def draw_tile(ax, area_id: int, face_color: str, edge_color: str, alpha: float = 0.4, linewidth: float = 1.5, 
              zorder: int = 3, label: str | None = None) -> patches.Rectangle:
    
    area = NAV[MAP_NAME][area_id]
    nw_x   = position_transform(MAP_NAME, area["northWestX"], "x")
    nw_y   = position_transform(MAP_NAME, area["northWestY"], "y")
    se_x   = position_transform(MAP_NAME, area["southEastX"], "x")
    se_y   = position_transform(MAP_NAME, area["southEastY"], "y")

    rect = patches.Rectangle(
        (min(nw_x, se_x), min(nw_y, se_y)),
        abs(se_x - nw_x), abs(se_y - nw_y),
        linewidth=linewidth,
        edgecolor=edge_color,
        facecolor=face_color,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )
    
    ax.add_patch(rect)
    return rect

def draw_centroid_link(ax, area_id: int, player_px: float, player_py: float, color: str, zorder: int = 4):
    
    area = NAV[MAP_NAME][area_id]
    cx_raw = (area["northWestX"] + area["southEastX"]) / 2
    cy_raw = (area["northWestY"] + area["southEastY"]) / 2
    cx_px = position_transform(MAP_NAME, cx_raw, "x")
    cy_px = position_transform(MAP_NAME, cy_raw, "y")

    ax.scatter(cx_px, cy_px, c=color, s=12, marker="x", linewidths=0.9, zorder=zorder + 2)
    ax.plot([cx_px, player_px], [cy_px, player_py], color=color, linewidth=0.9, linestyle="--", alpha=0.65, zorder=zorder)

def save_comparison_plot(
    row: pd.Series,
    area_id_a: int,
    area_id_b: int,
    dist_a: float,
    dist_b: float,
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
    title_line1: str,
    title_line2: str,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
 
    fig, ax = plot_map(MAP_NAME, map_type="simpleradar", dark=True)
 
    # Draw tile A (e.g. FCA or lower z_tol)
    rect_a = draw_tile(
        ax, area_id_a,
        face_color=color_a, edge_color=color_a,
        alpha=0.38, linewidth=2.0, zorder=3,
    )
    # Draw tile B (e.g. KDTree or higher z_tol)
    rect_b = draw_tile(
        ax, area_id_b,
        face_color=color_b, edge_color=color_b,
        alpha=0.38, linewidth=2.0, zorder=3,
    )
 
    # Player position dot
    px = position_transform(MAP_NAME, row["x"], "x")
    py = position_transform(MAP_NAME, row["y"], "y")
    ax.scatter(px, py, c="white", s=8, zorder=8, edgecolors="black", linewidths=0.7)
 
    # Centroid markers + dashed lines
    # draw_centroid_link(ax, area_id_a, px, py, color=color_a, zorder=4)
    # draw_centroid_link(ax, area_id_b, px, py, color=color_b, zorder=4)
 
    # Legend
    legend_handles = [
        patches.Patch(facecolor=color_a, alpha=0.7, edgecolor="gray",
                      label=f"{label_a}: Area {area_id_a} '{area_name(area_id_a)}' "
                            f"(dist={dist_a:.1f})"),
        patches.Patch(facecolor=color_b, alpha=0.7, edgecolor="gray",
                      label=f"{label_b}: Area {area_id_b} '{area_name(area_id_b)}' "
                            f"(dist={dist_b:.1f})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=7, label="Player position"),
        Line2D([0], [0], marker="x", color=color_a, markersize=6,
               linestyle="none", label=f"{label_a} centroid"),
        Line2D([0], [0], marker="x", color=color_b, markersize=6,
               linestyle="none", label=f"{label_b} centroid"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=5.0,
        framealpha=0.80,
        facecolor="#111111",
        labelcolor="white",
        edgecolor="#555555",
        handlelength=1.4,
    )
 
    # Title (two lines)
    ax.set_title(
        f"{title_line1}\n{title_line2}",
        color="white", fontsize=6.0, pad=3, linespacing=1.4,
    )
    ax.axis("off")
 
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="black")
    plt.close(fig)

def sort_by_diagnostic_priority(diff: pd.DataFrame,
                                 area_col_a: str, area_col_b: str,
                                 dist_col_a: str, dist_col_b: str) -> pd.DataFrame:
    """
    Sort disagreeing rows to surface the most informative cases first:
      1. Rows where the named place differs (tile names disagree — more severe)
      2. Among those, rows with the largest correction (abs distance difference)
      3. Then remaining rows by distance correction magnitude
    This means plots are generated for the cases where z_tol made a meaningful
    geographic difference, not just an adjacent-tile swap within the same named area.
    """
    diff = diff.copy()
    diff["_name_a"]   = diff[area_col_a].apply(area_name)
    diff["_name_b"]   = diff[area_col_b].apply(area_name)
    diff["_name_diff"]= diff["_name_a"] != diff["_name_b"]
    diff["_dist_diff"]= (diff[dist_col_a] - diff[dist_col_b]).abs()
    return (
        diff
        .sort_values(["_name_diff", "_dist_diff"], ascending=[False, False])
        .drop(columns=["_name_a", "_name_b", "_name_diff", "_dist_diff"])
    )

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
 
print("=" * 70)
print("  LOADING DATA")
print("=" * 70)
 
# FCA baseline (round 1 only)
fca_df = pd.read_csv(FCA_CSV)
if "Unnamed: 0" in fca_df.columns:
    fca_df = fca_df.drop(columns=["Unnamed: 0"])
fca_df["current_areaId"] = fca_df["current_areaId"].astype(int)
print(f"\n  FCA baseline:  {FCA_CSV}  ({len(fca_df)} rows, all rounds)")
 
# Sweep CSVs — load all files matching the naming convention
sweep_files  = sorted(glob.glob(os.path.join(SWEEP_DIR, "player_df_ztol_*.csv")))
sweep_frames = {} # dict: z_tol (float) -> DataFrame (round 1 only, for FCA comparison)
 
for filepath in sweep_files:
    fname  = os.path.basename(filepath)
    # Extract z_tol from filename: "player_df_ztol_100.csv" -> 100.0
    try:
        z_tol = float(fname.replace("player_df_ztol_", "").replace(".csv", ""))
    except ValueError:
        print(f"  WARNING: could not parse z_tol from filename '{fname}' — skipping")
        continue
 
    df = pd.read_csv(filepath)
    df["current_areaId"] = df["current_areaId"].astype(int)
 
    # Validate embedded z_tol column matches filename
    if "z_tol" in df.columns:
        embedded = df["z_tol"].iloc[0]
        if embedded != z_tol:
            print(f"  WARNING: {fname} — filename z_tol={z_tol} does not match "
                  f"embedded z_tol={embedded}. Using filename value.")
 
    sweep_frames[z_tol] = df
    print(f"  z_tol={z_tol:>6.1f}:     {filepath}  ({len(df)} rows)")
 
if not sweep_frames:
    print(f"\n  ERROR: no sweep CSVs found in {SWEEP_DIR}. "
          f"Run ztol_sweep.py first.")
    raise SystemExit(1)
 
z_tol_values = sorted(sweep_frames.keys())
print(f"\n  Loaded {len(z_tol_values)} z_tol values: {z_tol_values}")
 
# ---------------------------------------------------------------------------
# JOIN EACH Z_TOL (round 1) WITH FCA BASELINE
# ---------------------------------------------------------------------------
 
# merged_vs_fca[z_tol]: the inner join on (player_name, frame_idx) for round 1
merged_vs_fca = {}
 
for z_tol in z_tol_values:
    merged = fca_df.merge(
        sweep_frames[z_tol][["player_name", "round", "frame_idx",
                              "current_areaId", "mapping_distance"]],
        on=["player_name", "round", "frame_idx"],
        how="inner",
        suffixes=("_fca", "_kdt"),
    )
    merged_vs_fca[z_tol] = merged
 
    if len(merged) != len(fca_df):
        print(f"\n  WARNING z_tol={z_tol}: joined {len(merged)} rows "
          f"but FCA baseline has {len(fca_df)}. "
          f"Check for player name mismatches or missing frames.")
 
# ---------------------------------------------------------------------------
# SECTION 1: SUMMARY TABLE — each z_tol vs FCA
# ---------------------------------------------------------------------------
 
print(f"\n{'='*70}")
print("  SECTION 1 — Agreement vs FCA Baseline (All Rounds)")
print(f"{'='*70}")
print(f"\n  {'z_tol':>7}  {'Total rows':>11}  {'Agree w/ FCA':>13}  "
      f"{'Disagree':>10}  {'Disagree %':>11}")
print(f"  {'-'*7}  {'-'*11}  {'-'*13}  {'-'*10}  {'-'*11}")
 
agree_counts    = {}
disagree_counts = {}
 
for z_tol in z_tol_values:
    merged   = merged_vs_fca[z_tol]
    agree    = (merged["current_areaId_fca"] == merged["current_areaId_kdt"]).sum()
    disagree = len(merged) - agree
    agree_counts[z_tol]    = agree
    disagree_counts[z_tol] = disagree
    pct = 100 * disagree / len(merged) if len(merged) > 0 else 0.0
    print(f"  {z_tol:>7.1f}  {len(merged):>11}  {agree:>13}  {disagree:>10}  {pct:>10.1f}%")
 
print()
print("  Interpretation:")
print("  • Higher z_tol → fewer corrections vs FCA (more permissive z-filter,")
print("    closer to FCA's no-z-awareness behaviour).")
print("  • Lower z_tol → more corrections (more aggressive, may overcorrect).")
print("  • z_tol at which disagree % drops to ~0 is where the filter stops doing")
print("    anything — that is the effective upper bound for calibration.")
 
# ---------------------------------------------------------------------------
# SECTION 2: PAIRWISE AGREEMENT MATRIX (round 1, all z_tol pairs)
# ---------------------------------------------------------------------------
 
print(f"\n{'='*70}")
print("  SECTION 2 — Pairwise Agreement Matrix (Round 1)")
print(f"{'='*70}")
print("\n  Each cell shows the number of round-1 rows where the two z_tol values")
print("  produced the SAME area_id assignment. Diagonal = total row count.")
print()
 
# Build a square agreement matrix
n = len(z_tol_values)
agreement_matrix = np.zeros((n, n), dtype=int)
 
# Precompute round-1 area_id series per z_tol, aligned by (player_name, frame_idx)
# Use the FCA-merged DataFrames so all are aligned on the same join key
kdt_area_ids = {}
for z_tol in z_tol_values:
    kdt_area_ids[z_tol] = merged_vs_fca[z_tol]["current_areaId_kdt"]
 
# For off-diagonal cells, merge each pair on (player_name, frame_idx)
for i, zt_a in enumerate(z_tol_values):
    for j, zt_b in enumerate(z_tol_values):
        if i == j:
            # Diagonal: total row count (all rows trivially agree with themselves)
            agreement_matrix[i, j] = len(merged_vs_fca[zt_a])
        elif j > i:
            # Compare area_ids directly — both are already aligned via the FCA join
            # (same inner join key: (player_name, frame_idx) for round 1)
            agree = (kdt_area_ids[zt_a].values == kdt_area_ids[zt_b].values).sum()
            agreement_matrix[i, j] = agree
            agreement_matrix[j, i] = agree  # symmetric
 
# Print header
col_w = 7
header_labels = [f"z{int(zt)}" for zt in z_tol_values]
print("  " + " " * 8 + "  ".join(f"{lbl:>{col_w}}" for lbl in header_labels))
print("  " + " " * 8 + "  ".join(["-" * col_w] * n))
 
for i, zt in enumerate(z_tol_values):
    row_vals = "  ".join(
        f"{agreement_matrix[i, j]:>{col_w}}" for j in range(n)
    )
    print(f"  {f'z{int(zt)}':<8}{row_vals}")
 
print()
print("  Adjacent-pair disagreement counts (most diagnostic for calibration):")
for i in range(len(z_tol_values) - 1):
    zt_a     = z_tol_values[i]
    zt_b     = z_tol_values[i + 1]
    n_rows   = len(merged_vs_fca[zt_a])
    agree    = agreement_matrix[i, i + 1]
    disagree = n_rows - agree
    print(f"    z_tol={zt_a:.0f} vs z_tol={zt_b:.0f}: {disagree} disagreements "
          f"({100*disagree/n_rows:.1f}%)")
 
# ---------------------------------------------------------------------------
# SECTION 3: CORRECTION AREA-NAME BREAKDOWN — vs FCA
# ---------------------------------------------------------------------------
 
print(f"\n{'='*70}")
print("  SECTION 3 — Correction Area-Name Breakdown vs FCA (Round 1)")
print(f"{'='*70}")
print()
print("  For each z_tol, shows the top named places that are REASSIGNED relative")
print("  to FCA. FCA_areaName → KDT_areaName (count of frames corrected).")
print("  z-overlap zones (Catwalk, Underpass, Connector) appearing in the FCA")
print("  column indicate the filter is doing its intended job.")
print()
 
for z_tol in z_tol_values:
    merged    = merged_vs_fca[z_tol]
    diff_mask = merged["current_areaId_fca"] != merged["current_areaId_kdt"]
    diff_rows = merged[diff_mask].copy()
 
    if len(diff_rows) == 0:
        print(f"  z_tol={z_tol:.0f}: no corrections vs FCA baseline.")
        continue
 
    diff_rows["fca_name"] = diff_rows["current_areaId_fca"].apply(area_name)
    diff_rows["kdt_name"] = diff_rows["current_areaId_kdt"].apply(area_name)
    correction_pairs      = (
        diff_rows
        .groupby(["fca_name", "kdt_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(8)
    )
 
    print(f"  z_tol={z_tol:.0f}  ({len(diff_rows)} corrections):")
    for _, row in correction_pairs.iterrows():
        print(f"    {row['fca_name']:>20}  →  {row['kdt_name']:<20}  "
              f"({row['count']} frame(s))")
    print()
 
# ---------------------------------------------------------------------------
# SECTION 4: PER-PLAYER DISAGREEMENT COUNTS — vs FCA
# ---------------------------------------------------------------------------
 
print(f"\n{'='*70}")
print("  SECTION 4 — Per-Player Disagreement Counts vs FCA (Round 1)")
print(f"{'='*70}\n")
 
# Build a per-player table: rows = players, columns = z_tol values
all_players = sorted(fca_df["player_name"].unique())
col_labels  = [f"z{int(zt)}" for zt in z_tol_values]
player_disagree = pd.DataFrame(index=all_players, columns=col_labels, dtype=int).fillna(0)
 
for z_tol in z_tol_values:
    merged      = merged_vs_fca[z_tol]
    diff_mask   = merged["current_areaId_fca"] != merged["current_areaId_kdt"]
    per_player  = merged[diff_mask].groupby("player_name").size()
    col_label   = f"z{int(z_tol)}"
    for player, count in per_player.items():
        if player in player_disagree.index:
            player_disagree.loc[player, col_label] = count
 
# Add total frames per player as context
total_frames = fca_df.groupby("player_name").size().rename("total_frames")
player_disagree = player_disagree.join(total_frames)
 
print("  (count = number of round-1 frames where this z_tol disagrees with FCA)\n")
print(f"  {'Player':<14}", end="")
for lbl in col_labels:
    print(f"  {lbl:>6}", end="")
print(f"  {'Total':>7}")
print(f"  {'-'*14}", end="")
for _ in col_labels:
    print(f"  {'-'*6}", end="")
print(f"  {'-'*7}")
 
for player in all_players:
    row = player_disagree.loc[player]
    print(f"  {player:<14}", end="")
    for lbl in col_labels:
        print(f"  {int(row[lbl]):>6}", end="")
    print(f"  {int(row['total_frames']):>7}")
 
# ---------------------------------------------------------------------------
# SECTION 5: COMPARISON PLOTS — each z_tol vs FCA
# ---------------------------------------------------------------------------
'''
print(f"\n{'='*70}")
print(f"  SECTION 5 — Comparison Plots: Each z_tol vs FCA")
print(f"{'='*70}")
print(f"\n  Generating up to {MAX_PLOTS_VS_FCA} plots per z_tol.")
print(f"  Priority: named-area changes first, then largest distance correction.")
print(f"  Output root: {PLOTS_DIR}/vs_fca/\n")
 
total_plots_saved = 0
 
for z_tol in z_tol_values:
    merged    = merged_vs_fca[z_tol]
    diff_mask = merged["current_areaId_fca"] != merged["current_areaId_kdt"]
    diff_rows = merged[diff_mask].copy()
 
    if len(diff_rows) == 0:
        print(f"  z_tol={z_tol:.0f}: no disagreements vs FCA — no plots generated.")
        continue
 
    # Sort by diagnostic priority and take the top MAX_PLOTS_VS_FCA
    diff_rows = sort_by_diagnostic_priority(
        diff_rows,
        area_col_a="current_areaId_fca",
        area_col_b="current_areaId_kdt",
        dist_col_a="mapping_distance_fca",
        dist_col_b="mapping_distance_kdt",
    )
    sample = diff_rows.head(MAX_PLOTS_VS_FCA)
 
    z_tol_dir    = os.path.join(PLOTS_DIR, "vs_fca", f"ztol_{z_tol:.0f}")
    plots_for_zt = 0
 
    for _, row in sample.iterrows():
        player    = row["player_name"]
        frame_idx = int(row["frame_idx"])
        fca_id    = int(row["current_areaId_fca"])
        kdt_id    = int(row["current_areaId_kdt"])
        fca_dist  = float(row["mapping_distance_fca"])
        kdt_dist  = float(row["mapping_distance_kdt"])
 
        # Skip if same area (shouldn't happen in diff_rows, but defensive)
        if fca_id == kdt_id:
            continue
 
        fca_label = "FCA (find_closest_area)"
        kdt_label = f"KDTree z_tol={z_tol:.0f}"
 
        title1 = (f"{player}  |  Frame {frame_idx}  |  "
                  f"z={row['z']:.1f}  |  Side: {row['side']}")
        title2 = (f"FCA → Area {fca_id} '{area_name(fca_id)}' (dist={fca_dist:.1f})    "
                  f"KDT → Area {kdt_id} '{area_name(kdt_id)}' (dist={kdt_dist:.1f})")
 
        safe_player = player.replace(" ", "_").replace("/", "_")
        save_path   = os.path.join(
            z_tol_dir, f"{safe_player}_frame{frame_idx:04d}.png"
        )
 
        save_comparison_plot(
            row=row,
            area_id_a=fca_id,
            area_id_b=kdt_id,
            dist_a=fca_dist,
            dist_b=kdt_dist,
            label_a=fca_label,
            label_b=kdt_label,
            color_a="#FF6B6B",   # coral red for FCA
            color_b="#4ECDC4",   # teal cyan for KDTree
            title_line1=title1,
            title_line2=title2,
            save_path=save_path,
        )
        plots_for_zt    += 1
        total_plots_saved += 1
 
    if plots_for_zt > 0:
        print(f"  z_tol={z_tol:.0f}: {plots_for_zt} plot(s) saved to {z_tol_dir}/")
    else:
        print(f"  z_tol={z_tol:.0f}: no plottable disagreements "
              f"(all {len(diff_rows)} disagreements are same-area-id — unexpected).")
 
print(f"\n  Total vs-FCA comparison plots saved: {total_plots_saved}")
 
# ---------------------------------------------------------------------------
# SECTION 6: COMPARISON PLOTS — adjacent z_tol pairs
# ---------------------------------------------------------------------------
 
print(f"\n{'='*70}")
print(f"  SECTION 6 — Comparison Plots: Adjacent z_tol Pairs")
print(f"{'='*70}")
print(f"\n  Frames where adjacent z_tol values produce different assignments are")
print(f"  the most informative calibration cases — the tolerance boundary falls")
print(f"  right near those tile centroid z-differences.")
print(f"  Generating up to {MAX_PLOTS_PAIRS} plots per adjacent pair.")
print(f"  Output root: {PLOTS_DIR}/adjacent_pairs/\n")
 
adj_plots_saved = 0
 
for idx in range(len(z_tol_values) - 1):
    zt_a    = z_tol_values[idx]
    zt_b    = z_tol_values[idx + 1]
 
    # Both are already aligned on (player_name, frame_idx) via the FCA join.
    # Join them directly on that alignment.
    merged_a = merged_vs_fca[zt_a][
        ["player_name", "round", "frame_idx", "x", "y", "z", "side",
        "current_areaId_kdt", "mapping_distance_kdt"]
    ].rename(columns={
        "current_areaId_kdt":   f"area_{zt_a:.0f}",
        "mapping_distance_kdt": f"dist_{zt_a:.0f}",
    })
    merged_b = merged_vs_fca[zt_b][
        ["player_name", "round", "frame_idx",
        "current_areaId_kdt", "mapping_distance_kdt"]
    ].rename(columns={
        "current_areaId_kdt":   f"area_{zt_b:.0f}",
        "mapping_distance_kdt": f"dist_{zt_b:.0f}",
    })
 
    pair_merged = merged_a.merge(merged_b, on=["player_name", "round", "frame_idx"], how="inner")
    diff_mask   = pair_merged[f"area_{zt_a:.0f}"] != pair_merged[f"area_{zt_b:.0f}"]
    diff_rows   = pair_merged[diff_mask].copy()
 
    pair_label  = f"ztol{zt_a:.0f}_vs_ztol{zt_b:.0f}"
    pair_dir    = os.path.join(PLOTS_DIR, "adjacent_pairs", pair_label)
 
    if len(diff_rows) == 0:
        print(f"  z_tol={zt_a:.0f} vs z_tol={zt_b:.0f}: no disagreements — no plots generated.")
        continue
 
    diff_rows = sort_by_diagnostic_priority(
        diff_rows,
        area_col_a=f"area_{zt_a:.0f}",
        area_col_b=f"area_{zt_b:.0f}",
        dist_col_a=f"dist_{zt_a:.0f}",
        dist_col_b=f"dist_{zt_b:.0f}",
    )
    sample        = diff_rows.head(MAX_PLOTS_PAIRS)
    plots_for_pair = 0
 
    for _, row in sample.iterrows():
        player    = row["player_name"]
        frame_idx = int(row["frame_idx"])
        id_a      = int(row[f"area_{zt_a:.0f}"])
        id_b      = int(row[f"area_{zt_b:.0f}"])
        dist_a    = float(row[f"dist_{zt_a:.0f}"])
        dist_b    = float(row[f"dist_{zt_b:.0f}"])
 
        if id_a == id_b:
            continue
 
        title1 = (f"{player}  |  Frame {frame_idx}  |  "
                  f"z={row['z']:.1f}  |  Side: {row['side']}")
        title2 = (f"z_tol={zt_a:.0f} → Area {id_a} '{area_name(id_a)}' (dist={dist_a:.1f})    "
                  f"z_tol={zt_b:.0f} → Area {id_b} '{area_name(id_b)}' (dist={dist_b:.1f})")
 
        safe_player = player.replace(" ", "_").replace("/", "_")
        save_path   = os.path.join(
            pair_dir, f"{safe_player}_frame{frame_idx:04d}.png"
        )
 
        save_comparison_plot(
            row=row,
            area_id_a=id_a,
            area_id_b=id_b,
            dist_a=dist_a,
            dist_b=dist_b,
            label_a=f"z_tol={zt_a:.0f}",
            label_b=f"z_tol={zt_b:.0f}",
            color_a="#FF9F43",   # orange for the tighter (lower) z_tol
            color_b="#A29BFE",   # violet/lavender for the looser (higher) z_tol
            title_line1=title1,
            title_line2=title2,
            save_path=save_path,
        )
        plots_for_pair += 1
        adj_plots_saved += 1
 
    print(f"  z_tol={zt_a:.0f} vs z_tol={zt_b:.0f}: {len(diff_rows)} disagreements, "
          f"{plots_for_pair} plot(s) saved to {pair_dir}/")
 
print(f"\n  Total adjacent-pair comparison plots saved: {adj_plots_saved}")
'''
# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
 
print(f"\n{'='*70}")
print("  VERIFICATION COMPLETE")
print(f"{'='*70}\n")
print(f"  Sweep CSVs compared:    {len(z_tol_values)} z_tol values from {SWEEP_DIR}/")
print(f"  FCA baseline:           {FCA_CSV}")
# print(f"  vs-FCA plots saved:     {total_plots_saved}  →  {PLOTS_DIR}/vs_fca/")
# print(f"  Adjacent-pair plots:    {adj_plots_saved}  →  {PLOTS_DIR}/adjacent_pairs/")
print()
print("  Recommended next steps:")
print("  1. Review SECTION 3 (area-name breakdown) — if Catwalk/Underpass/Connector")
print("     appear frequently in the FCA column, those corrections are on-target.")
print("     If CTSpawn dominates, the z-filter may be confusing large-tile size")
print("     for z-overlap; inspect those plots specifically.")
print("  2. Open vs_fca/ plots for each z_tol — check that the KDTree-assigned tile")
print("     (teal) sits geographically closer to the player than the FCA tile (red).")
print("     A teal tile on the opposite side of the radar from the player = overcorrection.")
print("  3. Open adjacent_pairs/ plots — these show the sensitivity boundary. The")
print("     pair where corrections start to look geographically wrong sets your")
print("     upper bound; the pair where they all still look correct sets the floor.")
print("  4. Pick the z_tol that maximises correct corrections (Catwalk/Underpass)")
print("     while minimising incorrect ones (large-tile confusion in flat areas).")
print("     Update Z_TOL in ztol_sweep.py and re-run for confirmation.\n")
