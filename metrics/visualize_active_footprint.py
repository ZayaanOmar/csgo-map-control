"""
visualize_active_footprint.py

Generates per-frame heatmap images for a chosen match and round, overlaying
the active-footprint nodes (ct_mu + t_mu > EPSILON) in a transparent purple
on top of the normal sigmoid-normalised control field heatmap.

This is primarily an interpretive/diagnostic visualisation: it makes the
"active footprint" concept concrete by showing exactly which nav tiles are
being included in the active-footprint control margin and which are being
excluded, in the context of an actual frame from an actual round.

WHAT EACH FRAME SHOWS
-----------------------
  Base heatmap layer:
    Each nav tile is coloured by its control_val at that frame, using a
    diverging colormap centred at 0.5 (neutral).  Tiles with control_val
    close to 1 are deep cyan (CT control); tiles close to 0 are deep
    orange/red (T control); tiles near 0.5 are close to white.

  Active-footprint overlay:
    Tiles where (ct_mu + t_mu) > EPSILON at that frame are additionally
    drawn in semi-transparent purple (default alpha 0.35).  Tiles NOT
    highlighted by the purple overlay are the "dead weight" nodes sitting
    at neutral purely because no player is nearby — the ones the
    active-footprint metric actively excludes.

  Colorbar + annotation:
    A colorbar shows the cyan→white→orange scale.  A text box in the
    corner reports the frame's active node count, active fraction, and
    raw control values in both the full-map and active-footprint versions
    so you can compare them frame by frame.

USAGE
------
  # All frames in round 5 of a match, saved as PNGs + a GIF:
  python visualize_active_footprint.py --match FaZe_VS_BIG_4bab9a63 --round 5

  # A single specific frame only:
  python visualize_active_footprint.py --match FaZe_VS_BIG_4bab9a63 --round 5 --frame 12

  # Tighter active-footprint threshold (2*sigma cutoff instead of 3*sigma):
  python visualize_active_footprint.py --match FaZe_VS_BIG_4bab9a63 --round 5 --epsilon 0.135

  # Suppress GIF, keep PNGs only:
  python visualize_active_footprint.py --match FaZe_VS_BIG_4bab9a63 --round 5 --no-gif

INPUTS
-------
  outputs/control_fields/{match_name}.csv
      round, frame_idx, area_id, ct_mu, t_mu, control_val
      ct_mu/t_mu required — run updated team_agg.py before this script.

OUTPUTS
--------
  outputs/active_footprint_viz/{match_name}/round_{n}/
      frame_{idx:04d}.png  — one PNG per frame
      round_{n}_active_footprint.gif  — animated GIF of the full round
          (omitted if --no-gif or only one frame was generated)
"""

import argparse
import os
import glob

import imageio.v3 as imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

from awpy.data import NAV, MAP_DATA
from awpy.visualization.plot import plot_map, position_transform

# ---------------------------------------------------------------------------
# Constants — all match metric3_control_margin.py defaults
# ---------------------------------------------------------------------------
MAP_NAME = "de_mirage"
SIGMA = 500.0
EPSILON_DEFAULT = float(np.exp(-2))   # 3*sigma cutoff, same as Metric 3

CONTROL_FIELD_PATH = "outputs/control_fields"
POSITION_MAPS_PATH = "outputs/position_maps"
OUTPUT_BASE_DIR = "outputs/active_footprint_viz"

# Heatmap colormap — matches generate_heatmap_gifs.py exactly:
#   orange (T=0) → dark radar background (neutral=0.5) → cyan (CT=1)
# The neutral midpoint is the dark radar background color, NOT white, so that
# uncontested tiles blend into the map image rather than standing out as a
# bright white wash.
DARK_BG  = np.array([1.0 / 255.0, 19.0 / 255.0, 48.0 / 255.0])  # dark navy
CT_COLOR = np.array([0.0, 0.9, 0.9])      # vivid cyan  — matches CONTROL_CMAP
T_COLOR  = np.array([1.0, 0.5, 0.0])      # vivid orange — matches CONTROL_CMAP
NEUTRAL  = DARK_BG                         # alias for clarity in control_val_to_rgb

# Player and bomb markers — match generate_heatmap_gifs.py exactly
PLAYER_COLORS      = {"CT": "#00AAAA", "T": "#CC6600"}
PLAYER_MARKER_SIZE = 20
BOMB_COLOR         = "#FF0000"
BOMB_MARKER_SIZE   = 14 / 1.5   # ~9.33 — 1.5x smaller than players

# Active-footprint overlay
ACTIVE_COLOR   = "#9B30FF"   # purple — high contrast against both cyan and orange
ACTIVE_ALPHA   = 0.9        # transparent enough to see the heatmap underneath

# GIF frame duration in milliseconds
GIF_FRAME_DURATION_MS = 200  # 5 fps

# Fixed figure size in inches — must NOT vary frame-to-frame.
# bbox_inches='tight' in savefig trims each frame to its content bounding
# box, which changes slightly between frames (info-box numbers, colorbar
# label placement), producing inconsistent pixel dimensions that imageio
# cannot stack into a GIF. The fix (same as generate_heatmap_gifs.py) is
# a fixed figsize here plus NO bbox_inches='tight' in savefig.
DEFAULT_FIG_SIZE = (9.5, 8.5)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def control_val_to_rgb(val: float) -> tuple[float, float, float]:
    """Map a sigmoid-normalised control value in [0, 1] to an RGB triple.

    0.5 (neutral) → dark radar background (DARK_BG) — matches generate_heatmap_gifs.py
    1.0 (full CT) → vivid cyan
    0.0 (full T)  → vivid orange

    Uses a two-segment linear interpolation so the neutral midpoint is
    always exactly DARK_BG regardless of how the two side-colours are
    defined, rather than a single colormap that might not pass through the
    intended neutral color at exactly 0.5.
    """
    val = float(np.clip(val, 0.0, 1.0))
    if val >= 0.5:
        t = (val - 0.5) / 0.5       # 0 at neutral, 1 at full CT
        rgb = (1 - t) * NEUTRAL + t * CT_COLOR
    else:
        t = (0.5 - val) / 0.5       # 0 at neutral, 1 at full T
        rgb = (1 - t) * NEUTRAL + t * T_COLOR
    return tuple(np.clip(rgb, 0.0, 1.0))


def make_colorbar_mappable() -> ScalarMappable:
    """Build a ScalarMappable for the colorbar using a ListedColormap that
    matches control_val_to_rgb's two-segment interpolation exactly, so the
    colorbar legend is consistent with the tile colors in the figure.
    """
    n = 256
    vals = np.linspace(0, 1, n)
    rgba = np.array([(*control_val_to_rgb(v), 1.0) for v in vals])
    cmap = mcolors.ListedColormap(rgba)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


# ---------------------------------------------------------------------------
# Tile geometry helpers
# ---------------------------------------------------------------------------
def tile_rect_params(area_id: int) -> tuple[float, float, float, float] | None:
    """Return (x_nw, y_se, width, height) in radar-pixel coordinates for
    one nav tile, suitable for passing directly to patches.Rectangle.

    Returns None if the area_id is not in NAV (safety guard — should not
    happen with a verified patched graph, but worth checking).
    """
    if area_id not in NAV[MAP_NAME]:
        return None

    area = NAV[MAP_NAME][area_id]

    # Transform all four corners individually — do NOT assume width/height
    # transform as simple scalar multiples, since the Y axis is a flip-and-
    # shift rather than a pure scale (see awpy_unified_reference.md §7.2).
    x_nw = position_transform(MAP_NAME, area["northWestX"], "x")
    x_se = position_transform(MAP_NAME, area["southEastX"], "x")
    y_nw = position_transform(MAP_NAME, area["northWestY"], "y")
    y_se = position_transform(MAP_NAME, area["southEastY"], "y")

    # Use min/max to guarantee a positive width/height regardless of
    # coordinate-ordering sign conventions (see awpy_unified_reference.md §1.2).
    x_left   = min(x_nw, x_se)
    y_bottom = min(y_nw, y_se)
    width    = abs(x_se - x_nw)
    height   = abs(y_nw - y_se)

    return x_left, y_bottom, width, height


# ---------------------------------------------------------------------------
# Per-frame rendering
# ---------------------------------------------------------------------------
def render_frame(
    frame_df: pd.DataFrame,
    pos_frame_df: pd.DataFrame,
    frame_idx: int,
    round_num: int,
    match_name: str,
    epsilon: float,
    out_path: str,
) -> None:
    """Render one frame as a PNG.

    frame_df:     rows from control_fields for this (round, frame_idx),
                  columns: area_id, ct_mu, t_mu, control_val.
    pos_frame_df: rows from position_maps for this (round, frame_idx),
                  columns: player_name, side, current_x, current_y,
                  [bomb_x, bomb_y, bomb_planted] — bomb columns optional,
                  skipped gracefully if absent (matching generate_heatmap_gifs.py).
    """
    has_raw_mu = {"ct_mu", "t_mu"}.issubset(frame_df.columns)

    if has_raw_mu:
        frame_df = frame_df.copy()
        frame_df["summed_mu"] = frame_df["ct_mu"] + frame_df["t_mu"]
        frame_df["is_active"] = frame_df["summed_mu"] > epsilon
        n_active = int(frame_df["is_active"].sum())
        n_total  = len(frame_df)
        active_frac = n_active / n_total if n_total else 0.0
        full_mean   = float(frame_df["control_val"].mean())
        active_mean = float(frame_df.loc[frame_df["is_active"], "control_val"].mean()) \
                      if n_active > 0 else float("nan")
    else:
        has_raw_mu = False
        n_active, n_total, active_frac = 0, len(frame_df), 0.0
        full_mean, active_mean = float(frame_df["control_val"].mean()), float("nan")

    # --- Base radar image ---
    fig, ax = plot_map(MAP_NAME, map_type="simpleradar", dark=True)
    # Enforce a fixed pixel size across every frame — see DEFAULT_FIG_SIZE note.
    fig.set_size_inches(DEFAULT_FIG_SIZE)
    ax.set_aspect("equal")

    # --- Heatmap tiles (base layer) ---
    for _, row in frame_df.iterrows():
        rect_params = tile_rect_params(int(row["area_id"]))
        if rect_params is None:
            continue
        x, y, w, h = rect_params
        color = control_val_to_rgb(float(row["control_val"]))
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=0,
            facecolor=color,
            alpha=0,
            zorder=2,
        )
        ax.add_patch(rect)

    # --- Active-footprint overlay (purple, drawn on top) ---
    if has_raw_mu:
        active_rows = frame_df[frame_df["is_active"]]
        for _, row in active_rows.iterrows():
            rect_params = tile_rect_params(int(row["area_id"]))
            if rect_params is None:
                continue
            x, y, w, h = rect_params
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=0,
                facecolor=ACTIVE_COLOR,
                alpha=ACTIVE_ALPHA,
                zorder=3,
            )
            ax.add_patch(rect)

    # --- Player positions ---
    # One dot per alive player, colored by side, using the same marker
    # colors and size as generate_heatmap_gifs.py (PLAYER_COLORS, PLAYER_MARKER_SIZE).
    # position_maps stores world coordinates as current_x/current_y — these
    # need position_transform before plotting on the radar canvas.
    if not pos_frame_df.empty:
        for side, color in PLAYER_COLORS.items():
            side_df = pos_frame_df[pos_frame_df["side"] == side]
            if side_df.empty:
                continue
            px = [position_transform(MAP_NAME, x, "x") for x in side_df["current_x"]]
            py = [position_transform(MAP_NAME, y, "y") for y in side_df["current_y"]]
            ax.scatter(px, py, c=color, s=PLAYER_MARKER_SIZE, zorder=6,
                       edgecolors="black", linewidths=0.5)

    # --- Bomb position ---
    # Sourced from bomb_x/bomb_y columns in position_maps (added per
    # data_storage_alterations.md). Skipped gracefully if columns are absent,
    # matching the generate_heatmap_gifs.py convention.
    has_bomb_cols = {"bomb_x", "bomb_y"}.issubset(pos_frame_df.columns)
    if has_bomb_cols and not pos_frame_df.empty:
        bomb_row = pos_frame_df.iloc[0]  # bomb position is the same for all players in frame
        bx = bomb_row["bomb_x"]
        by = bomb_row["bomb_y"]
        if pd.notna(bx) and pd.notna(by):
            bx_px = position_transform(MAP_NAME, float(bx), "x")
            by_px = position_transform(MAP_NAME, float(by), "y")
            ax.scatter(bx_px, by_px, c=BOMB_COLOR, s=BOMB_MARKER_SIZE,
                       marker="s", zorder=7, edgecolors="white", linewidths=0.5)

    # --- Colorbar ---
    sm = make_colorbar_mappable()
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Control value  c(v,f)", fontsize=8, color="white")
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["T (0.0)", "0.25", "Neutral (0.5)", "0.75", "CT (1.0)"])

    # --- Info text box ---
    info_lines = [
        f"Round {round_num}  |  Frame {frame_idx}",
        f"Full-map mean:     {full_mean:.4f}",
    ]
    if has_raw_mu:
        info_lines += [
            f"Active mean:       {active_mean:.4f}",
            f"Active nodes:      {n_active} / {n_total}  ({active_frac:.1%})",
            f"EPSILON = {epsilon:.4f}  (σ={SIGMA:.0f})",
            "",
            "■ Purple = active footprint",
            "  (ct_mu + t_mu > EPSILON)",
        ]
    else:
        info_lines.append("(ct_mu/t_mu not available — re-run team_agg.py)")

    info_text = "\n".join(info_lines)
    ax.text(
        0.01, 0.01, info_text,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.65, edgecolor="none"),
        color="white",
        zorder=10,
        family="monospace",
    )

    # --- Legend ---
    from matplotlib.lines import Line2D
    legend_handles = []
    if has_raw_mu:
        legend_handles.append(
            patches.Patch(facecolor=ACTIVE_COLOR, alpha=0.7, label="Active footprint")
        )
    legend_handles += [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PLAYER_COLORS["CT"],
               markeredgecolor="white", markeredgewidth=0.5,
               markersize=6, label="CT player"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PLAYER_COLORS["T"],
               markeredgecolor="white", markeredgewidth=0.5,
               markersize=6, label="T player"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=BOMB_COLOR,
               markeredgecolor="white", markeredgewidth=0.5,
               markersize=5, label="Bomb"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              framealpha=0.65, facecolor="black", labelcolor="white",
              edgecolor="none")

    ax.set_title(
        f"{match_name}  —  Round {round_num}, Frame {frame_idx}",
        fontsize=9, color="white", pad=6,
    )
    ax.axis("off")
    fig.patch.set_facecolor("black")

    # No bbox_inches='tight' — fixed figsize above guarantees stable pixel
    # dimensions across all frames, which is required for GIF assembly.
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate active-footprint overlay heatmaps for a chosen match and round. "
            "Tiles are coloured by control_val (cyan=CT, orange=T); "
            "active nodes (ct_mu + t_mu > EPSILON) are additionally highlighted in purple."
        )
    )
    parser.add_argument(
        "--match", required=True,
        help="Match name stem (filename without .csv extension) as it appears "
             "in outputs/control_fields/ — e.g. FaZe_VS_BIG_4bab9a63",
    )
    parser.add_argument(
        "--round", dest="round_num", type=int, required=True,
        help="Round number to visualise.",
    )
    parser.add_argument(
        "--frame", dest="frame_idx", type=int, default=None,
        help="Optional: a single frame index to render. "
             "If omitted, all frames in the round are rendered.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=EPSILON_DEFAULT,
        help=(
            f"Active-node threshold on (ct_mu + t_mu), applied BEFORE the sigmoid. "
            f"Default {EPSILON_DEFAULT:.4f} (d_eff = 3*sigma, sigma={SIGMA}). "
            f"Pass 0.135 (~exp(-2)) for a tighter 2*sigma cutoff."
        ),
    )
    parser.add_argument(
        "--no-gif", dest="no_gif", action="store_true",
        help="Skip GIF generation — keep PNG frames only.",
    )
    args = parser.parse_args()

    control_path = os.path.join(CONTROL_FIELD_PATH, f"{args.match}.csv")
    if not os.path.exists(control_path):
        raise FileNotFoundError(
            f"Control field CSV not found at {control_path}. "
            f"Check --match matches a filename in {CONTROL_FIELD_PATH}/"
        )

    print(f"Loading control field: {control_path}")
    control_df = pd.read_csv(control_path)

    if "control_val" not in control_df.columns:
        raise ValueError(
            "control_fields CSV lacks a 'control_val' column — "
            "check that team_agg.py has been run for this match."
        )
    has_raw_mu = {"ct_mu", "t_mu"}.issubset(control_df.columns)
    if not has_raw_mu:
        print(
            "WARNING: ct_mu/t_mu columns not found — the purple active-footprint "
            "overlay will not be drawn. Re-run team_agg.py (updated version) to "
            "add these columns to the control_fields CSV."
        )

    # --- Position maps (player positions + bomb) ---
    pos_path = os.path.join(POSITION_MAPS_PATH, f"{args.match}.csv")
    if os.path.exists(pos_path):
        print(f"Loading position maps: {pos_path}")
        pos_df = pd.read_csv(pos_path)
        if {"bomb_x", "bomb_y"}.issubset(pos_df.columns):
            print("  Bomb position columns found — bomb marker will be drawn.")
        else:
            print("  NOTE: bomb_x/bomb_y not in position_maps — bomb marker will be skipped.")
    else:
        print(
            f"WARNING: position_maps file not found at {pos_path} — "
            f"player and bomb markers will not be drawn."
        )
        pos_df = pd.DataFrame()  # empty — render_frame handles this gracefully

    round_df = control_df[control_df["round"] == args.round_num]
    if round_df.empty:
        available = sorted(control_df["round"].unique())
        raise ValueError(
            f"Round {args.round_num} not found in {args.match}. "
            f"Available rounds: {available}"
        )

    if args.frame_idx is not None:
        frame_df_all = round_df[round_df["frame_idx"] == args.frame_idx]
        if frame_df_all.empty:
            available = sorted(round_df["frame_idx"].unique())
            raise ValueError(
                f"Frame {args.frame_idx} not found in round {args.round_num}. "
                f"Available frame indices: {available}"
            )
        frames_to_render = [args.frame_idx]
    else:
        frames_to_render = sorted(round_df["frame_idx"].unique())

    out_dir = os.path.join(OUTPUT_BASE_DIR, args.match, f"round_{args.round_num}")
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"Rendering {len(frames_to_render)} frame(s) for "
        f"{args.match} / round {args.round_num}  →  {out_dir}/"
    )

    round_pos_df = pos_df[pos_df["round"] == args.round_num] if not pos_df.empty else pd.DataFrame()

    png_paths = []
    for frame_idx in frames_to_render:
        frame_df = round_df[round_df["frame_idx"] == frame_idx]
        pos_frame_df = round_pos_df[round_pos_df["frame_idx"] == frame_idx] \
                       if not round_pos_df.empty else pd.DataFrame()
        out_path = os.path.join(out_dir, f"frame_{frame_idx:04d}.png")

        render_frame(
            frame_df=frame_df,
            pos_frame_df=pos_frame_df,
            frame_idx=frame_idx,
            round_num=args.round_num,
            match_name=args.match,
            epsilon=args.epsilon,
            out_path=out_path,
        )
        png_paths.append(out_path)
        print(f"  frame {frame_idx:4d}  →  {out_path}")

    # --- GIF ---
    if not args.no_gif and len(png_paths) > 1:
        gif_path = os.path.join(out_dir, f"round_{args.round_num}_active_footprint.gif")
        print(f"\nBuilding GIF  →  {gif_path}")
        frames_imgs = [imageio.imread(p) for p in png_paths]
        imageio.imwrite(gif_path, frames_imgs, duration=GIF_FRAME_DURATION_MS, loop=0)
        print("GIF saved.")

    print(f"\nDone. {len(png_paths)} frame(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
