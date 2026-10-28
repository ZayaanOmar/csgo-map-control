"""
generate_heatmaps.py

Generates two animated GIF heatmaps per qualifying round of a match:
  1. The influence-field-derived team control heatmap
     (from outputs/control_fields/{match_name}.csv, produced by team_agg.py)
  2. The Voronoi geodesic-nearest-player baseline heatmap
     (from outputs/voronoi_control_fields/{match_name}.csv, produced by
     voronoi_baseline.py — same schema as control_fields: round, frame_idx,
     area_id, control_value)

Each frame also draws:
  - The bomb position as a small red square, drawn above every other layer
    (tiles, players). Sourced from the bomb_x/bomb_y columns in
    outputs/position_maps/{match_name}.csv (schema addition per
    data_storage_alterations.md). Skipped gracefully (with one warning) if
    those columns are not present in a given match's position_maps file.
  - The two team names ("TeamA VS TeamB") above the heatmap, sourced from
    outputs/round_metadata/{match_name}.csv for the specific round being
    rendered (ct_team/t_team can differ by round due to side switches).
  - The frame number and match_id, left-aligned in small print below the
    heatmap.

A "qualifying round" is one with strictly more than MIN_FRAMES (120) frames,
determined from outputs/position_maps/{match_name}.csv — the canonical
per-frame record for the match.

Output layout:
    outputs/heatmaps/{match_name}/{round_num}/control_heatmap.gif
    outputs/heatmaps/{match_name}/{round_num}/voronoi_heatmap.gif

This script does NOT recompute team_agg_comp or the Voronoi assignment logic
— it only reads already-produced pipeline outputs and renders them. This is
intentionally heatmap-generation ONLY, adapted from the render_frame /
make_control_gif code in the earlier (pre-refactor) team_agg.py, consistent
with the project convention that heatmap generation is a local/manual,
per-round step excluded from the automated pipeline scripts.
"""

import argparse
import os

import imageio.v3 as imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from awpy.data import NAV
from awpy.visualization.plot import plot_map, position_transform

MAP_NAME = "de_mirage"
MIN_FRAMES = 120  # rounds with frame_idx.nunique() <= this are skipped

CONTROL_FIELDS_PATH_TMPL = "outputs/control_fields/{match_name}.csv"
POSITION_MAPS_PATH_TMPL = "outputs/position_maps/{match_name}.csv"
VORONOI_PATH_TMPL = "outputs/voronoi_control_fields/{match_name}.csv"
ROUND_METADATA_PATH_TMPL = "outputs/round_metadata/{match_name}.csv"
OUTPUT_ROOT = "outputs/heatmaps"

# Fixed figure size (inches) — wider than a square map to leave a stable
# column on the right for the colorbar. Must not vary frame-to-frame or GIF
# assembly breaks (see the fig.set_size_inches note in render_frame).
DEFAULT_FIG_SIZE = (9.5, 8.5)

# Matches the dark radar background so the colorbar/legend/caption whitespace
# doesn't show up as a jarring white margin around the map.
DARK_BG = (1.0 / 255.0, 19.0 / 255.0, 48.0 / 255.0)

PLAYER_MARKER_SIZE = 20
BOMB_MARKER_SIZE = 14  # smaller than PLAYER_MARKER_SIZE, per request
BOMB_COLOR = "#FF0000"

# orange (T) -> dark radar background (contested/neutral) -> cyan (CT)
CONTROL_CMAP = LinearSegmentedColormap.from_list(
    "control",
    [
        (0.0, (1.0, 0.5, 0.0)),
        (0.5, DARK_BG),
        (1.0, (0.0, 0.9, 0.9)),
    ],
)

PLAYER_COLORS = {
    "CT": "#00AAAA",
    "T": "#CC6600",
}


# --------------------------------------------------------------------------- #
# Loading & round selection
# --------------------------------------------------------------------------- #

def load_position_maps(match_name: str) -> pd.DataFrame:
    path = POSITION_MAPS_PATH_TMPL.format(match_name=match_name)
    if not os.path.exists(path):
        msg = f"Position maps file not found: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path)


def load_control_fields(match_name: str) -> pd.DataFrame:
    path = CONTROL_FIELDS_PATH_TMPL.format(match_name=match_name)
    if not os.path.exists(path):
        msg = f"Control fields file not found: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path)


def load_voronoi_control_fields(match_name: str) -> "pd.DataFrame | None":
    """Same schema as load_control_fields: round, frame_idx, area_id, control_value."""
    path = VORONOI_PATH_TMPL.format(match_name=match_name)
    if not os.path.exists(path):
        print(
            f"[WARN] Voronoi control fields file not found, skipping voronoi "
            f"heatmaps for this match: {path}"
        )
        return None
    df = pd.read_csv(path)
    if "control_value" not in df.columns:
        msg = (
            f"Expected column 'control_value' not found in {path}. "
            f"Available columns: {df.columns.tolist()}."
        )
        raise ValueError(msg)
    return df


def load_round_metadata(match_name: str) -> "pd.DataFrame | None":
    """Columns (per data_storage_alterations.md): match_id, round, winning_side,
    ct_team, t_team, round_end_reason, end_tick, freeze_time_end_tick.
    """
    path = ROUND_METADATA_PATH_TMPL.format(match_name=match_name)
    if not os.path.exists(path):
        print(
            f"[WARN] Round metadata file not found — team names and match_id "
            f"will fall back to placeholders: {path}"
        )
        return None
    return pd.read_csv(path)


def get_qualifying_rounds(pos_df: pd.DataFrame, min_frames: int = MIN_FRAMES) -> list:
    """Rounds with strictly more than `min_frames` unique frame_idx values."""
    frame_counts = pos_df.groupby("round")["frame_idx"].nunique()
    qualifying = frame_counts[frame_counts > min_frames].index.tolist()
    return sorted(int(r) for r in qualifying)


def get_match_context(
    match_name: str, round_num: int, pos_df: pd.DataFrame, round_meta_df: "pd.DataFrame | None"
) -> dict:
    """Resolve ct_team/t_team/match_id for one round, with graceful fallbacks
    if round_metadata is missing or doesn't have a row for this round.
    """
    ct_team, t_team, match_id = None, None, None

    if round_meta_df is not None:
        round_row = round_meta_df[round_meta_df["round"] == round_num]
        if not round_row.empty:
            ct_team = round_row.iloc[0].get("ct_team")
            t_team = round_row.iloc[0].get("t_team")
            match_id = round_row.iloc[0].get("match_id")

    if (match_id is None or pd.isna(match_id)) and "match_id" in pos_df.columns and not pos_df.empty:
        match_id = pos_df["match_id"].iloc[0]

    return {
        "ct_team": ct_team if ct_team and not pd.isna(ct_team) else "Team A",
        "t_team": t_team if t_team and not pd.isna(t_team) else "Team B",
        "match_id": match_id if match_id and not pd.isna(match_id) else "unknown",
    }


def get_bomb_position(pos_frame_df: pd.DataFrame) -> "tuple | None":
    """Bomb x/y for this frame, or None if unavailable. bomb_x/bomb_y are
    repeated across every player row for a given frame (schema addition per
    data_storage_alterations.md), so any row's value is representative.
    """
    if pos_frame_df.empty:
        return None
    if not {"bomb_x", "bomb_y"}.issubset(pos_frame_df.columns):
        return None
    row = pos_frame_df.iloc[0]
    bx, by = row["bomb_x"], row["bomb_y"]
    if pd.isna(bx) or pd.isna(by):
        return None
    return (bx, by)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_frame(
    value_df: pd.DataFrame,
    pos_frame_df: pd.DataFrame,
    round_num: int,
    frame_idx: int,
    value_col: str,
    title: str,
    ct_team: str,
    t_team: str,
    match_id,
    bomb_xy: "tuple | None" = None,
    fig_size=DEFAULT_FIG_SIZE,
):
    """Render one frame: tile heatmap (value_col in [0,1]) + player positions
    + bomb marker + colorbar + titles/caption/legend.

    `value_df` must already be filtered to (round_num, frame_idx) and contain
    `area_id` plus `value_col`, normalised to [0, 1]
    (1.0 = full CT, 0.0 = full T, 0.5 = contested/neutral/tie).
    """
    fig, ax = plot_map(map_name=MAP_NAME, map_type="simpleradar", dark=True)
    # Fix the figure size BEFORE saving, and never use bbox_inches='tight' on
    # savefig below — variable per-frame crop dimensions break GIF assembly
    # (documented project gotcha). Same reasoning applies to everything added
    # below (colorbar, legend, text): every call here must be deterministic
    # given identical inputs so every frame renders to identical pixel dims.
    fig.set_size_inches(*fig_size)
    fig.patch.set_facecolor(DARK_BG)

    for _, row in value_df.iterrows():
        area_id = int(row["area_id"])
        val = row[value_col]

        if area_id not in NAV[MAP_NAME]:
            continue

        area = NAV[MAP_NAME][area_id]

        nwx = position_transform(MAP_NAME, area["northWestX"], "x")
        nwy = position_transform(MAP_NAME, area["northWestY"], "y")
        sex = position_transform(MAP_NAME, area["southEastX"], "x")
        sey = position_transform(MAP_NAME, area["southEastY"], "y")

        left, right = min(nwx, sex), max(nwx, sex)
        bottom, top = min(nwy, sey), max(nwy, sey)

        rect = patches.Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            linewidth=0,
            facecolor=CONTROL_CMAP(val),
            alpha=0.9,
        )
        ax.add_patch(rect)

    for _, player in pos_frame_df.iterrows():
        px = position_transform(MAP_NAME, player["current_x"], "x")
        py = position_transform(MAP_NAME, player["current_y"], "y")
        color = PLAYER_COLORS.get(player["side"], "white")

        ax.scatter(
            px, py, s=PLAYER_MARKER_SIZE, c=color, edgecolors="black", linewidths=0.8, zorder=5
        )

    # Bomb marker — small red square, above every other layer (tiles, players).
    if bomb_xy is not None:
        bx = position_transform(MAP_NAME, bomb_xy[0], "x")
        by = position_transform(MAP_NAME, bomb_xy[1], "y")
        ax.scatter(
            bx, by, marker="s", s=BOMB_MARKER_SIZE, c=BOMB_COLOR,
            edgecolors="black", linewidths=0.6, zorder=10,
        )

    ax.axis("off")

    # Vertical colorbar on the right, same gradient/domain as the tile fill.
    sm = ScalarMappable(cmap=CONTROL_CMAP, norm=Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0 (T)", "0.5", "1 (CT)"])
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.outline.set_edgecolor("white")

    # Teams above the heatmap.
    fig.suptitle(f"{ct_team} VS {t_team}", color="white", fontsize=12, y=0.97)
    ax.set_title(f"{title} — Round {round_num}", color="white", fontsize=9)

    # Frame number + match_id, left-aligned, small print, below the heatmap.
    fig.text(
        0.02, 0.02, f"Frame: {frame_idx}    Match ID: {match_id}",
        color="white", fontsize=6, ha="left", va="bottom",
    )

    # Player/bomb icon key, right-aligned, below the heatmap.
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6,
               markerfacecolor=PLAYER_COLORS["CT"], markeredgecolor="black", label="CT"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6,
               markerfacecolor=PLAYER_COLORS["T"], markeredgecolor="black", label="T"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=5,
               markerfacecolor=BOMB_COLOR, markeredgecolor="black", label="Bomb"),
    ]
    legend = fig.legend(
        handles=legend_handles, loc="lower right", bbox_to_anchor=(0.98, 0.015),
        fontsize=6, frameon=False, ncol=3,
    )
    for text in legend.get_texts():
        text.set_color("white")

    return fig


def make_heatmap_gif(
    value_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    round_num: int,
    value_col: str,
    title: str,
    match_context: dict,
    output_path: str,
    fps: int = 5,
    fig_size=DEFAULT_FIG_SIZE,
) -> None:
    round_val_df = value_df[value_df["round"] == round_num].copy()
    round_pos_df = pos_df[pos_df["round"] == round_num].copy()

    frame_indices = sorted(round_val_df["frame_idx"].unique())
    if not frame_indices:
        print(
            f"[WARN] No frames found for round {round_num} in this value "
            f"source — skipping {output_path}"
        )
        return

    tmp_dir = os.path.join(os.path.dirname(output_path), "_tmp_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    image_paths = []

    try:
        for fidx in frame_indices:
            frame_df = round_val_df[round_val_df["frame_idx"] == fidx]
            pos_frame_df = round_pos_df[round_pos_df["frame_idx"] == fidx]
            bomb_xy = get_bomb_position(pos_frame_df)

            fig = render_frame(
                frame_df,
                pos_frame_df,
                round_num,
                fidx,
                value_col,
                title,
                ct_team=match_context["ct_team"],
                t_team=match_context["t_team"],
                match_id=match_context["match_id"],
                bomb_xy=bomb_xy,
                fig_size=fig_size,
            )

            img_path = os.path.join(tmp_dir, f"frame_{fidx:05d}.png")
            # Fixed figsize + no bbox_inches='tight' => stable pixel dims
            # across every frame, required for imageio GIF assembly.
            fig.savefig(img_path, dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            image_paths.append(img_path)

        images = [imageio.imread(p) for p in image_paths]
        # .gif per explicit request. Note: this reintroduces the 256-colour
        # palette banding risk on smooth heatmaps that MP4/libx264 avoided
        # elsewhere in this project — accepted trade-off since a GIF was
        # specifically asked for here.
        imageio.imwrite(output_path, images, duration=1000 / fps)
        print(f"GIF written to {output_path} ({len(images)} frames at {fps} fps)")
    finally:
        for p in image_paths:
            if os.path.exists(p):
                os.remove(p)
        if os.path.isdir(tmp_dir):
            os.rmdir(tmp_dir)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def generate_heatmaps_for_match(match_name: str, min_frames: int = MIN_FRAMES, fps: int = 5) -> None:
    pos_df = load_position_maps(match_name)
    control_df = load_control_fields(match_name)
    voronoi_df = load_voronoi_control_fields(match_name)  # may be None
    round_meta_df = load_round_metadata(match_name)  # may be None

    qualifying_rounds = get_qualifying_rounds(pos_df, min_frames=min_frames)

    if not qualifying_rounds:
        print(f"No rounds with > {min_frames} frames found for {match_name}. Nothing to do.")
        return

    print(f"{match_name}: generating heatmaps for rounds {qualifying_rounds}")

    # Both sources already store control_value normalised into (0,1) with
    # 1.0 = full CT, 0.0 = full T — no transform needed, just alias the
    # column so both value sources share a common `value_col` name downstream.
    control_plot_df = control_df.copy()
    control_plot_df["control_norm"] = control_plot_df["control_val"]

    if voronoi_df is not None:
        voronoi_plot_df = voronoi_df.copy()
        voronoi_plot_df["voronoi_norm"] = voronoi_plot_df["control_value"]
    else:
        voronoi_plot_df = None

    for round_num in qualifying_rounds:
        round_dir = os.path.join(OUTPUT_ROOT, match_name, str(round_num))
        os.makedirs(round_dir, exist_ok=True)

        match_context = get_match_context(match_name, round_num, pos_df, round_meta_df)

        control_gif_path = os.path.join(round_dir, "control_heatmap.gif")
        if os.path.exists(control_gif_path):
            print(f"Skipping control heatmap for round {round_num} — already exists")
        else:
            make_heatmap_gif(
                value_df=control_plot_df,
                pos_df=pos_df,
                round_num=round_num,
                value_col="control_norm",
                title="Control Field",
                match_context=match_context,
                output_path=control_gif_path,
                fps=fps,
            )

        if voronoi_plot_df is not None:
            voronoi_gif_path = os.path.join(round_dir, "voronoi_heatmap.gif")
            if os.path.exists(voronoi_gif_path):
                print(f"Skipping voronoi heatmap for round {round_num} — already exists")
            else:
                make_heatmap_gif(
                    value_df=voronoi_plot_df,
                    pos_df=pos_df,
                    round_num=round_num,
                    value_col="voronoi_norm",
                    title="Voronoi (Geodesic Nearest)",
                    match_context=match_context,
                    output_path=voronoi_gif_path,
                    fps=fps,
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate control-field and Voronoi heatmap GIFs for one match."
    )
    parser.add_argument(
        "match_name",
        type=str,
        help="Match name (the filename stem used across outputs/position_maps, "
        "outputs/control_fields, outputs/voronoi_control_fields, etc.)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=MIN_FRAMES,
        help="Frame-count threshold; only rounds with strictly more frames than "
        "this are rendered (default: 120).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="GIF playback frames per second (default: 5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_heatmaps_for_match(args.match_name, min_frames=args.min_frames, fps=args.fps)
