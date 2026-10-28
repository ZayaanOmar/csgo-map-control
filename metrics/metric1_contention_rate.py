"""
Metric 1 — Contention Rate

Definition (Research_Proposal.pdf, §4.4.3, "Metric 1: Contention Rate"):

    Let V be the set of all nav mesh nodes for de_mirage and F the set of
    all game frames in the dataset after preprocessing. Let c(v, f) in
    [0, 1] be the control value at node v for frame f. A node is defined
    as contested as:

        1_contested(v, f) = 1  if c(v, f) in [0.4, 0.6]
                             0  otherwise

    The Contention Rate for frame f is:

        rho_contested(f) = (1 / |V|) * sum_{v in V} 1_contested(v, f)

    and the mean Contention Rate across all frames is:

        rho_bar = (1 / |F|) * sum_{f in F} rho_contested(f)

This is a structural property of the control field — it requires no ground
truth, and answers RQ1 ("does a continuous control field produce a higher
proportion of nodes classified as contested than a Voronoi segmentation?").
The proposal itself states the Voronoi-baseline value for this metric is
*exactly* 0 by construction (every node is hard-assigned to one team), so
no separate Voronoi run is needed to make that comparison — it is recorded
directly in the summary output below.

-------------------------------------------------------------------------
ACTIVE-FOOTPRINT EXTENSION (added after empirical review)
-------------------------------------------------------------------------
This dilution problem is more severe for Contention Rate than it is for
Metric 3 (Control Margin). A node far from every player has ct_mu -> 0 and
t_mu -> 0, so c(v,f) = expit(K * (ct_mu - t_mu)) -> expit(0) = 0.5 exactly
— and 0.5 sits INSIDE the [0.4, 0.6] contested band by definition. Under
the full-map version, an empty node nobody is near is therefore
indistinguishable from a genuinely evenly-fought-over node: both count
identically as "contested." With an empirically measured mean active node
fraction of ~0.68 on de_mirage (see metric3_control_margin.py), roughly a
third of every frame's nodes are contributing to rho_bar purely because
they are unreached by any player, not because anything is being contested
there.

The fix restricts the numerator AND denominator to the "active footprint"
— nodes with non-negligible raw per-player influence reaching them, BEFORE
the sigmoid step:

    V_active(f) = { v in V : ct_mu(v, f) + t_mu(v, f) > EPSILON }

    rho_contested_active(f) = |{v in V_active(f) : c(v,f) in [0.4,0.6]}| / |V_active(f)|

EPSILON is defined relative to SIGMA (the Gaussian influence-decay
parameter in mu = exp(-d_eff^2 / 2*SIGMA^2)), NOT the sigmoid steepness
constant K — see metric3_control_margin.py's module docstring for the
full derivation. Same default as Metric 3: EPSILON = exp(-1) ~= 0.135,
the 2*SIGMA cutoff.

This reframes Contention Rate from "what fraction of the whole map sits
near-neutral" (which conflates emptiness with genuine contest) to "of the
nodes players are actually near or fighting over, what fraction are truly
up-for-grabs rather than decisively held." The active-footprint rho_bar is
expected to move LOWER than the full-map rho_bar, since the excluded
nodes were inflating the "contested" count rather than compressing it (the
opposite direction from Metric 3's active-footprint correction, which
increased the group-mean separation rather than the raw level).

IMPORTANT — upstream dependency: same as Metric 3, this filter requires
ct_mu and t_mu in outputs/control_fields/{match_name}.csv, which
team_agg.py must be updated to persist alongside control_val. This script
degrades gracefully: it always computes the full-map version (unchanged),
and computes the active-footprint version only for matches whose
control_fields CSV already has ct_mu/t_mu, printing a NOTE for any match
that doesn't.

The Voronoi baseline of exactly 0.0 holds under BOTH restrictions — a hard
nearest-player assignment produces c(v,f) in {0, 1} for every node
(including active-only ones), never landing in [0.4, 0.6] regardless of
which subset of V is considered — so no separate active-footprint Voronoi
computation is needed either.

Input:
    outputs/control_fields/{match_name}.csv  — columns: round, frame_idx,
        area_id, control_val, [ct_mu, t_mu]  <- ct_mu/t_mu required for
        the active version; script falls back to full-map-only if absent
    (produced by team_agg.py; one file per match, one row per (round, frame_idx, area_id))

-------------------------------------------------------------------------
PER-AREA BREAKDOWN
-------------------------------------------------------------------------
Where on the map the contention happens. Every node is assigned to one of
27 named areas using data/de_mirage_area_assignment.csv (awpy's 26 names
plus ApartmentsRamp). This file covers all 903 nodes; awpy's own areaName
is blank for 72 of them, so a breakdown on raw areaName would drop or lump
those tiles. For each area, per frame:

    rate_full(a, f)   = contested tiles in a / tiles in a
    active_frac(a, f) = active tiles in a / tiles in a
    rate_active(a, f) = active contested tiles in a / active tiles in a
                        (NaN if the area has no active tile in that frame)

and the dataset-level table averages each over frames (two-stage, so every
frame counts equally; rate_active averages only over frames where the area
had at least one active tile). Reading the columns:

  - mean_active_fraction: how often the area is "in play" at all.
  - mean_contention_rate_active: when it IS in play, how often a tile is
    genuinely balanced. The full-map rate for the same area is also given;
    the gap between them is the inflation from empty tiles.
  - share_of_active_contested vs share_of_active_tile_frames, and their
    ratio (contested_to_presence_ratio): > 1 means the area contributes
    more than its share of contested tiles relative to how often it is
    occupied; < 1 means it is usually held decisively when occupied.

Per-area rates over small areas are noisy (an area with 1-2 tiles can only
take a few distinct values); n_tiles is reported for that reason. This is
descriptive and carries no significance test.

Output:
    outputs/metrics/contention_rate/{match_name}.csv  — per-frame
        rho_contested(f) (full-map) and rho_contested_active(f)
        (active-footprint, NaN if unavailable), plus active_node_fraction
    outputs/metrics/contention_rate/_summary.json     — dataset-level
        rho_bar and stats, computed separately for full-map and
        active-footprint versions
    outputs/metrics/contention_rate/by_area/{match_name}.csv — per-match,
        per-area sums and frame counts (exact inputs to the dataset table)
    outputs/metrics/contention_rate/by_area/_area_summary.csv — dataset-level
        per-area table described above (skipped, with a warning, if the
        assignment file cannot be loaded; everything else still runs)
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd

# --/ Path consts
CONTROL_FIELD_PATH = "outputs/control_fields"
OUTPUT_DIR = "outputs/metrics/contention_rate"
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "_summary.json")

# --/ Per-area breakdown consts
# Per-area files live in a sub-folder on purpose: visualize_metric1_contention_rate.py
# globs OUTPUT_DIR/*.csv for per-match FRAME files, and a per-area CSV placed
# alongside them would be loaded as if it were a match.
AREA_ASSIGNMENT_PATH = "data/de_mirage_area_assignment.csv"
AREA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "by_area")
AREA_SUMMARY_PATH = os.path.join(AREA_OUTPUT_DIR, "_area_summary.csv")
AREA_PER_MATCH_COLS = {
    "assigned_area", "n_tiles", "n_formerly_blank_tiles", "n_frames",
    "sum_rate_full", "sum_contested_tiles", "sum_active_fraction",
    "sum_rate_active", "n_frames_rate_active", "sum_active_tiles",
    "sum_active_contested_tiles",
}

# --/ Contention band, exactly as defined in the proposal
CONTESTED_LOW = 0.4
CONTESTED_HIGH = 0.6

# --/ Active-footprint consts — same values as metric3_control_margin.py
SIGMA = 500.0                      # must match sigma used to build the influence field
EPSILON = float(np.exp(-2))      # d_eff = 2*SIGMA cutoff; see module docstring
REQUIRED_ACTIVE_COLS = {"ct_mu", "t_mu"}


def compute_frame_contention(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rho_contested(f) for every (round, frame_idx) in one match's control field.

    A node is contested iff control_val falls in [CONTESTED_LOW, CONTESTED_HIGH]
    inclusive on both ends, matching the proposal's indicator function
    literally (it does not specify open/closed bounds, and using an
    inclusive band is the more permissive and defensible reading — it
    counts a node sitting exactly at 0.4 or 0.6 as contested rather than
    silently excluding it).

    |V| is taken as the actual row count per (round, frame_idx) group
    rather than a hardcoded 903, so this remains correct even if a future
    frame has partial node coverage for any reason. The same applies to
    |V_active(f)| below — taken as the actual active-row count per frame,
    since it genuinely varies frame to frame as players move.

    If ct_mu/t_mu are present, also computes the active-footprint version:
    contention restricted to V_active(f) = {v : ct_mu(v,f)+t_mu(v,f) > EPSILON}.
    A frame with zero active nodes (rare, but possible in principle) gets
    NaN for contention_rate_active rather than a divide-by-zero.

    Returns:
        pd.DataFrame(round, frame_idx, num_nodes, num_contested, contention_rate,
                     num_active_nodes, num_active_contested, contention_rate_active,
                     active_node_fraction)
        (active_* columns are NaN throughout if ct_mu/t_mu were not present)
    """
    df = df.copy()
    df["is_contested"] = df["control_val"].between(
        CONTESTED_LOW, CONTESTED_HIGH, inclusive="both"
    )

    has_active_cols = REQUIRED_ACTIVE_COLS.issubset(df.columns)
    agg_kwargs = dict(
        num_nodes=("area_id", "size"),
        num_contested=("is_contested", "sum"),
    )

    if has_active_cols:
        df["summed_mu"] = df["ct_mu"] + df["t_mu"]
        df["is_active"] = df["summed_mu"] > EPSILON
        df["is_active_contested"] = df["is_active"] & df["is_contested"]
        agg_kwargs["num_active_nodes"] = ("is_active", "sum")
        agg_kwargs["num_active_contested"] = ("is_active_contested", "sum")

    frame_stats = (
        df.groupby(["round", "frame_idx"], sort=False)
        .agg(**agg_kwargs)
        .reset_index()
    )
    frame_stats["contention_rate"] = (
        frame_stats["num_contested"] / frame_stats["num_nodes"]
    )

    if has_active_cols:
        frame_stats["contention_rate_active"] = np.where(
            frame_stats["num_active_nodes"] > 0,
            frame_stats["num_active_contested"] / frame_stats["num_active_nodes"],
            np.nan,
        )
        frame_stats["active_node_fraction"] = (
            frame_stats["num_active_nodes"] / frame_stats["num_nodes"]
        )
    else:
        frame_stats["num_active_nodes"] = np.nan
        frame_stats["num_active_contested"] = np.nan
        frame_stats["contention_rate_active"] = np.nan
        frame_stats["active_node_fraction"] = np.nan

    return frame_stats[
        [
            "round", "frame_idx",
            "num_nodes", "num_contested", "contention_rate",
            "num_active_nodes", "num_active_contested", "contention_rate_active",
            "active_node_fraction",
        ]
    ]


def load_area_labels(path: str) -> tuple[dict, set]:
    """Return ({area_id: assigned_area}, {area_ids that had a blank awpy areaName}).

    Prefers the project's own
    pipeline/area_assignment.py::load_assignment; if that cannot be imported
    from here, reads the CSV directly, taking the node-id column from the
    first of a few likely names and raising a clear error rather than
    guessing if none matches.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Area assignment not found at {path}")

    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline")
    if os.path.isdir(scripts_dir) and scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from area_assignment import load_assignment  # type: ignore
        labels, formerly_blank = load_assignment(path)
        return {int(k): v for k, v in labels.items()}, {int(i) for i in formerly_blank}
    except ImportError:
        pass

    df = pd.read_csv(path)
    id_col = next((c for c in ("area_id", "areaId", "areaID", "node", "node_id") if c in df.columns), None)
    if id_col is None:
        raise ValueError(
            f"Could not find a node-id column in {path} (columns: {list(df.columns)}). "
            f"Make pipeline/area_assignment.py importable or add the column name to load_area_labels."
        )
    labels = {int(i): a for i, a in zip(df[id_col], df["assigned_area"])}
    blank = df["original_areaName"].fillna("") == ""
    return labels, {int(i) for i in df.loc[blank, id_col]}


def compute_area_contention(df: pd.DataFrame, labels: dict, formerly_blank: set) -> pd.DataFrame:
    """Per-area contention for one match: per-frame, per-area rates, reduced to
    per-area SUMS and frame counts so the dataset-level mean over frames can be
    reconstructed exactly across matches (a mean of per-match means would
    weight short matches too heavily).

    Active-footprint columns are NaN if ct_mu/t_mu are absent.
    """
    keep = ["round", "frame_idx", "area_id", "control_val"] + [
        c for c in ("ct_mu", "t_mu") if c in df.columns
    ]
    df = df[keep].copy()
    df["assigned_area"] = df["area_id"].map(labels).fillna("UNASSIGNED")
    df["is_contested"] = df["control_val"].between(
        CONTESTED_LOW, CONTESTED_HIGH, inclusive="both"
    )

    has_active = REQUIRED_ACTIVE_COLS.issubset(df.columns)
    agg = dict(n_tiles=("area_id", "size"), n_contested=("is_contested", "sum"))
    if has_active:
        df["is_active"] = (df["ct_mu"] + df["t_mu"]) > EPSILON
        df["is_active_contested"] = df["is_active"] & df["is_contested"]
        agg["n_active"] = ("is_active", "sum")
        agg["n_active_contested"] = ("is_active_contested", "sum")

    # one row per (round, frame, area)
    fa = df.groupby(["round", "frame_idx", "assigned_area"], sort=False).agg(**agg).reset_index()
    fa["rate_full"] = fa["n_contested"] / fa["n_tiles"]
    if has_active:
        fa["active_frac"] = fa["n_active"] / fa["n_tiles"]
        # NaN where the area has no active tile in that frame
        fa["rate_active"] = (fa["n_active_contested"] / fa["n_active"]).where(fa["n_active"] > 0)

    # static per-area tile counts (constant across frames)
    tile_info = df[["area_id", "assigned_area"]].drop_duplicates()
    tile_info["is_blank"] = tile_info["area_id"].astype(int).isin(formerly_blank)
    static = tile_info.groupby("assigned_area").agg(
        n_tiles=("area_id", "size"), n_formerly_blank_tiles=("is_blank", "sum")
    )

    g = fa.groupby("assigned_area")
    per_area = pd.DataFrame({
        "n_frames": g["rate_full"].size(),
        "sum_rate_full": g["rate_full"].sum(),
        "sum_contested_tiles": g["n_contested"].sum(),
    })
    if has_active:
        per_area["sum_active_fraction"] = g["active_frac"].sum()
        per_area["sum_rate_active"] = g["rate_active"].sum()        # skips NaN
        per_area["n_frames_rate_active"] = g["rate_active"].count() # frames with a defined rate
        per_area["sum_active_tiles"] = g["n_active"].sum()
        per_area["sum_active_contested_tiles"] = g["n_active_contested"].sum()
    else:
        for col in ("sum_active_fraction", "sum_rate_active", "n_frames_rate_active",
                    "sum_active_tiles", "sum_active_contested_tiles"):
            per_area[col] = np.nan

    return static.join(per_area).reset_index()


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN wherever den is missing or not positive."""
    return (num / den).where(den > 0)


def summarise_by_area(per_match_tables: list) -> pd.DataFrame:
    """Combine per-match per-area sums into the dataset-level per-area table."""
    big = pd.concat(per_match_tables, ignore_index=True)
    g = big.groupby("assigned_area")
    sum_cols = [
        "n_frames", "sum_rate_full", "sum_active_fraction", "sum_rate_active",
        "n_frames_rate_active", "sum_active_tiles", "sum_active_contested_tiles",
    ]
    tot = g[sum_cols].sum(min_count=1)   # all-NaN (no active data) stays NaN, not 0

    out = pd.DataFrame(index=tot.index)
    out["n_tiles"] = g["n_tiles"].max()
    out["n_formerly_blank_tiles"] = g["n_formerly_blank_tiles"].max()
    out["n_frames"] = tot["n_frames"]
    out["mean_contention_rate_full"] = _safe_div(tot["sum_rate_full"], tot["n_frames"])
    out["mean_active_fraction"] = _safe_div(tot["sum_active_fraction"], tot["n_frames"])
    out["mean_contention_rate_active"] = _safe_div(tot["sum_rate_active"], tot["n_frames_rate_active"])
    out["inflation_full_minus_active"] = (
        out["mean_contention_rate_full"] - out["mean_contention_rate_active"]
    )
    out["share_of_active_tile_frames"] = _safe_div(
        tot["sum_active_tiles"], pd.Series(tot["sum_active_tiles"].sum(), index=tot.index)
    )
    out["share_of_active_contested"] = _safe_div(
        tot["sum_active_contested_tiles"],
        pd.Series(tot["sum_active_contested_tiles"].sum(), index=tot.index),
    )
    out["contested_to_presence_ratio"] = _safe_div(
        out["share_of_active_contested"], out["share_of_active_tile_frames"]
    )

    sort_col = (
        "share_of_active_contested"
        if out["share_of_active_contested"].notna().any()
        else "mean_contention_rate_full"
    )
    return out.sort_values(sort_col, ascending=False)


def control_has_active_cols(control_field_file: str) -> bool:
    """Header-only check: does this control_fields CSV carry ct_mu/t_mu now?"""
    cols = pd.read_csv(control_field_file, nrows=0).columns
    return REQUIRED_ACTIVE_COLS.issubset(cols)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    control_field_files = sorted(glob.glob(os.path.join(CONTROL_FIELD_PATH, "*.csv")))
    print(f"Found {len(control_field_files)} control field files\n")

    # collected per-match frame-level rates for the dataset-level summary.
    # Each per-match CSV is small (rounds x frames rows, not rounds x frames
    # x 903), so holding all of them in memory simultaneously is fine —
    # this is a different scale than the control_fields/influence_fields
    # data itself.
    # Per-area breakdown needs the area assignment. If it cannot be loaded the
    # per-area part is skipped with a warning; everything else runs as before.
    labels, formerly_blank = None, set()
    try:
        labels, formerly_blank = load_area_labels(AREA_ASSIGNMENT_PATH)
        os.makedirs(AREA_OUTPUT_DIR, exist_ok=True)
        print(
            f"Loaded area assignment: {len(set(labels.values()))} areas over "
            f"{len(labels)} nodes ({len(formerly_blank)} formerly blank)\n"
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"WARNING: per-area breakdown disabled — {e}\n")

    all_frame_stats = []
    all_area_stats = []
    any_match_missing_active_cols = False

    for control_field_file in control_field_files:
        mname = os.path.splitext(os.path.basename(control_field_file))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{mname}.csv")
        area_output_path = os.path.join(AREA_OUTPUT_DIR, f"{mname}.csv")
        source_has_active = control_has_active_cols(control_field_file)

        # Skip-if-exists, BUT only if the cache is usable:
        #  - it must have the active-footprint columns (a cache written before
        #    this change would otherwise leave them permanently NaN), and
        #  - those columns must not be all-NaN while the source control_fields
        #    CSV NOW carries ct_mu/t_mu (a cache written before team_agg.py was
        #    patched has the columns but no values, and would otherwise be
        #    trusted forever).
        frame_stats = None
        if os.path.exists(output_path):
            cached = pd.read_csv(output_path)
            has_cols = {"contention_rate_active", "active_node_fraction"}.issubset(cached.columns)
            empty_active = has_cols and cached["contention_rate_active"].isna().all()
            if has_cols and not (empty_active and source_has_active):
                print(f"Skipping frame stats for {mname} - already exists with active-footprint columns")
                if empty_active:
                    any_match_missing_active_cols = True
                frame_stats = cached
            else:
                print(f"Recomputing frame stats for {mname} - cache is stale")

        area_stats = None
        if labels is not None and os.path.exists(area_output_path):
            cached_area = pd.read_csv(area_output_path)
            has_cols = AREA_PER_MATCH_COLS.issubset(cached_area.columns)
            empty_active = has_cols and cached_area["sum_active_tiles"].isna().all()
            if has_cols and not (empty_active and source_has_active):
                area_stats = cached_area

        need_frame = frame_stats is None
        need_area = labels is not None and area_stats is None
        if not need_frame and not need_area:
            all_frame_stats.append((mname, frame_stats))
            if area_stats is not None:
                all_area_stats.append(area_stats)
            continue

        written = []  # files written in THIS iteration; only these are removed on failure
        try:
            print(f"Processing {mname}")

            df = pd.read_csv(control_field_file)
            if not REQUIRED_ACTIVE_COLS.issubset(df.columns):
                any_match_missing_active_cols = True
                print(
                    f"  NOTE: {mname} control_fields lacks ct_mu/t_mu — "
                    f"active-footprint columns will be NaN for this match"
                )

            if need_frame:
                frame_stats = compute_frame_contention(df)
                frame_stats.to_csv(output_path, index=False)
                written.append(output_path)
                print(f"  Saved {len(frame_stats)} frame rows to {output_path}")

            if need_area:
                area_stats = compute_area_contention(df, labels, formerly_blank)
                area_stats.to_csv(area_output_path, index=False)
                written.append(area_output_path)
                print(f"  Saved {len(area_stats)} area rows to {area_output_path}")

            all_frame_stats.append((mname, frame_stats))
            if area_stats is not None:
                all_area_stats.append(area_stats)

            del df

        except Exception as e:
            print(f"\n FAILED: {mname} - {e}\n")
            for p in written:
                if os.path.exists(p):
                    os.remove(p)
            continue

    if not all_frame_stats:
        print("\nNo matches processed — nothing to summarise")
        return

    # --/ Dataset-level per-area table
    area_breakdown_info = None
    if all_area_stats:
        area_summary = summarise_by_area(all_area_stats)
        area_summary.to_csv(AREA_SUMMARY_PATH)
        print(f"\nSaved per-area summary ({len(area_summary)} areas) to {AREA_SUMMARY_PATH}")
        show_cols = [
            "n_tiles", "mean_active_fraction", "mean_contention_rate_full",
            "mean_contention_rate_active", "share_of_active_contested",
            "contested_to_presence_ratio",
        ]
        print(area_summary[show_cols].head(10).round(3).to_string())
        area_breakdown_info = {
            "file": AREA_SUMMARY_PATH,
            "n_areas": int(len(area_summary)),
            "top_areas_by_share_of_active_contested": [
                str(a) for a in area_summary.index[:5]
            ],
        }

    if any_match_missing_active_cols:
        print(
            "\nWARNING: at least one match's control_fields CSV lacked "
            "ct_mu/t_mu. The active-footprint metric is only as complete "
            "as its input — see module docstring for the required "
            "team_agg.py change (write ct_mu and t_mu alongside control_val)."
        )

    # --/ Dataset-level summary
    per_match_means = {}
    total_rate_sum_full = 0.0
    total_frame_count_full = 0
    total_rate_sum_active = 0.0
    total_frame_count_active = 0
    total_active_frac_sum = 0.0
    total_active_frac_count = 0

    for mname, frame_stats in all_frame_stats:
        if frame_stats.empty:
            continue

        match_mean_full = float(frame_stats["contention_rate"].mean())
        active_vals = frame_stats["contention_rate_active"].dropna()
        match_mean_active = float(active_vals.mean()) if len(active_vals) else None
        active_frac_vals = frame_stats["active_node_fraction"].dropna()
        match_active_frac = float(active_frac_vals.mean()) if len(active_frac_vals) else None

        per_match_means[mname] = {
            "mean_contention_rate_full": match_mean_full,
            "mean_contention_rate_active": match_mean_active,
            "mean_active_node_fraction": match_active_frac,
            "n_frames": int(len(frame_stats)),
        }

        total_rate_sum_full += frame_stats["contention_rate"].sum()
        total_frame_count_full += len(frame_stats)

        total_rate_sum_active += active_vals.sum()
        total_frame_count_active += len(active_vals)

        total_active_frac_sum += active_frac_vals.sum()
        total_active_frac_count += len(active_frac_vals)

    rho_bar_full = (
        float(total_rate_sum_full / total_frame_count_full)
        if total_frame_count_full > 0 else None
    )
    rho_bar_active = (
        float(total_rate_sum_active / total_frame_count_active)
        if total_frame_count_active > 0 else None
    )
    mean_active_node_fraction = (
        float(total_active_frac_sum / total_active_frac_count)
        if total_active_frac_count > 0 else None
    )

    summary = {
        "n_matches": len(all_frame_stats),
        "n_frames_total": total_frame_count_full,
        "epsilon": EPSILON,
        "sigma": SIGMA,
        "mean_active_node_fraction": mean_active_node_fraction,
        "voronoi_baseline_contention_rate": 0.0,
        "full_map": {
            "rho_bar_contention_rate": rho_bar_full,
            "hypothesis_check": {
                "description": (
                    "H1 / RQ1: the continuous control field should classify a "
                    "greater proportion of nodes as contested than a Voronoi "
                    "segmentation, whose contention rate is analytically 0 by "
                    "construction (every node hard-assigned to one team). "
                    "Tested over ALL 903 nodes per frame (original definition). "
                    "Retained for continuity; inflated by nodes far from every "
                    "player sitting exactly at the neutral 0.5, which falls "
                    "inside the [0.4, 0.6] contested band by construction — "
                    "see 'active_footprint' for the primary result."
                ),
                "passes": bool(rho_bar_full is not None and rho_bar_full > 0.0),
            },
        },
        "active_footprint": {
            "rho_bar_contention_rate": rho_bar_active,
            "n_frames_active": total_frame_count_active,
            "hypothesis_check": {
                "description": (
                    "H1 / RQ1 tested over nodes with (ct_mu + t_mu) > EPSILON "
                    "only, i.e. nodes within meaningful pre-sigmoid Gaussian "
                    "influence range of at least one player. This is the "
                    "primary Metric 1 result going forward — it excludes "
                    "empty-map nodes that would otherwise be counted as "
                    "'contested' purely by sitting at the neutral 0.5 "
                    "midpoint. Will be all-None until team_agg.py is updated "
                    "to persist ct_mu/t_mu. The Voronoi baseline remains 0.0 "
                    "under this restriction too — a hard assignment never "
                    "lands in [0.4, 0.6] on any subset of nodes."
                ),
                "passes": bool(rho_bar_active is not None and rho_bar_active > 0.0),
            },
        },
        "area_breakdown": area_breakdown_info,
        "per_match": per_match_means,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Contention Rate summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_match"}, indent=2))
    print(f"\nSaved full summary (incl. per-match breakdown) to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
