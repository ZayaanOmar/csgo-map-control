"""
voronoi_baseline.py

Standalone Voronoi-style map control baseline for the CS:GO map control
project (de_mirage). Independent of the influence-field pipeline
(Tasks 3, 4, 6) -- the only shared inputs are the APSP distance cache
produced by Task 4 (`de_mirage_apsp_dict.pkl`) and the per-match position
CSVs produced by Task 3 (`outputs/position_maps/{match_name}.csv`). This
script does not import or depend on `inf_function.py` or `team_agg.py`.

Method
------
For each frame, every one of the 903 nav nodes is "hard-assigned" to
whichever side (CT or T) has an alive player closer to it by geodesic
graph distance -- the classic nearest-neighbour Voronoi partition, but
over the nav graph rather than raw Euclidean space. This is the
project's baseline comparison point for the continuous Gaussian
influence field: same underlying graph distance, but a hard
winner-takes-all assignment instead of a smooth exp(-d^2/2sigma^2) decay.

Distance convention -- read before changing anything here
-----------------------------------------------------------
The APSP matrix from Task 4 is DIRECTED (33.27% of finite pairs are
asymmetric -- see task4_implementation_notes.md S4.3), and that
directionality is preserved here, NOT symmetrized. This matches Task
4's own default (keep the matrix directed; only symmetrize per use
case with a specific documented reason -- see task4_implementation_
notes.md S2.3 and S6), and is what actually respects the map's
one-way geometry (drops, boosts, vents): if a player can drop from a
ledge cheaply but can only climb back up via a long detour, a node
below that ledge should read as "close" to a player standing above it,
and a node above the ledge should read as "far" from a player standing
below it. Taking min(d[u,v], d[v,u]) would silently give the bottom
player the cheap drop-distance to reach the top too, which is wrong.

Every distance lookup below is therefore `dist_mat[player_node,
target_node]` -- read directly off the raw APSP matrix as "the cost to
travel FROM the player's current node TO the target node" -- with no
min()/symmetrization step anywhere in this file. Do not add one without
updating this docstring and getting explicit sign-off, since it would
silently erase asymmetric map geometry (Ladder/Truck-style one-way
SCCs and any other directed edge) from every node's assignment.

Tie handling
------------
A node with ct_dist == t_dist (including the inf == inf case, e.g. a
node unreachable from both sides at once, as happens for the Ladder/
Truck SCCs relative to each other) is left UNRESOLVED (winner = 0).

Output -- THREE files per match, flat-by-type layout
------------------------------------------------------
1. outputs/voronoi_control_fields/{match_name}.csv   <-- primary output
    Schema-matched to team_agg.py's outputs/control_fields/{match_name}.csv
    so both can be fed into the same heatmap generator interchangeably:

        round, frame_idx, area_id, control_value

    control_value is a hard assignment rather than team_agg.py's
    continuous sigmoid, so it only ever takes three values:
        1.0  -- node strictly closer (geodesically) to an alive CT player
        0.0  -- node strictly closer to an alive T player
        0.5  -- tie or unresolved (both sides equidistant, or the node
                is unreachable from every alive player on both sides)
    0.5 is used for the neutral case rather than NaN/omission because
    it is the same neutral point team_agg.py's sigmoid produces when
    ct_mu == t_mu (expit(0) = 0.5) -- a heatmap colormap centred on 0.5
    as "contested/neutral" will therefore render both representations
    on a consistent scale.

    ASSUMPTION, stated explicitly: this assumes control_fields CSVs
    have exactly the four columns above and no match_id column (per
    the project note that match_id was deliberately removed from
    team_agg.py's output since each match already has its own file).
    If your actual control_fields schema differs, this is a one-line
    change in `build_control_field_rows()` below.

2. outputs/voronoi_node_assignments/{match_name}.parquet   <-- diagnostic
    One row per (round, frame_idx, area_id): winner (+1 CT / -1 T / 0
    tie), plus the raw ct_dist / t_dist candidate distances. Not needed
    for heatmap generation -- kept for the same kind of verification
    work done on the influence pipeline's own outputs (e.g. checking
    how often ties/unresolved nodes occur, or spot-checking a specific
    frame's assignment against the radar).

3. outputs/voronoi_metrics/{match_name}.csv   <-- diagnostic
    One row per (round, frame_idx): the area-weighted scalar control
    metric in [-1, +1] (+1 = complete CT control), for the same kind of
    round-level analysis Metric 3 does on the influence pipeline's
    output. Not needed for heatmap generation.

Usage
-----
    python voronoi_baseline.py
    python voronoi_baseline.py --positions-dir outputs/position_maps \
        --apsp-path data/de_mirage_apsp_dict.pkl

Every match already fully processed (all three output files present) is
skipped, matching the restart behaviour of the other pipeline scripts.
"""

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from awpy.data import NAV

MAP_NAME = "de_mirage"

APSP_PATH = "data/de_mirage_apsp_dict.pkl"
POSITION_MAPS_DIR = "outputs/position_maps"
CONTROL_FIELDS_DIR = "outputs/voronoi_control_fields"
NODE_ASSIGNMENTS_DIR = "outputs/voronoi_node_assignments"
METRICS_DIR = "outputs/voronoi_metrics"

# Hard-assignment control values -- see module docstring, "Output" S1.
CT_CONTROL_VALUE = 1.0
T_CONTROL_VALUE = 0.0
NEUTRAL_CONTROL_VALUE = 0.5


# ---------------------------------------------------------------------
# One-time, map-level setup (run once, reused across every match)
# ---------------------------------------------------------------------

def load_apsp(apsp_path: str = APSP_PATH) -> dict:
    """Load the Task 4 APSP cache.

    Returns the dict with keys "distance_matrix" (903x903 float64,
    directed, inf for unreachable pairs), "area_to_index_map", and
    "index_to_area_map" -- exactly as produced in Task 4.
    """
    with open(apsp_path, "rb") as f:
        apsp = pickle.load(f)
    return apsp


def compute_node_areas(ita_map: dict) -> np.ndarray:
    """Per-node XY footprint area, in the same row order as the APSP matrix.

    Uses NAV["de_mirage"][area_id]'s bounding-box corners directly
    (rather than importing awpy.analytics.nav.calculate_tile_area for a
    single formula). Same formula as awpy's own function:
    (northWestX - southEastX) * (northWestY - southEastY), abs()'d per
    the sign-convention warning in awpy_unified_reference.md S2.8 --
    the raw product can be negative depending on corner-naming
    convention and would corrupt the area-weighted sum if left signed.
    Only used for the diagnostic voronoi_metrics/ output.
    """
    n_nodes = len(ita_map)
    areas = np.zeros(n_nodes, dtype=np.float64)
    for idx in range(n_nodes):
        area_id = ita_map[idx]
        a = NAV[MAP_NAME][area_id]
        width = a["northWestX"] - a["southEastX"]
        height = a["northWestY"] - a["southEastY"]
        areas[idx] = abs(width * height)
    return areas


# ---------------------------------------------------------------------
# Per-frame assignment
# ---------------------------------------------------------------------

def assign_frame(
    ct_area_ids: list,
    t_area_ids: list,
    dist_mat: np.ndarray,
    ati_map: dict,
    n_nodes: int,
):
    """Hard-assign every node to CT, T, or neither, for a single frame.

    Parameters
    ----------
    ct_area_ids, t_area_ids : list[int]
        current_areaId values of alive players on each side in this
        frame. Either list may be empty (side fully eliminated).
    dist_mat : (903, 903) float array
        The RAW DIRECTED APSP matrix from Task 4, unmodified --
        dist_mat[u, v] is the cost to travel from node u to node v.
        See module docstring, "Distance convention", for why this is
        used as-is rather than symmetrized.
    ati_map : dict[int, int]
        area_to_index_map from the APSP cache.
    n_nodes : int
        903 for de_mirage.

    Returns
    -------
    ct_dist : (n_nodes,) float array
        For each node, the minimum directed cost FROM any alive CT
        player's current node TO that node (i.e. how quickly the
        nearest CT player could actually reach it, respecting one-way
        edges). All-inf if no CT players are alive.
    t_dist : (n_nodes,) float array
        Same for T.
    winner : (n_nodes,) int8 array
        +1 = CT strictly closer, -1 = T strictly closer, 0 = tie
        (including both-inf, i.e. unreachable from either side).
    """
    if ct_area_ids:
        ct_idx = [ati_map[a] for a in ct_area_ids]
        # dist_mat[ct_idx, :] : each row is "cost from this CT player's
        # node to every other node" -- min across rows (axis=0) gives,
        # per target node, the nearest CT player's directed travel cost.
        ct_dist = np.min(dist_mat[ct_idx, :], axis=0)
    else:
        ct_dist = np.full(n_nodes, np.inf)

    if t_area_ids:
        t_idx = [ati_map[a] for a in t_area_ids]
        t_dist = np.min(dist_mat[t_idx, :], axis=0)
    else:
        t_dist = np.full(n_nodes, np.inf)

    winner = np.zeros(n_nodes, dtype=np.int8)
    winner[ct_dist < t_dist] = 1
    winner[t_dist < ct_dist] = -1
    # ct_dist == t_dist (including inf == inf) stays 0 -- see module
    # docstring, "Tie handling".

    return ct_dist, t_dist, winner


def winner_to_control_value(winner: np.ndarray) -> np.ndarray:
    """Map the +1/-1/0 winner array to the control_value scale.

    1.0 = CT, 0.0 = T, 0.5 = tie/unresolved. See module docstring,
    "Output" S1, for why 0.5 was chosen for the neutral case.
    """
    control_value = np.full(winner.shape, NEUTRAL_CONTROL_VALUE, dtype=np.float64)
    control_value[winner == 1] = CT_CONTROL_VALUE
    control_value[winner == -1] = T_CONTROL_VALUE
    return control_value


# ---------------------------------------------------------------------
# Per-match processing
# ---------------------------------------------------------------------

def process_match(
    positions_csv: str,
    dist_mat: np.ndarray,
    ati_map: dict,
    ita_map: dict,
    node_areas: np.ndarray,
    control_field_out_path: str,
    node_out_path: str,
    metric_out_path: str,
) -> None:
    """Run the Voronoi baseline over every frame of one match.

    Writes the diagnostic node-assignment parquet incrementally (one
    903-row block per frame, via ParquetWriter) so peak memory for that
    output stays at a single block regardless of match length -- same
    pattern as `inf_comp` in Task 6, applied here independently. The
    control_value CSV and the scalar metrics CSV are both accumulated
    in memory and written once at the end, since they are the same
    order of magnitude as team_agg.py's own control_fields output
    (one row per (round, frame_idx, area_id) for the former, one row
    per (round, frame_idx) for the latter) -- small enough per match
    that this does not reintroduce the OOM problem Task 6 solved for
    the much larger per-player-per-node influence parquets.
    """
    n_nodes = len(ita_map)
    node_area_ids = np.array([ita_map[i] for i in range(n_nodes)], dtype=np.int32)

    df = pd.read_csv(positions_csv)

    required_cols = {"match_id", "round", "frame_idx", "side", "current_areaId"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = (
            f"{positions_csv}: missing required columns {missing}. "
            "Voronoi baseline needs match_id, round, frame_idx, side, "
            "and current_areaId from the Task 3 position_maps output."
        )
        raise ValueError(msg)

    # Guard against the float64-after-CSV-round-trip issue documented
    # in task3_implementation_notes.md (Bug 8) -- area IDs must be int
    # before being used as ati_map dict keys.
    df["current_areaId"] = df["current_areaId"].astype(int)

    match_id = str(df["match_id"].iloc[0])
    total_area = float(node_areas.sum())

    node_writer = None
    control_field_chunks = []
    metric_rows = []

    for (rnd, frame_idx), frame_df in df.groupby(["round", "frame_idx"], sort=False):
        ct_area_ids = frame_df.loc[frame_df["side"] == "CT", "current_areaId"].tolist()
        t_area_ids = frame_df.loc[frame_df["side"] == "T", "current_areaId"].tolist()

        ct_dist, t_dist, winner = assign_frame(
            ct_area_ids, t_area_ids, dist_mat, ati_map, n_nodes
        )
        control_value = winner_to_control_value(winner)

        # --- (1) primary output: control_fields-schema-matched CSV rows ---
        control_field_chunks.append(
            pd.DataFrame(
                {
                    "round": np.full(n_nodes, int(rnd), dtype=np.int32),
                    "frame_idx": np.full(n_nodes, int(frame_idx), dtype=np.int32),
                    "area_id": node_area_ids,
                    "control_value": control_value,
                }
            )
        )

        # --- (2) diagnostic: node-level assignment parquet ---
        block = pa.table(
            {
                "match_id": pa.array([match_id] * n_nodes),
                "round": pa.array([int(rnd)] * n_nodes, type=pa.int32()),
                "frame_idx": pa.array([int(frame_idx)] * n_nodes, type=pa.int32()),
                "area_id": pa.array(node_area_ids, type=pa.int32()),
                "ct_dist": pa.array(ct_dist.astype(np.float32), type=pa.float32()),
                "t_dist": pa.array(t_dist.astype(np.float32), type=pa.float32()),
                "winner": pa.array(winner, type=pa.int8()),
            }
        )
        if node_writer is None:
            node_writer = pq.ParquetWriter(node_out_path, block.schema)
        node_writer.write_table(block)

        # --- (3) diagnostic: frame-level scalar metric ---
        ct_area = float(node_areas[winner == 1].sum())
        t_area = float(node_areas[winner == -1].sum())
        resolved_area = ct_area + t_area

        if resolved_area > 0:
            metric = (ct_area - t_area) / resolved_area
        else:
            # No node was strictly closer to either side -- should not
            # occur mid-round with at least one alive player total, but
            # guard against divide-by-zero rather than assume it can't.
            metric = 0.0

        metric_rows.append(
            {
                "match_id": match_id,
                "round": int(rnd),
                "frame_idx": int(frame_idx),
                "n_ct_alive": len(ct_area_ids),
                "n_t_alive": len(t_area_ids),
                "ct_area": ct_area,
                "t_area": t_area,
                "unresolved_area": total_area - resolved_area,
                "control_metric": metric,
            }
        )

    if node_writer is not None:
        node_writer.close()

    pd.concat(control_field_chunks, ignore_index=True).to_csv(
        control_field_out_path, index=False
    )
    pd.DataFrame(metric_rows).to_csv(metric_out_path, index=False)


# ---------------------------------------------------------------------
# Full-dataset loop
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voronoi (nearest-player, hard-assignment) map-control "
        "baseline for de_mirage -- independent of the influence-field pipeline. "
        "Primary output matches team_agg.py's control_fields CSV schema so "
        "both can be fed into the same heatmap generator."
    )
    parser.add_argument("--apsp-path", default=APSP_PATH)
    parser.add_argument("--positions-dir", default=POSITION_MAPS_DIR)
    parser.add_argument("--control-field-out-dir", default=CONTROL_FIELDS_DIR)
    parser.add_argument("--node-out-dir", default=NODE_ASSIGNMENTS_DIR)
    parser.add_argument("--metric-out-dir", default=METRICS_DIR)
    args = parser.parse_args()

    os.makedirs(args.control_field_out_dir, exist_ok=True)
    os.makedirs(args.node_out_dir, exist_ok=True)
    os.makedirs(args.metric_out_dir, exist_ok=True)

    apsp = load_apsp(args.apsp_path)
    dist_mat = apsp["distance_matrix"]  # used as-is -- directed, not symmetrized
    ati_map = apsp["area_to_index_map"]
    ita_map = apsp["index_to_area_map"]

    node_areas = compute_node_areas(ita_map)

    positions_files = sorted(glob.glob(os.path.join(args.positions_dir, "*.csv")))
    if not positions_files:
        print(f"No position CSVs found in {args.positions_dir}")
        return

    for positions_csv in positions_files:
        match_name = os.path.splitext(os.path.basename(positions_csv))[0]
        control_field_out_path = os.path.join(
            args.control_field_out_dir, f"{match_name}.csv"
        )
        node_out_path = os.path.join(args.node_out_dir, f"{match_name}.parquet")
        metric_out_path = os.path.join(args.metric_out_dir, f"{match_name}.csv")

        already_done = (
            os.path.exists(control_field_out_path)
            and os.path.exists(node_out_path)
            and os.path.exists(metric_out_path)
        )
        if already_done:
            print(f"Skipping {match_name} -- already processed")
            continue

        try:
            process_match(
                positions_csv,
                dist_mat,
                ati_map,
                ita_map,
                node_areas,
                control_field_out_path,
                node_out_path,
                metric_out_path,
            )
            print(f"Done: {match_name}")
        except Exception as e:  # noqa: BLE001 -- broad by design, see inf_function.py precedent
            print(f"FAILED: {match_name} -- {e}")
            # Delete any partial output so a retry doesn't get skipped
            # by the exists() check above -- same reasoning as
            # inf_function.py's partial-parquet cleanup.
            for path in (control_field_out_path, node_out_path, metric_out_path):
                if os.path.exists(path):
                    os.remove(path)
            continue


if __name__ == "__main__":
    main()
