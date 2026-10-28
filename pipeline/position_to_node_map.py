import os
import re
import csv
import glob
import json
import lzma
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import KDTree

# --/ Paths
PATCHED_GRAPH_PATH = "data/de_mirage_patched_graph_v3.pkl"
APSP_PATH = "data/de_mirage_apsp_dict.pkl"
FILTERED_DEMOS_DIR = "data/filtered_demos"
POSITION_MAPS_DIR = "outputs/position_maps"
ROUND_METADATA_PATH = "outputs/round_metadata"
BOMB_PLANTS_PATH = "outputs/bomb_plants"

MAP_NAME = "de_mirage"

# --/ tunable consts
Z_TOL = 250.0
PROJ_THRESH = 350.0
KDTREE_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

# --/ Helper functions
def sanitise(name: str) -> str:
    """Replace non-alphanumeric characters with _"""
    return re.sub(r"[^A-Za-z0-9]", "_", name)

def create_match_name(ct: str, t: str, match_id: str) -> str:
    """Create canonical name for a match - ct_VS_t_matchId"""
    return f"{sanitise(ct)}_VS_{sanitise(t)}_{sanitise(match_id)}"

def write_csv(rows: list[dict], path: str, fieldnames: list[str]):
    """Save csv to path"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# --/ construct kd-tree
def build_kd_tree(graph: nx.DiGraph) -> tuple[KDTree, list]:
    """Build KDTree over 3D centroid coordinates of all nodes in graph
    Returns (tree, node_ids) with node_ids[i] coresopnding to row i in the KDTree's array
    """
    node_ids = list(graph.nodes())
    centroids = np.array([graph.nodes[nid]["center"] for nid in node_ids])
    return KDTree(centroids), node_ids

# --/ Position map
def map_position_kdtree(tree: KDTree, df: pd.DataFrame, G_patched: nx.DiGraph, node_ids: list, z_tol: float=Z_TOL, prefix: str="current_") -> None:
    """Map player positions to the nearest nav area using nearest-centroid based assignment with z-tolerance checking
    For each query point, the k nearest node centroids are fetched from the KDTree. The assigned node is the nearest candidate whose centroid z is within z_tol
    of the query point's z. Raw nearest centroid chosen as fallback.
    """
    x_col, y_col, z_col = f"{prefix}x", f"{prefix}y", f"{prefix}z"
    coords = df[[x_col, y_col, z_col]].values

    distances, indices = tree.query(coords, k=5, workers=KDTREE_WORKERS)

    area_ids = []
    mapping_dists = []

    for i in range(len(coords)):
        query_z = coords[i, 2]
        resolved = False

        # z-tol check
        for rank in range(5):
            cand_id = node_ids[indices[i, rank]]
            cand_z = G_patched.nodes[cand_id]["center"][2]

            if abs(query_z - cand_z) <= z_tol:
                area_ids.append(cand_id)
                mapping_dists.append(float(distances[i, rank]))
                resolved = True
                break

        # raw fallback - no passing z_tol check
        if not resolved:
            area_ids.append(node_ids[indices[i, 0]])
            mapping_dists.append(float(distances[i, 0]))

    df[f"{prefix}areaId"] = area_ids
    df[f"{prefix}mapping_distance"] = mapping_dists

# --/ Projection clamping
def clamp_projection(c_aid: int, p_aid: int, dist_matrix: np.ndarray, ati_map: dict, threshold: float=PROJ_THRESH) -> int:
    """return p_aid if graph distance from current node (c_aid) to projected node (p_aid) if within threshold
    return c_aid otherwise
    """
    dist = dist_matrix[ati_map[c_aid], ati_map[p_aid]]
    return c_aid if dist > threshold else p_aid

# --/ Per demo processing
def process_demo(demo: dict, tree: KDTree, node_ids: list, G_patched: nx.DiGraph, dist_matrix: np.ndarray, ati_map: dict) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Extract, project, and map all player positions from one demo
    Returns:
        player_df: one row per alive player per frame, with current and project positions mapped to nav node ids
        round_metadata: list of dicts for round_metadata table
        plant_events: list of dicts for plant_events table
    """
    match_id: str = demo["matchId"]
    rounds = demo["gameRounds"]

    player_info = []
    round_metadata = []
    plant_events = []

    for rnd in rounds:
        # Safety check
        if rnd.get("isWarmup", False):
            continue

        round_num = rnd["roundNum"]
        freeze_time_end_tick = rnd["freezeTimeEndTick"]
        end_tick = rnd["endTick"]

        round_metadata.append({
            "match_id": match_id,
            "round": round_num,
            "winning_side": rnd["winningSide"],
            "ct_team": rnd.get("ctTeam", ""),
            "t_team": rnd.get("tTeam", ""),
            "round_end_reason": rnd["roundEndReason"],
            "end_tick": end_tick,
            "freeze_time_end_tick": freeze_time_end_tick,
        })

        for event in rnd.get("bombEvents") or []:
            if event["bombAction"] == "plant":
                plant_events.append({
                    "match_id": match_id,
                    "round": round_num,
                    "bomb_site": event["bombSite"],
                    "tick": event["tick"],
                    "seconds": event["seconds"],
                    "planter_name": event["playerName"],
                })

        # per-frame player info extraction
        for frame_idx, frame in enumerate(rnd.get("frames") or []):
            for side, side_label in [("t", "T"), ("ct", "CT")]:
                for player in frame.get(side, {}).get("players", []):
                    if not player["isAlive"]:
                        continue

                    player_info.append({
                        "match_id": match_id,
                        "round": rnd["roundNum"],
                        "frame_idx": frame_idx,
                        "player_name": player["name"],
                        "side": side_label,
                        "current_x": player["x"], "current_y": player["y"], "current_z": player["z"],
                        "vx": player["velocityX"], "vy": player["velocityY"], "vz": player["velocityZ"],
                        "tick": frame["tick"],
                        "bomb_x": frame["bomb"]["x"],
                        "bomb_y": frame["bomb"]["y"],
                        "bomb_z": frame["bomb"]["z"],
                        "bomb_planted": frame["bombPlanted"],
                        "bomb_site": frame["bombsite"]
                    })

    player_df = pd.DataFrame(player_info)

    if player_df.empty:
        return player_df, round_metadata, plant_events

    # project positions forward one second
    player_df["projected_x"] = player_df["current_x"] + player_df["vx"]
    player_df["projected_y"] = player_df["current_y"] + player_df["vy"]
    player_df["projected_z"] = player_df["current_z"] + player_df["vz"]

    # map current and projected positions to nav areas
    map_position_kdtree(tree, player_df, G_patched, node_ids, z_tol=Z_TOL, prefix="current_")
    map_position_kdtree(tree, player_df, G_patched, node_ids, z_tol=Z_TOL, prefix="projected_")

    # clamp any projected overshoots
    player_df["projected_areaId"] = player_df.apply(
        lambda row: clamp_projection(
            int(row["current_areaId"]), 
            int(row["projected_areaId"]), 
            dist_matrix,
            ati_map, 
            PROJ_THRESH
        ),
        axis=1
    )

    return player_df, round_metadata, plant_events

# --/ main loop over entire dataset
def main() -> None:
    # load map constants
    with open(PATCHED_GRAPH_PATH, "rb") as f:
        G_patched = pickle.load(f)

    with open(APSP_PATH, "rb") as f:
        apsp = pickle.load(f)

    dist_matrix = apsp["distance_matrix"]
    ati_map = apsp["area_to_index_map"]

    tree, node_ids = build_kd_tree(G_patched)

    os.makedirs(POSITION_MAPS_DIR, exist_ok=True)
    os.makedirs(ROUND_METADATA_PATH, exist_ok=True)
    os.makedirs(BOMB_PLANTS_PATH, exist_ok=True)

    demo_files = sorted(glob.glob(os.path.join(FILTERED_DEMOS_DIR, "*.json.xz")))
    print(f"Found {len(demo_files)} filters demos in {FILTERED_DEMOS_DIR}\n")

    for demo_file in demo_files:
        # derive match name from file name
        # {ct_team}_VS_{t_team}_{match_id}
        mname = os.path.basename(demo_file).replace(".json.xz", "")

        positions_out = os.path.join(POSITION_MAPS_DIR, f"{mname}.csv")
        round_metadata_out = os.path.join(ROUND_METADATA_PATH, f"{mname}.csv")
        plant_events_out = os.path.join(BOMB_PLANTS_PATH, f"{mname}.csv")

        # skip if already exist for safe restarting of failed jobs
        if os.path.exists(positions_out):
            print(f"Skipping position mapping {mname} - already exists")
            continue

        try:
            print(f"Processing demo {mname}")

            with lzma.open(demo_file, "rt", encoding="utf-8") as f:
                demo: dict = json.load(f)

            player_df, round_metadata, plant_events = process_demo(demo, tree, node_ids, G_patched, dist_matrix, ati_map)

            if player_df.empty:
                print(f"\n WARNING - no player rows produced - skipping demo\n")
                continue

            # Write position map to CSV
            player_df.to_csv(positions_out, index=False)

            # Write round_metadata and plant_events if they dont exist
            if not os.path.exists(round_metadata_out):
                write_csv(round_metadata, round_metadata_out, fieldnames=["match_id", "round", "winning_side", "ct_team", "t_team", 
                                                                   "round_end_reason", "end_tick", "freeze_time_end_tick"])
                print(f"Round metadata written to {round_metadata_out}")

            if not os.path.exists(plant_events_out):
                write_csv(plant_events, plant_events_out, fieldnames=["match_id", "round", "bomb_site", "tick", "seconds", "planter_name"])            
                print(f"Plant events written to {plant_events_out} | Number of plants: {len(plant_events)}")

            # explicit memory clean up
            del demo, player_df, round_metadata, plant_events

        except Exception as e:
            print(f"\n FAILED: {mname} - {e}\n")
            # Remove any partially written outputs
            if os.path.exists(positions_out):
                os.remove(positions_out)
            continue

    print(f"\nPosition mapping complete")

if __name__ == "__main__":
    main()

