import networkx as nx
import pickle
import json
import lzma
import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from awpy.analytics.nav import point_in_area
from awpy.data import NAV

MAP_NAME = "de_mirage"
PATCHED_GRAPH_PATH = "data/de_mirage_patched_graph_v3.pkl"
DEMO_FILE = "data/raw_demos/1decf556-b9d7-4085-9081-31811f0496c3.json.xz"
OUTPUT_DIR = "outputs/ztol_sweep"

Z_TOL_VALS = [200.0, 250.0, 300.0, 350.0, 400.0, 450.0]

# --- / Mapping function - same as in position_to_node_map.py
def map_position_kdtree(tree: KDTree, df: pd.DataFrame, G_patched: nx.DiGraph, node_ids: list, z_tol: float=100.0) -> pd.DataFrame:

    coords = df[["x", "y", "z"]].values

    distances, indices = tree.query(coords, k=5, workers=-1)

    area_ids = []
    mapping_dists = []

    for i in range(len(coords)):
        query_z = coords[i, 2]
        resolved = False

        candidate_ztol_nodes = []

        # --- / Collect 5 candidates that pass the z-tol filter -> nearest-first order
        for rank in range(5):
            candidate_node_id = node_ids[indices[i, rank]]
            candidate_z = G_patched.nodes[candidate_node_id]["center"][2]

            if abs(query_z - candidate_z) <= z_tol:
                candidate_ztol_nodes.append(
                    (candidate_node_id, float(distances[i, rank]))
                )
        
        # --- / From the above list, check (and choose first instance of) point_in_area
        for node, dist in candidate_ztol_nodes:
            if point_in_area(MAP_NAME, node, coords[i].tolist()):
                area_ids.append(node)
                mapping_dists.append(dist)
                resolved = True
                break          

        # --- / If no point in area but pass z-tol, fallback to first (nearest) not-in-area point
        if not resolved and candidate_ztol_nodes:
            best_node_id, best_dist = candidate_ztol_nodes[0]  
            area_ids.append(best_node_id)
            mapping_dists.append(best_dist)
            resolved = True

        # --- / Fal back to raw nearest tile from KDTree
        if not resolved:
            area_ids.append(node_ids[indices[i, 0]])
            mapping_dists.append(float(distances[i, 0]))

    result_df                    = df.copy()
    result_df["current_areaId"]   = area_ids     
    result_df["mapping_distance"] = mapping_dists
    return result_df

def run_sweeps(player_df: pd.DataFrame, tree: KDTree, G_patched: nx.DiGraph, node_ids: list, z_tol_vals: list, output_dir: str, map_name: str = "de_mirage") -> dict:
    """
    Map player positions to nodes using various z_tol
    Append used z_tol to genereated DF for comparison checks later
    save to {output_dir}/player_df_ztol_{z_tol}.csv
    """

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for z_tol in z_tol_vals:
        print(f"  z_tol = {z_tol:>6.1f} — mapping {len(player_df)} rows...", end="", flush=True)

        mapped_df = map_position_kdtree(tree=tree, df=player_df, G_patched=G_patched, node_ids=node_ids, z_tol=z_tol)

        mapped_df["z_tol"] = z_tol 

        filename = f"player_df_ztol_{z_tol:.0f}.csv"
        filepath = os.path.join(output_dir, filename)

        mapped_df.to_csv(filepath, index=False)

        unique_areas = mapped_df["current_areaId"].nunique()
        print(f"done. {unique_areas} unique area ids -> {filepath}")

        results[z_tol] = mapped_df

    return results
    
if __name__ == "__main__":
    
    # --- / Load graph and what not
    with open(PATCHED_GRAPH_PATH, "rb") as f:
        G_PATCHED = pickle.load(f)

    with lzma.open(DEMO_FILE, "rt", encoding="utf-8") as f:
        demo = json.load(f)

    rounds = demo["gameRounds"]

    node_ids = list(G_PATCHED.nodes())
    centroids = np.array([G_PATCHED.nodes[nid]["center"] for nid in node_ids])
    tree = KDTree(centroids)

    # print("\nExtracting player positions from demo frames...")
    player_info = []
    for rnd in rounds:
        for frame_idx, frame in enumerate(rnd["frames"] or []):
            for side, side_label in [("t", "T"), ("ct", "CT")]:
                for player in frame.get(side, {}).get("players", []):
                    if player["isAlive"]:
                        player_info.append({
                            "round": rnd["roundNum"],
                            "frame_idx": frame_idx,
                            "player_name": player["name"],
                            "side": side_label,
                            "x": player["x"],           
                            "y": player["y"],
                            "z": player["z"],
                            "vx": player["velocityX"],
                            "vy": player["velocityY"],
                            "vz": player["velocityZ"],
                        })

    player_df = pd.DataFrame(player_info)

    print(f"Running z_tol sweep\n")

    results = run_sweeps(player_df=player_df, tree=tree, G_patched=G_PATCHED, node_ids=node_ids, z_tol_vals=Z_TOL_VALS, output_dir=OUTPUT_DIR, map_name=MAP_NAME)

