import os
import glob
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --/ Path consts
APSP_PATH = "data/de_mirage_apsp_dict.pkl"
POSITION_MAPS_DIR = "outputs/position_maps"
INFLUENCE_FIELDS_DIR = "outputs/influence_fields"

# --/ Sigma (Guassian decay parameter)
SIGMA = 500.0

# --/ load APSP
def load_apsp_dict(path: str) -> dict:
    """Load cached distance matrix and ati/ita maps
    Returns:
        "distance_matric" - np.ndarray shape (903, 903) dtype float64
        "area_to_index_map" - dict[int area_id -> int matrix_index]
        "index_to_area_map" - dict[int matrix_index -> int area_id]
    """
    with open(path, "rb") as f:
        return pickle.load(f)

# --/ Influence computation
def inf_comp(df: pd.DataFrame, dist_mat: np.ndarray, ati_map: dict, ita_map: dict, output_path: str, sigma: float=SIGMA) -> None:
    """Compute per-node influce for every player-frame row in df and incrementally write results to 
    parquet file at output_path
    For each player-frame row:\n
        1. Look up distance vector from current_areId to all 903 nodes
        2. Look up distance vector from projected_areaId to all 903 nodes
        3. Compute d_eff = (d_current + d_projected) / 2
        4. Replace inf values with 1e9
        5. Compute mu = exp(-d_eff**2) / (2 * sigma**2)
        6. Write 903-row pyarrow block to parquet file at output_path and release block
    """
    n_nodes = len(ita_map)
    node_area_ids = np.array([ita_map[i] for i in range(n_nodes)], dtype=np.int32)

    writer = None

    for _, row in df.iterrows():
        c_aid = int(row["current_areaId"])
        p_aid = int(row["projected_areaId"])

        c_dists = dist_mat[ati_map[c_aid], :]
        p_dists = dist_mat[ati_map[p_aid], :]

        d_eff = (c_dists + p_dists) / 2.0
        d_eff = np.where(np.isinf(d_eff), 1e9, d_eff)

        mu = np.exp(-(d_eff ** 2) / (2.0 * sigma ** 2))

        # build 903-row pyarrwo table
        block = pa.table({
            "match_id": pa.array([str(row["match_id"])] * n_nodes),
            "round": pa.array([int(row["round"])] * n_nodes, type=pa.int32()),
            "frame_idx": pa.array([int(row["frame_idx"])] * n_nodes, type=pa.int32()),
            "player_name": pa.array([str(row["player_name"])] * n_nodes),
            "side": pa.array([str(row["side"])] * n_nodes),
            "current_areaId": pa.array([int(row["current_areaId"])] * n_nodes, type=pa.int32()),
            "projected_areaId": pa.array([int(row["projected_areaId"])] * n_nodes, type=pa.int32()),
            "area_id": pa.array(node_area_ids, type=pa.int32()),
            "d_current": pa.array(c_dists.astype(np.float32), type=pa.float32()),
            "d_projected": pa.array(p_dists.astype(np.float32), type=pa.float32()),
            "d_eff": pa.array(d_eff.astype(np.float32), type=pa.float32()),
            "mu": pa.array(mu.astype(np.float32), type=pa.float32()),
        })

        if writer is None:
            writer = pq.ParquetWriter(output_path, block.schema)
        writer.write_table(block)

    if writer is not None:
        writer.close()

# --/ Loop over full dataset
def main() -> None:
    apsp = load_apsp_dict(APSP_PATH)
    dist_mat = apsp["distance_matrix"]
    ati_map = apsp["area_to_index_map"]
    ita_map = apsp["index_to_area_map"]

    os.makedirs(INFLUENCE_FIELDS_DIR, exist_ok=True)

    position_map_files = sorted(glob.glob(os.path.join(POSITION_MAPS_DIR, "*.csv")))
    print(f"Found {len(position_map_files)} position map CSV files in {POSITION_MAPS_DIR} \n")

    for position_map in position_map_files:
        mname = os.path.splitext(os.path.basename(position_map))[0]
        output_path = os.path.join(INFLUENCE_FIELDS_DIR, f"{mname}.parquet")

        # skip if already exists check
        if os.path.exists(output_path):
            print(f"Skipping {mname} - already exists")
            continue

        try:
            print(f"Processing demo {mname}")

            rnd_info_df = pd.read_csv(position_map)

            if "match_id" not in rnd_info_df.columns:
                raise ValueError(
                    f"'match_id' column missing from {position_map}" 
                    f"Found columns: {rnd_info_df.columns.tolist()}"
                )

            # cast area_id columns to int
            rnd_info_df["current_areaId"] = rnd_info_df["current_areaId"].astype(int)
            rnd_info_df["projected_areaId"] = rnd_info_df["projected_areaId"].astype(int)

            inf_comp(rnd_info_df, dist_mat, ati_map, ita_map, output_path, SIGMA)

            # explicitly free
            del rnd_info_df            

        except Exception as e:
            print(f" FAILED - {mname} - {e}")

            # delete partial writes
            if os.path.exists(output_path):
                os.remove(output_path)
            continue

    print("\nInfluence field computation complete")

if __name__ == "__main__":
    main()

