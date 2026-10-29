import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import imageio.v3 as imageio
import awpy.data
from awpy.analytics.nav import area_distance

CSV_PATH = "outputs/player_positions_projected.csv"
PATCHED_GRAPH_PATH = "data/de_mirage_patched_graph_v3.pkl"

def run_checks(df: pd.DataFrame, g: nx.DiGraph):
    # --- / Check 1 - Structural verification
    print("=" * 70)
    print("CHECK 1 - STRUCTURAL VERIFICATION")
    print("=" * 70)
    print(f"Does the DataFrame contain null values: {df.isnull().values.any()}")
    print(f"DataFrane Column Data Types: {df.dtypes}")
    print(f"DataFrame Shape: {df.shape}")
    print()

    # --- / Check 2 - projection math verification
    print("=" * 70)
    print("CHECK 2 - PROJECTION CALCULATION CORRECTNESS")
    print("=" * 70)
    sample_count = 20
    sample_df = df.sample(sample_count, random_state=42)
    assert np.allclose(sample_df["projected_x"], sample_df["current_x"] + sample_df["vx"])
    assert np.allclose(sample_df["projected_y"], sample_df["current_y"] + sample_df["vy"])
    assert np.allclose(sample_df["projected_z"], sample_df["current_z"] + sample_df["vz"])
    print()
    print(f"Velocity Description:\n {df[['vx', 'vy', 'vz']].describe()}")
    print()

    # --- / Check 3 - Area IDs actually exist in graph
    print("=" * 70)
    print("CHECK 3 - AREA ID VALIDATION")
    print("=" * 70)
    valid_nodes = set(g.nodes())
    current_ids = set(df["current_areaId"].unique())
    projected_ids = set(df["projected_areaId"].unique())

    new_current = current_ids - valid_nodes
    new_projected = projected_ids - valid_nodes

    print(f"Unkown Current Area IDs: {new_current}")
    print(f"Unkown Projected Area IDs: {new_projected}")
    print()

    # -- / Check 4 - reaslistic posiotion projections
    print("=" * 70)
    print("CHECK 4 - POSITION PROJECTION FEASIBILITY")
    print("=" * 70)
    dx = df["projected_x"] - df["current_x"]
    dy = df["projected_y"] - df["current_y"]
    dz = df["projected_z"] - df["current_z"]
    disp = np.sqrt(dx**2 + dy**2 + dz**2)
    print(f"Description of projection distances:\n{disp.describe()}")
    print()

    sample_count = 200
    sample_df = df.sample(sample_count, random_state=42)
    hop_dists = []
    for _, row in sample_df.iterrows():
        try:
            d = area_distance("de_mirage", row["current_areaId"], row["projected_areaId"], dist_type="graph")
            hop_dists.append(d["distance"])
        except Exception:
            hop_dists.append(float("inf"))
    
    print(f"Description of number of hops between current and projected areas:\n{pd.Series(hop_dists).describe()}")
    print()
    inf_rows = sample_df[pd.Series(hop_dists, index=sample_df.index) == float("inf")]
    print(inf_rows[["current_areaId", "projected_areaId"]].value_counts())
    print()

    # Check 5 - Stationary players stay in same area
    print("=" * 70)
    print("CHECK 5 - STATIONARY PLAYERS STAY IN SAME AREA")
    print("=" * 70)
    stationary = df[np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2) < 5.0]
    mismatched = stationary[stationary["current_areaId"] != stationary["projected_areaId"]]
    print(f"Stationary players with different current/projected area IDs: {len(mismatched)} / {len(stationary)}")
    print(f"Mismatch Rate: {len(mismatched) / len(stationary):.2%}")
    print()

df = pd.read_csv(CSV_PATH)
with open(PATCHED_GRAPH_PATH, "rb") as f:
    graph = pickle.load(f)

awpy.data.NAV_GRAPHS["de_mirage"] = graph
run_checks(df, graph)