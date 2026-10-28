import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from awpy.visualization.plot import plot_map, position_transform_all

APSP_PATH = "outputs/de_mirage_apsp_dict.pkl"
PATCHED_GRAPH_PATH = "outputs/de_mirage_patched_graph_v3.pkl"

results = {"passed": 0, "failed": 0}

# --- / Helper function - define structure of verification tests
def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
    if condition:
        results["passed"] += 1
    else:
        results["failed"] += 1

def main():
    
    with open(APSP_PATH, "rb") as f:
        apsp_dict: dict = pickle.load(f)
    
    with open(PATCHED_GRAPH_PATH, "rb") as f:
        G_patched = pickle.load(f)

    dist_matrix: np.matrix = apsp_dict["distance_matrix"]
    area_to_index = apsp_dict["area_to_index_map"]
    idx_to_area = apsp_dict["index_to_area_map"]

    num_inf = np.isinf(dist_matrix).sum()
    print(num_inf)

    # --- / DATA INTEGRITY CHECKS (SHAPE, DIAGONAL 0s, etc.)
    print("=" * 70)
    print("CHECK 1 : DATA INTEGRITY")
    print("=" * 70)
    check(
        "Dist matrix should have shape (903x903)", 
        np.shape(dist_matrix) == (903, 903)
    )
    check(
        "Diagonal entries should be 0",
        not np.any(dist_matrix.diagonal())
    )
    check(
        "Number of inf entries should be 3 (3 strongly connected components)",
        np.count_nonzero(dist_matrix == np.inf) == 6296, # comes from condenstation matrix
                                                         # 3 scc's, of length 894, 4, 3. edges from 896->4,3 but not vice-versa
        f"Number of inf: {np.count_nonzero(dist_matrix == np.inf)}"
    )
    check(
        "No negative distances",
        not np.any(dist_matrix[np.isfinite(dist_matrix)]) < 0
    )

if __name__ == "__main__":
    main()