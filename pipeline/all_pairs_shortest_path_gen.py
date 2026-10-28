import os
import pickle
import networkx as nx
import numpy as np
import pandas as pd
from awpy.data import NAV_GRAPHS
from awpy.analytics.nav import area_distance

MAP_NAME = "de_mirage"
PATCHED_GRAPH_PATH = "outputs/de_mirage_patched_graph_v3.pkl"
G_og = NAV_GRAPHS["de_mirage"]

with open(PATCHED_GRAPH_PATH, "rb") as f:
    G_patched: nx.DiGraph = pickle.load(f)

nodes = sorted(G_patched.nodes)
area_to_idx = {a: i for i, a in enumerate(nodes)} # mapping area ids to indices -> there are skipped ints in area ids (e.g. no area with id of 51)
idx_to_area = {i: a for i, a in enumerate(nodes)} # inverse of above

N = len(nodes)
dist = np.full((N, N), np.inf, dtype=np.float64)

for u, dists in nx.all_pairs_dijkstra_path_length(G_patched, weight="weight"):
    u_idx = area_to_idx[u]
    for v, d in dists.items():
        v_idx = area_to_idx[v]
        dist[u_idx, v_idx] = d

apsp = {"distance_matrix": dist, "area_to_index_map": area_to_idx, "index_to_area_map": idx_to_area}

with open("outputs/de_mirage_apsp_dict.pkl", "wb") as f:
    pickle.dump(apsp, f)

print(dist[area_to_idx[49], area_to_idx[800]])
awpy_dist = area_distance(MAP_NAME, 41, 800, dist_type="geodesic")

# awpy_path = awpy_dist["areas"]
# patched_path = nx.dijkstra_path(G_patched, 41, 800, weight="weight")

# # print(awpy_path)
# # print(patched_path)

# import matplotlib.pyplot as plt
# # import networkx as nx
# from awpy.visualization.plot import plot_map, position_transform_all

# def path_to_xy(G, path, map_name):
#     xs, ys = [], []

#     for n in path:
#         x, y, z = G.nodes[n]["center"]
#         rx, ry, _ = position_transform_all(map_name, (x, y, z))
#         xs.append(rx)
#         ys.append(ry)

#     return xs, ys


# def plot_two_paths(G_og, G_patched, awpy_path, patched_path, map_name="de_mirage"):
#     # --- base map ---
#     fig, ax = plot_map(map_name, map_type="simpleradar", dark=True)

#     # --- convert paths ---
#     awpy_x, awpy_y = path_to_xy(G_og, awpy_path, map_name)
#     pat_x, pat_y = path_to_xy(G_patched, patched_path, map_name)

#     # --- plot AWPY path ---
#     ax.plot(
#         awpy_x,
#         awpy_y,
#         color="red",
#         linewidth=2,
#         label="AWPY (A*)"
#     )

#     # --- plot patched path ---
#     ax.plot(
#         pat_x,
#         pat_y,
#         color="cyan",
#         linewidth=2,
#         label="Patched (Dijkstra)"
#     )

#     # --- start/end markers ---
#     ax.scatter(awpy_x[0], awpy_y[0], color="white", s=80, marker="o")
#     ax.scatter(awpy_x[-1], awpy_y[-1], color="white", s=80, marker="X")

#     ax.scatter(pat_x[0], pat_y[0], color="yellow", s=80, marker="o")
#     ax.scatter(pat_x[-1], pat_y[-1], color="yellow", s=80, marker="X")

#     ax.legend()
#     ax.set_title("AWPY vs Patched Graph Path Comparison")

#     plt.savefig("outputs/path_comp.png")
#     plt.close()

#     return fig, ax

# plot_two_paths(G_og, G_patched, awpy_path, patched_path)