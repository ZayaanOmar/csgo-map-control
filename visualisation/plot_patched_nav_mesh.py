"""
plot_patched_nav_mesh.py

Visualizes the PATCHED de_mirage nav mesh (tiles, original edges and the bridge edges added by
pipeline/patch_de_mirage_graph_v3.py).

Notes:
  - Loads the patched graph from data/de_mirage_patched_graph_v3.pkl instead of pulling
    NAV_GRAPHS[map_name] directly from awpy.data (which still holds awpy's
    original, orphaned graph -- this script does NOT monkeypatch that global,
    it just uses the patched graph object on its own for plotting).
  - Tile geometry (bounding boxes) still comes from awpy.data.NAV, since the
    patch only added edges -- it never touched area bounding boxes/areaName,
    so NAV itself doesn't need patching.
  - Since the patched graph is a single connected component, the original
    component-coloring scheme (main vs. disconnected) is no longer
    informative on its own. Instead, this script reads data/de_mirage_bridge_log_v3.json
    (produced by patch_de_mirage_graph_v3.py) and highlights:
      - tiles that were orphans before patching (solid color fill, by type:
        main-mesh-bridge target side vs. orphan side)
      - the specific bridge edges that were added (drawn in a distinct color
        and thicker than the original mesh edges, so they stand out from the
        existing 2979 edges)

Run from the project root inside the project venv:
    python3 visualisation/plot_patched_nav_mesh.py
Output: outputs/nav_mesh/mirage_navmesh_patched.png
"""

import json
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

from awpy.data import NAV
from awpy.visualization.plot import plot_map, position_transform

MAP_NAME = "de_mirage"
MAP_TYPE = "simpleradar"
DARK = True

PATCHED_GRAPH_PATH = "data/de_mirage_patched_graph_v3.pkl"
EDGE_LOG_PATH = "data/de_mirage_bridge_log_v3.json"
OUT_PATH = "outputs/nav_mesh/mirage_navmesh_patched.png"


def load_patched_graph() -> nx.DiGraph:
    with open(PATCHED_GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def load_edge_log() -> list:
    with open(EDGE_LOG_PATH, "r") as f:
        return json.load(f)


def get_patch_info(edge_log: list) -> tuple[set, set]:
    """Returns (orphan_area_ids, bridge_edge_pairs) from the saved edge log.

    orphan_area_ids: the set of area IDs that were orphans before patching
        (i.e. the "orphan" side of every main_mesh_bridge entry).
    bridge_edge_pairs: set of frozenset({a, b}) pairs representing the actual
        bridge edges added, so we can draw them distinctly regardless of
        which direction they're iterated in when walking graph.edges().
    """
    orphan_ids = set()
    bridge_pairs = set()
    for entry in edge_log:
        if entry.get("type") == "main_mesh_bridge":
            orphan_ids.add(entry["orphan_area_id"])
            bridge_pairs.add(frozenset({entry["orphan_area_id"], entry["bridged_to_area_id"]}))
        elif entry.get("type") == "internal_mst":
            orphan_ids.add(entry["node_a"])
            orphan_ids.add(entry["node_b"])
            bridge_pairs.add(frozenset({entry["node_a"], entry["node_b"]}))
    return orphan_ids, bridge_pairs


def plot_patched_nav_mesh(map_name: str = MAP_NAME, map_type: str = MAP_TYPE, dark: bool = DARK):
    nav_areas = NAV[map_name]
    graph = load_patched_graph()
    edge_log = load_edge_log()
    orphan_ids, bridge_pairs = get_patch_info(edge_log)

    components = list(nx.weakly_connected_components(graph))
    print(f"Patched graph: {len(components)} weakly connected component(s)")
    print(f"Previously-orphaned tiles being highlighted: {len(orphan_ids)}")
    print(f"Bridge/patch edges being highlighted: {len(bridge_pairs)}")

    fig, ax = plot_map(map_name=map_name, map_type=map_type, dark=dark)

    # --- Draw tiles ---
    for area_id, area in nav_areas.items():
        if area_id in orphan_ids:
            # Previously orphaned -- now patched in. Flag clearly.
            face_color = "lime"
            edge_color = "lime"
            alpha = 0.55
            lw = 0.3
            zorder = 3
        else:
            face_color = "yellow"
            edge_color = "yellow"
            alpha = 0.4
            lw = 0.25
            zorder = 2

        nx1 = position_transform(map_name, area["northWestX"], "x")
        ny1 = position_transform(map_name, area["northWestY"], "y")
        sx2 = position_transform(map_name, area["southEastX"], "x")
        sy2 = position_transform(map_name, area["southEastY"], "y")

        x0, x1 = min(nx1, sx2), max(nx1, sx2)
        y0, y1 = min(ny1, sy2), max(ny1, sy2)

        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=lw,
            edgecolor=edge_color,
            facecolor=face_color,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(rect)

    # --- Draw original mesh edges (faint, as before) ---
    drawn_pairs = set()
    for u, v in graph.edges():
        if u not in nav_areas or v not in nav_areas:
            continue
        pair = frozenset({u, v})
        if pair in bridge_pairs:
            continue  # drawn separately below, in a distinct style
        if pair in drawn_pairs:
            continue  # graph is directed -> avoid drawing each undirected pair twice
        drawn_pairs.add(pair)

        cu = graph.nodes[u]["center"]
        cv = graph.nodes[v]["center"]
        ux = position_transform(map_name, cu[0], "x")
        uy = position_transform(map_name, cu[1], "y")
        vx = position_transform(map_name, cv[0], "x")
        vy = position_transform(map_name, cv[1], "y")

        ax.plot([ux, vx], [uy, vy], color="cyan", linewidth=0.25, alpha=0.4, zorder=1)

    # --- Draw bridge/patch edges distinctly, on top of everything ---
    for pair in bridge_pairs:
        u, v = tuple(pair)
        if u not in graph.nodes or v not in graph.nodes:
            continue
        cu = graph.nodes[u]["center"]
        cv = graph.nodes[v]["center"]
        ux = position_transform(map_name, cu[0], "x")
        uy = position_transform(map_name, cu[1], "y")
        vx = position_transform(map_name, cv[0], "x")
        vy = position_transform(map_name, cv[1], "y")

        ax.plot([ux, vx], [uy, vy], color="orange", linewidth=0.3, alpha=0.95, zorder=4)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(
        f"{map_name} patched nav mesh — {len(components)} component(s), "
        f"{len(orphan_ids)} previously-orphaned tiles patched in "
        f"(lime fill = patched tile, orange line = patch edge)"
    )

    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_patched_nav_mesh()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=600, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")
