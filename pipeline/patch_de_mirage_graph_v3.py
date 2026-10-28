"""
patch_de_mirage_graph_v3.py

Fixes de_mirage's 72 orphaned nav areas using pure geometric adjacency --
"every orphan connects to every area (orphan or main-mesh) that it actually
touches" -- rather than nearest-centroid bridging or an MST.

Key fix vs. de_mirage_patch_fixed.py: the touching test there required the
perpendicular-axis overlap to be STRICTLY GREATER than `tol`, which silently
rejects tile pairs that touch via a thin/near-zero-overlap edge (common on a
diagonally-stepped staircase, where consecutive steps are offset enough that
they only share a sliver of edge). That's why the 42-tile CT stairs orphan
group fractured into multiple smaller touching-clusters (20 total orphan
clusters instead of ~2 big, sensible ones) and produced the disconnected-
looking gaps in the screenshot.

Fix: replace the two-condition touch test with the standard axis-aligned
rectangle GAP DISTANCE formula. This gives a single number -- 0 if the
rectangles overlap or share a boundary, positive if there's a real gap -- and
treating "touching" as gap_distance <= tol cleanly covers edge-touching,
corner-touching, and small real gaps in one consistent rule, with no separate
vertical/horizontal special-casing to get subtly wrong.

This version also checks orphan-to-MAIN-MESH adjacency directly (not just
orphan-to-orphan), so an orphan with multiple real-world touching main-mesh
tiles gets edges to ALL of them, not just one nearest-centroid pick. This
satisfies "every orphan node should connect to every node adjacent to it."

A fallback nearest-centroid bridge is still applied to any orphan that, after
the geometric pass, still isn't reachable from the main mesh (e.g. a tile with
a real gap larger than `tol` to its nearest neighbor) -- this guarantees full
connectivity even if `tol` doesn't catch every legitimate case.

"""

import json
import pickle

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from awpy.data import NAV, NAV_GRAPHS

MAP_NAME = "de_mirage"
TOUCH_TOL = 2.0  # max gap (game units) between two rectangles to call them "touching"


def area_bounds(area):
    min_x = min(area["northWestX"], area["southEastX"])
    max_x = max(area["northWestX"], area["southEastX"])
    min_y = min(area["northWestY"], area["southEastY"])
    max_y = max(area["northWestY"], area["southEastY"])
    return min_x, max_x, min_y, max_y


def rect_gap_distance(bounds_a, bounds_b) -> float:
    """Standard axis-aligned rectangle gap distance. Returns 0.0 if the
    rectangles overlap or share a boundary; otherwise the straight-line
    distance of the actual gap between them. Correctly handles edge-touching,
    corner-touching, and overlapping cases without needing separate logic
    for each, unlike the original areas_touch().
    """
    ax1, ax2, ay1, ay2 = bounds_a
    bx1, bx2, by1, by2 = bounds_b

    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float(np.hypot(dx, dy))


def areas_touch(bounds_a, bounds_b, tol: float = TOUCH_TOL) -> bool:
    return rect_gap_distance(bounds_a, bounds_b) <= tol


def get_weakly_connected_components(G: nx.DiGraph) -> list:
    comps = list(nx.weakly_connected_components(G))
    comps.sort(key=len, reverse=True)
    return comps


def patch_orphans(G: nx.DiGraph):
    G = G.copy()
    nav = NAV[MAP_NAME]

    components = get_weakly_connected_components(G)
    main_nodes = set(components[0])
    all_orphans = set()
    for comp in components[1:]:
        all_orphans.update(comp)

    print(f"Main component size before patching: {len(main_nodes)}")
    print(f"Total orphan nodes before patching: {len(all_orphans)}")

    # Precompute bounding boxes once for every area in the graph (orphan AND
    # main), since each orphan needs to be checked against BOTH other orphans
    # and the main mesh.
    all_node_ids = list(G.nodes)
    bounds_lookup = {area_id: area_bounds(nav[area_id]) for area_id in all_node_ids
                     if area_id in nav}

    edge_log = []
    edges_added = set()  # frozenset pairs, to avoid duplicate work/logging

    def add_edge(a, b, reason):
        pair = frozenset({a, b})
        if pair in edges_added:
            return
        edges_added.add(pair)
        dist = float(np.linalg.norm(
            np.array(G.nodes[a]["center"]) - np.array(G.nodes[b]["center"])
        ))
        G.add_edge(a, b, weight=dist)
        G.add_edge(b, a, weight=dist)
        edge_log.append({
            "type": reason,
            "node_a": int(a),
            "node_b": int(b),
            "distance": dist,
        })

    # --- Geometric adjacency pass: every orphan vs. EVERY other area ---
    # (orphan-orphan AND orphan-main). 72 orphans x ~903 areas is ~65k pair
    # checks -- trivially fast, no need for a spatial index at this scale.
    orphan_list = list(all_orphans)
    for orphan in orphan_list:
        if orphan not in bounds_lookup:
            continue
        orphan_bounds = bounds_lookup[orphan]
        for other in all_node_ids:
            if other == orphan or other not in bounds_lookup:
                continue
            if areas_touch(orphan_bounds, bounds_lookup[other]):
                add_edge(orphan, other, "geometric_adjacency")

    n_geometric = len(edge_log)
    print(f"Geometric adjacency edges added: {n_geometric}")

    # --- Fallback: any orphan still unreachable from main mesh gets a single
    # nearest-centroid bridge, in case its real gap to every neighbor exceeds
    # TOUCH_TOL. ---
    components_after_geo = get_weakly_connected_components(G)
    main_after_geo = set(components_after_geo[0])
    still_orphaned = set()
    for comp in components_after_geo[1:]:
        still_orphaned.update(comp)

    if still_orphaned:
        print(f"Orphans remaining after geometric pass: {len(still_orphaned)} "
              f"(gap to nearest neighbor exceeds TOUCH_TOL={TOUCH_TOL}; "
              f"applying fallback nearest-centroid bridge)")
        main_node_list = list(main_after_geo)
        main_centroids = np.array([G.nodes[n]["center"] for n in main_node_list])
        main_tree = cKDTree(main_centroids)

        for orphan in still_orphaned:
            orphan_center = np.array(G.nodes[orphan]["center"])
            dist, idx = main_tree.query(orphan_center, k=1)
            target = main_node_list[idx]
            add_edge(orphan, target, "fallback_centroid_bridge")
    else:
        print("No fallback bridges needed -- geometric adjacency alone fully "
              "connected every orphan.")

    return G, edge_log


def main():
    G_original = NAV_GRAPHS[MAP_NAME]
    print(f"Original graph: {G_original.number_of_nodes()} nodes, "
          f"{G_original.number_of_edges()} edges")

    G_patched, edge_log = patch_orphans(G_original)

    print(f"\nPatched graph: {G_patched.number_of_nodes()} nodes, "
          f"{G_patched.number_of_edges()} edges")
    print(f"Edges added (undirected pairs): {len(edge_log)}")

    final_components = list(nx.weakly_connected_components(G_patched))
    print(f"\nWeakly connected components after patching: {len(final_components)}")
    if len(final_components) == 1:
        print(f"SUCCESS: single component, {len(final_components[0])} nodes")
    else:
        sizes = sorted((len(c) for c in final_components), reverse=True)
        print(f"WARNING: still {len(final_components)} components, sizes: {sizes}")

    geo_edges = [e for e in edge_log if e["type"] == "geometric_adjacency"]
    fallback_edges = [e for e in edge_log if e["type"] == "fallback_centroid_bridge"]
    print(f"\nGeometric adjacency edges: {len(geo_edges)}")
    print(f"Fallback centroid-bridge edges: {len(fallback_edges)}")

    if fallback_edges:
        print("\nFallback bridges used:")
        for e in fallback_edges:
            print(f"  {e['node_a']} <-> {e['node_b']}, dist={e['distance']:.1f}")

    out_path = "../outputs/de_mirage_patched_graph_v3.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(G_patched, f)
    print(f"\nSaved patched graph to {out_path}")

    with open("../outputs/de_mirage_bridge_log_v3.json", "w") as f:
        json.dump(edge_log, f, indent=2)
    print("Saved edge log to /outputs/de_mirage_bridge_log.json")

    # Also emit the new edges in the SAME comma-separated, one-pair-per-line,
    # directed format used by awpy's own nav/de_mirage.txt, in case you want
    # to fold these directly into the bundled file instead of (or in addition
    # to) using the pickled graph at runtime.
    with open("../outputs/de_mirage_additional_edges.txt", "w") as f:
        for e in edge_log:
            a_key = "node_a" if "node_a" in e else "orphan_area_id" if "orphan_area_id" in e else None
            b_key = "node_b" if "node_b" in e else "bridged_to_area_id" if "bridged_to_area_id" in e else None
            a = e.get("node_a", e.get("orphan_area_id"))
            b = e.get("node_b", e.get("bridged_to_area_id"))
            # Write both directions, matching the directed-pair-per-line format.
            f.write(f"{a},{b}\n")
            f.write(f"{b},{a}\n")
    print("Saved /outputs/de_mirage_additional_edges.txt (same format as awpy's "
          "nav/de_mirage.txt -- append these lines there if you want the fix "
          "baked into the installed package instead of using the pickle)")


if __name__ == "__main__":
    main()
