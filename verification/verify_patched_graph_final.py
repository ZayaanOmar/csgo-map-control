"""
verify_patched_graph_final.py

Rigorous, non-visual verification of de_mirage_patched_graph.pkl (produced by
patch_de_mirage_graph_v3.py). This goes beyond "is it one component" and
checks every structural property that Task 3 (player position -> node
mapping) will implicitly depend on. Run this once and keep the printed
output as your record that the graph is sound before building anything on
top of it.

Checks performed:
  1. Node set integrity -- the patched graph's nodes exactly match
     awpy.data.NAV["de_mirage"]'s area IDs (no nodes silently added/dropped).
  2. No isolated nodes -- every node has in_degree > 0 AND out_degree > 0.
  3. Single weakly connected component covering all 903 nodes.
  4. Original edges preserved -- every edge in awpy's original (unpatched)
     graph is still present with an unchanged weight in the patched graph
     (the patch should only ADD edges, never remove or reweight existing
     ones).
  5. Added edges are genuinely new -- the edge set added by the patch is
     disjoint from the original edge set (sanity check against double-
     counting / accidental no-ops).
  6. Bidirectionality of patch edges -- for every (u, v) added by the patch,
     (v, u) is also present (since the whole patching approach assumed it was
     safe to add edges bidirectionally for these specific orphans).
  7. Edge weight sanity -- every edge weight is a finite, non-negative float
     (catches NaN/inf/negative weights that would silently break
     nx.shortest_path / A* distance calculations downstream).
  8. Reachability spot-check -- every node that was an orphan in the
     ORIGINAL graph has a finite-length path to a known main-mesh reference
     node in the PATCHED graph.
  9. Weight-vs-geometry consistency -- spot-checks that edge weights roughly
     match the Euclidean distance between the two nodes' 'center' attributes
     (since that's the convention awpy itself uses, and the convention this
     project's patch edges were built to match).

Run on the VM inside the project venv:
    python3 verify_patched_graph_final.py
"""

import pickle
import sys

import networkx as nx
import numpy as np

from awpy.data import NAV, NAV_GRAPHS

MAP_NAME = "de_mirage"
PATCHED_GRAPH_PATH = "data/de_mirage_patched_graph_v3.pkl"
REFERENCE_MAIN_NODE = 513  # known main-mesh area (CTSpawn) from earlier bridge logs

results = {"passed": 0, "failed": 0}


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
    if condition:
        results["passed"] += 1
    else:
        results["failed"] += 1


def main():
    G_original = NAV_GRAPHS[MAP_NAME]
    with open(PATCHED_GRAPH_PATH, "rb") as f:
        G_patched = pickle.load(f)

    nav_area_ids = set(NAV[MAP_NAME].keys())
    original_nodes = set(G_original.nodes)
    patched_nodes = set(G_patched.nodes)

    print("=" * 70)
    print("1. NODE SET INTEGRITY")
    print("=" * 70)
    check(
        "Patched graph node count matches NAV area count",
        len(patched_nodes) == len(nav_area_ids),
        f"patched={len(patched_nodes)}, NAV={len(nav_area_ids)}",
    )
    check(
        "Patched graph nodes are an exact match for NAV area IDs",
        patched_nodes == nav_area_ids,
        f"missing={nav_area_ids - patched_nodes or 'none'}, "
        f"extra={patched_nodes - nav_area_ids or 'none'}",
    )
    check(
        "No nodes were dropped relative to the original graph",
        original_nodes.issubset(patched_nodes),
        f"dropped={original_nodes - patched_nodes or 'none'}",
    )

    print()
    print("=" * 70)
    print("2. NO ISOLATED NODES")
    print("=" * 70)
    zero_in = [n for n in G_patched.nodes if G_patched.in_degree(n) == 0]
    zero_out = [n for n in G_patched.nodes if G_patched.out_degree(n) == 0]
    check(
        "Every node has in_degree > 0",
        len(zero_in) == 0,
        f"nodes with in_degree=0: {zero_in[:10]}{'...' if len(zero_in) > 10 else ''}",
    )
    check(
        "Every node has out_degree > 0",
        len(zero_out) == 0,
        f"nodes with out_degree=0: {zero_out[:10]}{'...' if len(zero_out) > 10 else ''}",
    )

    print()
    print("=" * 70)
    print("3. SINGLE WEAKLY CONNECTED COMPONENT")
    print("=" * 70)
    components = list(nx.weakly_connected_components(G_patched))
    check(
        "Exactly one weakly connected component",
        len(components) == 1,
        f"found {len(components)} component(s), sizes="
        f"{sorted((len(c) for c in components), reverse=True)}",
    )
    if components:
        check(
            "The single component covers every node",
            len(components[0]) == G_patched.number_of_nodes(),
            f"component size={len(components[0])}, total nodes={G_patched.number_of_nodes()}",
        )

    print()
    print("=" * 70)
    print("4. ORIGINAL EDGES PRESERVED (UNCHANGED)")
    print("=" * 70)
    original_edges = set(G_original.edges())
    patched_edges = set(G_patched.edges())
    missing_original = original_edges - patched_edges
    check(
        "Every original edge is still present in the patched graph",
        len(missing_original) == 0,
        f"{len(missing_original)} original edges missing"
        + (f" (sample: {list(missing_original)[:5]})" if missing_original else ""),
    )

    weight_mismatches = []
    for u, v in original_edges & patched_edges:
        w_orig = G_original[u][v].get("weight")
        w_patch = G_patched[u][v].get("weight")
        if w_orig is not None and w_patch is not None and not np.isclose(w_orig, w_patch):
            weight_mismatches.append((u, v, w_orig, w_patch))
    check(
        "No original edge weights were altered",
        len(weight_mismatches) == 0,
        f"{len(weight_mismatches)} mismatches"
        + (f" (sample: {weight_mismatches[:5]})" if weight_mismatches else ""),
    )

    print()
    print("=" * 70)
    print("5. ADDED EDGES ARE GENUINELY NEW")
    print("=" * 70)
    added_edges = patched_edges - original_edges
    check(
        "At least one edge was actually added",
        len(added_edges) > 0,
        f"{len(added_edges)} directed edges added "
        f"({len(added_edges) // 2} bidirectional pairs, approx)",
    )

    print()
    print("=" * 70)
    print("6. BIDIRECTIONALITY OF ADDED EDGES")
    print("=" * 70)
    non_reciprocal = [(u, v) for (u, v) in added_edges if (v, u) not in patched_edges]
    check(
        "Every added edge has a reciprocal edge in the other direction",
        len(non_reciprocal) == 0,
        f"{len(non_reciprocal)} non-reciprocal added edges"
        + (f" (sample: {non_reciprocal[:5]})" if non_reciprocal else ""),
    )

    print()
    print("=" * 70)
    print("7. EDGE WEIGHT SANITY (ALL EDGES)")
    print("=" * 70)
    bad_weights = []
    for u, v, data in G_patched.edges(data=True):
        w = data.get("weight")
        if w is None or not np.isfinite(w) or w < 0:
            bad_weights.append((u, v, w))
    check(
        "Every edge weight is finite and non-negative",
        len(bad_weights) == 0,
        f"{len(bad_weights)} bad weights"
        + (f" (sample: {bad_weights[:5]})" if bad_weights else ""),
    )

    print()
    print("=" * 70)
    print("8. REACHABILITY SPOT-CHECK (FORMERLY ORPHANED NODES)")
    print("=" * 70)
    original_components = list(nx.weakly_connected_components(G_original))
    original_components.sort(key=len, reverse=True)
    original_orphans = set()
    for comp in original_components[1:]:
        original_orphans.update(comp)

    check(
        "Reference main-mesh node exists in patched graph",
        REFERENCE_MAIN_NODE in G_patched,
        f"REFERENCE_MAIN_NODE={REFERENCE_MAIN_NODE}",
    )

    unreachable = []
    for orphan in original_orphans:
        try:
            nx.shortest_path(G_patched, source=orphan, target=REFERENCE_MAIN_NODE)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            unreachable.append(orphan)
    check(
        f"All {len(original_orphans)} originally-orphaned nodes can reach "
        f"the reference main-mesh node",
        len(unreachable) == 0,
        f"{len(unreachable)} unreachable"
        + (f" (sample: {unreachable[:10]})" if unreachable else ""),
    )

    print()
    print("=" * 70)
    print("9. WEIGHT-VS-GEOMETRY CONSISTENCY (SPOT-CHECK ON ADDED EDGES)")
    print("=" * 70)
    geometry_mismatches = []
    sample_added = list(added_edges)[:200]  # cap for speed; full graph is small anyway
    for u, v in sample_added:
        w = G_patched[u][v].get("weight")
        if w is None:
            continue
        cu = np.array(G_patched.nodes[u]["center"])
        cv = np.array(G_patched.nodes[v]["center"])
        euclidean = float(np.linalg.norm(cu - cv))
        if not np.isclose(w, euclidean, atol=1.0):
            geometry_mismatches.append((u, v, w, euclidean))
    check(
        "Sampled added-edge weights match Euclidean center distance "
        "(awpy's own convention)",
        len(geometry_mismatches) == 0,
        f"{len(geometry_mismatches)} mismatches out of {len(sample_added)} sampled"
        + (f" (sample: {geometry_mismatches[:5]})" if geometry_mismatches else ""),
    )

    print()
    print("=" * 70)
    print(f"SUMMARY: {results['passed']} passed, {results['failed']} failed")
    print("=" * 70)
    if results["failed"] == 0:
        print("All checks passed. The patched graph is structurally sound.")
    else:
        print("One or more checks failed -- review before relying on this "
              "graph for Task 3.")
        sys.exit(1)


if __name__ == "__main__":
    main()
