# Task 4 — All-Pairs Shortest Path Caching: Implementation Notes

Reference document for the CS:GO map control project. Describes the design
decisions, implementation approach, verification methodology, and confirmed
findings for Task 4 (all-pairs shortest-path caching over the patched
de_mirage nav graph), so this does not need to be re-derived in a future
session.

---

## 1. Purpose and motivation

The influence-field formula at the core of this project is:

```
µᵢ(x,t) = exp(−d_eff(i,x,t)² / 2σ²)
d_eff(i,x,t) = (d_G(pᵢ,x) + d_G(p̂ᵢ,x)) / 2
```

where `d_G` is the graph shortest-path distance over the de_mirage nav mesh.
Evaluating `µᵢ(x,t)` for a single frame requires computing `d_G(player, node)`
for every alive player and every one of the 903 nav nodes — then repeating
this across every frame of every round of every demo in the ESTA dataset.
Doing this via live shortest-path calls (`nx.dijkstra_path_length` or
`nx.astar_path` on each query) would mean tens of millions of live graph
searches, making the pipeline computationally intractable.

Task 4 solves this by pre-computing the **full 903×903 distance matrix once**,
over the patched de_mirage graph, and caching it to disk. Task 5 and all
downstream pipeline steps then load this matrix once per run and perform
O(1) lookups — `dist[area_to_idx[u], area_to_idx[v]]` — instead of live
graph traversals.

This is exactly the gap that `AREA_DIST_MATRIX` fills in awpy's own
architecture — and confirmed empty for de_mirage in the bundled install,
meaning every live `area_distance()` call for de_mirage was taking the slow
path. Task 4 fills that gap on the **patched** graph.

---

## 2. Prerequisites

Two decisions had to be made explicitly before building the matrix:

### 2.1 Which graph to use

**Decision: load `de_mirage_patched_graph_v3.pkl` directly via `pickle.load()`
(Option A from the patch summary).**

Running APSP on the unpatched `NAV_GRAPHS["de_mirage"]` would bake `inf`
distances to all 72 formerly-orphaned nodes permanently into the cached file —
corrupting the matrix for the full pipeline run. The pickle is the most
reproducible approach and avoids any risk of the monkeypatched state being
stale or inconsistent across script runs. This matches the standing Option A
convention from the graph patch work.

### 2.2 Which distance metric

**Decision: geodesic distance (weighted shortest path, summing edge `weight`
attributes via Dijkstra).**

The influence decay formula `exp(−d²/2σ²)` requires `σ` to be calibrated in
some unit. Geodesic distance preserves the physical scale of the map (edge
weights are Euclidean centroid-to-centroid distances in game units, set by
awpy's own `create_nav_graphs` convention). Hop-count distance (`"graph"` mode
in awpy) would make `σ` dimensionless and meaningless — two corridors with
identical hop counts but very different physical lengths would produce identical
influence decay, which is wrong. The geodesic choice is also consistent with
the Fernández & Bornn [2018] and Xenopoulos et al. [2020] methodological
framing: the whole point of using graph distance over awpy's Euclidean
influence field is to make `d_G` wall-aware while still preserving physical
scale.

### 2.3 Directed vs. symmetrized storage

**Decision: store the full directed matrix.**

Because the nav graph is a `DiGraph` and 33.27% of all finite ordered pairs
have asymmetric distances (see §4), discarding directionality at the caching
layer would silently bake in a loss of real map-geometry information. Keeping
the directed matrix preserves the option to symmetrize in Task 5 if needed
(e.g. using `min(dist[u,v], dist[v,u])` to match awpy's own
`_graph_based_position_distance` convention), while the reverse is not
possible — once symmetrized, the one-way traversal cost information is gone.

Unreachable ordered pairs are stored as `float('inf')`, not as a large finite
sentinel. Task 5's influence code will treat `inf` distances as producing zero
influence (`exp(-inf) = 0`), which should be made an explicit rather than
accidental decision in that step's implementation.

---

## 3. Implementation

### 3.1 Script: APSP computation

**Script:** (not reproduced in full here — see project working directory)

Core approach:

```python
import pickle
import networkx as nx
import numpy as np

PATCHED_GRAPH_PATH = "outputs/de_mirage_patched_graph_v3.pkl"

with open(PATCHED_GRAPH_PATH, "rb") as f:
    G_patched: nx.DiGraph = pickle.load(f)

nodes = sorted(G_patched.nodes)
area_to_idx = {a: i for i, a in enumerate(nodes)}
idx_to_area = {i: a for i, a in enumerate(nodes)}

N = len(nodes)
dist = np.full((N, N), np.inf, dtype=np.float64)

for u, dists in nx.all_pairs_dijkstra_path_length(G_patched, weight="weight"):
    u_idx = area_to_idx[u]
    for v, d in dists.items():
        v_idx = area_to_idx[v]
        dist[u_idx, v_idx] = d

apsp = {
    "distance_matrix": dist,
    "area_to_index_map": area_to_idx,
    "index_to_area_map": idx_to_area,
}

with open("outputs/de_mirage_apsp_dict.pkl", "wb") as f:
    pickle.dump(apsp, f)
```

**Why `nx.all_pairs_dijkstra_path_length` over awpy's
`generate_area_distance_matrix`:**

awpy's own function uses N² individual shortest-path calls in a nested Python
loop, and computes all three distance types (graph, geodesic, euclidean)
simultaneously. `nx.all_pairs_dijkstra_path_length` runs one Dijkstra pass per
source node (903 passes total), which for a sparse graph of this size (~3,505
edges) is substantially faster, and produces only the geodesic distances
needed. It also avoids the string-keyed nested-dict output format of
`AreaMatrix`, which would require repeated `str(area_id)` conversions on every
lookup in the influence-field loop.

**Node ordering:** `sorted(G_patched.nodes)` produces a stable, deterministic
ordering of the 903 area IDs. Note that area IDs are *not* a contiguous
integer range (e.g. there is no area with ID 51) — the sorted ordering is
therefore not the same as `range(903)`, and `area_to_idx`/`idx_to_area` are
the required translation layer for all matrix lookups.

### 3.2 Output artifact: `de_mirage_apsp_dict.pkl`

A single pickled dict with three keys:

| Key | Type | Description |
|---|---|---|
| `"distance_matrix"` | `np.ndarray`, shape (903, 903), dtype float64 | Geodesic shortest-path distances in game units. `dist[i][j]` is the cost from the node at matrix index `i` to the node at index `j`. Unreachable pairs are `float('inf')`. |
| `"area_to_index_map"` | `dict[int, int]` | Maps awpy area ID (int) → matrix row/column index (0-based). |
| `"index_to_area_map"` | `dict[int, int]` | Inverse of above. Maps matrix index → awpy area ID. |

**Usage pattern in Task 5:**

```python
import pickle
import numpy as np

with open("outputs/de_mirage_apsp_dict.pkl", "rb") as f:
    apsp = pickle.load(f)

dist = apsp["distance_matrix"]
area_to_idx = apsp["area_to_index_map"]

# O(1) lookup
d = dist[area_to_idx[area_u], area_to_idx[area_v]]
```

---

## 4. Graph structure findings

Several structural properties of the patched de_mirage graph were confirmed
during Task 4's diagnostic pass. These are findings about the graph itself —
they inform downstream methodology decisions and are worth preserving as facts.

### 4.1 Strong connectivity: 3 components

```
nx.number_strongly_connected_components(G_patched) = 3
SCC sizes: [3, 4, 896]
```

The graph is **not** strongly connected. It has three strongly connected
components:

| SCC | Area IDs | Size | areaName | Interpretation |
|---|---|---|---|---|
| Main mesh | (all remaining 896 nodes) | 896 | various | The fully interconnected main nav mesh |
| "Ladder" cluster | [833, 834, 835, 836] | 4 | `"Ladder"` | Top vent area above Sniper's Nest |
| "Truck" cluster | [363, 742, 743] | 3 | `"Truck"` | Top of truck at Bombsite B |

**Condensation DAG edges:** `(1, 0)` and `(2, 0)` — meaning both small
clusters can reach the main mesh (896 nodes), but the main mesh cannot
reach back into either small cluster.

**These are legitimate one-way map geometry features — not patch defects.** Both
the "Ladder" and "Truck" tiles are reachable from surrounding areas (jump up,
boost onto a prop) but exit via a drop/fall that routes to a *different*
surrounding area than the one you climbed from — a classic one-way directed
edge pattern. This is confirmed by the fact that:

- Both clusters carry real, human-assigned `areaName` values (`"Ladder"`,
  `"Truck"`), and are therefore **not** among the 72 blank-named originally
  orphaned areas that the patch script targeted.
- The 526 directed edges added by the patch script were confirmed 100%
  bidirectional in the Task 3/patch verification (15/15 checks). These SCCs
  are therefore an original, pre-patch property of the `nav/de_mirage.txt`
  edge-list file, unchanged by any project work.

**Do not confuse the "Ladder" cluster (areas 833–836) with the "Apartments
ramp" cluster (areas 873–880)** discussed in `de_mirage_navmesh_findings.md`.
Those are a different set of tiles — blank-named, originally orphaned, now
patched — that were *incorrectly* provisionally labeled "Ladder" early in the
project before being identified as the Apartments ramp by direct map knowledge.
The actual named "Ladder" tiles (833–836) are distinct in both area ID range
and physical location.

### 4.2 Expected inf count: 6,296

From the condensation structure, the exact number of unreachable ordered pairs
(off-diagonal `inf` entries) can be derived analytically:

| Direction | Reachable? | Pair count |
|---|---|---|
| Ladder (4) → Main (896) | Yes (condensation edge) | 4 × 896 = 3,584 |
| Truck (3) → Main (896) | Yes (condensation edge) | 3 × 896 = 2,688 |
| Main (896) → Ladder (4) | **No** | 896 × 4 = **3,584** |
| Main (896) → Truck (3) | **No** | 896 × 3 = **2,688** |
| Ladder (4) → Truck (3) | **No** | 4 × 3 = **12** |
| Truck (3) → Ladder (4) | **No** | 3 × 4 = **12** |

```
Total inf entries (off-diagonal) = 3,584 + 2,688 + 12 + 12 = 6,296
```

**This matches the matrix exactly** — confirming the matrix faithfully
represents the patched graph's actual reachability structure.

Note: a verification script bug introduced a `- N` subtraction that caused
the count to print as `5,393` rather than `6,296`. The matrix is correct;
the script was wrong. The fix is to drop the erroneous `- N`:

```python
# Wrong (underreports by N=903):
n_inf = int(np.sum(~finite_mask)) - N

# Correct:
n_inf = int(np.sum(~finite_mask))
```

### 4.3 Distance asymmetry: 33.27% of finite pairs

Of all finite off-diagonal distance pairs, **33.27%** satisfy
`dist[u,v] ≠ dist[v,u]` (i.e. the directed shortest-path cost differs by
direction). This is a property of the *original* 2,979-edge graph, not of the
526 patch edges (which are confirmed 100% bidirectional). A meaningful fraction
of the original edge-list appears to encode genuine one-way traversal (ledge
drops, boost spots, one-way vents, etc.).

This is the primary justification for the **directed storage decision** (§2.3):
symmetrizing the matrix would silently discard this real gameplay information
for 33% of all reachable pairs — a non-negligible loss, not a rounding detail.

Note: the asymmetry computation (`np.abs(dist - dist.T)`) triggers a
`RuntimeWarning: invalid value encountered in subtract` due to `inf - inf = nan`
for pairs where *both* directions are unreachable. This is benign (nan values
are correctly excluded by the subsequent `np.isfinite` filter), but can be
silenced with:

```python
both_finite = np.isfinite(dist) & np.isfinite(dist.T)
sym_diff_finite = np.abs(dist - dist.T)[both_finite]
```

### 4.4 Distance distribution

Over all finite, non-zero (off-diagonal) pairs:

| Statistic | Value |
|---|---|
| Minimum | 25.0 game units |
| Maximum | 7,968.7 game units |
| Mean | 2,893.1 game units |

These figures are consistent with de_mirage's physical scale. The minimum
(~25 units) corresponds to adjacent tile centroids; the maximum (~7,969 units)
is plausible for the longest map-diagonal route.

---

## 5. Verification

### 5.1 Verification script

**Script:** `apsp_claude.py` (development verification script; not included in the final repository.
The repository's APSP verifier is `verification/verify_apsp.py`, with its log in
`docs/verification_logs/verify_apsp_out.txt`.)

14 checks total across three categories:

**Data integrity (Checks 1–4):**

| Check | Result |
|---|---|
| 1: Matrix shape (903, 903) | PASS |
| 2a: `area_to_idx` ↔ `idx_to_area` is a proper bijection | PASS |
| 2b: `area_to_idx` covers exactly the graph's node set | PASS |
| 2c: Index values are exactly 0..N-1 with no gaps/dupes | PASS |
| 3: Diagonal (self-distances) all exactly 0.0 | PASS |
| 4: No negative distances | PASS |

**Structural checks (Checks 5–6, informational):**

| Check | Result |
|---|---|
| 5: Multiple SCCs present — unreachable pairs expected | WARN (informational) |
| 6: 33.27% asymmetry in finite pairs — one-way edges present | WARN (informational) |

**Correctness cross-validation (Checks 7–10):**

| Check | Result |
|---|---|
| 7: Matrix matches `nx.dijkstra_path_length` on patched graph (30-pair sample) | PASS |
| 8: Matrix distance never exceeds direct edge weight for connected pairs (50-edge sample) | PASS |
| 9: Matrix matches awpy's `area_distance(..., dist_type="geodesic")` on monkeypatched patched graph (30-pair sample) | PASS |
| 10: Reconstructed path cost matches cached matrix distance (10-pair sample) | PASS |

**Distribution / dtype (Checks 11–12):**

| Check | Result |
|---|---|
| 11: Distance distribution plausible (informational) | WARN (informational) |
| 12: dtype is float64, array is C-contiguous | PASS |

**Final tally: 11 passed, 0 failed, 3 informational/warnings**

### 5.2 Key correctness evidence

Checks 7, 9, and 10 are the strongest correctness signals because they
cross-validate the matrix against fully independent computations:

- **Check 7** re-runs `nx.dijkstra_path_length` directly on the patched graph
  for 30 random pairs — same algorithm, same weights, completely independent of
  the matrix-building code path. Zero mismatches confirms no indexing or
  algorithm bug in the APSP script.
- **Check 9** calls awpy's own `area_distance()` (A* + summed edge weights) on
  the monkeypatched patched graph — a different algorithm (`astar_path` vs.
  `dijkstra_path_length`), same graph. Zero mismatches confirms the matrix is
  consistent with awpy's own production distance computation.
- **Check 10** reconstructs the actual node path for 10 pairs via
  `nx.dijkstra_path` and sums edge weights directly — checks that the stored
  scalar distance corresponds to a real, walkable sequence of edges in the
  graph. Zero mismatches confirms no silent corruption in the stored values.

### 5.3 Path comparison plot

`outputs/task4_path_comparison.png`

A visual comparison of two paths between the same source/target pair (CT Spawn
area → near T-side A site approach area):

- **Green (patched, matrix-implied):** path reconstructed via
  `nx.dijkstra_path` on the patched graph. **Cost: 2,480.6 game units, 31
  nodes.**
- **Red (unpatched, awpy `area_distance()`):** path from awpy's
  `area_distance()` called against the original `NAV_GRAPHS["de_mirage"]`
  (no monkeypatch). **Cost: 11,115.3 game units, 87 nodes (before endpoint).**

The patched path routes directly through areas `536, 540, 554, 562` — all
within the CT-stairs cluster (area IDs `525–558, 561–566`), the 42-tile
formerly-orphaned cluster identified in `de_mirage_navmesh_findings.md` as
"CT stairs up to A site." The unpatched path, lacking any edges into this
cluster, detours through unrelated parts of the map to reach the same
destination.

**Key figures:**

| Metric | Patched (matrix) | Unpatched (awpy) | Ratio |
|---|---|---|---|
| Path cost (game units) | 2,480.6 | 11,115.3 | 4.48× |
| Path length (nodes) | 31 | 87 | 2.81× |

This is not just a connectivity demonstration — it confirms the patch recovers
the *geometrically correct, shortest* route for one of de_mirage's most
heavily-trafficked CT-side paths. Without this patch, any frame where a CT
player traverses the stairs to A site would produce `d_G` distances
systematically ~4.5× too large to every other point on the map, directly
corrupting `µᵢ(x,t)` for a large fraction of real gameplay frames.

---

## 6. Outstanding decisions for Task 5

Two items from this task carry forward as explicit, documented decisions Task 5
must respect:

1. **Directed distances:** the matrix stores `dist[u,v]` and `dist[v,u]`
   separately. Task 5 must decide, per use case, whether to use the directed
   value (true cost of traversal in that direction) or symmetrize (e.g.
   `min(dist[u,v], dist[v,u])`). If symmetrizing, this choice must be
   documented explicitly in the methodology — not silently baked into the code.

2. **Inf handling:** `dist[u,v] = float('inf')` for unreachable pairs.
   Task 5's influence function will effectively treat this as zero influence
   (`exp(-inf) = 0`), which is likely correct behaviour — a player standing on
   the Truck or Ladder top cannot reach most of the map in the reverse
   direction, so their influence on those tiles should be zero. This should be
   confirmed as an explicit intent rather than an accidental numerical outcome.

---

## 7. Key files

| File | Description |
|---|---|
| `data/de_mirage_apsp_dict.pkl` | The APSP cache: dense 903×903 float64 matrix + bidirectional area-ID-to-index mappings |
| `data/de_mirage_patched_graph_v3.pkl` | Source graph used for the APSP computation — must match any future re-run |
| `verification/verify_apsp.py` | APSP verification script included in the repository (the 14-check development script above is not included) |
| `outputs/task4_path_comparison.png` | Visual comparison: patched matrix path vs. unpatched awpy path, CT Spawn → A site approach |

---

*Compiled from project session following Task 3 completion. All verification
results are from a confirmed run on `csgo-research-vm` (Ubuntu, Python 3.11,
venv), awpy 1.3.1.*
