# de_mirage Nav Graph Patch — Process Summary & Final Approach

Reference document for the CS:GO map control project. Describes the full
process used to eliminate de_mirage's 72 orphaned nav areas, including the
two failed intermediate approaches and why they failed, so this doesn't need
to be re-derived from scratch in a future session. This supersedes the
"patch script" section of `de_mirage_navmesh_findings.md` — that file's
nav-mesh *analysis* (orphan counts, spatial clusters, CT-stairs/Apartments-
ramp identification) is still accurate and still the reference for those
facts; only the *patching approach* described there has been superseded by
what's documented here.

---

## 1. Problem recap

awpy's bundled `NAV_GRAPHS["de_mirage"]` (`nx.DiGraph`, 903 nodes, 2979
edges) has 72 areas with zero in-edges and zero out-edges — confirmed via
direct parse of the underlying `nav/de_mirage.txt` edge-list file that awpy
reads at import time (see §1 of `de_mirage_navmesh_findings.md` for the full
data-provenance trace: `NAV_CSV` → `transform_csv_to_json` → `NAV`, then
`create_nav_graphs(NAV, data_path)` → `NAV_GRAPHS`, reading edges from a
static per-map text file, not computed from tile geometry).

Two of the 72 orphans form large, tactically significant spatial clusters
(42-tile CT stairs to A site; 8–10-tile Apartments ramp from T spawn); the
remainder are ~22 scattered singletons/small groups. Leaving them
disconnected would corrupt position data for important areas in Task 3
(player position → nav node mapping), since any frame where a player stands
on one of these tiles would map to an unreachable node.

---

## 2. Approaches tried, in order, and why the first two were insufficient

### Attempt 1 — Nearest-centroid bridging with a "boundary node" heuristic

**Approach:** For each of the original graph's weakly-connected orphan
components, build a `cKDTree` over main-component centroids, then try to
identify a small number of "boundary" nodes per cluster (the node closest to
the main mesh, plus any node whose nearest main-mesh target was >500 units
from that one) and bridge only those, to avoid one redundant bridge edge per
tile in a 42-tile cluster.

**Why it failed:** Every one of the 72 orphans is its OWN singleton
weakly-connected component in the original graph — per
`de_mirage_navmesh_findings.md` §3, *"Zero orphans form an isolated
sub-cluster connected only to each other — no internal edges among orphans
exist at all."* The DBSCAN-identified "CT stairs" / "Apartments ramp"
groupings are a purely *spatial* clustering, not a *graph* clustering. Since
the patch script's loop was keyed off graph components (each of size 1), the
"boundary node" logic and a planned internal MST step both operated on
trivial 1-node "clusters" and did nothing — every orphan ended up bridged
individually anyway (`entry_group_size=1` for all 72, 0 MST edges added).
Functionally this produced a connected graph (1 component, 903 nodes), but
not the intended minimal/sensible bridging — confirmed by inspecting the
printed edge log, where 50 of 72 bridges were individual CT-stairs/
Apartments-ramp tiles each bridging separately to nearby main-mesh tiles
(`CTSpawn`, `TicketBooth`, `SideAlley`, `House`).

### Attempt 2 — Geometric touching test (`areas_touch`) with strict overlap condition

**Approach** (`de_mirage_patch_fixed.py`): Reconstruct orphan-to-orphan
adjacency directly from nav-area bounding-box geometry — two areas "touch"
if their boundaries are within `tol` along one axis AND they have positive
overlap (greater than `tol`) along the perpendicular axis — then connect
each resulting touching-cluster to the main mesh via a single nearest-
centroid bridge.

**Why it failed:** `areas_touch`'s overlap condition required the
perpendicular-axis overlap to be **strictly greater than `tol`**
(`> tol`, not `>= 0` or `>= -tol`). On a staircase, consecutive step-tiles
are frequently offset enough that they only share a thin sliver of edge —
sometimes near-zero overlap. The strict `> tol` condition silently rejected
many of these genuine touches. Result (confirmed by run output): 20 orphan
touching-clusters from 72 nodes (average cluster size ~3.6), instead of the
expected ~2 large clusters (CT stairs, Apartments ramp) plus the ~16 known
small/singleton clusters. Visually, this fractured the 42-tile CT stairs
group into multiple disconnected-looking sub-clusters, each independently
bridged to the main mesh at a different point — exactly the "still
disconnected near the far right and middle-right of the cluster" gaps
visible in the diagnostic plot.

---

## 3. Final approach — geometric adjacency via rectangle-gap distance (used now)

**Script:** `patch_de_mirage_graph_v3.py`

### Core fix: rectangle gap-distance instead of separate touch/overlap conditions

Replaced the two-condition `areas_touch` test with the standard axis-aligned
rectangle gap-distance formula:

```
dx = max(bx1 - ax2, ax1 - bx2, 0)
dy = max(by1 - ay2, ay1 - by2, 0)
gap_distance = sqrt(dx² + dy²)
```

This single value is `0` if the two rectangles overlap or share any part of
a boundary, and a positive real distance otherwise. "Touching" is then just
`gap_distance <= tol` (`tol = 2.0` game units) — one consistent rule that
correctly handles edge-touching, corner-touching, and small real gaps in one
pass, with no risk of the asymmetric strict-inequality bug from Attempt 2.

### Structural change: orphan-to-EVERYTHING adjacency, not orphan-to-orphan-then-single-bridge

Rather than first clustering orphans among themselves and then picking one
representative bridge per cluster, this version checks **every orphan
against every other area in the graph** (both other orphans AND main-mesh
areas) and adds an edge for every pair that satisfies the touching test.
This directly implements "every orphan node connects to every node adjacent
to it" — an orphan with several real touching main-mesh neighbors gets
edges to all of them, not just the single nearest one. At de_mirage's scale
(72 orphans × ~903 areas ≈ 65k pair checks) this is computationally trivial
and needs no spatial index.

### Fallback safety net

After the geometric pass, weak connectivity is recomputed. Any orphan that
still isn't reachable from the main mesh (i.e. its true gap to every
neighbor exceeds `TOUCH_TOL`) gets a single nearest-centroid `cKDTree`
bridge as a fallback — guaranteeing full connectivity is reached even if
`tol` doesn't happen to catch every legitimate case.

### Edge weight convention (unchanged throughout all three attempts)

Every added edge — geometric-adjacency or fallback — uses
`weight = euclidean_distance(center_u, center_v)`, matching awpy's own
convention in `create_nav_graphs` (straight-line distance between
bounding-box midpoints). This keeps `area_distance(..., dist_type="geodesic")`
and any A* search using the `"center"`-based heuristic behaving consistently
across original and patched edges.

### Outputs produced

- `de_mirage_patched_graph_v3.pkl` — the patched `nx.DiGraph`, pickled, ready to
  load standalone with `pickle.load()`.
- `de_mirage_bridge_log_v3.json` — every edge added, tagged by type
  (`"geometric_adjacency"` or `"fallback_centroid_bridge"`), with endpoint
  area IDs/names and distance.
- `de_mirage_additional_edges.txt` — the same new edges written in the exact
  `area_id_1,area_id_2` (one directed pair per line) format awpy's own
  `nav/de_mirage.txt` uses, in case the fix is folded directly into the
  installed package's bundled data file instead of (or in addition to)
  loading the pickle at runtime (see §4 below for the tradeoffs of each).

---

## 4. Two ways to use the patched graph going forward

### Option A — Load the pickle directly (recommended for this project's own code)

```python
import pickle
with open("de_mirage_patched_graph_v3.pkl", "rb") as f:
    G_patched = pickle.load(f)
```

Use this graph object directly in any of your own scripts (Task 3's
`fast_map_position_to_area`, Task 4's distance caching, etc.). No
dependency on awpy's global state — fully self-contained and reproducible
from the pickle alone.

### Option B — Monkeypatch into `awpy.data.NAV_GRAPHS` (only if reusing awpy's own functions)

```python
import awpy.data as awpy_data
awpy_data.NAV_GRAPHS["de_mirage"] = G_patched
```

Needed only if you want to keep calling awpy's own higher-level functions
(`area_distance`, `point_distance`, `generate_position_token`,
`generate_area_distance_matrix`, etc.) or reuse earlier scripts (e.g.
`plot_nav_tiles.py`) that import `NAV_GRAPHS` directly from `awpy.data`.
This is **process-local and non-persistent** — it must be re-applied at the
top of every new script/session, since it's just overwriting a dict entry
in memory, not modifying anything on disk inside the awpy package.

### Option C — Edit the bundled `nav/de_mirage.txt` directly (not yet applied; optional)

`de_mirage_additional_edges.txt` contains the new edges in the same format
awpy's own file uses. Appending those lines to
`site-packages/awpy/data/nav/de_mirage.txt` would make every fresh awpy
import already include the fix, with no runtime patching needed at all.
**Not yet done** — tradeoffs to weigh before doing this:
- Pro: simplest possible downstream usage; no pickle, no monkeypatch,
  works identically across the dev VM and the HPC cluster as long as the
  same edit is applied to both installs.
- Con: edits an installed package file, not project source — will be lost
  on any awpy reinstall/upgrade, and isn't naturally version-controlled
  with the rest of the project unless the diff is kept and reapplied
  manually after any environment rebuild (e.g. setting up the HPC cluster
  environment from scratch).

---

## 5. Verification

`verify_patched_graph_final.py` performs a non-visual, structural audit of
`de_mirage_patched_graph_v3.pkl` covering: node-set integrity against
`NAV["de_mirage"]`, absence of isolated nodes, single weakly-connected
component, preservation of all original edges/weights unchanged, that added
edges are genuinely new and fully bidirectional, that all edge weights are
finite/non-negative, that every originally-orphaned node can now reach a
known main-mesh reference node, and that a sample of added-edge weights are
consistent with the Euclidean center-distance convention.

### Result: 15/15 checks passed, 0 failed

Confirmed on a clean run against the patched graph produced by
`patch_de_mirage_graph_v3.py`:

- **Node set**: patched graph has exactly the same 903 area IDs as
  `NAV["de_mirage"]` — none dropped, none extra.
- **No isolated nodes**: every node has both in-degree > 0 and out-degree > 0.
- **Single component**: exactly 1 weakly connected component, covering all
  903 nodes.
- **Original edges untouched**: all 2979 original edges are still present
  with unchanged weights — the patch only ever added edges, never modified
  or removed existing ones.
- **526 directed edges added** (≈263 bidirectional pairs) by the geometric
  adjacency pass. The fallback nearest-centroid bridge path was available
  but **not needed** — every orphan found at least one genuine geometric
  neighbor (orphan or main-mesh) within `TOUCH_TOL = 2.0` units, so 0
  fallback bridges were used. This is a meaningfully different (and more
  geometrically faithful) outcome than either of the two earlier attempts:
  Attempt 1 produced 72 individual bridges with no shared structure between
  same-cluster tiles; Attempt 2 produced 20 fragmented touching-clusters
  each independently bridged. This final version's 526 edges instead
  reflect dense, mutual, real-geometry adjacency, including orphan-to-orphan
  AND orphan-to-main-mesh connections wherever they actually touch.
- **Bidirectionality**: 0 non-reciprocal added edges — every added edge has
  its reverse-direction counterpart, as intended.
- **Edge weights**: all finite and non-negative across the entire graph (not
  just added edges).
- **Reachability**: all 72 originally-orphaned nodes can reach a known
  main-mesh reference node (`area 513`, CTSpawn) — 0 unreachable.
- **Weight-geometry consistency**: a 200-edge sample of added edges all had
  weights matching the Euclidean distance between their `center` attributes
  to within 1.0 unit, confirming the edge-weight convention was applied
  correctly throughout.

This is the structural confirmation that the visual check (orange/lime
highlighting in `plot_patched_nav_mesh.py`'s output) was correct at the
logical level too — not just "looks connected," but verified connected,
verified non-destructive of the original graph, and verified consistent
with awpy's own distance conventions.

---

## 6. Status / what's next

- The patched graph is structurally connected and **verified sound at the
  logical level** (15/15 checks passed, see §5) — not just visually correct
  via the plotting script.
- **Still undecided:** whether to apply Option C (editing the bundled
  `nav/de_mirage.txt`) or continue with Option A/B (pickle + optional
  monkeypatch) as the project's standing convention, going forward. This
  remains an open decision as of this writing — revisit before Task 4
  (distance-matrix caching) to avoid maintaining two slightly different
  versions of "the de_mirage graph" across scripts, and especially before
  setting up the HPC cluster environment, where Option C would need to be
  reapplied to a separate awpy install if chosen.
- Next step: proceed to Task 3 (`fast_map_position_to_area`) using Option A
  (load the pickle directly), per the design already discussed — `cKDTree`
  over patched-graph node centers, k=5 candidates, z-tolerance filter
  (~100 units, to be tuned empirically against known overlap-zone frames),
  validated against `point_in_area` spot-checks.

---

*Compiled from project chat history. Describes the patching process applied
to de_mirage specifically — the underlying methodology (rectangle gap-
distance adjacency, geometric-pass-then-fallback-bridge) should generalize
to other ESTA maps if/when this project extends beyond de_mirage, but has
only been run and verified against de_mirage so far.*
