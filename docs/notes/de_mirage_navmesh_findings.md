# de_mirage Nav Mesh — Consolidated Findings

Reference document for the CS:GO map control / graph-based influence field project.
Built from awpy 1.3.1, verified at runtime in `csgo-research-vm` (Python 3.11, venv).
Use this to jumpstart new chats without re-deriving everything from scratch.

---

## 1. Environment & data provenance

- **awpy version:** 1.3.1, CS:GO-only (not awpy 2.x / CS2). Methodology depends on
  `awpy.data.NAV_GRAPHS`, which has no CS2 equivalent — this is why the project stays
  on awpy 1.x despite CS2/awpy 2.x being more actively maintained upstream.
- **Install location (Zayaan's VM):**
  `/home/zayaan/csgo-map-control/venv/lib/python3.11/site-packages/awpy/`
- **Data load pipeline (traced from source, confirmed via `inspect.getsource`):**
  1. `NAV_CSV` — flat pandas DataFrame, 26,307 rows, all maps combined. Columns:
     `mapName, areaId, areaName, northWestX/Y/Z, southEastX/Y/Z`.
  2. `transform_csv_to_json(NAV_CSV)` (lives in `awpy/utils.py`, not `data/__init__.py`)
     → builds `NAV`, a `dict[map_name][area_id] -> Area dict`.
  3. `create_nav_graphs(NAV, data_path)` (lives in `awpy/data/__init__.py`) → builds
     `NAV_GRAPHS`, a `dict[map_name] -> nx.DiGraph`, by:
     - Adding one node per area in `NAV[map_name]`, with attributes `mapName, areaID,
       areaName, northWestX/Y/Z, southEastX/Y/Z, center, size`.
       - `center` = bounding-box midpoint: `((nwX+seX)/2, (nwY+seY)/2, (nwZ+seZ)/2)`.
       - `size` = **3D bounding-box diagonal length** (`sqrt(dx²+dy²+dz²)`), **not**
         tile area or volume. Don't use this as a footprint-area proxy.
     - Reading a **separate per-map edge-list text file** at
       `PATH/nav/{map_name}.txt` (one `area_id_1,area_id_2` pair per line, directed)
       and adding each as a directed edge with
       `weight = scipy.spatial.distance.euclidean(center_u, center_v)` — i.e. pure
       straight-line distance between bounding-box midpoints. This is what awpy calls
       "geodesic" distance elsewhere (`analytics.nav.area_distance`) — it is **not**
       wall-aware or surface-aware, just A* over these straight-line edge weights.
  4. **Critical implication:** graph connectivity is determined entirely by whatever
     is (or isn't) recorded in the static `nav/de_mirage.txt` edge-list file — it is
     *not* derived from tile adjacency/geometry at runtime. Disconnection is a property
     of this bundled source file, not of any analysis choice made in this project.

- **`MAP_DATA["de_mirage"]`** = `{'pos_x': -3230.0, 'pos_y': 1713.0, 'scale': 5.0}`.
  **No `z_cutoff` key.** Confirmed both empirically and structurally (the `MapData`
  TypedDict defines `z_cutoff` as `NotRequired[float]` — i.e. optional, map-dependent).
  **De_mirage has no vertical-split radar handling in awpy 1.3.1** — unlike de_nuke/
  de_vertigo, there is no built-in `y += 1024` offset mechanism for this map. Any
  z-overlap disambiguation (e.g. Catwalk-over-Mid, Underpass/Connector) must be
  handled by this project's own logic; it cannot be borrowed from awpy's plotting code.

- **`AREA_DIST_MATRIX`**: empty for all maps (0 populated) — every tile-level distance
  call for de_mirage takes the slow live `nx.shortest_path`/`nx.astar_path` path, not
  a cached fast path. Relevant for Task 4 (all-pairs shortest path) performance planning.
- **`PLACE_DIST_MATRIX`**: populated for 10 maps, **and `de_mirage` is one of them.**
  Named-place-level (not tile-level) distances for de_mirage already have a fast
  precomputed path available. Worth deciding which parts of the pipeline can tolerate
  place-level granularity to skip the expensive `generate_area_distance_matrix` run
  (docstring-documented 20min–5h runtime, 50–300MB output) entirely.

---

## 2. Graph structure for de_mirage

- **903 nodes, 2979 edges, directed (`nx.DiGraph`).**
- Edge count cross-validated two independent ways: (a) `NAV_GRAPHS["de_mirage"]
  .number_of_edges()`, (b) raw line count of `nav/de_mirage.txt` — both = 2979.
- **76 weakly-connected components**: 1 main component (814 nodes) + 75 other
  components holding 89 nodes total — *this was the original, broader estimate before
  the orphan analysis below narrowed and confirmed the exact figures.*
- **Confirmed, narrower figure from direct edge-list analysis: 72 areas are fully
  disconnected** — they appear as neither source nor destination in any line of
  `nav/de_mirage.txt`. (The slight discrepancy between "89 nodes across 75 components"
  and "72 fully isolated areas" has not yet been fully reconciled — worth checking
  whether the remaining ~17 nodes belong to small multi-node components that aren't
  pure isolates, e.g. 2–3 node clusters connected to each other but not the main mesh.
  This is a TODO, not yet resolved.)

---

## 3. Orphan tile classification (de_mirage)

Verified by direct parse of `nav/de_mirage.txt` (no awpy dependency needed for this
check — pure text parsing), cross-checked against the live graph:

- **All 72 orphans are fully isolated**: zero appear as a source, zero as a
  destination, in any edge-list line.
- **Zero orphans are one-directional** (no destination-only "sinks", no
  source-only "exits").
- **Zero orphans form an isolated sub-cluster connected only to each other** —
  no internal edges among orphans exist at all.
- **Practical implication:** there is no existing directionality to preserve when
  patching. Any KD-tree-based bridge from an orphan to the main component can safely
  be added bidirectionally without risk of overriding a deliberate one-way design
  choice (e.g. a drop-down ledge), because no such choice was ever encoded for these
  72 tiles in the first place.
- **All 72 orphans have a blank (`""`) `areaName`** in `NAV["de_mirage"]`. Blank names
  also exist elsewhere in the dataset on fully-connected tiles on other maps (e.g.
  `de_bank` area `2370`), so blank name is not an automatic orphan predictor in general
  — but for de_mirage specifically, it appears that whoever curated the edge-list file
  and whoever assigned area names skipped the same tile set. Not yet checked: whether
  *all* blank-named de_mirage tiles are orphaned, or just some (open question, see §5).

### Spatial clustering of the 72 orphans (DBSCAN, eps=150 units, XY-only on centroids)

18 clusters found. Two dominate:

| Cluster | Area IDs | Size | Identified as (confirmed by Zayaan, ground truth) |
|---|---|---|---|
| Largest | 511, 512, 525–558, 561–566 | 42 tiles | **CT stairs up to A site** |
| Second | 873–880 (possibly incl. 773, 774 — unconfirmed, see below) | 8–10 tiles | **Apartments ramp from T spawn** |

Remaining ~22 orphans are scattered across 16 small clusters/singletons — *not* part
of a few large dropped regions; this fraction looks like genuine scattered individual-
tile omissions rather than wholesale region drops.

**Geometric signatures observed (for future shape-based heuristics, with the caveat
below):**
- CT stairs cluster: narrow, elongated XY footprint (~150×300 units), smoothly
  varying z (range ~−167 to −196) — consistent with a sloped/stepped walkable surface.
- Apartments ramp cluster: tight XY footprint (~340×115 units) but **large z range**
  (~126 units, from −135.97 to −261.97) — tight horizontal footprint + large vertical
  spread is the shared signature of *any* steep vertical-traversal feature (ramp,
  ladder, or stairs at a steep angle) — **shape alone could not distinguish "ramp" from
  "ladder" here; only Zayaan's direct map knowledge resolved this.** Earlier in this
  project this cluster was provisionally guessed as a "Ladder" — that guess was
  **wrong** and has been corrected. Do not reuse "Ladder" for this cluster.
- **Unresolved:** areas 773/774 (a separate 2-tile cluster at eps=150, located at
  nearly identical y/z to the Apartments-ramp cluster, ~312 units away in x — just
  outside the eps=150 cutoff) may actually be part of the same ramp feature (e.g. a
  landing) rather than a distinct dropped object. Not yet confirmed with Zayaan's map
  knowledge. Re-running DBSCAN with eps=250–300 to check whether clusters merge is a
  pending TODO, as is checking whether areas 881/882 (adjacent IDs to the ramp cluster,
  but spatially ~1700+ units away — a clear example of ID-adjacency NOT implying
  spatial adjacency) might pair with each other under a looser radius.
- **General caution validated by this exercise:** nav area ID adjacency does **not**
  reliably predict spatial adjacency on this map (881/882 numerically next to 873–880
  but geometrically unrelated to it, and to each other only loosely).

---

## 4. Visualization — working plotting approach

Validated pattern for accurately plotting tiles/graph on the de_mirage radar:

- Use `awpy.visualization.plot.position_transform(map_name, value, axis)` (or
  reimplement identically) to convert each tile's `northWestX/Y` and `southEastX/Y`
  corners individually into canvas space — do **not** assume width/height deltas
  transform as simple scalar multiples, since the y-axis transform is
  `(start_y - position) / scale`, a flip-and-shift, not a pure scale.
- Since de_mirage has no `z_cutoff`, the `y += 1024` offset branch is a no-op for this
  map — confirmed not needed, but the offset code can stay in scripts harmlessly for
  portability to other maps.
- Tile rectangles: build from `min`/`max` of transformed corners rather than assuming
  a consistent corner-ordering sign convention.
- Connectivity overlay: draw `NAV_GRAPHS[map_name]` edges between `center` attributes;
  color tiles/edges by `nx.weakly_connected_components` membership (treat graph as
  directed but use *weak* connectivity for "is this reachable from the main mesh"
  framing, since one-way edges still count as connected for this purpose).
- For distinguishing multiple disconnected components from each other (not just
  binary main/non-main), use a qualitative colormap like `tab20` rather than a single
  flag color — this was needed to even notice the CT-stairs/Apartments-ramp split.

---

## 5. Open questions / TODOs for future sessions

1. Reconcile "89 orphan nodes across 75 components" (early estimate) vs. "72 fully
   isolated areas, 18 spatial clusters" (later, more rigorous direct edge-list check).
   Likely the early estimate was a rougher pass; confirm by recomputing weakly-connected
   components directly and diffing node sets.
2. Confirm whether 773/774 belong to the Apartments-ramp cluster (rerun DBSCAN at
   eps≈250–300; then verify with map knowledge).
3. Confirm whether 881/882 should be treated as a pair (rerun DBSCAN at looser eps).
4. Determine, for ALL blank-`areaName` tiles on de_mirage (not just the 72 orphans),
   what fraction are also orphaned — needed to know if blank-name is a strong or weak
   predictor specifically on this map (script for this was drafted but not yet run).
5. Decide patch strategy per cluster type:
   - Large named clusters (CT stairs, Apartments ramp): bridge only boundary tiles to
     the main mesh via KD-tree nearest-neighbor, rather than patching every tile in
     the cluster independently (avoids a tangle of redundant bridge edges into one
     room/corridor).
   - Remaining ~22 scattered singletons/small clusters: per-tile/per-small-group
     KD-tree nearest-neighbor bridging, as originally planned.
6. Decide whether the KD-tree distance metric for patching should include z (not just
   XY) given confirmed z-overlap risk elsewhere on this map (Catwalk/Mid,
   Underpass/Connector) despite the *orphan* clusters themselves not being z-overlap
   cases.

---

## 6. Key scripts produced so far (for reference, not reproduced in full here)

- `nav_tile_plot.py` — tile + connectivity overlay plot on the radar, colored by
  component (main vs. disconnected), with z_cutoff handling included (currently inert
  for de_mirage, but kept for portability).
- `inspect_awpy_data.py` — full introspection of `awpy.data` exports (classifies
  every name, dumps class annotations, summarizes bundled data objects, prints full
  source of `create_nav_graphs`/`transform_csv_to_json`).
- Direct `de_mirage.txt` parser — classifies every area ID as fully-isolated /
  destination-only / source-only / orphaned-but-bidirectionally-connected-to-other-
  orphans, with no awpy dependency.
- `get_orphan_centroids.py` — computes and exports centroid (bounding-box midpoint)
  per orphan area to CSV, cross-validated against the graph's own `center` node
  attribute (100% match confirmed).
- `plot_orphan_clusters.py` — DBSCAN spatial clustering + labeled plot of just the
  72 orphan tiles, colored by cluster membership.

---

*Compiled from project chat history. Last updated after confirming CT-stairs and
Apartments-ramp identifications with Zayaan directly (ground truth, not inferred).*
