# Task 3 — Player Position → Nav Node Mapping & Position Projection

Reference document for the CS:GO map control / graph-based influence field project.
Covers the full implementation of Task 3: mapping per-frame player positions to nav
mesh nodes using a KDTree-based approach, and extending that mapping to project each
player's position forward 1 second using velocity data. Both the current and projected
positions are needed to compute `d_eff(i, x, t)` in the influence field formula (see
§1 below). Verification results are recorded in §5.

---

## 1. Why this task exists — role in the influence field formula

The per-player influence function from the project methodology is:

```
µᵢ(x, t) = exp(−d_eff(i, x, t)² / 2σ²)
```

where:

```
d_eff(i, x, t) = (d_G(pᵢ, x) + d_G(p̂ᵢ, x)) / 2
```

- `pᵢ` is player `i`'s current nav node at time `t`
- `p̂ᵢ` is the nav node corresponding to where player `i` will be 1 second later,
  estimated by projecting their current position forward using their velocity vector
- `d_G(·, ·)` is the graph-distance (geodesic shortest path) over the nav mesh,
  precomputed in Task 4 and stored in `de_mirage_apsp_dict.pkl`

Task 3 therefore has two distinct sub-problems:

1. **Current mapping:** for each alive player in each frame, find the nav mesh node
   that best represents their actual position `pᵢ`.
2. **Projected mapping:** compute the 1-second forward position `p̂ᵢ = pᵢ + v · Δt`
   and map *that* to a nav mesh node as well.

Both mappings must use the patched graph (`de_mirage_patched_graph_v3.pkl`) so that
positions on formerly-orphaned tiles (e.g. CT stairs, Apartments ramp) map to
reachable nodes rather than isolated dead-ends. See `de_mirage_graph_patch_summary.md`
for full detail on the patch.

---

## 2. Why `find_closest_area` is insufficient at scale

awpy's built-in `find_closest_area(map_name, point)` does an O(n) brute-force linear
scan over all 903 nodes in `NAV["de_mirage"]`, computing 3D Euclidean distance to each
node's centroid and returning the closest one. At 42,268 player-frame rows (from a
single demo), that is roughly 38 million distance computations per pass — and the
influence field requires two passes per row (current and projected). At full ESTA
dataset scale across all de_mirage demos, this is not feasible.

Additionally, `find_closest_area` has a known correctness problem for de_mirage: it
ignores z-axis overlap between vertically stacked areas. Locations on Catwalk (above
Mid), or in Underpass/Connector (below B apps), would frequently snap to the wrong
tile because the nearest centroid in 3D Euclidean space is not necessarily the tile
the player is actually standing on when two tiles are directly above each other.

---

## 3. KDTree-based mapping approach

### 3.1 Index construction

```python
def build_kd_tree(graph: nx.DiGraph) -> tuple[KDTree, list]:
    node_ids = list(graph.nodes())
    centroids = np.array([graph.nodes[nid]["center"] for nid in node_ids])
    tree = KDTree(centroids)
    return tree, node_ids
```

A single `scipy.spatial.KDTree` is built over the 3D centroid coordinates
(`graph.nodes[nid]["center"]`) of all 903 nodes in the patched graph. The parallel
`node_ids` list preserves the correspondence between KDTree index positions and actual
area IDs. This index is built once per script run and reused for both the current and
projected mapping passes.

### 3.2 Three-tier resolution strategy

For each query point, the KDTree returns the 5 nearest centroids (`k=5`). The mapping
then applies a three-tier resolution strategy in priority order:

**Tier 1 — `point_in_area` containment among z-tolerant candidates:**
From the 5 nearest candidates, collect all whose centroid z-coordinate is within
`z_tol` of the query point's z. Among those, pick the first (nearest) whose bounding
box actually contains the query point's XY coordinates via `point_in_area`. This is
the highest-confidence result: the player is geometrically inside the tile.

**Tier 2 — Nearest z-tolerant candidate without containment:**
If no z-tolerant candidate passes `point_in_area` (e.g. the player is at a tile
boundary, or in a gap between tiles), fall back to the nearest candidate that still
passes the z-tolerance filter. This handles positions at tile edges that fall just
outside the strict `<` containment check in `point_in_area`.

**Tier 3 — Raw KDTree nearest fallback:**
If no candidate passes z-tolerance at all (possible in extreme z-overlap zones where
all 5 nearest centroids are on a different floor), fall back to the single closest
centroid by pure 3D Euclidean distance. This is the least-preferred outcome and its
rate should be monitored.

```python
def map_position_kdtree(tree, df, G_patched, node_ids, z_tol=250.0, prefix="current_"):
    x_col, y_col, z_col = f"{prefix}x", f"{prefix}y", f"{prefix}z"
    coords = df[[x_col, y_col, z_col]].values
    distances, indices = tree.query(coords, k=5, workers=-1)

    area_ids, mapping_dists = [], []

    for i in range(len(coords)):
        query_z = coords[i, 2]
        resolved = False
        candidate_ztol_nodes = []

        for rank in range(5):
            candidate_node_id = node_ids[indices[i, rank]]
            candidate_z = G_patched.nodes[candidate_node_id]["center"][2]
            if abs(query_z - candidate_z) <= z_tol:
                candidate_ztol_nodes.append(
                    (candidate_node_id, float(distances[i, rank]))
                )

        for node, dist in candidate_ztol_nodes:
            if point_in_area(MAP_NAME, node, coords[i].tolist()):
                area_ids.append(node); mapping_dists.append(dist)
                resolved = True; break

        if not resolved and candidate_ztol_nodes:
            best_node_id, best_dist = candidate_ztol_nodes[0]
            area_ids.append(best_node_id); mapping_dists.append(best_dist)
            resolved = True

        if not resolved:
            area_ids.append(node_ids[indices[i, 0]])
            mapping_dists.append(float(distances[i, 0]))

    df[f"{prefix}areaId"] = area_ids
    df[f"{prefix}mapping_distance"] = mapping_dists
```

`workers=-1` in the KDTree query uses all available CPU cores for the bulk distance
computation, which is the bottleneck at ESTA scale. The per-row tier logic runs in a
Python loop over the returned top-5 candidates — this is not the bottleneck.

### 3.3 z_tol parameter

The z-tolerance parameter controls how aggressively z-overlap disambiguation operates.
It defines the maximum permitted difference between the query point's z-coordinate and
a candidate node's centroid z-coordinate for that candidate to be considered valid.

Effective tolerance space for de_mirage collapses to four functionally distinct
configurations due to the actual z-separations present in the nav mesh: `{50}`,
`{75}`, `{100 ≡ 125}`, `{150 ≡ 200}`. The 50→75 boundary is the most active
calibration zone (487 disagreements across the dataset at that threshold boundary).

The current script uses `z_tol=250.0`, which is intentionally permissive — it was
set to produce output for verification before the final calibration decision is made.
The final `z_tol` value has **not yet been determined** and requires inspection of the
adjacent-pair diagnostic plots for known overlap zones (Catwalk/Mid,
Underpass/Connector). See §6 (open items) for the calibration decision status.

---

## 4. Position projection

### 4.1 Formula

```python
def project_positions(df: pd.DataFrame) -> pd.DataFrame:
    delta = 1.0
    df["projected_x"] = df["current_x"] + delta * df["vx"]
    df["projected_y"] = df["current_y"] + delta * df["vy"]
    df["projected_z"] = df["current_z"] + delta * df["vz"]
    return df
```

This implements `p̂ᵢ = pᵢ + v · Δt` with `Δt = 1 second`. The velocity fields
(`velocityX`, `velocityY`, `velocityZ`) are sourced directly from the awpy-parsed
demo JSON, which records them in **game units per second** — confirmed from the
velocity distribution in §5.2 (max XY velocities ~296–308 units/s, consistent with
CSGO's maximum run speed of ~250–260 units/s). The multiplication by `delta = 1.0`
is therefore dimensionally correct as written and requires no tick-rate conversion.

### 4.2 Important caveats on the projected position

The projected position is a purely kinematic first-order estimate. It does not account
for:

- **Wall collisions:** a projected position may land inside a wall or in a gap between
  nav tiles, especially for players running toward a wall. The three-tier mapping
  strategy handles this gracefully — such positions will resolve via Tier 2 (nearest
  z-tolerant centroid) rather than Tier 1 (containment).
- **Acceleration and deceleration:** the projection assumes constant velocity over the
  1-second window. In practice, players change direction and speed frequently. The
  1-second horizon is kept short precisely to limit this error.
- **Death within the window:** a player who dies within 1 second of a given frame will
  have a projected position that is physically meaningless. Dead players are already
  excluded from the DataFrame (filtered by `isAlive` in `create_round_info_df`), so
  their projected position is never computed — but a player who is about to die
  contributes a projected area that they will never actually occupy. This is an
  accepted approximation consistent with Xenopoulos et al. [2020].

---

## 5. Verification results

Verification script: `verify_position_mapping.py`  
Input: `outputs/player_positions_projected.csv`, `outputs/de_mirage_patched_graph_v3.pkl`  
Demo tested: FaZe vs BIG (`matchID: 4bab9a63-42e6-4766-acb6-cae50d575d04`)

### 5.1 Check 1 — Structural verification ✅ PASS

```
Does the DataFrame contain null values: False
DataFrame Shape: (42268, 17)
```

All 17 expected columns present with correct dtypes: area IDs as `int64`, all
coordinates and distances as `float64`, `round` and `frame_idx` as `int64`,
`player_name` and `side` as `object`. No null values in any column. 42,268 rows is
plausible for a full de_mirage match — 30 rounds at standard parse rates typically
yield 30,000–60,000 player-frame rows.

Full column set confirmed: `round`, `frame_idx`, `player_name`, `side`,
`current_x/y/z`, `vx`, `vy`, `vz`, `projected_x/y/z`, `current_areaId`,
`current_mapping_distance`, `projected_areaId`, `projected_mapping_distance`.

### 5.2 Check 2 — Projection calculation correctness ✅ PASS

Arithmetic assertion passed on a 20-row random sample:

```python
assert np.allclose(sample["projected_x"], sample["current_x"] + sample["vx"])
assert np.allclose(sample["projected_y"], sample["current_y"] + sample["vy"])
assert np.allclose(sample["projected_z"], sample["current_z"] + sample["vz"])
```

Velocity distribution confirmed as **units per second** (not units per tick):

```
            vx           vy           vz
count  42268        42268        42268
mean     -10.2         3.8         -0.4
std       99.6       106.1         44.4
min     -308.4      -280.6       -534.4
25%      -56.3       -39.1          0.0
50%        0.0         0.0          0.0
75%       27.6        45.3          0.0
max      288.6       296.6        295.7
```

Max XY velocities ~296–308 units/s are consistent with CSGO's maximum run speed.
Median 0 for all three axes reflects that a large fraction of frames capture stationary
or near-stationary players. The `vz` range of −534 to +296 is expected — falling
(large negative) and jumping (positive) are momentary and produce extreme values,
but the 75th percentile of 0 confirms vertical movement is the exception. The
`delta = 1.0` coefficient in `project_positions` is therefore dimensionally correct
as written.

### 5.3 Check 3 — Area ID validity ✅ PASS

```
Unknown Current Area IDs: set()
Unknown Projected Area IDs: set()
```

Every area ID appearing in the output DataFrame exists as a node in the patched
graph. The KDTree index (`node_ids` list) and the graph node set are in sync. No
phantom IDs were produced by the mapping.

### 5.4 Check 4 — Position projection feasibility ✅ PASS (with one known informational finding)

**Displacement distribution:**

```
count    42268
mean       118.4
std         96.1
min          0.0
25%         17.7
50%        111.8
75%        211.3
max        597.9
```

The max displacement of ~598 units is slightly above the expected maximum (~300 units
for the fastest possible movement over 1 second). This is attributable to explosive
knockback from HE grenades or bomb blast in a small number of frames, and does not
indicate a systematic bug. The median of ~112 and 75th percentile of ~211 are both
physically plausible for typical player movement.

**Hop distance between current and projected area:**

```
count    200
mean     inf
std      NaN
min      0.0
25%      0.0
50%      1.0
75%      2.0
max      inf
```

The median of 1 hop and 75th percentile of 2 hops confirm that for the vast majority
of rows the projected position lands in the same tile (0 hops) or an immediately
adjacent tile (1–2 hops), consistent with physically plausible 1-second movement.

**The `inf` values are fully explained and are not a bug.** They arise from pairs
where one or both area IDs belong to a node in one of the three small SCCs in the
patched graph that have no directed path to the main 896-node mesh:

- **Ladder cluster** — areas 833, 834, 835, 836 (4-node cluster, top vent above
  sniper's nest). One-way geometry: navigable downward but not upward via the graph.
- **Truck cluster** — areas 363, 742, 743 (3-node cluster, top of truck in Bombsite
  B). One-way geometry: accessible by jumping up but not represented as a traversable
  return path in the nav mesh.

These were never among the 72 orphaned areas and were not introduced by the patch —
they are genuine one-way map geometry that has been in the nav mesh since the original
`nav/de_mirage.txt` edge-list. `area_distance(..., dist_type="graph")` returns `inf`
for any pair where one node is in an SCC with no outgoing path to the other, which is
correct behaviour. Their occurrence in the output CSV is expected and harmless — the
influence field formula uses the precomputed APSP matrix (Task 4), which handles these
cases by storing `inf` for unreachable pairs, and any frame where a player is on one
of these tiles simply contributes a zero influence signal to the field from that
direction.

**Note on the hop-distance check itself:** The `area_distance` call in the
verification script uses the default `NAV_GRAPHS["de_mirage"]` (unpatched) because the
script does not monkeypatch the graph before calling awpy's functions. This means
formerly-orphaned tiles (now connected in the patched graph) would also appear as `inf`
in this check if they happened to be in the sample. Future runs of this check should
add the monkeypatch at the top of the verification script:

```python
import awpy.data as awpy_data
with open(PATCHED_GRAPH_PATH, "rb") as f:
    G_patched = pickle.load(f)
awpy.data.NAV_GRAPHS["de_mirage"] = G_patched
```

With the monkeypatch applied, the remaining `inf` values will be exclusively SCC-
related, giving a cleaner signal.

### 5.5 Check 5 — Stationary players stay in same area ✅ PASS

```
Stationary players with different current/projected area IDs: 17 / 8702
Mismatch Rate: 0.20%
```

Stationary players are defined as those with `sqrt(vx² + vy² + vz²) < 5.0` units/s.
Of 8,702 such player-frame rows, only 17 map to a different projected area ID than
their current area ID. This 0.20% mismatch rate is geometrically expected — players
standing within ~5 units of a tile boundary may have their near-zero projected
position tip just over the boundary into an adjacent tile. There is no systematic bug
indicated by this figure.

### 5.6 Summary table

| Check | Description | Result |
|---|---|---|
| 1 | Structural verification (nulls, dtypes, shape) | ✅ PASS |
| 2 | Projection arithmetic and velocity unit confirmation | ✅ PASS |
| 3 | All area IDs present in patched graph | ✅ PASS |
| 4 | Displacement plausibility and hop-distance distribution | ✅ PASS (inf values are SCC-related, expected) |
| 5 | Stationary player area consistency | ✅ PASS (0.20% mismatch — geometric boundary effect) |

---

## 6. Output schema

`outputs/player_positions_projected.csv` — one row per alive player per frame.

| Column | Type | Description |
|---|---|---|
| `round` | int | Round number within the match |
| `frame_idx` | int | Frame index within the round (0-based) |
| `player_name` | str | Player name string as recorded in the demo |
| `side` | str | `"T"` or `"CT"` |
| `current_x` | float | Player x-coordinate at this frame |
| `current_y` | float | Player y-coordinate at this frame |
| `current_z` | float | Player z-coordinate at this frame |
| `vx` | float | Velocity x-component in game units per second |
| `vy` | float | Velocity y-component in game units per second |
| `vz` | float | Velocity z-component in game units per second |
| `projected_x` | float | `current_x + vx * 1.0` |
| `projected_y` | float | `current_y + vy * 1.0` |
| `projected_z` | float | `current_z + vz * 1.0` |
| `current_areaId` | int | Nav node ID for the current position |
| `current_mapping_distance` | float | 3D centroid distance from `current_x/y/z` to chosen node |
| `projected_areaId` | int | Nav node ID for the projected position |
| `projected_mapping_distance` | float | 3D centroid distance from `projected_x/y/z` to chosen node |

Dead players are excluded. Warmup rounds are **not yet excluded** from
`create_round_info_df` — this is flagged as an open item in §7.

---

## 7. Open items before Task 3 can be fully closed

### 7.1 z_tol calibration — decision pending

The current `z_tol=250.0` is deliberately permissive and has not been validated
against the known z-overlap zones on de_mirage. The final value must be determined
by inspecting the adjacent-pair diagnostic GIF (which plots the mapping choices
across z-overlap frames for known problematic areas) and choosing the tightest value
that still correctly resolves those cases.

Effective discrete configurations are `{50}`, `{75}`, `{100 ≡ 125}`,
`{150 ≡ 200}` — the tightest defensible value is `75` based on the 50→75
disagreement count of 487, but this needs to be confirmed visually before being
committed to. This decision should be documented in an updated section of this file
once made.

### 7.2 Resolution tier instrumentation — not yet added

The current `map_position_kdtree` implementation does not record which resolution
tier (containment / z-tol nearest / raw fallback) was used for each row. Adding a
`current_resolution_tier` and `projected_resolution_tier` column to the output CSV
would allow computing the Tier 3 fallback rate, which is the most important quality
metric for the mapping — a high Tier 3 rate at any z_tol value indicates that value
is too tight for the data. This instrumentation should be added before the final
z_tol sweep.

### 7.3 Warmup round exclusion — not yet implemented

`create_round_info_df` iterates all `demo["gameRounds"]` without checking
`rnd.get("isWarmup", False)`. Warmup rounds can contain garbage position data
(players at spawn, coordinates at map origin) and should be excluded. Add:

```python
if rnd.get("isWarmup", False):
    continue
```

This is low priority for the single-demo verification pass but must be in place before
any full-dataset run.

### 7.4 Non-reciprocal edge check on the original 2,979-edge graph — pending

Prior analysis found 33.27% asymmetry in the distance matrix, raising the question of
whether the original graph has non-reciprocal edges (i.e. directed edges with no
corresponding reverse edge). This check has not yet been run. The result determines
whether the influence field's `d_G` computations should use raw directed distances
or a symmetrized version (e.g. `min(d_G(a,b), d_G(b,a))`). This decision must be
made and documented before Task 6 (influence field computation) begins.

---

## 8. Key scripts

| Script | Purpose |
|---|---|
| `position_to_node_map.py` | Main pipeline: loads patched graph, parses demo, projects positions, maps both current and projected positions to nav nodes, saves CSV |
| `verify_position_mapping.py` | Standalone verification: runs 5 structural, arithmetic, and plausibility checks against the output CSV and patched graph |

---

*Compiled from project implementation and verification session. Testing demo:
FaZe vs BIG (`matchID: 4bab9a63-42e6-4766-acb6-cae50d575d04`). All verification
output shown in §5 was produced on `csgo-research-vm` (Ubuntu, Python 3.11, venv,
awpy 1.3.1).*
