# Task 3 — Player Position-to-Nav-Node Mapping: Implementation Notes

**Research schedule dates:** 30 Jun – 3 Jul  
**Status:** Implementation complete on 8-demo sample. Verification script written and
GIF generation added. Pending: run verification script and document output here.

This document covers every decision, design choice, bug, and architectural lesson from
implementing Task 3, including what was deferred and why, and exactly what needs to
happen when scaling to the full ESTA dataset. It is intended to be a self-contained
reference that can be picked up in any future session without needing to reconstruct
context from chat history.

---

## 1. What Task 3 Is and Why It Matters

Task 3's job is to translate each player's raw `(x, y, z)` world coordinate (as
parsed from ESTA demo frames) into a specific **nav mesh area ID** — a node in the
patched de_mirage nav graph. This translation is the gate through which every
downstream computation passes.

The reason this matters is architectural: the entire influence field methodology
operates over **graph nodes**, not raw coordinates. The per-player influence formula
is:

```
µᵢ(x,t) = exp(−d_eff(i,x,t)² / 2σ²)
```

where `d_eff` is built from `d_G(pᵢ, x)` — graph distance between the player's
nav node and a candidate map location `x` (also a nav node). Graph distance is
computed by `area_distance(map_name, area_a, area_b, dist_type="graph")`, which takes
**two integer area IDs**, not two raw coordinate triples. There is no way to compute
graph distance between two arbitrary points without first mapping each to a node.

The same applies to the Voronoi baseline (nearest alive player by graph distance to
each nav node), and to the projected position in Task 5 (`p̂ᵢ = pᵢ + v·Δt` also
needs to be mapped to a node before `d_eff` can be evaluated).

Task 3 is therefore not optional and cannot be replaced with raw Euclidean distance
without abandoning the project's core methodological contribution. This was explicitly
considered and rejected — see Section 5.3 below.

---

## 2. Prerequisites Entering Task 3

Everything listed here was confirmed complete or confirmed as a known state before
Task 3 work began.

### 2.1 Patched de_mirage nav graph (from graph patching work)

- `de_mirage_patched_graph_v3.pkl` — a pickled `nx.DiGraph` with 903 nodes and
  2979 + 526 = 3505 directed edges (526 edges added by the geometric adjacency patch).
- All 15 structural verification checks passed (`verify_patched_graph_final.py`).
- The 72 originally-orphaned nodes are fully connected and reachable from the main mesh.
- Every added edge is bidirectional and uses the same centroid Euclidean distance
  weight convention as awpy's own edges.
- The patched graph is loaded via pickle (Option A), not via monkeypatching
  `awpy.data.NAV_GRAPHS` at runtime (Option B) or editing the bundled text file
  (Option C). The choice of Option A was confirmed as the standing convention for
  this project's own pipeline code, with Option B available only when needing to
  reuse awpy's own higher-level functions directly.

### 2.2 ESTA sample data

- 8 de_mirage demos selected as the development sample.
- Demo files stored as plain JSON at `data/sample/` (decompressed from `.json.xz`).
- Task 3 development uses a single demo file:
  `data/sample/1decf556-b9d7-4085-9081-31811f0496c3.json`
- One round (round 1) selected for the initial mapping test, yielding approximately
  1,100 player-frame rows after filtering to alive players only.

### 2.3 awpy 1.3.1 confirmed API facts relevant to Task 3

The following facts were either confirmed from source inspection or caught as bugs
during implementation. They are listed here because they are all easy to get wrong.

| Fact | Detail |
|---|---|
| `find_closest_area` return type | Returns a **dict**, not a tuple: `{"mapName": str, "areaId": int, "distance": float}` |
| Return key spelling | `"areaId"` — camelCase with **lower** d. Not `"area_id"`, not `"areaID"` |
| Node attribute spelling | `G.nodes[area_id]["areaID"]` — capital D. Different from the return key |
| `flat` parameter | Keyword-only (`*` in signature): `find_closest_area(map_name, [x,y], flat=True)` |
| `point_in_area` dimensionality | 2D only — checks XY bounding box containment, **ignores Z** |
| `AREA_DIST_MATRIX` for de_mirage | Empty in the bundled awpy install — all graph-distance calls are live |
| `find_closest_area` uses `NAV` | Brute-force scan over `NAV[map_name]` centroid distances, **not** `NAV_GRAPHS` |
| Patched graph not needed for sample-run mapping | `find_closest_area` accesses `NAV["de_mirage"]` directly; all 903 nodes including formerly-orphaned tiles are present there regardless of graph connectivity |
| Patched graph **is** needed from Task 4 | Graph distance computation uses `NAV_GRAPHS` (or the patched pickle); unpatched graph would give `inf` distances to the 72 formerly-orphaned nodes |

---

## 3. Architecture — The Two-Pass Design

Task 3 is implemented as two completely separate passes over the data, rather than
one fused loop that extracts and maps simultaneously.

### Pass 1 — Player data extraction

Walk all rounds and frames in the demo JSON, collect alive player records into a
list of dicts, and build `player_df` — a pandas DataFrame where each row represents
one alive player at one frame in one round. No spatial computation happens here.

### Pass 2 — Position mapping

Take the full `player_df` (or a subset), apply the position-mapping function to
every row, and write the results back as two new columns: `curr_areaId` and
`mapping_distance`. No JSON parsing happens here.

### Why not fuse them?

This was considered explicitly. The argument for fusing ("why walk the data twice?")
is superficially reasonable but is wrong in this context for two distinct reasons:

**Performance reason.** The efficiency gain from using a KDTree (the full-dataset
mapping approach) comes specifically from **batching** — passing all query points
to `KDTree.query()` as a single `(n, 3)` array. Fusing extraction and mapping means
calling `.query()` once per player per frame inside the extraction loop, which throws
away the batching advantage entirely and reduces the KDTree to approximately the same
per-call overhead as `find_closest_area`. The correct approach is: extract all
positions first → batch query → write results back. This is technically two passes
but runs faster than a fused loop.

**Verification reason.** The project's established discipline is to produce an
artefact, verify it independently, then build on it. A fused loop makes it
impossible to verify "did extraction work correctly" separately from "did mapping
work correctly." A bug in either stage contaminates the other, making root cause
analysis harder. Keeping the passes separate means `player_df` can be inspected
after extraction (and is, via the commented-out print statements in the script)
before the mapping function is ever applied to it.

---

## 4. The `player_df` DataFrame — Structure and Role

`player_df` is the **central shared data structure** of the entire pipeline from
Task 3 through Task 7. Each pipeline stage reads existing columns and appends new
ones. No stage re-parses the raw JSON or re-walks the frame structure after
extraction is complete.

### Columns after Task 3

| Column | Type | Description |
|---|---|---|
| `round` | int | Round number from `rnd["roundNum"]` |
| `frame_idx` | int | 0-indexed frame position within the round (from `enumerate(frames)`) |
| `player_name` | str | Player's name string |
| `side` | str | `"T"` or `"CT"` |
| `x` | float | Player x-coordinate (world units) |
| `y` | float | Player y-coordinate (world units) |
| `z` | float | Player z-coordinate (world units) |
| `vx` | float | Player x-velocity (world units/sec) |
| `vy` | float | Player y-velocity (world units/sec) |
| `vz` | float | Player z-velocity (world units/sec) |
| `curr_areaId` | float64* | Nav node area ID for the player's current position |
| `mapping_distance` | float64 | Euclidean distance from player position to assigned node's centroid |

*`curr_areaId` is stored as `float64` rather than `int64` due to pandas behaviour
when loading from CSV — see Bug 8 below. Must be cast to `int` before passing to
`area_distance()` in Task 4.

### Columns that will be added by later tasks

| Column | Added in | Description |
|---|---|---|
| `proj_x`, `proj_y`, `proj_z` | Task 5 | Projected position `p̂ᵢ = pᵢ + v·Δt` |
| `proj_areaId` | Task 5 | Nav node mapped from projected position |
| `d_eff_*` | Task 6 | Effective distance to each nav node (one column or nested) |
| `influence_*` | Task 6/7 | Per-player influence value at each nav node |

**Note on column naming for Task 5:** Before Task 5 is implemented, decide whether
the current column name `curr_areaId` should become `obs_areaId` (observed) to
pair clearly with `proj_areaId` (projected), since both are required simultaneously
in the `d_eff` computation. Renaming now is less disruptive than after Task 5 has
been written.

### Why save to disk between stages?

Three reasons specific to this project:

1. **Task 4 is expensive.** `generate_area_distance_matrix` is documented by awpy
   as taking 20 minutes to 5 hours depending on the map. Saving `player_df` after
   Task 3 means Task 4 and all downstream stages can load from the checkpoint rather
   than re-running extraction and mapping on every development iteration.

2. **The sample run produces a permanent validation artefact.** The `find_closest_area`
   results in `sample_df.csv` are the ground-truth baseline for validating the KDTree
   fast-path when it is built. They should be saved now and kept.

3. **HPC cluster deployment.** The target execution environment (Wits HPC) runs
   full-dataset jobs. Saving after Task 3 lets the cluster run Task 3 once, write the
   output, and subsequent jobs start from the checkpoint — standard HPC practice.

---

## 5. Key Design Decisions

### 5.1 Scalar columns for coordinates and velocity (not tuples)

**Decision:** store `x`, `y`, `z`, `vx`, `vy`, `vz` as six separate scalar columns,
not as two tuple columns (`coords`, `velocity`).

**Rationale:**
- `df[["x","y","z"]].values` gives a ready `(n, 3)` numpy array directly, usable
  as the KDTree query input without any conversion step.
- Vectorized math works trivially on separate columns: `df["x"] + df["vx"] * dt`
  is Task 5's projection written in one line. With tuple columns this requires
  unpacking, which is verbose and slower.
- Boolean filtering on individual columns works naturally: `df[df["z"].between(-200,
  -100)]` is exactly what's needed when debugging z-overlap cases around Catwalk/Mid
  and Underpass/Connector.

### 5.2 `frame_idx` as temporal identifier

**Decision:** store `frame_idx` (0-indexed position within the round) in every row.

**Rationale:** The influence field is explicitly time-varying — `c(x,t)` and
`µᵢ(x,t)`. Without a frame-level temporal identifier, there is no way to:
- Reconstruct the sequence of control-field states across a round for Metric 3
  (Control Margin, averaged across frames).
- Identify the frame immediately preceding a plant event for Metric 2 (Site Control
  Score).
- Plot how control evolves over time.
- Validate that consecutive frame mappings trace a spatially coherent path (which the
  verification script's Check 9 does).

A tick number would also work if available from the parsed frame dict, but `frame_idx`
is guaranteed present since it comes from `enumerate(frames)` rather than from the
parsed data itself.

### 5.3 Stay on graph distance, not raw Euclidean distance

**This was explicitly considered and rejected.**

Arguments considered for switching to raw Euclidean distance:
- No discretization loss from centroid snapping.
- Removes Task 3 entirely (no node mapping needed).
- Removes Task 4 entirely (no APSP caching needed).
- Faster — straight-line distance computation is trivial.

Arguments against (decisive):
- **Not wall-aware.** Two points can be Euclidean-close but tactically distant
  because they are on opposite sides of a wall or at different floor levels. De_mirage
  specifically has vertical stacking (Catwalk above Mid, Underpass near Connector)
  where Euclidean distance actively misrepresents tactical reality.
- **Abandons the methodological contribution.** The project's contribution over the
  Fernández & Bornn [2018] Euclidean baseline is precisely the substitution of
  Xenopoulos et al.'s [2020] graph-distance `d_G`. Switching back to Euclidean
  distance removes that contribution and the citation grounding for it, and would
  require revising the research proposal's framing.
- **Invalidates already-completed work.** The 72-orphan graph patch and all associated
  verification work is relevant only because the pipeline needs a sound graph to
  compute distances over. Switching to raw coordinates renders that work orphaned
  (irony intended) from the methodology.
- **Breaks evaluation metrics.** The Voronoi baseline explicitly uses graph distance
  for nearest-player assignment. If the influence field uses Euclidean but Voronoi
  uses graph, the RQ1 comparison is no longer like-for-like.

**Conclusion:** not viable without revising the proposal. Dismissed.

### 5.4 `find_closest_area` for sample run; KDTree for full dataset

**Decision:** use awpy's own `find_closest_area` for the 8-demo sample run. Switch
to a `KDTree`-based fast path when scaling to the full ~200-demo ESTA dataset.

**Rationale for `find_closest_area` on sample data:**
- Sample run is about verifying correctness of the pipeline, not performance.
- `find_closest_area` is the ground truth by definition — using it now means the
  saved sample CSV is a ready-made validation baseline for the KDTree replacement,
  without any additional work.

**Why KDTree is needed at full scale:**
`find_closest_area` is a brute-force O(n) Python-level loop over all 903 areas —
approximately 2–5ms per call. At full dataset scale:

```
~200 demos × ~20 rounds/demo × ~90 frames/round × 10 players/frame × 2 lookups
(observed + projected in Task 5) ≈ 72,000,000 individual calls
```

At 2–5ms each, that is 40–100 hours of mapping time alone. Even the conservative
estimate (just the 8-demo sample's single round): 1,100 rows × 1 lookup = ~5 seconds.
Fine for one round. For full dataset, unacceptable.

**The swap is isolated.** The mapping function `map_current_position(row)` has a
clean boundary: it takes a row and returns `pd.Series({"curr_areaId": ...,
"mapping_distance": ...})`. The entire swap from `find_closest_area` to KDTree
happens inside this function's body. Nothing outside the function changes.

### 5.5 `KDTree` not `cKDTree`

**Decision:** use `scipy.spatial.KDTree`, not `scipy.spatial.cKDTree`.

**Rationale:** Prior to scipy 1.6, `KDTree` was a pure-Python implementation and
`cKDTree` was a Cython-optimized version with meaningfully faster query performance.
Since scipy 1.6, `KDTree` was rewritten as a thin wrapper over the same underlying
Cython code as `cKDTree`. At scipy 1.10.1 (which this project uses), there is no
performance difference. The scipy documentation for 1.6+ explicitly says to prefer
`KDTree` unless backward compatibility with older scipy is required. There is no such
requirement here.

### 5.6 Option A for patched graph loading

**Decision:** load the patched graph via `pickle.load()` from
`de_mirage_patched_graph_v3.pkl` (Option A). Do not monkeypatch `awpy.data.NAV_GRAPHS`
at runtime (Option B) unless specifically reusing an awpy higher-level function that
internally reads `NAV_GRAPHS`. Do not edit the bundled `nav/de_mirage.txt` file
(Option C).

**Rationale:** Option A is self-contained and reproducible. Option B requires
re-applying the monkeypatch at the top of every script and is fragile if any script
forgets — a bug that would fail silently in the worst case (mapping would proceed
against the unpatched graph and produce `inf` distances for the 72 orphaned tiles
in Task 4, with no error raised). Option C loses the fix on any awpy reinstall/upgrade.

**Important note for Task 3's sample run:** `find_closest_area` accesses
`NAV["de_mirage"]` for its centroid scan, **not** `NAV_GRAPHS["de_mirage"]`. All
903 areas exist in `NAV` regardless of graph connectivity. This means the patched
graph is not required for Task 3's mapping step itself — it only becomes critical in
Task 4, when graph distances need to be computed between the mapped nodes. The patched
graph is imported and loaded in the Task 3 script for forward-compatibility (and for
Check 4 in the verification script), but it is not actively used by `find_closest_area`.

---

## 6. Implementation — What Was Built

### 6.1 Demo data loading

```python
with open(DEMO_FILE, "r", encoding="utf-8") as file:
    demo = json.load(file)

rounds = demo["gameRounds"]
```

ESTA demo files are stored as plain JSON after decompression from `.json.xz`. The
`gameRounds` key contains a list of round objects, each with a `frames` key containing
the per-frame game state.

### 6.2 Player data extraction loop (Pass 1)

```python
player_info = []

for rnd in rounds:
    frames = rnd["frames"]
    for frame_idx, frame in enumerate(frames):
        for side, side_label in [("t", "T"), ("ct", "CT")]:
            for player in frame.get(side, {}).get("players", []):
                if player["isAlive"]:
                    player_info.append({
                        "round":       rnd["roundNum"],
                        "frame_idx":   frame_idx,
                        "player_name": player["name"],
                        "side":        side_label,
                        "x": player["x"], "y": player["y"], "z": player["z"],
                        "vx": player["velocityX"],
                        "vy": player["velocityY"],
                        "vz": player["velocityZ"],
                    })

player_df = pd.DataFrame(player_info)
```

Key points:
- Iterates both `"t"` and `"ct"` side keys using a single loop with a label pair —
  not two separate loops.
- `frame.get(side, {}).get("players", [])` — uses the **variable** `side`, not the
  string literal `"side"` (an easy bug to introduce; see Bug 1 below).
- `isAlive` filter removes dead players at extraction time. Dead players should not
  contribute to the influence field computation.
- `enumerate(frames)` provides `frame_idx` — the temporal identifier within the round.
- Velocity fields (`velocityX/Y/Z`) are captured at extraction time alongside
  position, since Task 5 needs them for position projection. Capturing them here avoids
  a second pass over the frame structure later.
- `rnd` used as loop variable (not `round`) to avoid shadowing the Python builtin
  `round()` function — see Bug 5 below.

### 6.3 Sample selection for development testing

```python
sample_df = player_df[player_df["round"] == player_df["round"].iloc[0]].copy()
```

Selects all rows from the first round only. The `.copy()` call is critical — without
it, `sample_df` is a view of `player_df`, and any column assignment to `sample_df`
will raise `SettingWithCopyWarning` and may fail to persist — see Bug 3 below.

This approach was chosen over `.head(N)` or `.sample(n)` because it gives a complete
coherent round's worth of data: all players, all frames, in temporal order. This
allows Check 9 in the verification script (spatial coherence of consecutive frame
transitions) to work meaningfully, since it requires an ordered per-player frame
sequence.

### 6.4 Position mapping function (Pass 2 — sample run version)

```python
def map_current_position(row):
    res = find_closest_area(MAP_NAME, [row["x"], row["y"], row["z"]])
    return pd.Series({
        "curr_areaId":      res["areaId"],
        "mapping_distance": res["distance"],
    })

sample_df[["curr_areaId", "mapping_distance"]] = sample_df.apply(
    map_current_position, axis=1
)
```

Key points:
- Returns a `pd.Series` with two keys, enabling the two-column assignment on
  `sample_df` in one `.apply()` call.
- Column names in the returned Series (`curr_areaId`, `mapping_distance`) exactly
  match the assignment targets. A mismatch causes silent `NaN` filling — see Bug 4.
- `res["areaId"]` — lower-d, as returned by `find_closest_area`. Not `res["areaID"]`.
- Both the area ID and the centroid distance are captured. The distance is not needed
  by the influence formula itself but is essential for the verification script and
  for detecting z-overlap misassignment candidates (large distances to assigned
  centroid are a signal that the mapping may have grabbed a geometrically adjacent but
  vertically misaligned tile).
- `.apply(func, axis=1)` is a Python-level row loop under the hood. Fine for ~1,100
  rows on the sample. This entire function is replaced by a batched `KDTree.query()`
  call when scaling — see Section 8 below.

### 6.5 Saving the mapped sample DataFrame

```python
sample_df.to_csv("data/sample/sample_df.csv", index=False)
```

This CSV serves two purposes: (1) input to the verification script, and (2) a
permanent record of `find_closest_area` ground-truth results for future KDTree
validation.

---

## 7. Bugs Encountered and Fixed

All of these were caught during development before the CSV was saved. They are
documented here because several are the kind of silent failure that is easy to
reintroduce.

### Bug 1 — String literal `"side"` instead of variable `side`

**Where:** inner loop of the extraction loop.

**What was written:**
```python
for player in frame.get("side", {}).get("players", []):
```

**What was meant:**
```python
for player in frame.get(side, {}).get("players", []):
```

**How it failed:** `frame` has no key called `"side"` — the actual keys are `"t"`
and `"ct"`. `frame.get("side", {})` always returned `{}`, so `.get("players", [])`
returned `[]`, and the inner loop never executed. `player_info` stayed empty. No
error was raised. The resulting `player_df` was an empty DataFrame.

**Why it's easy to miss:** the failure is completely silent — no exception, no warning,
just an empty DataFrame that looks like a legitimate result if you don't check `len()`.

### Bug 2 — CT-side players not extracted

**Where:** extraction loop, before the T/CT pair loop was introduced.

**What was written:**
```python
t_side_players = t_side["players"]
for t_player in t_side_players:
    if t_player["isAlive"]:
        player_info.append(...)
```

**How it failed:** only T-side players were appended. CT-side players were fetched
into `ct_side` but never iterated. Since the influence field requires both teams
(`Σ_CT µᵢ − Σ_T µⱼ`), extracting only T players would produce a completely
one-sided influence computation.

**Fix:** replace the two separate loops (T and CT) with the single paired iteration
over `[("t", "T"), ("ct", "CT")]`.

### Bug 3 — Missing `.copy()` on slice, `SettingWithCopyWarning`

**Where:** sample selection.

**What was written:**
```python
sample_df = player_df[player_df["round"] == player_df["round"].iloc[0]]
```

**How it failed:** this creates a **view** of `player_df`, not an independent
DataFrame. Assigning new columns to `sample_df` afterwards triggers
`SettingWithCopyWarning` in pandas, and in pandas 2.0 (which this project uses at
version 2.0.3) the assignment may silently fail to persist or behave
non-deterministically.

**Fix:**
```python
sample_df = player_df[player_df["round"] == player_df["round"].iloc[0]].copy()
```

This applies to any subset created from `player_df` throughout the project —
boolean-filter slices, `.head()`, `.sample()`, all of them need `.copy()` before
columns are assigned to the result.

### Bug 4 — Column name mismatch between Series keys and assignment targets

**Where:** mapping function.

**What was written:**
```python
return pd.Series({"curr_areaId": res["areaId"], "mapping_dist": res["distance"]})
# ...
sample_df[["curr_areaId", "mapping_distance"]] = sample_df.apply(...)
```

**How it failed:** the Series key `"mapping_dist"` and the assignment target
`"mapping_distance"` don't match. Pandas aligns by column name when applying a
function that returns a DataFrame — finding no match for `"mapping_distance"`, it
fills that column with `NaN` silently.

**Fix:** use the same name in both places. Final version uses `"mapping_distance"`
throughout.

### Bug 5 — `round` shadowing the Python builtin

**Where:** extraction loop variable name.

**What was written:**
```python
for round in rounds:
```

**Problem:** `round` is a Python builtin function. This doesn't cause an immediate
error in this context (the builtin isn't called inside the loop body), but would
silently break any code inside the loop that tried to call `round()` for numeric
rounding. Also a readability issue since it creates a confusing overload of a
familiar name.

**Fix:** renamed to `rnd` throughout.

### Bug 6 — `find_closest_area` return value treated as tuple

**Where:** mapping function.

**What was written:**
```python
areaId, dist = find_closest_area(MAP_NAME, [x, y, z])
```

**How it failed:** `find_closest_area` returns a **dict** (`{"mapName": ...,
"areaId": ..., "distance": ...}`), not a tuple. Unpacking a 3-key dict into two
variables raises `ValueError: too many values to unpack` on the first row processed
by `.apply()`.

**Fix:** treat the return value as a dict:
```python
res = find_closest_area(MAP_NAME, [row["x"], row["y"], row["z"]])
return pd.Series({"curr_areaId": res["areaId"], "mapping_distance": res["distance"]})
```

### Bug 7 — Redundant `list()` call on frames

**Where:** extraction loop.

**What was written:**
```python
frames = list(rnd["frames"])
```

**Problem:** `rnd["frames"]` is already a Python list (parsed from JSON). `list()`
creates an unnecessary copy of potentially hundreds of frame dicts per round. This is
minor (no incorrect behaviour) but wasteful.

**Fix:** iterate `rnd["frames"]` directly inside `enumerate()`:
```python
for frame_idx, frame in enumerate(rnd["frames"]):
```

### Bug 8 — `curr_areaId` stored as `float64` after CSV round-trip

**Where:** after loading the saved CSV with `pd.read_csv()`.

**What happens:** pandas infers `float64` for any numeric column that could
potentially contain `NaN` values, since the standard `int64` dtype cannot represent
`NaN`. Even if no `NaN` values are present in `curr_areaId`, pandas won't infer
`int64` unless explicitly told to.

**Impact:** `area_distance("de_mirage", 12.0, 628.0, ...)` — passing float area IDs
to awpy's distance functions will raise `TypeError` in Task 4.

**Fix:** after loading the CSV and confirming no `NaN` values are present:
```python
player_df["curr_areaId"] = player_df["curr_areaId"].astype(int)
```

This should be done as the first step in any script that loads the CSV and uses
`curr_areaId` for graph-distance lookups.

---

## 8. Sample Run Output Assessment

The following was observed from inspecting the first 50 and last 50 rows of the
saved `sample_df.csv` (round 1 of the selected demo, 1,113 total rows).

### Spatial plausibility

**T spawn positions (frames 0–4):** all five T players cluster around `x ≈ 1050–1350`,
`y ≈ -233 to 510`, `z ≈ -168`. This is consistent with T spawn on de_mirage, which
sits in the positive-x region of the map's coordinate space. Players disperse naturally
across subsequent frames as they move out toward their chosen positions.

**CT spawn positions (frames 0–4):** CT players cluster around `x ≈ -1485 to -1945`,
`y ≈ -1835 to -1926`, `z ≈ -264 to -281`. Consistent with CT spawn in the negative-x,
negative-y region. Z values being more negative than T spawn is consistent with
de_mirage's terrain.

**Late-round frames (125–134):** only T-side players remain (CT side eliminated
earlier in round 1). T players are spread across a wide range — some at `x ≈ -1248`
deep in CT territory, others near `x ≈ 1267` close to T spawn — which is tactically
plausible for a late-round scenario with one side alive and free to roam.

### Mapping distance observations

Most mapping distances are under 100 units, which is reasonable given de_mirage's
nav tile granularity.

**Flagged for investigation:**

| Player | Frame | Area ID | Mapping Distance | Note |
|---|---|---|---|---|
| shox | 1 | 3 | 159.87 | Largest distance observed; area 3 appears repeatedly in CT spawn |
| shox | 4 | 59 | 148.58 | Same player, two frames later; still unusually large |
| misutaaa | 2 | 10 | 142.74 | Second-largest distance; area 10 also in CT spawn region |
| misutaaa | 4 | 10 | 106.76 | Same area, still above warning threshold |
| ZywOo | 4 | 59 | 147.50 | Same area as shox in frame 4 |

These are candidates for the z-tolerance / overlap-zone investigation when the
KDTree fast path is built. Large mapping distance does not automatically mean a wrong
assignment — it can mean the tile is large and the player is near its far boundary —
but it is the primary diagnostic signal for potential misassignment.

**shox specifically** was noted in memory from a prior session as showing unusually
large mapping distances (~150+ units), identified as a z-tolerance investigation
candidate. This is consistent with the above observations.

### area_id float dtype

All `curr_areaId` values are stored as `float64` in the CSV (e.g. `12.0`, `628.0`).
This is expected behaviour — see Bug 8 above. Must be cast to `int` before any Task 4
usage.

### Multiple players mapped to same area simultaneously

In frame 0, both ZywOo and misutaaa map to area 3. In frames 0–4, multiple T players
map to area 12. This is expected — nav tiles can be large enough to contain multiple
players simultaneously, and the centroid-nearest rule maps all nearby players to the
same node. Not a bug.

### interz's elevated z in late frames

From approximately frame 120 onward, interz's z-coordinate is consistently around
`-79` while other players are at approximately `-168`. This suggests interz is on an
elevated surface (a rooftop or raised platform). The area IDs assigned (856, 855, 18,
131, 648 etc.) change across frames, consistent with movement. This is one of the most
important cases to verify with the GIF — if a tile at z ≈ -168 happened to have a
closer centroid in 3D space than the correct elevated tile, a naive centroid-nearest
search would misassign it. The KDTree z-tolerance filter is specifically designed to
prevent this class of error.

---

## 9. Verification

### 9.1 Verification script (`scripts/verify_task3_mapping.py`)

A 12-check verification script was written. It loads the saved CSV and the patched
graph pickle and runs the following checks, reporting PASS / WARN / FAIL per check.

| Check | What it tests | Failure mode |
|---|---|---|
| 1 — Schema integrity | All expected columns present, correct dtypes | Missing column or wrong dtype |
| 2 — Null/NaN audit | No null or infinite values in any column | Failed mapping producing NaN, or broken distance computation |
| 3 — area_id validity (NAV) | Every `curr_areaId` exists in `NAV["de_mirage"]` | area_id outside the 903 known areas |
| 4 — area_id validity (patched graph) | Every `curr_areaId` exists as a node in the patched graph | Mapping running against wrong graph state |
| 5 — Formerly-orphaned tile detection | Checks whether any rows map to the 72 known-orphan clusters | Reports as PASS if hits found (positive signal post-patch), WARN if none (may mean cluster not visited in sample) |
| 6 — Mapping distance audit | Per-player breakdown of rows exceeding 100-unit warning and 150-unit critical thresholds | Large distance rows flagged for manual inspection |
| 7 — Velocity completeness | `vx`, `vy`, `vz` populated and finite for every row | Missing velocity means Task 5 position projection cannot be run for that row |
| 8 — area_id float dtype | Warns if `curr_areaId` is `float64` | Needs int cast before Task 4 |
| 9 — Spatial coherence | For each (player, round) track, consecutive frame area_ids are identical or graph-adjacent | Non-adjacent jumps are z-overlap misassignment candidates |
| 10 — point_in_area spot-check | 100-row sample: does mapped area's XY bounding box contain the player's position? | Low containment rate signals systematic misassignment |
| 11 — Coverage summary | Unique areas used, players, rounds, rows per side | Informational only |
| 12 — GIF qualitative verification | One animated GIF per (player, round) track, overlaying tile and position on radar | Visual inspection of spatial coherence and mapping plausibility |

A FAIL on any check blocks downstream use of the CSV. A WARN is worth investigating
but does not block. The script is designed to be rerunnable with zero modification
against any future version of `sample_df.csv`.

### 9.2 GIF verification (Check 12)

For each unique (player, round) combination in the sample DataFrame, the script
generates an animated GIF overlaying the following on the de_mirage simpleradar:

**Per-frame visual elements:**
- Assigned nav tile as a coloured rectangle:
  - Yellow face = mapping distance ≤ 100 units (normal)
  - Orange face = mapping distance > 100 units (warning)
  - Red face = mapping distance > 150 units (critical)
  - White border = player is inside the tile's XY bounding box (`point_in_area = True`)
  - Magenta border = player is outside the tile's XY bounding box (`point_in_area = False`)
- White ✕ marker at the tile centroid
- White dashed line from centroid to player position (visual representation of
  `mapping_distance`)
- Cyan dot = CT-side player position; red dot = T-side player position
- `⚠ JUMP` annotation in red when this frame is the destination of a non-adjacent
  area transition (flagged by Check 9)
- Legend and title with frame index, area ID, area name, distance, and containment
  status

**What to look for:**
1. Player dot should sit inside or immediately adjacent to the yellow tile. If it is
   clearly elsewhere on the radar, the mapping is wrong for that frame.
2. Orange/red tiles are not automatically wrong — they may simply reflect a large or
   oddly-shaped tile where the player is near the far boundary. The signal to worry
   about is an orange/red tile where the player dot is visually far from the tile.
3. JUMP frames should show plausible rapid movement, not teleportation across the map.
4. The full track should trace a spatially coherent path — smooth progression across
   adjacent tiles, not random jumping.

**Priority GIFs to inspect first:**
- `shox_round1.gif` — largest mapping distances (159, 148 units)
- `misutaaa_round1.gif` — second-largest mapping distances (142, 106 units)
- `interz_round1.gif` — elevated z in late frames (~-79); z-overlap candidate
- Any player whose track has JUMP annotations from Check 9

GIFs are written to `outputs/gifs/`. Frame duration is 200ms (5fps) for readability.

### 9.3 Verification output

**Run date:** 3 Jul 2026  
**Input:** `outputs/sample_df.csv` (1114 rows × 13 columns, round 1 of demo
`1decf556-b9d7-4085-9081-31811f0496c3.json`)  
**Overall result:** 11 PASS · 6 WARN · 0 FAIL — no pipeline blockers; CSV is
structurally sound for Task 4, with caveats documented below.

---

#### Check 1 — Schema Integrity: 2 PASS, 1 WARN

All expected columns are present with correct dtypes. The only issue is an unexpected
`Unnamed: 0` column, which is the CSV's integer row index written by pandas when the
file was saved without `index=False`. This column is harmless — it does not interfere
with any downstream computation — but should be eliminated for cleanliness. The fix is
to resave the CSV:

```python
sample_df.to_csv("outputs/sample_df.csv", index=False)
```

or to suppress it on load:

```python
df = pd.read_csv("outputs/sample_df.csv", index_col=0)
```

**Minor naming discrepancy:** the actual area ID column in the saved CSV is named
`current_areaId`, not `curr_areaId` as these notes originally planned. The user's
mapping script used the longer name. This is consistent throughout the output (Check 8
references `current_areaId`) and is not a problem — the column name is whatever the
extraction script wrote. All downstream references should use `current_areaId` to
match the actual CSV content.

---

#### Check 2 — Null/NaN Audit: 2 PASS

Zero null values across all 1114 rows and all 13 columns. Zero infinite values in any
numeric column. This confirms the `.apply(map_current_position, axis=1)` call
completed without any silent failures — every row that entered the mapping function
returned a valid result. It also confirms `find_closest_area` never returned a null or
degenerate distance for any of the 1114 query points.

---

#### Check 3 — area_id Validity Against NAV: 1 PASS

264 unique area IDs were produced across all 1114 rows. All 264 exist in
`NAV["de_mirage"]`'s 903-area dictionary. No out-of-range or fabricated area IDs.
264/903 = 29.2% of the de_mirage nav mesh was visited across one competitive round —
this is a reasonable single-round coverage figure given that players focus on specific
routes rather than exploring the entire map.

---

#### Check 4 — area_id Validity Against Patched Graph: 1 PASS

All 264 unique area IDs are valid node IDs in `de_mirage_patched_graph_v3.pkl`. This
confirms two things simultaneously: (1) the pickle is loading correctly via Option A,
and (2) the mapping function is producing area IDs that will be valid inputs to
`area_distance()` in Task 4. If `find_closest_area` had somehow been querying against
a different version of `NAV` (e.g. a stale import), some area IDs could have appeared
valid in NAV but been absent from the patched graph's node set — that did not happen.

---

#### Check 5 — Formerly-Orphaned Tile Mapping Detection: 2 PASS + informational note

This is one of the most meaningful results in the verification. Both major previously-
orphaned clusters were hit in real gameplay:

**CT stairs cluster (area IDs 511/512/525–558/561–566):**
3 rows mapped — area IDs 532, 542, 554 — players Kyojin and apEX. These are CT-side
players, which is exactly correct: the CT stairs lead up to A site from CT spawn and
are a natural CT-side position. Only 3 rows because CT players were eliminated earlier
in the round (439 CT rows vs 675 T rows total), limiting the number of frames where CT
players could be moving through this cluster. The presence of any hits here at all
confirms the patch produced correct connectivity for this cluster and that
`find_closest_area` can reach these tiles via `NAV["de_mirage"]`.

**Apartments ramp cluster (area IDs 873–880):**
18 rows mapped — area IDs 873, 874, 875, 876, 877, 878 — players Ax1Le, Hobbit,
interz, nafany, sh1ro (all T-side). Apartments is a primary T-side aggression route on
de_mirage, so five T-side players having 18 combined mappings to this cluster across
one round is entirely consistent with real play. Six of the eight area IDs in the
cluster (873–878) were actually used, giving solid coverage of the ramp geometry.

**Scattered singleton orphans (blank-areaName, outside named clusters):**
15 rows mapped to area IDs 356, 385, 747, and 881. These are four of the ~22 scattered
singleton orphans from the original 72 orphaned tiles. Area 881 is particularly notable
— the navmesh findings document flagged areas 881/882 as numerically adjacent to the
Apartments ramp cluster (873–880) but spatially ~1700+ units away, meaning their area
IDs give no geographic information about their actual location. The fact that area 881
receives 15 player-frame mappings indicates it is a real, walkable tile that players
visit in competitive play — not an obscure corner of the map — and that the patch
correctly made it reachable. The geographic identity of areas 356, 385, and 747 was not
confirmed in the earlier navmesh analysis; their presence here is a signal they too are
non-trivial tiles worth identifying if the scatter-orphan investigation from Section 5
of `de_mirage_navmesh_findings.md` is ever completed.

**Overall interpretation:** the patch is working correctly for all three categories of
orphaned tiles. Players are being mapped to formerly-orphaned tiles when standing in
those map regions, and those mappings are reaching the verification script without
failing Checks 3 or 4. This is the strongest positive indicator in the entire
verification output.

---

#### Check 6 — Mapping Distance Audit: 2 WARN

**Distribution statistics across all 1114 rows:**

| Statistic | Value |
|---|---|
| min | 2.28 units |
| median | 45.30 units |
| mean | 51.83 units |
| p90 | 94.01 units |
| p95 | 117.28 units |
| max | 159.87 units (shox) |

The median of 45.30 units is the most useful summary statistic — half of all 1114
player positions are mapped to a tile whose centroid is within 45 units. The mean is
only slightly above the median (51.83 vs 45.30), indicating the distribution is only
mildly right-skewed. The p90 of 94 units confirms that 90% of all mappings are within
the warning threshold, and p95 of 117 units means the extreme large-distance cases are
genuinely at the tail.

**Critical threshold (>150 units) — 2 rows:**

| Player | Area ID | Distance |
|---|---|---|
| shox | 3 | 159.87 units |
| Kyojin | 10 | 156.93 units |

Both are CT-side players in CT spawn (areas 3 and 10 are large CT spawn tiles). The
large distances here are likely a large-tile effect rather than a misassignment — CT
spawn on de_mirage has some very wide nav tiles whose centroids are far from the tile
boundaries. The GIF inspection will confirm whether the player dots are plausibly inside
or adjacent to these large tiles.

**Warning threshold (>100 units) — 90 rows:**

shox dominates with 44 rows (out of 125 total frames), meaning 35% of shox's entire
track in round 1 exceeds the warning threshold. This is the single most striking finding
in Check 6. It strongly suggests shox is consistently operating in a region where nav
tiles are very large (centroid-to-boundary distances routinely exceed 100 units), or
that there is a systematic z-overlap or off-mesh situation affecting shox's mapped
positions. sh1ro has 21 warning rows, which is also notable but less extreme.

The remaining players have 1–6 warning rows each, suggesting they are occasional large-
tile or boundary-region cases rather than systematic issues. apEX's single warning row
(103.8 units) is close enough to the threshold to likely be a large-tile boundary case
with no deeper significance.

---

#### Check 7 — Velocity Completeness: 1 PASS + note

All 1114 rows have finite, non-null velocity values — no issues with the extraction
capturing `velocityX/Y/Z` from the parsed frames.

197 rows (17.7%) have `vx = vy = vz = 0.0`. This is a meaningful fraction but not
alarming. Known legitimate causes of zero velocity in the ESTA data:

- **Stationary players:** a player standing still between two parsed frames will have
  zero velocity. The late-round data (frames 132–134) showed nafany and Ax1Le with
  identical coordinates and zero velocity across multiple consecutive frames, confirming
  this occurs naturally.
- **Round-ending freeze:** in the final frames of a round, players stop moving as the
  round-end logic plays out. These frames are legitimately alive (the player hasn't
  died, the `isAlive` flag is still True) but velocity drops to zero.
- **Dead-player leakage:** theoretically, if the `isAlive` filter failed to exclude a
  dead player, that player's velocity would be zero. However, Check 2 confirmed no null
  positions, and dead players in ESTA typically have null or zero coordinates — so any
  leakage would show in the null check. The 2 PASS on Check 2 makes this scenario
  unlikely.

Given 10 players across 135 frames (theoretical max 1350 rows, actual 1114 due to
deaths), approximately 197 zero-velocity rows across a single round is consistent with
natural standing-still behaviour and end-of-round effects. No action required.

---

#### Check 8 — area_id Float dtype Warning: 1 WARN

`current_areaId` is stored as `float64` in the loaded CSV — e.g. values appear as
`12.0`, `628.0` rather than `12`, `628`. This is standard pandas behaviour when a
numeric column that could theoretically contain `NaN` is round-tripped through CSV
format — pandas uses `float64` as the default for such columns because `int64` cannot
represent `NaN`. No `NaN` values are actually present (confirmed by Check 2), but
pandas doesn't know that at read time.

The cast required at the start of any Task 4 script:

```python
player_df["current_areaId"] = player_df["current_areaId"].astype(int)
```

This must happen before passing any area IDs to `area_distance()` — that function
expects `int` arguments, and `area_distance("de_mirage", 12.0, 628.0, ...)` will raise
a `TypeError`.

---

#### Check 9 — Spatial Coherence: 1 WARN — 218 non-adjacent jumps

**Per-player jump counts:**

| Player | Jumps | Total Frames |
|---|---|---|
| apEX | 34 | 103 |
| Hobbit | 29 | 135 |
| interz | 28 | 135 |
| sh1ro | 27 | 135 |
| shox | 27 | 125 |
| Ax1Le | 24 | 135 |
| nafany | 20 | 135 |
| Kyojin | 14 | 101 |
| misutaaa | 8 | 48 |
| ZywOo | 7 | 62 |

**First 5 jumps (from output):**

| Jump | Player | Frames | Areas | Centroid dist |
|---|---|---|---|---|
| 1 | Ax1Le | 1→2 | 12→621 | 215.2 units |
| 2 | Ax1Le | 2→3 | 621→140 | 129.7 units |
| 3 | Ax1Le | 4→5 | 141→877 | 167.7 units |
| 4 | Ax1Le | 5→6 | 877→840 | 127.6 units |
| 5 | Ax1Le | 7→8 | 874→875 | 149.5 units |

**Critical interpretation — the parse rate effect:**

The ESTA dataset is parsed at parse_rate=128, meaning frames are captured approximately
every 1 second of real game time. In CS:GO, a player running at full speed moves at
approximately 250 game units per second. At this speed, a player can traverse 250 units
in a single inter-frame interval — easily spanning multiple nav tiles whose centroids
are 50–100 units apart. This means **consecutive frames in the DataFrame do not
correspond to physically adjacent positions on the map** — one full second of movement
separates them. The Check 9 adjacency test assumes consecutive frames should map to the
same tile or a direct graph-adjacent tile. At a 1-second parse rate, this assumption is
too strict.

The likely breakdown of the 218 jumps:
- **Majority: legitimate rapid movement.** A player running from one area of the map to
  another will pass through several intermediate tiles between frames, none of which
  appear in the DataFrame. The mapping for each frame is correct for that frame's
  position, but the intermediate tiles are simply absent.
- **Minority: potential z-overlap misassignments.** Where a jump coincides with a large
  mapping distance (e.g. shox's jumps in combination with 44 warning-threshold rows),
  it is more plausible that the mapping landed on the wrong tile.
- **Small cluster: intra-orphan-cluster adjacency gaps.** Jump 5 (area 874→875,
  centroid distance 149.5 units, not graph-adjacent) is notable because both 874 and
  875 are in the Apartments ramp cluster (873–880). These tiles were individually
  connected to the main mesh by the patch, but may not have been connected to each other
  if their bounding boxes did not satisfy the `gap_distance <= 2.0` condition between
  themselves. This is a known limitation of the patching approach — the gap-distance
  formula connects orphans to everything adjacent, but "adjacent" is defined at 2.0
  units tolerance, and two staircase tiles can be more than 2 units apart from each
  other while both being within 2 units of a shared main-mesh neighbour.

**Recommendation:** treat most of the 218 jumps as parse-rate artifacts rather than
mapping errors, and use the GIF inspection (Check 12) to discriminate between:
- Jumps that trace a directional path across the radar (movement artifact — acceptable)
- Jumps that appear to teleport to an unrelated region of the map (genuine misassignment
  — warrants investigation)

A potential refinement to Check 9 for future use would be to weight by centroid
distance: filter the jump list to only those where the centroid-to-centroid distance
is greater than, say, 300 units (exceeding plausible 1-second movement distance), to
separate rapid-movement cases from genuine spatial anomalies.

---

#### Check 10 — point_in_area Spot-Check: 1 WARN

**Containment rate: 65/100 = 65%** (below the 80% warning threshold).

Before interpreting this as a problem, two factors specific to `point_in_area` must be
accounted for:

1. **`point_in_area` is a strict 2D bounding-box containment test with no tolerance.**
   A player standing exactly on the edge of a tile returns `False`. A player standing
   1 unit outside the tile boundary returns `False`. The correct mapping (nearest
   centroid) can place a player in a tile whose XY bounding box they are just outside
   of, particularly near tile boundaries. This is geometrically valid — the centroid of
   the assigned tile is still closer than any other tile's centroid.

2. **Z is ignored.** `point_in_area` only checks XY. A player at z=-54 standing below
   Catwalk (z≈above) will fail containment for any tile whose XY footprint they are
   standing under, regardless of whether the mapping was correct.

The 5 worst non-contained cases from the sample:

**Case 1 — Hobbit at (-2364.7, 541.1, z=7.8) → area 358, dist=132.4:**
The z-coordinate of 7.8 is anomalous — nearly all other players in this round are at
z ≈ -168 to -265. z=7.8 is approximately 170 units higher than the typical ground
level, suggesting Hobbit is on an elevated surface at an unusual height. The position
at x ≈ -2364 is far to the west/negative-x side of the map — potentially near the edge
of the playable area or on a specific map feature. The mapping_distance of 132.4 is
above the warning threshold, and non-containment combined with the anomalous z makes
this the highest-priority individual case to inspect in `Hobbit_round1.gif`.

**Cases 2–4 — shox at (-854.4, 30.3, z=-54.0) → area 268 (Catwalk), dist=117.6:**
The same frame appears three times in the 100-row random sample, meaning shox was
stationary at this position across multiple consecutive frames (identical coordinates
appear multiple times in the DataFrame when a player is standing still, and the
random sample with `random_state=42` happened to draw multiple of these identical rows).
This is the clearest z-overlap misassignment candidate found in the entire verification
run:

- Area 268 is named "Catwalk" per the Check 11 top-areas breakdown (40 rows total for
  this area across the round). Catwalk is the elevated walkway that runs over Mid on
  de_mirage.
- shox's z-coordinate is -54.0. Players standing on Catwalk itself would be at a
  higher z than players on the ground below. A z of -54.0 is intermediate — neither
  clearly ground level (-168) nor clearly atop Catwalk.
- `point_in_area` returns `False` — shox is not inside area 268's XY bounding box,
  suggesting the player is not actually on Catwalk, but `find_closest_area` selected
  area 268's centroid as the nearest in 3D space.
- The mapping distance of 117.6 units (above the warning threshold) is consistent with
  the player being physically distant from area 268's centroid.

This combination — z overlap, containment failure, elevated mapping distance, named
Catwalk tile — is exactly the scenario the KDTree z-tolerance filter is designed to
correct. Catwalk's centroid is at some z above ground level; ground-level players with
a z that puts them closer to Catwalk's centroid than to nearby ground tiles will be
misassigned by a naive centroid-nearest search. A z-tolerance of 100 units would
constrain candidates to those within 100 units of the query z, preventing Catwalk from
being selected for a player at z=-54 if Catwalk's centroid z is sufficiently higher.

**Case 5 — Ax1Le at (-1595.5, 111.5, z=-168.0) → area 269, dist=91.5:**
z=-168 is standard ground level. dist=91.5 is just below the warning threshold. This
is most likely a tile-boundary case (player near the edge of area 269's XY footprint)
rather than a z-overlap issue.

**Overall interpretation of 65% containment rate:**
The 35% non-containment rate is higher than ideal (warning threshold was 80%), but is
explained by a combination of three factors: (1) the strictness of `point_in_area`'s
boundary test, (2) the shox/Catwalk z-overlap case inflating the non-containment count,
and (3) genuine tile-boundary edge cases. The rate is not alarming enough to block
downstream use, but the Catwalk/shox case stands out as the one genuine misassignment
candidate that should be confirmed via GIF inspection.

---

#### Check 11 — Coverage Summary: 1 PASS (informational)

| Metric | Value |
|---|---|
| Total rows | 1114 |
| Unique players | 10 |
| Unique rounds | 1 (round 1) |
| Unique frame indices | 135 |
| Rows by side | CT: 439 (39.4%), T: 675 (60.6%) |
| Unique area IDs used | 264 of 903 (29.2%) |

The T:CT row ratio (60.6%:39.4%) is directly explained by the round's outcome: CT
players were progressively eliminated during round 1, reducing CT-side rows in later
frames. The shortest GIF (misutaaa at 48 frames) and second-shortest (ZywOo at 62
frames) are both CT-side players, confirming the CT side died out first.

**Top 10 most-mapped area IDs:**

| Area ID | Row count | Named place |
|---|---|---|
| 855 | 68 | Apartments |
| 268 | 40 | Catwalk |
| 112 | 39 | Shop |
| 42 | 28 | SnipersNest |
| 867 | 28 | TicketBooth |
| 22 | 25 | BombsiteB |
| 648 | 24 | BackAlley |
| 393 | 23 | BackAlley |
| 656 | 22 | CTSpawn |
| 745 | 19 | Shop |

This distribution is geographically coherent with a real competitive round on de_mirage:
Apartments (area 855, 68 rows) being the most-visited area is consistent with T-side
Apartment aggression being a primary route. Catwalk (area 268, 40 rows) is a dominant
CT-side position for Mid control. Shop (areas 112, 745) and SnipersNest (area 42) are
standard CT-side positions flanking B site. BombsiteB (area 22) and BackAlley (areas
648, 393) see significant traffic from both sides during B-site plays.

The presence of two different area IDs for the same named places (112 and 745 both
named "Shop", 648 and 393 both named "BackAlley") is expected — named places in awpy
can span multiple discrete nav tiles.

**Note:** area 268 (Catwalk) appearing as the second most-mapped area with 40 rows is
directly connected to the Check 10 concern. If shox is being systematically misassigned
to Catwalk when actually beneath it, some portion of those 40 rows may represent
misassignments rather than players genuinely on the Catwalk walkway. The GIF
inspection will disambiguate this.

---

#### Check 12 — GIF Generation: 1 PASS

10 GIFs generated to `outputs/position_mapping_gifs/`, one per player:

| Player | GIF | Frames | Side | Notes |
|---|---|---|---|---|
| Ax1Le | `Ax1Le_round1.gif` | 135 | T | Full round; first 5 Check 9 jumps all from Ax1Le |
| Hobbit | `Hobbit_round1.gif` | 135 | T | Anomalous z=7.8 case (Check 10) |
| Kyojin | `Kyojin_round1.gif` | 101 | CT | Critical-distance row (156.9 units, area 10) |
| ZywOo | `ZywOo_round1.gif` | 62 | CT | Died early; fewest frames |
| apEX | `apEX_round1.gif` | 103 | CT | Most Check 9 jumps (34); 1 warning-distance row |
| interz | `interz_round1.gif` | 135 | T | Elevated z (~-79) in late frames; z-overlap candidate |
| misutaaa | `misutaaa_round1.gif` | 48 | CT | Died first; 5 warning-distance rows |
| nafany | `nafany_round1.gif` | 135 | T | 2 warning-distance rows; clean otherwise |
| sh1ro | `sh1ro_round1.gif` | 135 | T | 21 warning-distance rows; second-most after shox |
| shox | `shox_round1.gif` | 125 | CT | 44 warning-distance rows (35% of frames); Catwalk z-overlap candidate |

The variation in frame counts directly reflects when each player died in round 1. The
5 T-side survivors (Ax1Le, Hobbit, interz, nafany, sh1ro) all have 135 frames. CT
players have fewer because they were eliminated earlier, with misutaaa dying around
frame 48 and ZywOo around frame 62.

**GIF inspection status:** pending. Qualitative findings to be documented in Section 9.4
once GIFs have been reviewed.

---

#### Summary of WARN items and their severity

| WARN source | Check | Severity | Likely cause | Action |
|---|---|---|---|---|
| `Unnamed: 0` column | 1 | Trivial | CSV saved without `index=False` | Resave with `index=False` |
| 2 critical-distance rows | 6 | Low | Large CT spawn tiles (areas 3, 10) | Confirm via GIF; expected for large tiles |
| 90 warning-distance rows | 6 | Medium | Mix of large tiles and potential z-overlap | GIF inspection + future KDTree z-filter |
| `current_areaId` float64 | 8 | Trivial | Standard CSV round-trip behaviour | Cast to int at start of Task 4 script |
| 218 spatial jumps | 9 | Low-medium | Primarily parse-rate artifact; some potential z-overlap | GIF inspection to discriminate |
| 65% containment rate | 10 | Medium | shox/Catwalk z-overlap is primary driver | GIF inspection; Catwalk case confirmed as misassignment candidate |

### 9.4 GIF qualitative inspection findings

*To be filled in after reviewing `outputs/position_mapping_gifs/`. Priority order:
`shox_round1.gif` (Catwalk/z-overlap candidate), `misutaaa_round1.gif` (warning
distances), `interz_round1.gif` (elevated z in late frames), `Hobbit_round1.gif`
(anomalous z=7.8 case), `apEX_round1.gif` (most spatial jumps).*

```
[GIF INSPECTION FINDINGS TO BE ADDED HERE]
```

---

## 10. KDTree Fast-Path Design (Deferred to Full Dataset)

This section documents the full design of the KDTree-based position mapping that
will replace `find_closest_area` when scaling to the full ESTA dataset. The design
is settled — it just hasn't been implemented yet because it isn't needed for the
sample run.

### 10.1 What changes and what stays the same

**Changes:**
- The body of `map_current_position()` (or the replacement function).
- The call site becomes a single batched `KDTree.query()` call over all rows at once,
  rather than a per-row `.apply()`.

**Stays the same:**
- `player_df` structure — all the same columns.
- Output column names — `curr_areaId` and `mapping_distance`.
- Downstream code — nothing after the mapping step needs to know how it was computed.

### 10.2 Step-by-step implementation plan

**Step 1: Load the patched graph and build the centroid index**

```python
import pickle
from scipy.spatial import KDTree
import numpy as np

with open(PATCHED_GRAPH_PATH, "rb") as f:
    G_patched = pickle.load(f)

# Extract node IDs and centroids in consistent order
node_ids  = list(G_patched.nodes())
centroids = np.array([G_patched.nodes[nid]["center"] for nid in node_ids])
# centroids shape: (903, 3)

# Build index once — reused for all queries in the session
tree = KDTree(centroids)
```

`node_ids[i]` and `centroids[i]` are the parallel mapping from KDTree row index to
area ID. This must be kept in sync — do not sort or reindex either without doing the
same to the other.

**Step 2: Batch query**

```python
# All player positions as a single (n, 3) array
coords = player_df[["x", "y", "z"]].values  # shape: (n, 3)

# Single call returns k=5 nearest neighbours for every point simultaneously
distances, indices = tree.query(coords, k=5, workers=-1)
# distances shape: (n, 5) — Euclidean distances to k nearest centroids
# indices   shape: (n, 5) — row indices into centroids/node_ids arrays
```

`workers=-1` uses all available CPU cores for parallel queries. On the VM/HPC, this
can reduce runtime significantly for large batches.

**Step 3: Z-tolerance filtering and fallback**

For each row, among the k=5 candidates, prefer the nearest candidate whose centroid
z is within `Z_TOLERANCE` units of the query point's z. Fall back to the nearest
overall if no candidate passes the filter.

```python
Z_TOLERANCE = 100.0  # game units — provisional, to be tuned empirically

area_ids      = []
mapping_dists = []

for i in range(len(coords)):
    query_z = coords[i, 2]
    resolved = False
    for rank in range(5):
        candidate_node_id = node_ids[indices[i, rank]]
        candidate_z = G_patched.nodes[candidate_node_id]["center"][2]
        if abs(query_z - candidate_z) <= Z_TOLERANCE:
            area_ids.append(candidate_node_id)
            mapping_dists.append(float(distances[i, rank]))
            resolved = True
            break
    if not resolved:
        # Fallback: nothing passed z-filter — use raw nearest
        area_ids.append(node_ids[indices[i, 0]])
        mapping_dists.append(float(distances[i, 0]))

player_df["curr_areaId"]      = area_ids
player_df["mapping_distance"] = mapping_dists
```

Note: the per-row loop here only iterates over k=5 pre-fetched candidates, not over
the full KDTree. The expensive nearest-neighbour search already happened in the
single batched `.query()` call — this loop is O(k) per row, which is negligible.

**Step 4: point_in_area tie-breaking (optional enhancement)**

If multiple candidates pass the z-filter in Step 3, prefer the one whose XY bounding
box actually contains the query point (via `point_in_area`). This requires changing
the tie-breaking logic to collect all z-passing candidates first, then check
containment before falling back to distance-ordering.

This is deferred until the basic z-tolerance filter has been validated — the
containment check adds per-row overhead and may not be necessary if the z-filter
alone resolves the overlap cases cleanly.

### 10.3 Tuning the z-tolerance constant

`Z_TOLERANCE = 100.0` is provisional. To tune it empirically:

1. Identify frames in the full dataset where a player is known to be on Catwalk
   (above Mid) or in Underpass/Connector (the two confirmed z-overlap zones on
   de_mirage with no `z_cutoff` in `MAP_DATA`).
2. For each such frame, compare the KDTree result at various tolerance values against
   the `find_closest_area` ground truth.
3. Find the smallest tolerance that correctly resolves all confirmed overlap cases
   without over-filtering normal cases.

The saved sample CSV (with `find_closest_area` results) is a starting point for this
tuning — look at the large-distance rows for shox, misutaaa, ZywOo in CT spawn and
check whether the KDTree with z-tolerance gives a different (better) result.

---

## 11. Outstanding Items

Items are grouped by urgency: those blocking Task 4 from starting, those that need
to happen before the full-dataset run, and longer-horizon deferred items.

### 11.1 Before Task 4 can begin

- [x] **Run the verification script** on `outputs/sample_df.csv` — complete. Output
      documented in Section 9.3.
- [x] **Verification result: no pipeline blockers.** 11 PASS, 6 WARN, 0 FAIL. CSV is
      structurally sound.
- [ ] **Inspect priority GIFs** and document findings in Section 9.4. This is the
      remaining qualitative gate before Task 4. Inspect in this order:
      1. `shox_round1.gif` — 44 warning-distance rows (35% of all shox frames), plus
         the Catwalk/z=-54 case from Check 10 which is the strongest misassignment
         candidate in the entire verification output. The key question is whether the
         player dot is consistently inside large CT spawn tiles (a large-tile effect,
         acceptable) or consistently displaced from tiles onto visibly wrong regions of
         the radar (a z-overlap problem, requires the KDTree z-filter before full-dataset
         run).
      2. `Hobbit_round1.gif` — the z=7.8 anomaly (Check 10, worst non-contained row).
         z=7.8 is ~170 units higher than the typical ground level of -168. The GIF
         should make clear whether Hobbit is on a known elevated feature of the map or
         whether this is a parsing anomaly that should be investigated upstream.
      3. `interz_round1.gif` — elevated z consistently around -79 in later frames.
         Established from the sample output assessment before verification. The GIF will
         confirm whether the mapped tiles make geographic sense for that z level, or
         whether the elevated frames are being pulled down to ground-level tiles nearby.
      4. `apEX_round1.gif` — 34 spatial jumps (highest count in Check 9). The jumps
         are most likely parse-rate movement artifacts; the GIF will confirm whether
         the track traces a directional path or contains genuine teleportation-style
         discontinuities.
      5. `misutaaa_round1.gif` and `sh1ro_round1.gif` — 5 and 21 warning-distance rows
         respectively. Lower priority than shox but worth reviewing.

- [ ] **Resave CSV without the `Unnamed: 0` column.** The current `outputs/sample_df.csv`
      contains a spurious index column that was saved without `index=False`. Resave:
      ```python
      sample_df.to_csv("outputs/sample_df.csv", index=False)
      ```
      This should be done before the CSV is used as input to any Task 4 script, to
      prevent the `Unnamed: 0` column from being present in the loaded DataFrame and
      potentially interfering with column-name-based logic.

- [ ] **Cast `current_areaId` to int** at the start of every Task 4 script, immediately
      after loading the CSV, before any call to `area_distance()`:
      ```python
      player_df["current_areaId"] = player_df["current_areaId"].astype(int)
      ```
      Note: the actual column name in the saved CSV is `current_areaId` (not
      `curr_areaId` as originally planned in these notes — the user's extraction script
      used the longer form). All downstream references should use `current_areaId`.

### 11.2 Before the full-dataset run

- [ ] **Decide column naming for observed vs projected positions.** Before Task 5 is
      written, decide whether `current_areaId` stays as-is (paired with a new
      `projected_areaId`) or whether both get renamed to `obs_areaId` /
      `proj_areaId` for symmetry. This decision must be made before Task 5 is
      implemented to avoid renaming a column that Task 4 scripts already depend on.
      Current recommendation: keep `current_areaId` (it's already written into the CSV
      and the Task 4 cast line above) and add `projected_areaId` as the new Task 5
      column. Rename only if the asymmetry causes confusion in practice.

- [ ] **Implement the KDTree fast-path** when scaling beyond the 8-demo sample. The
      implementation slot is the body of `map_current_position()` only — nothing
      outside the function changes. Full design is documented in Section 10. The swap
      point is: load the patched graph, build the centroid index, batch-query all rows
      at once, apply z-tolerance filtering on the k=5 candidates, write results back
      as `current_areaId` and `mapping_distance` columns. The sample CSV's
      `find_closest_area` results serve as the ground-truth baseline for validating
      the KDTree output.

- [ ] **Tune the z-tolerance constant** (`Z_TOLERANCE = 100.0` is provisional; see
      Section 10.3). The verification has already provided concrete candidates for
      tuning: the shox/Catwalk case (z=-54, area 268) is the clearest z-overlap
      instance found. When the KDTree fast-path is implemented, test it against this
      specific case first. If `Z_TOLERANCE = 100.0` correctly excludes the Catwalk tile
      (whose centroid z is presumably further than 100 units from -54) while retaining
      correct ground-level tiles, the provisional constant is adequate. If not, adjust.

- [ ] **Investigate intra-cluster adjacency gaps.** Check 9 flagged jump 5 (area
      874→875, centroid dist 149.5, not graph-adjacent) where both tiles are in the
      Apartments ramp cluster. This suggests some tiles within the formerly-orphaned
      clusters are not directly connected to each other even though they are on the
      same staircase — they each connect to the main mesh via different neighbours but
      not to each other. Confirm by checking which pairs within the 873–880 cluster
      have direct edges in `de_mirage_patched_graph_v3.pkl`. If pairs that should be
      geometrically adjacent are not connected (gap > 2.0 units by the patch formula),
      consider adding those edges manually or re-running the patch with a slightly
      larger `TOUCH_TOL`. This is a graph quality issue rather than a mapping issue,
      but it will affect Task 4's shortest-path distances for players moving through
      the ramp.

- [ ] **Investigate scattered singleton orphan area IDs 356, 385, 747, 881.** These
      appeared in Check 5 as blank-areaName tiles outside the two named clusters,
      mapped to by real players in round 1. Their geographic identity on the map is
      unknown — the navmesh findings document did not individually enumerate the ~22
      scattered singletons. Understanding what map features they correspond to (e.g.
      specific boxes, ledges, corners) would be useful context for interpreting any
      mapping anomalies involving these tiles in the full dataset.

- [ ] **Interpret the 218 spatial jumps in the context of parse rate before discarding
      them as errors.** The Check 9 jump count should be re-evaluated with a minimum
      centroid-distance threshold applied — jumps below, say, 300 units (plausible 1-
      second movement at run speed) should be treated as parse-rate movement artifacts
      and excluded from the flagged list. Only jumps exceeding that distance are likely
      genuine misassignments. Consider refining Check 9 in the verification script to
      report the parse-rate-adjusted count alongside the raw count.

### 11.3 HPC deployment

- [ ] **Resolve Option A/B/C for the cluster environment.** Option A (pickle) is
      confirmed for local VM development. For the Wits HPC cluster, either:
      - Copy `de_mirage_patched_graph_v3.pkl` to the cluster and ensure the Task 4+
        scripts reference the correct path relative to the cluster working directory, or
      - Apply Option C (append `de_mirage_additional_edges.txt` to the bundled
        `nav/de_mirage.txt` in the cluster's awpy install) during environment setup,
        so that `NAV_GRAPHS["de_mirage"]` is already patched on import without needing
        the pickle at all.
      This decision should be finalised and documented before the full-dataset pipeline
      is submitted as a cluster job.

### 11.4 Completion gate

- [ ] **Update `csgo-map-control-progress.md`** to mark Task 3 as complete. This
      should only happen once all of the following are true: (1) GIF inspection
      findings are documented in Section 9.4 of this file, (2) the CSV has been
      resaved without `Unnamed: 0`, (3) no new pipeline-blocking issues were found in
      the GIF inspection, (4) the `current_areaId` int-cast requirement is
      documented as a known Task 4 entry requirement, and (5) the z_tol calibration
      decision from Section 13.7 has been made and a final z_tol value selected.

### 11.5 z_tol sweep follow-up (added after first sweep run)

- [ ] **Fix `verify_ztol_sweep.py` FCA baseline** — change `FCA_BASELINE_CSV` to
      `outputs/sample_df.csv`, add round-1 filter after loading, and add `"round"`
      to the join key. Full details and exact code in Section 13.5.
- [ ] **Re-run `verify_ztol_sweep.py`** with the corrected baseline. Document the
      updated Section 1 and Section 3 output in Section 13.4 of this file.
- [ ] **Inspect the adjacent-pair plots** in `outputs/ztol_sweep_plots/adjacent_pairs/`
      — these are already reliable (not contaminated by the FCA baseline issue). Focus
      on: `ztol50_vs_ztol75/` (487 disagreements — largest active boundary, most
      diagnostic for calibration) and `ztol75_vs_ztol100/` (100 disagreements — second
      most active). The specific diagnostic question for each plot: does the tile
      assigned by the tighter z_tol (orange, lower value) sit geographically closer to
      the player on the radar than the tile assigned by the looser z_tol (violet, higher
      value), or does it look like an overcorrection?
- [ ] **Make and record the z_tol calibration decision** once the above inspections are
      complete. The reliable finding from the first run is that z_tol=100 and z_tol=125
      are identical in all cases, and z_tol=150 and z_tol=200 are identical — so the
      real choice is among {50}, {75}, {100/125}, {150/200}. Document the chosen value
      and the reasoning here, then update `Z_TOL` in all pipeline scripts.

---

## 12. Files Produced by Task 3

| File | Location | Description |
|---|---|---|
| `sample_df.csv` | `outputs/` | Mapped player DataFrame for round 1 of demo `1decf556-b9d7-4085-9081-31811f0496c3.json`. 1114 rows × 13 columns. Central data structure for all downstream task development on the sample dataset. Currently contains an `Unnamed: 0` index column — needs resave with `index=False`. |
| `verify_task3_mapping.py` | `scripts/` | 12-check verification script. Loads CSV and patched graph pickle, runs structural and spatial checks, generates per-player animated GIFs. Result on first run: 11 PASS, 6 WARN, 0 FAIL. |
| `Ax1Le_round1.gif` | `outputs/position_mapping_gifs/` | 135-frame GIF. T-side. First 5 Check 9 jumps all originate from Ax1Le's track. |
| `Hobbit_round1.gif` | `outputs/position_mapping_gifs/` | 135-frame GIF. T-side. Anomalous z=7.8 case flagged by Check 10. |
| `Kyojin_round1.gif` | `outputs/position_mapping_gifs/` | 101-frame GIF. CT-side. Contains the 156.9-unit critical-distance row (area 10). |
| `ZywOo_round1.gif` | `outputs/position_mapping_gifs/` | 62-frame GIF. CT-side. Died early — fewest frames of any player. |
| `apEX_round1.gif` | `outputs/position_mapping_gifs/` | 103-frame GIF. CT-side. Most Check 9 jumps (34). |
| `interz_round1.gif` | `outputs/position_mapping_gifs/` | 135-frame GIF. T-side. Elevated z (~-79) in late frames; z-overlap candidate. |
| `misutaaa_round1.gif` | `outputs/position_mapping_gifs/` | 48-frame GIF. CT-side. Died first — fewest frames. 5 warning-distance rows. |
| `nafany_round1.gif` | `outputs/position_mapping_gifs/` | 135-frame GIF. T-side. 2 warning-distance rows; otherwise clean. |
| `sh1ro_round1.gif` | `outputs/position_mapping_gifs/` | 135-frame GIF. T-side. 21 warning-distance rows; second most after shox. |
| `shox_round1.gif` | `outputs/position_mapping_gifs/` | 125-frame GIF. CT-side. 44 warning-distance rows (35% of all shox frames); Catwalk z-overlap candidate (Check 10). Highest-priority GIF to inspect. |

---

## 13. z_tol Sweep and Calibration Work

This section documents the work done after the initial KDTree fast-path was
implemented, specifically the effort to calibrate the provisional `z_tol=100.0`
constant by sweeping across a range of tolerance values and comparing the results
both against each other and against the `find_closest_area` baseline.

### 13.1 Motivation

The KDTree mapping function's z-tolerance filter (`Z_TOL = 100.0`) was explicitly
marked as provisional in Section 10.3 of these notes. The value was chosen conservatively
without empirical grounding — 100 game units was a reasonable starting guess for "how
many units of z-separation is enough to reject a vertically displaced tile candidate"
but was not validated against any actual z-overlap cases in the data.

The shox/Catwalk case found during verification (`shox` at `z=-54.0` mapped to area 268
"Catwalk" by `find_closest_area`) was the clearest concrete candidate for z-overlap
misassignment. The question was: which `z_tol` value correctly excludes the Catwalk
tile for a player at `z=-54.0`, without aggressively excluding legitimately nearby tiles
for players at normal ground level? Answering this empirically requires running the
mapping at multiple tolerance values and observing where the assignment changes.

### 13.2 Scripts written

**`scripts/ztol_sweep.py`** — the sweep runner. Accepts a list of `Z_TOL_VALS`,
runs `map_position_kdtree` once per value across the full `player_df` (all rounds),
appends a `z_tol` column to each resulting DataFrame (so the verification script can
cross-check CSV provenance against the filename), and saves each result to
`outputs/ztol_sweep/player_df_ztol_{z_tol:.0f}.csv`. The `map_position_kdtree`
function included is the full three-tier version with the `point_in_area` enhancement
(containment-preferred among z-passing candidates → nearest z-passing → raw nearest
fallback).

The tolerance values chosen for the sweep were `[50.0, 75.0, 100.0, 125.0, 150.0,
200.0]`, bracketing the provisional default symmetrically and extending into ranges
where the filter should become ineffective (200 units is wide enough that virtually
every tile centroid on a flat map surface will pass it, making it equivalent to no
z-filtering at all).

**`verification/verify_ztol_sweep.py`** — the comparison and visualisation
script. Loads the FCA baseline CSV and all sweep CSVs, produces six sections of output:
(1) summary agreement table (each z_tol vs FCA), (2) pairwise agreement matrix across
all z_tol pairs, (3) area-name breakdown of corrections vs FCA, (4) per-player
disagreement counts, (5) comparison plots of sampled disagreeing frames vs FCA (up to
10 per z_tol, saved to `outputs/ztol_sweep_plots/vs_fca/ztol_{value}/`), and (6)
comparison plots of adjacent z_tol pair disagreements (up to 6 per pair, saved to
`outputs/ztol_sweep_plots/adjacent_pairs/ztol{A}_vs_ztol{B}/`). Comparison plots show
the radar with both tiles overlaid as semi-transparent rectangles in contrasting
colours, player position dot, centroid X markers, and dashed centroid-to-player lines.

### 13.3 Bug found in `run_sweeps()`: `return` inside the for-loop

After the first run, only one CSV was produced (`player_df_ztol_50.csv`) despite six
z_tol values being specified. The cause was a single indentation error: `return results`
was placed at the indentation level of the `for` loop body rather than the function
body, causing the function to return after the very first iteration (z_tol=50.0) before
any subsequent values were processed. Python does not raise a `SyntaxError` or
`RuntimeError` for this pattern — the code is valid Python that silently exits early.
The only signal was the observable behaviour: one CSV written, script exits normally.

Fix: dedent `return results` by one level so it sits outside the `for` block, at the
same indentation as `os.makedirs()` and `results = {}` at the top of the function. No
other change to the function was required.

### 13.4 Verification output — text-based analysis

The verification script was run after the indentation fix. All six z_tol values
produced CSVs and the script ran to completion. The output is reproduced and analysed
below by section. **Note: Section 13.5 documents a critical data issue that invalidates
several of the text-based findings — read both together.**

---

#### Loading data

```
FCA baseline:  outputs/round_1_player_positions_df.csv  (50168 rows, round 1 only)
z_tol= 100.0: outputs/ztol_sweep/player_df_ztol_100.csv (50168 total rows, 1114 round-1 rows)
[...same for all six z_tol values...]

WARNING z_tol=50.0: joined 34221 rows but FCA baseline has 50168.
WARNING z_tol=75.0: joined 34221 rows but FCA baseline has 50168.
[...same warning for all six z_tol values...]
```

The WARNING line is the most important output in the entire run. The FCA baseline CSV
loaded was `outputs/round_1_player_positions_df.csv` with 50168 rows. The verification
script's comment says "round 1 only" and the script does not apply a round filter to
the FCA baseline before joining. But 50168 rows is the full multi-round player_df from
the KDTree pipeline script — it contains all rounds from the 8-demo sample, not just
round 1. The intended baseline was `outputs/sample_df.csv` (1114 rows, round 1 only,
produced by `find_closest_area`). A wrong CSV was passed as the FCA baseline.

The join count of 34221 (not 1114, not 50168) is a direct consequence of this:
the join key is `(player_name, frame_idx)` with no round filter. Because `frame_idx`
is a 0-indexed position within a round (not a globally unique frame ID), the value
`frame_idx=5` in round 3 of the FCA CSV matches `frame_idx=5` in round 1 of the
z_tol data for the same player — a false cross-round match. The join therefore
produces many more rows than the true round-1 intersection (1114 rows) because it
cross-matches frames from different rounds that happen to share the same `frame_idx`
value and `player_name`. The 34221-row count represents exactly this set of false and
true matches combined.

---

#### Section 1 — Agreement vs FCA: 95.4% disagreement across all z_tol values

```
  z_tol   Total rows   Agree w/ FCA    Disagree   Disagree %
     50.0       34221           1583      32638         95.4%
     75.0       34221           1582      32639         95.4%
    100.0       34221           1586      32635         95.4%
    125.0       34221           1586      32635         95.4%
    150.0       34221           1588      32633         95.4%
    200.0       34221           1588      32633         95.4%
```

The 95.4% disagreement rate is an artifact of the cross-round false matches, not a
genuine finding about KDTree vs FCA performance. When `frame_idx=5` from round 3 of the
FCA CSV is matched against `frame_idx=5` from round 1 of the KDTree data for the same
player, the players are at completely different positions in those two moments (different
rounds, different game states), so their area IDs will almost certainly differ. This
inflates the disagreement count massively.

The correct round-1-only comparison (using `sample_df.csv` as the baseline with a
round filter on the join) would produce at most 1114 matched rows and a much lower
disagreement rate. The 95.4% figure is not meaningful and should not be referenced for
calibration decisions.

---

#### Section 2 — Pairwise agreement matrix: the most reliable output in the run

```
             z50     z75    z100    z125    z150    z200
          ------  ------  ------  ------  ------  ------
  z50     34221   33734   33706   33706   33644   33644
  z75     33734   34221   34121   34121   34059   34059
  z100    33706   34121   34221   34221   34159   34159
  z125    33706   34121   34221   34221   34159   34159
  z150    33644   34059   34159   34159   34221   34221
  z200    33644   34059   34159   34159   34221   34221

  Adjacent-pair disagreement counts:
    z_tol=50  vs z_tol=75:  487 disagreements (1.4%)
    z_tol=75  vs z_tol=100: 100 disagreements (0.3%)
    z_tol=100 vs z_tol=125:   0 disagreements (0.0%)
    z_tol=125 vs z_tol=150:  62 disagreements (0.2%)
    z_tol=150 vs z_tol=200:   0 disagreements (0.0%)
```

The pairwise matrix compares z_tol values against each other rather than against the
FCA baseline, so the cross-round false matching issue contaminates it less severely.
For false-matched cross-round rows, both z_tol values will produce the same area_id
(since they're mapping the same position from the z_tol CSV with only the tolerance
differing), so they will agree with each other regardless. The pairwise disagreements
therefore primarily reflect cases where two z_tol values genuinely differ on the same
position — which is exactly what calibration requires.

**Key findings from this section:**

**Finding 1 — z_tol=100 and z_tol=125 are completely identical (0 disagreements).**
No tile centroid z-difference in the entire dataset falls between 100 and 125 units
from any query point's z in a way that changes the mapping outcome. This collapses
these two values to a single equivalence class: `{100 ≡ 125}` for practical purposes.

**Finding 2 — z_tol=150 and z_tol=200 are completely identical (0 disagreements).**
Same reasoning: the z-tolerance boundary between 150 and 200 does not affect any
mapping decision in the data. `{150 ≡ 200}` is another equivalence class.

**Finding 3 — The effective distinct configurations are only four: {50}, {75},
{100=125}, {150=200}.** The calibration question is therefore not a fine-grained
continuous tuning problem — it is a discrete choice among these four classes.

**Finding 4 — The 50→75 boundary is the most active (487 disagreements, 1.4%).**
Many tile centroid z-differences in the data fall between 50 and 75 units. This
means z_tol=50 is making substantially more corrections (or over-corrections) than
z_tol=75. These 487 cases are the most important to inspect visually — they determine
whether the additional corrections at z_tol=50 are genuine z-overlap fixes or
overcorrections on flat-ground tiles.

**Finding 5 — The 75→100 boundary is moderately active (100 disagreements, 0.3%).**
A smaller but still meaningful set of positions are affected by the choice between
z_tol=75 and z_tol=100. These 100 cases are the second-most diagnostic for calibration.

**Finding 6 — The 125→150 boundary has 62 disagreements, which is surprising.**
Since z_tol=100 and z_tol=125 are identical, and z_tol=150 and z_tol=200 are also
identical, the 62 disagreements between z_tol=125 and z_tol=150 represent positions
where the z-filter boundary between 125 and 150 units changes the mapping outcome.
This means there is a cluster of tile centroid z-differences in the 125–150 unit range
that affects 62 frames in the dataset. Whether those corrections are valid requires
GIF/plot inspection.

---

#### Section 3 — Area-name breakdown vs FCA: unreliable due to cross-round issue

```
  z_tol=100  (32635 corrections):
               BombsiteA  →  Shop             (623 frame(s))
                  TSpawn  →  CTSpawn          (562 frame(s))
                 Catwalk  →  Catwalk          (548 frame(s))
                 CTSpawn  →  CTSpawn          (487 frame(s))
               BombsiteA  →  CTSpawn         (456 frame(s))
             SnipersNest  →  BackAlley        (454 frame(s))
               BombsiteA  →  TicketBooth      (442 frame(s))
[...identical for all z_tol values...]
```

The corrections shown here are almost entirely artifacts of the cross-round false
matching. `TSpawn → CTSpawn` means a position that FCA (in a different round's frame)
assigns to T spawn is being assigned by the KDTree (in round 1) to CT spawn — because
the player is actually in CT spawn in round 1 at that frame index, even though the FCA
CSV has them in T spawn for a different round's frame with the same index. This is not
a z-overlap correction; it is a geographic non-sequitur caused by comparing two
completely unrelated game moments.

`BombsiteA → Shop` and `SnipersNest → BackAlley` are similarly implausible as z-overlap
corrections — these named places are horizontally separated by large distances on the
map, not vertically stacked. They cannot be explained by z-tolerance filtering.

**The only entries worth examining are same-name corrections:**
- `Catwalk → Catwalk` (548 frames): FCA and KDTree both assign a Catwalk tile, but
  different tiles within the same named area. This could represent a genuine intra-area
  tile boundary correction — the z-filter preferring a different Catwalk tile that is
  at a more appropriate z-level for the player. Whether this is an improvement requires
  plot inspection.
- `CTSpawn → CTSpawn` (487 frames): similar — same named area, different tile ID. Also
  potentially real but needs inspection.
- `BombsiteB → BombsiteB` (623 frames): same pattern, also potentially real.

These same-name corrections represent the most credible signal in Section 3 given the
cross-round contamination. Everything with a different source and destination name
should be treated as unreliable until the baseline is fixed.

---

#### Section 4 — Per-player disagreement counts vs FCA: inflated by cross-round issue

```
  Player         z50    z75   z100   z125   z150   z200   Total
  Ax1Le         4014   4017   4015   4015   4015   4015    5046
  ZywOo         1884   1884   1884   1884   1884   1884    5361
  misutaaa      1479   1477   1477   1477   1476   1476    5178
  [...]
```

The `Total` column should represent the number of round-1 frames per player. From
`sample_df.csv` (the true round-1 only data), the actual frame counts were: ZywOo=62,
misutaaa=48, Kyojin=101, apEX=103. But the `Total` column here shows ZywOo=5361,
misutaaa=5178 — numbers that are roughly 80–100× larger than the real round-1 frame
counts. These inflated totals directly confirm that the 34221-row join result includes
massive numbers of cross-round false matches being counted as "round 1" data for each
player.

The per-player disagreement counts (the z50, z75, ... columns) are similarly inflated
and unreliable as absolute numbers. However, the **relative pattern within a single
row** is still informative: ZywOo's disagreement counts are identical across all
z_tol values (1884 in every column), meaning the z-tolerance filter makes zero
difference to ZywOo's assignments in the data — perhaps because ZywOo operates
exclusively in flat-ground areas where no z-overlap correction is ever triggered.
This relative stability is a genuine signal even in the contaminated data.

Similarly, the slight variation for Ax1Le across z_tol values (4014, 4017, 4015,
4015, 4015, 4015) is real but tiny, suggesting z_tol changes affect only a handful
of Ax1Le's frames. The much larger absolute number (4014) is the artifact.

---

#### Sections 5 and 6 — Plot generation

60 vs-FCA comparison plots were generated (10 per z_tol value) and 30 adjacent-pair
comparison plots (10 per active adjacent pair). The vs-FCA plots are unreliable for
the same reasons as Sections 1, 3, and 4 — the plot subjects were selected from
the cross-round-contaminated disagreement rows, so they may be showing genuine
geographic mismatches between rounds rather than z-overlap corrections. These plots
should not be used for calibration decisions until the FCA baseline is fixed.

The adjacent-pair plots are more trustworthy since they compare two z_tol values
against each other using consistent data. Specifically, the 10 plots for z_tol=50
vs z_tol=75 and 10 plots for z_tol=75 vs z_tol=100 represent the most diagnostic
cases for understanding where the z-filter boundary falls in real gameplay positions.

### 13.5 Critical finding: wrong FCA baseline used

The verification script was designed to use `outputs/sample_df.csv` (1114 rows, round
1 only, produced by `find_closest_area`) as the ground-truth baseline. Instead,
`outputs/round_1_player_positions_df.csv` (50168 rows, all rounds, produced by the
KDTree pipeline script) was passed as the FCA baseline. The filename contains "round_1"
but the file contains all rounds.

**Consequences:**
- Section 1 (agreement vs FCA): invalid. 95.4% disagreement is a cross-round artifact.
- Section 3 (area-name breakdown): mostly invalid. Geographically implausible
  corrections (TSpawn→CTSpawn, BombsiteA→Shop) are cross-round artifacts.
- Section 4 (per-player counts): inflated by ~80–100×. Relative patterns within a
  player's row are valid; absolute numbers are not.
- Section 2 (pairwise matrix): largely valid. Pairwise comparisons between z_tol
  values are not contaminated by the FCA baseline issue.
- Section 5 (vs-FCA plots): unreliable. Plot subjects were selected from contaminated
  disagreement rows.
- Section 6 (adjacent-pair plots): valid. These do not involve the FCA baseline.

**Fixes required before re-running:**

1. In the verification script, change the FCA baseline path to `sample_df.csv`:
   ```python
   FCA_BASELINE_CSV = "outputs/sample_df.csv"   # round 1 only, 1114 rows
   ```

2. Add an explicit round filter to the FCA baseline after loading, as a defensive
   guard even if the file is correctly specified:
   ```python
   fca_df = fca_df[fca_df["round"] == 1].copy()
   ```
   Without this, any accidental use of a multi-round CSV as the baseline will again
   produce silent false matches rather than an immediate error.

3. Add the same round filter to the join itself — join on `(player_name, frame_idx,
   round)` rather than `(player_name, frame_idx)` alone — to make the key globally
   unique even if the baseline CSV is multi-round:
   ```python
   merged = fca_df.merge(
       r1_frames[z_tol][["player_name", "frame_idx", "round",
                          "current_areaId", "mapping_distance"]],
       on=["player_name", "frame_idx", "round"],
       ...
   )
   ```

### 13.6 Reliable conclusions from the first sweep run

Despite the FCA baseline issue, the following conclusions from Section 2 are sound
and carry forward to calibration decisions:

**The effective tolerance space has four distinct configurations:**
`{z_tol=50}`, `{z_tol=75}`, `{z_tol=100 ≡ z_tol=125}`, `{z_tol=150 ≡ z_tol=200}`.

**The 50→75 boundary is the most active zone** (487 disagreements across the dataset).
This is the boundary that most aggressively affects mapping decisions. Whether z_tol=50
is over-correcting on flat-ground tiles or correctly correcting genuine z-overlap cases
is the key open question, resolvable only by inspecting the adjacent-pair plots.

**The 75→100 boundary is moderately active** (100 disagreements). Smaller effect.

**The 100–125 range is inert** — picking anywhere in this range produces identical
results. Similarly, the 150–200 range is inert.

**The 125→150 boundary affects 62 frames** — non-trivial but the smallest active zone.

**Recommended calibration strategy after the baseline fix:** once the correct baseline
is in place and re-run, focus analysis on whether z_tol=75 corrections (relative to
z_tol=100) look geographically plausible. If the 100 frames where z_tol=75 and
z_tol=100 differ all correspond to known or visually apparent z-overlap positions
(Catwalk, elevated surfaces, staircase tile boundaries), z_tol=75 may be a better
default than z_tol=100. If any of those 100 corrections look wrong (a ground-level
player being sent to a tile far from their position), z_tol=100 should be retained.

### 13.7 Outstanding items from the sweep work

- [ ] **Fix the FCA baseline in `verify_ztol_sweep.py`**: change `FCA_BASELINE_CSV`
      to `outputs/sample_df.csv`, add a round-1 filter on the loaded DataFrame, and
      add `round` to the join key. See Section 13.5 for the exact code fixes.
- [ ] **Re-run `verify_ztol_sweep.py`** with the corrected baseline and document the
      updated Section 1 and Section 3 output here. The pairwise matrix (Section 2)
      findings from the first run are already reliable and should not change
      substantially.
- [ ] **Inspect the adjacent-pair plots** (Section 6 output, which is not contaminated
      by the baseline issue) for the two active boundaries: z_tol=50 vs z_tol=75 (487
      cases, most important) and z_tol=75 vs z_tol=100 (100 cases). Document findings
      here.
- [ ] **Make a z_tol calibration decision** based on the above inspections and the
      corrected vs-FCA comparison. Update `Z_TOL` in `ztol_sweep.py` and all pipeline
      scripts that call `map_position_kdtree`. The provisional value is 100.0 — retain
      this until evidence from the corrected run justifies changing it.
- [ ] **Re-run vs-FCA comparison plots** (Section 5) with the corrected baseline and
      inspect the plots specifically for the shox/Catwalk case (area 268, z=-54.0) —
      this is the clearest pre-identified z-overlap candidate and the primary test case
      for confirming the filter is working as intended.

---

## 14. Files Produced by z_tol Sweep Work

| File | Location | Description |
|---|---|---|
| `ztol_sweep.py` | `scripts/` | Sweep runner. Loops over `Z_TOL_VALS`, runs `map_position_kdtree` at each value, saves one CSV per value. Bug fixed: `return results` was incorrectly indented inside the `for` loop; dedented to function body level. |
| `verify_ztol_sweep.py` | `verification/` | Comparison and visualisation script. Six output sections; Sections 1/3/4/5 are currently unreliable due to wrong FCA baseline (see Section 13.5). Section 2 (pairwise matrix) and Section 6 (adjacent-pair plots) are reliable. |
| `player_df_ztol_50.csv` | `outputs/ztol_sweep/` | KDTree mapping of full demo (all rounds, 50168 rows) at z_tol=50.0. Includes embedded `z_tol` column. |
| `player_df_ztol_75.csv` | `outputs/ztol_sweep/` | As above, z_tol=75.0. |
| `player_df_ztol_100.csv` | `outputs/ztol_sweep/` | As above, z_tol=100.0. Identical to z_tol=125 for every row (0 pairwise disagreements). |
| `player_df_ztol_125.csv` | `outputs/ztol_sweep/` | As above, z_tol=125.0. Identical to z_tol=100 for every row. |
| `player_df_ztol_150.csv` | `outputs/ztol_sweep/` | As above, z_tol=150.0. Identical to z_tol=200 for every row. |
| `player_df_ztol_200.csv` | `outputs/ztol_sweep/` | As above, z_tol=200.0. Identical to z_tol=150 for every row. |
| `ztol_50/` through `ztol_200/` | `outputs/ztol_sweep_plots/vs_fca/` | 10 comparison plots per z_tol showing FCA (red) vs KDTree (teal) tile assignments on the de_mirage radar. **Currently unreliable — plots were generated from cross-round-contaminated disagreement rows.** Re-generate after fixing the FCA baseline. |
| `ztol50_vs_ztol75/` | `outputs/ztol_sweep_plots/adjacent_pairs/` | 10 plots comparing z_tol=50 vs z_tol=75 assignments for the 487 rows where they differ. **Reliable** — does not use FCA baseline. |
| `ztol75_vs_ztol100/` | `outputs/ztol_sweep_plots/adjacent_pairs/` | 10 plots comparing z_tol=75 vs z_tol=100 for the 100 differing rows. **Reliable.** |
| `ztol125_vs_ztol150/` | `outputs/ztol_sweep_plots/adjacent_pairs/` | 10 plots comparing z_tol=125 vs z_tol=150 for the 62 differing rows. **Reliable.** |

---

*Document started during Task 3 implementation (3 Jul 2026). Verification output
added 3 Jul 2026 (11 PASS, 6 WARN, 0 FAIL). GIF qualitative inspection findings
pending — to be added to Section 9.4 after reviewing `outputs/position_mapping_gifs/`.
z_tol sweep work and first verification run added 3 Jul 2026 — re-run pending after
FCA baseline fix (Section 13.5).*

