# ESTA Demo Filtering — de_mirage Dataset Preparation

Reference document for the CS:GO map control project. Describes the
implementation of the demo-filtering pipeline that produces the cleaned,
de_mirage-only dataset used by all downstream pipeline stages. Covers the
rationale for each filtering decision, the key implementation issues
encountered and resolved, the output schema, and the verification approach.

Corresponds to **Research Schedule Task 5**: "Filter dataset to de_mirage
rounds, remove dead players per frame, and strip buy-phase frames"
(scheduled 12 Jul – 18 Jul).

---

## 1. Background and motivation

The ESTA dataset contains professional CS:GO match demos across multiple
maps, stored as `.json.xz` files under `esta/data/lan/` and
`esta/data/online/`. Each file is a compressed JSON object whose top-level
structure follows awpy's `Game` TypedDict schema — a dict containing
match metadata and a `"gameRounds"` list, where each round contains a
`"frames"` list of per-tick game-state snapshots.

The influence field pipeline operates exclusively on `de_mirage`. It
consumes frames one at a time, extracting per-player world coordinates to
map to nav nodes and compute graph distances. Two categories of frames are
useless or actively harmful to include:

- **Buy-phase frames** (`tick < freezeTimeEndTick`): players are stationary
  in spawn, no tactical movement is occurring, and including these frames
  would artificially inflate CT/T control readings near spawn for every
  round. The influence field is only meaningful during live gameplay.

- **Post-round-result frames** (`tick > endTick`): the round outcome has
  already been decided. Players may be running around or spectating. These
  frames carry no tactical signal and would pollute per-round aggregations
  (e.g. Metric 3, Control Margin) with post-hoc noise.

Additionally, warmup rounds (`isWarmup == True`) are structurally identical
to real rounds in the JSON but contain no competitive gameplay. They must be
removed entirely before any analysis.

---

## 2. Filtering decisions

### 2.1 Live round-time window

The tick window kept per round is:

```
freezeTimeEndTick <= frame["tick"] <= endTick
```

- `freezeTimeEndTick` is the tick at which the buy phase ends and players
  can move. Frames strictly before this tick are buy-phase frames and are
  dropped.
- `endTick` is the tick at which the round result is decided (bomb
  explodes, defuse completes, time runs out, or last player eliminated).
  Frames strictly after this tick are post-round dead time and are dropped.
- The choice of `endTick` rather than `endOfficialTick` (which includes
  the restart-delay period, `mp_round_restart_delay`) was deliberate:
  `endOfficialTick` frames are entirely post-result and have no tactical
  content. Cutting at `endTick` keeps the round complete up to the moment
  of decision without padding in dead time.

### 2.2 Warmup round removal

Any round with `isWarmup == True` is dropped entirely — no frames from
warmup rounds are retained. Warmup rounds always precede round 1 in the
ESTA demo format; in practice they appear as one or more leading entries
in `gameRounds` before the first competitive round.

### 2.3 Dead-player removal — deliberately deferred

The research schedule task description includes "remove dead players per
frame" as part of this step. This was evaluated and deliberately deferred
to the position-projection step for the following reason:

`extract_teams_metadata` (awpy's `map_control.py`) and the equivalent
logic in the position-to-node mapping script both already filter to
`isAlive == True` players when reading a frame. Dead players in the raw
frame JSON are therefore harmlessly ignored by any downstream consumer
that follows the standard awpy reading pattern. Stripping them at this
stage would offer a marginal storage/parsing-speed benefit but zero
correctness benefit, and would destroy the raw dead-player position data
(HP-at-death, position-at-death snapshots) that could be useful for future
extensions. The current pipeline does not use this data, but preserving it
costs nothing.

### 2.4 Event lists (kills, damages, bombEvents, etc.) — not filtered

Each `GameRound` dict contains both `frames` and separate event lists
(`kills`, `damages`, `grenades`, `bombEvents`, `weaponFires`, `flashes`).
Each event carries its own `tick` field. The filtering pipeline touches
only `frames` — event lists are left intact regardless of whether their
individual event ticks fall within the kept window.

This is correct for the current methodology, which drives everything off
frame-by-frame position snapshots rather than event lists. The one place
event ticks matter downstream is **Metric 2 (Site Control Score)**, which
requires locating the frame immediately preceding each bomb plant event
by looking up the plant tick in `bombEvents`. Since plants logically occur
only during live round time, a plant tick falling outside the kept window
would indicate a malformed round rather than a consequence of this
filtering step. When Metric 2 is implemented, an assertion should confirm
each plant tick is within `[freezeTimeEndTick, endTick]` as a guard.

---

## 3. Output schema

The output of the filtering pipeline is a new `.json.xz` file per input
demo. The schema is **identical** to awpy's native `Game` TypedDict — the
output is a strict structural subset of the input, not a transformed
representation. Specifically:

- All top-level keys (`matchID`, `mapName`, `tickRate`, `serverVars`,
  `matchPhases`, etc.) are copied unchanged via dict spreading
  (`{**demo, "gameRounds": filtered_rounds}`).
- `"gameRounds"` is replaced with a new list containing only non-warmup
  rounds, in their original order.
- Within each surviving round, all round-level keys (`roundNum`, `endTick`,
  `ctTeam`, `tTeam`, `kills`, `bombEvents`, etc.) are copied unchanged via
  dict spreading (`{**rnd, "frames": kept_frames}`).
- `"frames"` within each round is replaced with a new list containing only
  frames whose tick falls within `[freezeTimeEndTick, endTick]`, in their
  original order.

Nothing is mutated, reordered, or renamed. Any script that consumes awpy
`Game`/`GameRound`/`GameFrame`-shaped dicts will work identically against
these filtered files as against the originals.

### Output filename convention

```
{ctTeam}_VS_{tTeam}_{matchID}.json.xz
```

- `ctTeam` and `tTeam` are taken from the first surviving (non-warmup)
  round, sanitised to replace non-alphanumeric characters with underscores.
  These are for human readability only.
- `matchID` (note: the ESTA JSON uses `matchId` with a lowercase `d`,
  unlike the awpy `Game` TypedDict which specifies `matchID` with a capital
  `D` — this is an ESTA schema extension, not a standard awpy field) is
  the actual uniqueness guarantee. Two demos from different matches between
  the same teams on the same map will have different `matchId` values.
- Output directory: `data/processed/` (relative to project root).

### ESTA schema extensions

The ESTA demo JSON contains several top-level keys that are not part of
awpy's standard `Game` TypedDict but are present in every ESTA demo:

| Key | Type | Description |
|---|---|---|
| `matchId` | str | Unique match identifier (note lowercase `d`) |
| `demoId` | str | Unique demo file identifier |
| `competitionName` | str | Tournament/league name |
| `hltvUrl` | str | HLTV match page URL |
| `matchDate` | str | ISO date string of the match |
| `matchName` | str | Human-readable match name |

These are preserved unchanged in the output. The filtering pipeline does
not depend on them other than using `matchId` for output filename
construction.

---

## 4. Key implementation bugs encountered and resolved

### 4.1 Frame-filter condition typo (both comparisons used `freezeTimeEndTick`)

The initial implementation had the frame-drop condition written as:

```python
if frame['tick'] < freezeTimeEndTick or frame['tick'] > freezeTimeEndTick:
    continue
```

Both comparisons referenced `freezeTimeEndTick` — the second should have
been `endTick`. As written, every frame whose tick was not exactly equal to
`freezeTimeEndTick` (i.e. virtually every frame) was dropped, producing
zero kept frames per round. The correct condition is:

```python
if frame['tick'] < freezeTimeEndTick or frame['tick'] > endTick:
    continue
```

Or equivalently (less prone to this class of typo):

```python
if freezeTimeEndTick <= frame['tick'] <= endTick:
    kept_frames.append(frame)
```

### 4.2 Team name extraction before warmup filtering

An early version of the parsing function extracted `ctTeam` and `tTeam`
from `gameRounds[0]` before the warmup-filtering loop ran. Round 0 in ESTA
demos is typically a warmup round, whose `ctTeam`/`tTeam` fields may be
`None` or misleading. The fix is to extract team names from
`filtered_rounds[0]` after the filtering loop completes, guaranteeing the
source is a real competitive round.

### 4.3 `matchId` capitalisation

The ESTA JSON key is `matchId` (lowercase `d`), not `matchID` (capital `D`)
as in awpy's standard `Game` TypedDict. Using `demo['matchID']` raises a
`KeyError` at runtime. The correct key for ESTA demos is `demo['matchId']`.

---

## 5. Verification

`verify_filtered_demo.py` performs a structural and logical audit of one
processed demo against its corresponding original source demo. The testing
demo used during development was:

- **Original:** `esta/data/lan/884ea5ad-9174-4475-bc6c-5e031cfe2987.json.xz`
- **Processed:** `data/processed/FaZe_VS_BIG_4bab9a63-42e6-4766-acb6-cae50d575d04.json.xz`
- **Match:** FaZe vs BIG (`matchId: 4bab9a63-42e6-4766-acb6-cae50d575d04`)

### Checks performed

**Check 1 — Key structure:** Top-level, round-level, and frame-level keys
are compared between original and processed as sets. The round-level
comparison uses the first non-warmup round from the original (not round 0,
which may be warmup) to ensure a like-for-like comparison. A PASS confirms
the dict-spreading operation did not drop or add any keys at any level.

**Check 2 — Map name:** `pd["mapName"] == "de_mirage"` — confirms the
filtering correctly selected only de_mirage demos.

**Check 3 — No warmup rounds:** Iterates all rounds in the processed demo
and asserts none have `isWarmup == True`.

**Check 4 — Round count plausibility:** Asserts the processed round count
equals the original's non-warmup round count (i.e. filtering dropped
exactly the warmup rounds and nothing else), and that the count falls
within the plausible range for a professional CS:GO match (16–48, covering
standard MR15 and multiple overtime periods).

**Check 5 — Tick window correctness, all rounds:** Iterates every frame
in every round of the processed demo and asserts
`freezeTimeEndTick <= tick <= endTick`. This check covers all rounds, not
just round 1 — an earlier version of the verification script had a bug
where `if not invalidTick: break` at the end of the outer round loop
caused it to exit after the first clean round, checking only round 1.
The corrected condition is `if invalidTick: break` (exit early only when
a violation is found).

**Check 6 — Strict ordered subset of original:** For each surviving round,
locates the matching round in the original by `roundNum`, then verifies:
- Every frame tick in the processed round appears in the original round's
  frame list, in the same relative order (enforced by advancing a search
  pointer rather than allowing arbitrary reordering).
- Key non-frame round fields (`roundNum`, `isWarmup`, `startTick`,
  `freezeTimeEndTick`, `endTick`, `ctTeam`, `tTeam`, `winningSide`,
  `winningTeam`, `roundEndReason`) are byte-for-byte identical between
  original and processed, confirming the dict-spreading introduced no
  mutations.

### Result: 6/6 checks passed, 0 failed, 0 warned

Confirmed on the FaZe vs BIG testing demo. This is the reference result
for the correctness of the filtering pipeline.

---

## 6. Visualisation — round GIF

As a qualitative sanity check, `plot_round1_gif.py` produces an animated
GIF of all player positions across every frame of round 1, plotted on the
de_mirage simpleradar image using `awpy.visualization.plot.position_transform`
for world-to-radar coordinate conversion.

The GIF is generated from the position-to-node CSV produced by
`position_to_node_map.py` (Task 3 output), not directly from the filtered
demo JSON. This means the visualisation simultaneously validates both the
demo filtering (correct frames present) and the position-to-node mapping
(correct world coordinates extracted per player per frame).

**Dead-player handling:** player presence in the CSV at a given `frame_idx`
is the proxy for `isAlive`. A player absent from the current frame but
present in a prior frame is marked dead — their last known world position
is retained and plotted at `alpha = 0.25` (versus `alpha = 1.0` for alive
players). Once marked dead, a player's alpha does not recover within the
round.

**Known limitation of this approach:** the CSV contains only alive players
(because `position_to_node_map.py` reads from `player["isAlive"]`-filtered
frame data). A player who was never alive in a given round (e.g. a
substitute not yet in the server) simply never appears in `player_states`
and is not plotted at all — this is correct behaviour, not a gap.

---

## 7. Scripts produced

| Script | Location | Purpose |
|---|---|---|
| `filter_de_mirage_demos.py` | `scripts/` | Batch-filters all ESTA demos to de_mirage, writes one `.json.xz` per matching demo to `data/processed/` |
| `verify_filtered_demo.py` | `scripts/` | Verifies one processed demo against its original source across 6 structural and logical checks |
| `plot_round1_gif.py` | `scripts/` | Generates an animated GIF of all player positions in round 1 from the position-to-node CSV |

---

## 8. Status and what's next

- Demo filtering is **complete and verified** (6/6 checks passed on the
  FaZe vs BIG testing demo).
- Dead-player removal is **deferred** to the position-projection step by
  design — no action needed here.
- The filtered demos in `data/processed/` are the canonical input for all
  remaining pipeline stages.
- **Open item before influence field work can proceed:** the directed vs.
  symmetrized distance decision for the APSP matrix (Task 4) must be
  resolved and documented. The 33.27% asymmetry finding in the distance
  matrix, and the non-reciprocal edge check on the original 2,979-edge
  graph, are the two outstanding blockers from Task 4 that feed directly
  into the influence field formula (`d_G` in `µᵢ(x,t)`).
- **Next pipeline stage:** Task 6 — project each player's position forward
  1 second using velocity data, map the projected position to a nav node,
  and use both the current and projected area IDs to compute `d_eff` for
  the influence field formula.

---

*Compiled from project chat history. Testing was performed against the
FaZe vs BIG de_mirage demo from ESTA LAN data. The filtering logic
generalises to any ESTA demo on any map by changing the `mapName` filter
target, but has only been run and verified against de_mirage.*
