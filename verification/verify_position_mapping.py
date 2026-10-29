"""
verify_task3_mapping.py
-----------------------
Structural and spatial verification of the Task 3 position-to-nav-node mapping
output (sample_df saved as CSV).

Checks performed (in order):
  1.  Schema integrity         — expected columns present, no extras, correct dtypes
  2.  Null/NaN audit           — no missing values in any column
  3.  area_id validity         — every mapped area_id exists in NAV["de_mirage"]
  4.  area_id patched-graph    — every mapped area_id exists as a node in the
                                 patched graph (confirms Option A loading and that
                                 mapping targets patched, not unpatched, graph)
  5.  Formerly-orphaned tiles  — checks whether any mapped area_ids fall within the
                                 72 originally-orphaned area IDs (CT stairs cluster
                                 511/512/525-558/561-566, Apartments ramp 873-880)
                                 and reports them — these are valid post-patch
                                 mappings and should be treated as a positive signal
                                 that players are being correctly assigned to
                                 formerly-orphaned but now-connected tiles
  6.  Mapping distance audit   — statistical summary of mapping_distance column;
                                 flags rows above WARNING_DIST_THRESHOLD (100 units)
                                 and CRITICAL_DIST_THRESHOLD (150 units) per player
  7.  Velocity completeness    — checks vx/vy/vz are populated (non-null, finite);
                                 frames with all-zero velocity are noted separately
                                 since this can be legitimate (player standing still)
                                 or a parsing gap
  8.  area_id float dtype      — warns if area_ids are stored as float64 (will need
                                 int cast before passing to area_distance() in Task 4)
  9.  Spatial coherence        — for each (player, round) track, checks that
                                 consecutive-frame area_ids are either identical or
                                 graph-adjacent in the patched graph; large jumps
                                 (non-adjacent, non-identical) are flagged as
                                 potential mapping errors or z-overlap misassignments
 10.  point_in_area spot-check — for a sample of rows, verifies that the mapped
                                 area's bounding box actually contains the player's
                                 (x, y) position using awpy's own point_in_area;
                                 reports containment rate as a quality metric
 11.  Coverage summary         — unique area_ids used, unique players, unique rounds,
                                 row counts per side
 12.  GIF qualitative verify  — generates one animated GIF per (player, round) track,
                                 overlaying the assigned nav tile and raw player position
                                 on the de_mirage radar. Tile face colour encodes
                                 mapping distance severity (yellow/orange/red); tile
                                 border colour encodes point_in_area containment
                                 (white=inside, magenta=outside); a dashed line from
                                 tile centroid to player dot visualises the distance;
                                 JUMP annotations mark non-adjacent frame transitions
                                 from Check 9. GIFs are written to GIF_OUTPUT_DIR.

Run from project root with venv active:
    python verification/verify_task3_mapping.py

Outputs:
    PASS / WARN / FAIL tags per check, plus a final summary.
    A FAIL stops the pipeline — the CSV should not be used downstream until fixed.
    A WARN is worth investigating but does not block downstream use.
    GIFs written to GIF_OUTPUT_DIR (default: outputs/gifs/) for qualitative review.
"""

import pickle
import sys
import json
import os
import shutil
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before ALL other
                       # matplotlib imports and before awpy.visualization imports,
                       # otherwise matplotlib may try to open a display connection
                       # and crash on a headless VM/HPC environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import imageio.v3 as imageio
import numpy as np
import pandas as pd
import networkx as nx

from awpy.data import NAV
from awpy.analytics.nav import point_in_area
from awpy.visualization.plot import plot_map, position_transform

# ---------------------------------------------------------------------------
# Configuration — adjust paths to match your project layout
# ---------------------------------------------------------------------------

CSV_PATH            = "outputs/position_maps/FaZe_VS_BIG_4bab9a63_42e6_4766_acb6_cae50d575d04.csv"
PATCHED_GRAPH_PATH  = "data/de_mirage_patched_graph_v3.pkl"
MAP_NAME            = "de_mirage"

# Mapping distance thresholds (game units).
# WARNING  : worth reviewing; tile may be large or player near a boundary
# CRITICAL : strong signal of a misassignment or z-overlap error
WARNING_DIST_THRESHOLD  = 100.0
CRITICAL_DIST_THRESHOLD = 150.0

# How many rows to spot-check with point_in_area (it's O(n) per call, slow at scale)
POINT_IN_AREA_SAMPLE_N = 100

# Confirmed originally-orphaned area ID sets (from de_mirage_navmesh_findings.md)
CT_STAIRS_ORPHANS = (
    {511, 512}
    | set(range(525, 559))   # 525–558 inclusive
    | set(range(561, 567))   # 561–566 inclusive
)  # 42 tiles — CT stairs up to A site

APARTMENTS_RAMP_ORPHANS = set(range(873, 881))  # 873–880 inclusive, 8 tiles
# Note: 773/774 membership in this cluster is unconfirmed — excluded here

ALL_KNOWN_ORPHANS = CT_STAIRS_ORPHANS | APARTMENTS_RAMP_ORPHANS
# Remaining ~22 scattered orphans are not individually listed here since their
# IDs were not enumerated in the findings doc — the check below will catch them
# too if they appear, labelled as "other orphan" vs known cluster members.

# GIF output configuration
GIF_OUTPUT_DIR     = "outputs/position_mapping_gifs"
                                           # (persistent — these are deliverables)
GIF_TMP_DIR        = "/tmp/task3_verify_gif_frames"  # staging dir for per-frame PNGs
                                                      # (cleared and deleted after use)
GIF_FRAME_DURATION = 400                  # milliseconds per frame → 5 fps (default 200) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                           # (slow enough to read per-frame annotations)
GIF_DPI            = 150                  # render resolution — balance between
                                           # visual clarity and output file size

# Expected columns and their pandas dtype categories
# ("int or float" because area_id may be float64 due to pandas nullable int behaviour)
EXPECTED_COLUMNS = {
    "round":            "numeric",
    "frame_idx":        "numeric",
    "player_name":      "object",
    "side":             "object",
    "x":                "numeric",
    "y":                "numeric",
    "z":                "numeric",
    "vx":               "numeric",
    "vy":               "numeric",
    "vz":               "numeric",
    "current_areaId":      "numeric",
    "mapping_distance": "numeric",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_COUNT = 0
WARN_COUNT = 0
FAIL_COUNT = 0

def header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def result(tag: str, msg: str) -> None:
    global PASS_COUNT, WARN_COUNT, FAIL_COUNT
    if tag == "PASS":
        PASS_COUNT += 1
    elif tag == "WARN":
        WARN_COUNT += 1
    elif tag == "FAIL":
        FAIL_COUNT += 1
    print(f"  [{tag}]  {msg}")

def note(msg: str) -> None:
    """Print an informational line that isn't a pass/warn/fail."""
    print(f"         {msg}")

# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------

header("LOADING ARTEFACTS")

# Load CSV
try:
    df = pd.read_csv(CSV_PATH)
    note(f"Loaded CSV: {CSV_PATH}")
    note(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print(f"  [FAIL]  CSV not found at {CSV_PATH}. Aborting.")
    sys.exit(1)

# Load patched graph (Option A)
try:
    with open(PATCHED_GRAPH_PATH, "rb") as f:
        G_patched = pickle.load(f)
    note(f"Loaded patched graph: {PATCHED_GRAPH_PATH}")
    note(f"Graph has {G_patched.number_of_nodes()} nodes, "
         f"{G_patched.number_of_edges()} edges")
except FileNotFoundError:
    print(f"  [FAIL]  Patched graph not found at {PATCHED_GRAPH_PATH}. Aborting.")
    sys.exit(1)

# Build patched-graph node set and adjacency set for later checks
patched_node_ids   = set(G_patched.nodes())
nav_area_ids       = set(NAV[MAP_NAME].keys())

# Build adjacency lookup: area_id -> set of direct graph neighbours (both directions,
# treating the directed graph as undirected for "reachable in one hop" purposes)
adjacency: dict[int, set] = defaultdict(set)
for u, v in G_patched.edges():
    adjacency[u].add(v)
    adjacency[v].add(u)

# ---------------------------------------------------------------------------
# CHECK 1: Schema integrity
# ---------------------------------------------------------------------------

header("CHECK 1 — Schema Integrity")

actual_cols = set(df.columns)
expected_cols = set(EXPECTED_COLUMNS.keys())

missing_cols = expected_cols - actual_cols
extra_cols   = actual_cols - expected_cols

if missing_cols:
    result("FAIL", f"Missing expected columns: {sorted(missing_cols)}")
else:
    result("PASS", "All expected columns present")

if extra_cols:
    result("WARN", f"Unexpected extra columns present: {sorted(extra_cols)} "
                   f"(harmless but worth noting)")
else:
    result("PASS", "No unexpected extra columns")

# Dtype check
dtype_issues = []
for col, expected_kind in EXPECTED_COLUMNS.items():
    if col not in df.columns:
        continue  # already flagged above
    if expected_kind == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
        dtype_issues.append(f"{col} (expected numeric, got {df[col].dtype})")
    elif expected_kind == "object" and not pd.api.types.is_object_dtype(df[col]):
        dtype_issues.append(f"{col} (expected object/string, got {df[col].dtype})")

if dtype_issues:
    result("FAIL", f"Dtype mismatches: {dtype_issues}")
else:
    result("PASS", "All column dtypes are as expected")

# ---------------------------------------------------------------------------
# CHECK 2: Null / NaN audit
# ---------------------------------------------------------------------------

header("CHECK 2 — Null / NaN Audit")

null_counts = df.isnull().sum()
cols_with_nulls = null_counts[null_counts > 0]

if cols_with_nulls.empty:
    result("PASS", "No null/NaN values in any column")
else:
    for col, count in cols_with_nulls.items():
        result("FAIL", f"Column '{col}' has {count} null values "
                       f"({100*count/len(df):.1f}% of rows)")

# Also check for inf values in numeric columns
inf_counts = {}
for col in EXPECTED_COLUMNS:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            inf_counts[col] = n_inf

if inf_counts:
    for col, count in inf_counts.items():
        result("FAIL", f"Column '{col}' has {count} infinite values — "
                       f"indicates a failed distance computation")
else:
    result("PASS", "No infinite values in any numeric column")

# ---------------------------------------------------------------------------
# CHECK 3: area_id validity against NAV["de_mirage"]
# ---------------------------------------------------------------------------

header("CHECK 3 — area_id Validity Against NAV[\"de_mirage\"]")

note(f"NAV[\"{MAP_NAME}\"] contains {len(nav_area_ids)} area IDs")

mapped_area_ids = set(df["current_areaId"].dropna().astype(int).unique())
invalid_ids = mapped_area_ids - nav_area_ids

if invalid_ids:
    result("FAIL", f"{len(invalid_ids)} area_id(s) in CSV do not exist in "
                   f"NAV[\"{MAP_NAME}\"]: {sorted(invalid_ids)}")
else:
    result("PASS", f"All {len(mapped_area_ids)} unique area_ids found in "
                   f"NAV[\"{MAP_NAME}\"]")

# ---------------------------------------------------------------------------
# CHECK 4: area_id validity against patched graph nodes
# ---------------------------------------------------------------------------

header("CHECK 4 — area_id Validity Against Patched Graph Nodes")

note(f"Patched graph has {len(patched_node_ids)} node IDs")

invalid_in_graph = mapped_area_ids - patched_node_ids

if invalid_in_graph:
    result("FAIL", f"{len(invalid_in_graph)} area_id(s) in CSV do not exist as "
                   f"nodes in the patched graph: {sorted(invalid_in_graph)}")
    note("This may indicate find_closest_area is running against the unpatched "
         "NAV_GRAPHS rather than the patched pickle — check your loading logic")
else:
    result("PASS", f"All {len(mapped_area_ids)} unique area_ids are valid nodes "
                   f"in the patched graph")

# ---------------------------------------------------------------------------
# CHECK 5: Formerly-orphaned tile mapping detection
# ---------------------------------------------------------------------------

header("CHECK 5 — Formerly-Orphaned Tile Mapping Detection")

note("Checks whether any player positions were correctly mapped to the 72")
note("originally-orphaned tiles (now connected via the patch). These are VALID")
note("mappings post-patch — their presence is a positive signal.")

df["_area_id_int"] = df["current_areaId"].astype(int)

ct_stairs_hits   = df[df["_area_id_int"].isin(CT_STAIRS_ORPHANS)]
apartments_hits  = df[df["_area_id_int"].isin(APARTMENTS_RAMP_ORPHANS)]

if len(ct_stairs_hits) > 0:
    result("PASS", f"{len(ct_stairs_hits)} row(s) mapped to CT stairs cluster "
                   f"(area IDs 511/512/525-558/561-566) — patch working correctly "
                   f"for this cluster")
    note(f"Unique CT-stairs area IDs used: "
         f"{sorted(ct_stairs_hits['_area_id_int'].unique())}")
    note(f"Players mapped here: "
         f"{sorted(ct_stairs_hits['player_name'].unique())}")
else:
    result("WARN", "No rows mapped to CT stairs orphan cluster — either no players "
                   "visited this region in the sample, or these tiles are being "
                   "skipped by find_closest_area (check NAV dict coverage)")

if len(apartments_hits) > 0:
    result("PASS", f"{len(apartments_hits)} row(s) mapped to Apartments ramp cluster "
                   f"(area IDs 873-880) — patch working correctly for this cluster")
    note(f"Unique Apartments-ramp area IDs used: "
         f"{sorted(apartments_hits['_area_id_int'].unique())}")
    note(f"Players mapped here: "
         f"{sorted(apartments_hits['player_name'].unique())}")
else:
    result("WARN", "No rows mapped to Apartments ramp orphan cluster — either no "
                   "players visited this region in the sample, or these tiles are "
                   "being skipped by find_closest_area (check NAV dict coverage)")

# Check for hits on the ~22 scattered orphans not individually listed above.
# We can't enumerate all 72 exactly without the full orphan ID list, but we
# can detect any hit on the known sets plus flag any blank-areaName areas used.
blank_name_hits = df[
    df["_area_id_int"].apply(
        lambda aid: NAV[MAP_NAME].get(aid, {}).get("areaName", None) == ""
    )
]
non_known_cluster_blanks = blank_name_hits[
    ~blank_name_hits["_area_id_int"].isin(ALL_KNOWN_ORPHANS)
]
if len(non_known_cluster_blanks) > 0:
    note(f"{len(non_known_cluster_blanks)} row(s) mapped to blank-areaName areas "
         f"outside the two named clusters — likely scattered orphan singletons:")
    note(f"  Area IDs: "
         f"{sorted(non_known_cluster_blanks['_area_id_int'].unique())}")
else:
    note("No rows mapped to blank-areaName areas outside the two named clusters")

# ---------------------------------------------------------------------------
# CHECK 6: Mapping distance audit
# ---------------------------------------------------------------------------

header("CHECK 6 — Mapping Distance Audit")

dist_col = df["mapping_distance"]
note(f"mapping_distance statistics across all {len(df)} rows:")
note(f"  min    = {dist_col.min():.2f} units")
note(f"  median = {dist_col.median():.2f} units")
note(f"  mean   = {dist_col.mean():.2f} units")
note(f"  p90    = {dist_col.quantile(0.90):.2f} units")
note(f"  p95    = {dist_col.quantile(0.95):.2f} units")
note(f"  max    = {dist_col.max():.2f} units")

# Rows exceeding warning threshold
warn_rows = df[dist_col > WARNING_DIST_THRESHOLD]
crit_rows = df[dist_col > CRITICAL_DIST_THRESHOLD]

if len(crit_rows) > 0:
    result("WARN", f"{len(crit_rows)} row(s) have mapping_distance > "
                   f"{CRITICAL_DIST_THRESHOLD} units (critical threshold) — "
                   f"strong z-overlap or off-mesh candidates:")
    # Group by player so it's readable
    per_player = crit_rows.groupby("player_name").agg(
        count=("mapping_distance", "count"),
        max_dist=("mapping_distance", "max"),
        area_ids=("_area_id_int", lambda x: sorted(x.unique()))
    )
    for player, row in per_player.iterrows():
        note(f"  {player}: {row['count']} row(s), max dist = {row['max_dist']:.1f}, "
             f"area_ids = {row['area_ids']}")
else:
    result("PASS", f"No rows exceed critical distance threshold "
                   f"({CRITICAL_DIST_THRESHOLD} units)")

if len(warn_rows) > 0:
    result("WARN", f"{len(warn_rows)} row(s) have mapping_distance > "
                   f"{WARNING_DIST_THRESHOLD} units (warning threshold) — "
                   f"worth inspecting in the per-player GIF")
    per_player_warn = warn_rows.groupby("player_name").agg(
        count=("mapping_distance", "count"),
        max_dist=("mapping_distance", "max"),
    )
    for player, row in per_player_warn.iterrows():
        note(f"  {player}: {row['count']} row(s), max dist = {row['max_dist']:.1f}")
else:
    result("PASS", f"No rows exceed warning distance threshold "
                   f"({WARNING_DIST_THRESHOLD} units)")

# ---------------------------------------------------------------------------
# CHECK 7: Velocity completeness
# ---------------------------------------------------------------------------

header("CHECK 7 — Velocity Completeness")

vel_cols = ["vx", "vy", "vz"]
any_null_vel = df[vel_cols].isnull().any(axis=1).sum()
any_inf_vel  = np.isinf(df[vel_cols].values).any(axis=1).sum()

if any_null_vel > 0:
    result("FAIL", f"{any_null_vel} row(s) have null velocity values — "
                   f"these players cannot have their positions projected in Task 5")
elif any_inf_vel > 0:
    result("FAIL", f"{any_inf_vel} row(s) have infinite velocity values")
else:
    result("PASS", "All velocity columns (vx, vy, vz) are populated and finite")

# All-zero velocity is valid (player standing still or dead — but dead players
# should have been filtered by isAlive check during extraction)
all_zero_vel = ((df["vx"] == 0) & (df["vy"] == 0) & (df["vz"] == 0)).sum()
pct_zero = 100 * all_zero_vel / len(df)
note(f"{all_zero_vel} row(s) ({pct_zero:.1f}%) have vx=vy=vz=0 (stationary or "
     f"end-of-round freeze) — review your late-round frames to confirm these are "
     f"legitimate standing-still cases and not dead-player leakage from extraction")

# ---------------------------------------------------------------------------
# CHECK 8: area_id dtype warning
# ---------------------------------------------------------------------------

header("CHECK 8 — area_id Float dtype Warning")

area_id_dtype = df["current_areaId"].dtype
if area_id_dtype == np.float64:
    result("WARN", f"current_areaId is stored as float64 (e.g. 12.0 instead of 12). "
                   f"This is harmless for inspection but will need to be cast to int "
                   f"before passing to area_distance() in Task 4. "
                   f"Cast with: df['current_areaId'] = df['current_areaId'].astype(int)")
elif pd.api.types.is_integer_dtype(area_id_dtype):
    result("PASS", f"current_areaId is already stored as integer dtype ({area_id_dtype})")
else:
    result("WARN", f"current_areaId has unexpected dtype {area_id_dtype} — "
                   f"verify this is correct before Task 4")

# ---------------------------------------------------------------------------
# CHECK 9: Spatial coherence — consecutive frame jumps
# ---------------------------------------------------------------------------

header("CHECK 9 — Spatial Coherence (Consecutive Frame Jumps)")

note("For each (player, round) track, checks that consecutive frames map to")
note("the same area OR a directly adjacent area in the patched graph.")
note("Non-adjacent jumps are flagged as potential misassignments.")

# Group by player and round, sort by frame_idx
jump_issues = []
tracks = df.groupby(["player_name", "round"])

for (player, rnd), group in tracks:
    group_sorted = group.sort_values("frame_idx")
    area_ids_seq = group_sorted["_area_id_int"].tolist()
    frame_idxs   = group_sorted["frame_idx"].tolist()

    for i in range(len(area_ids_seq) - 1):
        a = area_ids_seq[i]
        b = area_ids_seq[i + 1]
        if a == b:
            continue  # same tile, no movement — fine
        if b in adjacency[a]:
            continue  # direct graph neighbour — fine
        # Non-adjacent, non-identical consecutive mapping
        jump_issues.append({
            "player":     player,
            "round":      rnd,
            "frame_from": frame_idxs[i],
            "frame_to":   frame_idxs[i + 1],
            "area_from":  a,
            "area_to":    b,
        })

if len(jump_issues) == 0:
    result("PASS", "All consecutive-frame area transitions are same-tile or "
                   "graph-adjacent — no unexpected spatial jumps detected")
else:
    result("WARN", f"{len(jump_issues)} consecutive-frame non-adjacent jump(s) "
                   f"detected — these are candidates for z-overlap misassignment "
                   f"or large tile-boundary transitions:")
    # Summarise by player — don't print every row if there are many
    jump_counts = Counter(j["player"] for j in jump_issues)
    for player, count in jump_counts.most_common():
        note(f"  {player}: {count} jump(s)")
    # Print the worst few individually for inspection
    note("First 5 jumps (player, round, frame_from→frame_to, area_from→area_to):")
    for j in jump_issues[:5]:
        # Compute Euclidean centroid distance of the jump for context
        c_from = np.array(G_patched.nodes[j["area_from"]]["center"])
        c_to   = np.array(G_patched.nodes[j["area_to"]]["center"])
        dist   = float(np.linalg.norm(c_to - c_from))
        note(f"    {j['player']} | round {j['round']} | "
             f"frame {j['frame_from']}→{j['frame_to']} | "
             f"area {j['area_from']}→{j['area_to']} | "
             f"centroid dist = {dist:.1f} units")

# ---------------------------------------------------------------------------
# CHECK 10: point_in_area spot-check
# ---------------------------------------------------------------------------

header("CHECK 10 — point_in_area Spot-Check (Containment Validation)")

note(f"Sampling {POINT_IN_AREA_SAMPLE_N} rows and checking whether each mapped")
note(f"area's bounding box (XY only — awpy's point_in_area is 2D) contains")
note(f"the player's (x, y) position. Non-containment is not automatically wrong")
note(f"(centroid-nearest can map correctly to an adjacent tile whose centroid is")
note(f"closest even if the player is outside that specific tile's exact boundary)")
note(f"but a low containment rate is a signal worth investigating.")

sample_rows = df.sample(
    n=min(POINT_IN_AREA_SAMPLE_N, len(df)),
    random_state=42
).copy()

contained_count   = 0
not_contained     = []

for _, row in sample_rows.iterrows():
    area_id = int(row["_area_id_int"])
    point   = [row["x"], row["y"], row["z"]]
    try:
        inside = point_in_area(MAP_NAME, area_id, point)
    except ValueError:
        # area_id not in NAV — already caught in Check 3, skip here
        inside = False
    if inside:
        contained_count += 1
    else:
        not_contained.append({
            "player":   row["player_name"],
            "area_id":  area_id,
            "dist":     row["mapping_distance"],
            "x": row["x"], "y": row["y"], "z": row["z"],
        })

n_checked       = len(sample_rows)
containment_pct = 100 * contained_count / n_checked

note(f"Containment rate: {contained_count}/{n_checked} = {containment_pct:.1f}%")

if containment_pct >= 80:
    result("PASS", f"Containment rate {containment_pct:.1f}% — mapping is placing "
                   f"most players inside or very near their assigned tile")
elif containment_pct >= 60:
    result("WARN", f"Containment rate {containment_pct:.1f}% — lower than expected; "
                   f"inspect non-contained rows for z-overlap misassignments")
else:
    result("WARN", f"Containment rate {containment_pct:.1f}% — notably low; "
                   f"review your mapping function logic carefully")

if not_contained:
    note(f"{len(not_contained)} sampled row(s) where player is outside assigned "
         f"tile's bounding box (showing up to 5, sorted by mapping distance desc):")
    not_contained_sorted = sorted(not_contained, key=lambda r: r["dist"], reverse=True)
    for r in not_contained_sorted[:5]:
        note(f"  {r['player']} at ({r['x']:.1f}, {r['y']:.1f}, {r['z']:.1f}) "
             f"→ area {r['area_id']} | mapping_dist = {r['dist']:.1f}")

# ---------------------------------------------------------------------------
# CHECK 11: Coverage summary
# ---------------------------------------------------------------------------

header("CHECK 11 — Coverage Summary")

n_unique_areas   = df["_area_id_int"].nunique()
n_total_areas    = len(nav_area_ids)
coverage_pct     = 100 * n_unique_areas / n_total_areas

note(f"Total rows:                {len(df)}")
note(f"Unique players:            {df['player_name'].nunique()} "
     f"({sorted(df['player_name'].unique())})")
note(f"Unique rounds:             {df['round'].nunique()} "
     f"({sorted(df['round'].unique())})")
note(f"Frames per round:")

for rnd, grp in df.groupby("round"):
    note(f"  Round {rnd}: {grp['frame_idx'].nunique()} unique frame indices, "
         f"{len(grp)} player-frame rows")

note(f"Rows by side:")
for side, grp in df.groupby("side"):
    note(f"  {side}: {len(grp)} rows ({100*len(grp)/len(df):.1f}%)")

note(f"Unique area IDs mapped:    {n_unique_areas} of {n_total_areas} "
     f"({coverage_pct:.1f}% of de_mirage nav mesh) — low coverage is expected "
     f"for a single-round, single-demo sample")

# Most frequently mapped areas
top_areas = (
    df.groupby("_area_id_int")
    .size()
    .sort_values(ascending=False)
    .head(10)
)
note("Top 10 most-mapped area IDs (area_id: row count | areaName):")
for area_id, count in top_areas.items():
    area_name = NAV[MAP_NAME].get(area_id, {}).get("areaName", "UNKNOWN")
    note(f"  area {area_id:>5}: {count:>4} row(s) | '{area_name}'")

result("PASS", "Coverage summary complete (informational only — no pass/fail threshold)")

# ---------------------------------------------------------------------------
# CHECK 12: GIF-based qualitative verification
# ---------------------------------------------------------------------------
'''
header("CHECK 12 — GIF-based Qualitative Verification")

note("Generates one animated GIF per (player, round) track, overlaying the")
note("assigned nav tile and raw player position on the de_mirage simpleradar.")
note("")
note("Visual encoding per frame:")
note(f"  Tile face colour:")
note(f"    Yellow  — mapping_distance <= {WARNING_DIST_THRESHOLD:.0f} units (normal)")
note(f"    Orange  — mapping_distance >  {WARNING_DIST_THRESHOLD:.0f} units (warning threshold)")
note(f"    Red     — mapping_distance >  {CRITICAL_DIST_THRESHOLD:.0f} units (critical threshold)")
note("  Tile border colour:")
note("    White   — player is inside the tile's XY bounding box (point_in_area=True)")
note("    Magenta — player is outside the tile's XY bounding box (point_in_area=False)")
note("  Player dot:")
note("    Cyan    — CT-side player position")
note("    Red     — T-side player position")
note("  White X marker at tile centroid + white dashed line = visual representation")
note("    of mapping_distance (how far the player is from the assigned tile centroid)")
note("  WARNING JUMP annotation — this frame is the destination of a non-adjacent")
note("    area transition flagged in Check 9; warrants closer spatial inspection")
note("")
note("What to look for:")
note("  1. Player dot should sit inside or immediately adjacent to the tile rectangle.")
note("     If it is clearly elsewhere on the radar, the mapping is wrong for that frame.")
note("  2. Orange/red tiles are not automatically wrong — they may reflect a large or")
note("     oddly-shaped tile where the player is near the far edge from the centroid.")
note("     The signal to investigate is an orange/red tile where the dot is far from")
note("     the tile on the radar — that indicates a genuine misassignment.")
note("  3. JUMP frames should show plausible rapid movement across a tile boundary,")
note("     not sudden teleportation to an unrelated region of the map.")
note("  4. The full track across frames should trace a spatially coherent path —")
note("     smooth progression across adjacent tiles, not random jumping.")
note("  Priority GIFs to inspect first:")
note("    shox_round*.gif      — largest mapping distances flagged in Check 6")
note("    misutaaa_round*.gif  — second-largest mapping distances flagged in Check 6")
note("    interz_round*.gif    — elevated z (~-79 units) in late frames; potential")
note("                           z-overlap misassignment candidate")
note("    Any GIF with WARNING JUMP annotations — Check 9 non-adjacent transitions")
note(f"  Output directory: {GIF_OUTPUT_DIR}/")

# Build a fast-lookup set of (player, round, frame_idx) tuples that are the
# *destination* of a non-adjacent jump, so those frames can be annotated in the GIF.
# jump_issues was populated at module scope in Check 9 and is accessible here.
jump_frame_set = {
    (j["player"], int(j["round"]), int(j["frame_to"]))
    for j in jump_issues
}

# Create GIF output directory — persistent, these are deliverable artefacts
os.makedirs(GIF_OUTPUT_DIR, exist_ok=True)

# Create / clear temp staging directory for per-frame PNG files
if os.path.isdir(GIF_TMP_DIR):
    shutil.rmtree(GIF_TMP_DIR)
os.makedirs(GIF_TMP_DIR)

gif_count   = 0
skipped     = []
failed_gifs = []

# Precompute legend handles once and reuse in every frame.
# Doing this inside the loop would reconstruct the same list of patch/line
# objects thousands of times for no reason.
legend_handles = [
    patches.Patch(facecolor="yellow", edgecolor="gray", alpha=0.8,
                  label=f"dist <= {WARNING_DIST_THRESHOLD:.0f} (normal)"),
    patches.Patch(facecolor="orange", edgecolor="gray", alpha=0.8,
                  label=f"dist > {WARNING_DIST_THRESHOLD:.0f} (warning)"),
    patches.Patch(facecolor="red",    edgecolor="gray", alpha=0.8,
                  label=f"dist > {CRITICAL_DIST_THRESHOLD:.0f} (critical)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="cyan",
           markersize=6, label="CT player position"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
           markersize=6, label="T player position"),
    patches.Patch(facecolor="none", edgecolor="white",   linewidth=1.5,
                  label="inside tile (point_in_area)"),
    patches.Patch(facecolor="none", edgecolor="magenta", linewidth=1.5,
                  label="outside tile (point_in_area)"),
    Line2D([0], [0], marker="x", color="white", markersize=5,
           linestyle="none", label="tile centroid"),
    Line2D([0], [0], color="white", linewidth=0.8, linestyle="--",
           alpha=0.7, label="centroid to player (dist)"),
]

# ---------------------------------------------------------------------------
# Main GIF generation loop — one GIF per (player_name, round) combination
# ---------------------------------------------------------------------------

first_round = df["round"].min()

# for (player_name, rnd), track in df.groupby(["player_name", "round"]):
for (player_name, rnd), track in (df[df["round"] == first_round].groupby(["player_name", "round"])):

    track_sorted = track.sort_values("frame_idx").reset_index(drop=True)

    # Need at least 2 frames to make a meaningful animation
    if len(track_sorted) < 2:
        skipped.append((player_name, rnd, f"only {len(track_sorted)} frame(s)"))
        note(f"  Skipping {player_name} round {rnd} — only {len(track_sorted)} "
             f"frame(s), not enough for a meaningful animation")
        continue

    frame_png_paths = []

    for _, row in track_sorted.iterrows():

        frame_idx  = int(row["frame_idx"])
        area_id    = int(row["_area_id_int"])
        dist       = float(row["mapping_distance"])
        side_label = row["side"]

        # --- Radar background ---
        fig, ax = plot_map(MAP_NAME, map_type="simpleradar", dark=True)
        # Set standard figure size
        fig.set_size_inches(8, 6)

        # --- Retrieve tile bounding box from NAV ---
        area = NAV[MAP_NAME].get(area_id)
        if area is None:
            # area_id not in NAV — already caught by Check 3.
            # Skip this individual frame rather than crashing the whole GIF.
            plt.close(fig)
            continue

        # Transform all four bounding-box corners into radar pixel space individually.
        # Do NOT assume northWest < southEast in either axis — the coordinate sign
        # convention is not guaranteed by awpy (confirmed in the navmesh findings doc
        # and the awpy unified reference). Use min/max to derive a well-formed rectangle.
        nw_x_px = position_transform(MAP_NAME, area["northWestX"], "x")
        nw_y_px = position_transform(MAP_NAME, area["northWestY"], "y")
        se_x_px = position_transform(MAP_NAME, area["southEastX"], "x")
        se_y_px = position_transform(MAP_NAME, area["southEastY"], "y")

        rect_x = min(nw_x_px, se_x_px)
        rect_y = min(nw_y_px, se_y_px)
        rect_w = abs(se_x_px - nw_x_px)
        rect_h = abs(se_y_px - nw_y_px)

        # --- Tile face colour encodes mapping distance severity ---
        if dist > CRITICAL_DIST_THRESHOLD:
            face_color = "red"
            tile_alpha = 0.55
        elif dist > WARNING_DIST_THRESHOLD:
            face_color = "orange"
            tile_alpha = 0.45
        else:
            face_color = "yellow"
            tile_alpha = 0.35

        # --- Tile border colour encodes point_in_area containment ---
        # Run point_in_area independently for every frame being rendered — not just
        # the 100-row random sample from Check 10, but every frame for every player
        # being visualised. This gives full per-frame containment truth in the GIF.
        # Note: point_in_area is a 2D XY-only strict bounding-box check (Z ignored).
        contained = False
        try:
            contained = point_in_area(
                MAP_NAME, area_id, [row["x"], row["y"], row["z"]]
            )
        except (ValueError, KeyError):
            pass  # area_id already validated in Check 3; unexpected exception here
        edge_color = "white" if contained else "magenta"

        # --- Draw assigned nav tile as a filled rectangle ---
        tile_rect = patches.Rectangle(
            (rect_x, rect_y), rect_w, rect_h,
            linewidth=2.0,
            edgecolor=edge_color,
            facecolor=face_color,
            alpha=tile_alpha,
            zorder=3,
        )
        ax.add_patch(tile_rect)

        # --- Tile centroid marker (white X) ---
        # Pull centroid from the patched graph node attributes — these are the same
        # values the KDTree will eventually index over, so using them here keeps
        # the centroid display consistent with the fast-path mapping logic.
        center = G_patched.nodes[area_id]["center"]
        cx_px  = position_transform(MAP_NAME, center[0], "x")
        cy_px  = position_transform(MAP_NAME, center[1], "y")
        ax.scatter(cx_px, cy_px, c="white", s=16, marker="x",
                   linewidths=1.2, zorder=6)

        # --- Dashed line from centroid to player position ---
        # The visual length of this line on the radar is proportional to the
        # mapping_distance value. A long dashed line on an orange or red tile is
        # the primary visual signal that the frame warrants closer inspection.
        px = position_transform(MAP_NAME, row["x"], "x")
        py = position_transform(MAP_NAME, row["y"], "y")
        ax.plot(
            [cx_px, px], [cy_px, py],
            color="white", linewidth=0.9, linestyle="--", alpha=0.65, zorder=4,
        )

        # --- Player position dot ---
        dot_color = "cyan" if side_label == "CT" else "orange"
        ax.scatter(
            px, py,
            c=dot_color, s=8, zorder=7,
            edgecolors="white", linewidths=0.5,
        )

        # --- JUMP annotation ---
        # Appears when the previous frame's area_id and this frame's area_id are
        # neither identical nor adjacent in the patched graph (from Check 9).
        is_jump = (player_name, int(rnd), frame_idx) in jump_frame_set
        if is_jump:
            ax.text(
                0.02, 0.08,
                "WARNING JUMP",
                transform=ax.transAxes,
                color="red", fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75),
                zorder=10,
            )

        # --- Legend ---
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=4.5,
            framealpha=0.75,
            facecolor="#111111",
            labelcolor="white",
            edgecolor="#444444",
            handlelength=1.5,
        )

        # --- Frame title with full diagnostic info per frame ---
        area_name   = NAV[MAP_NAME].get(area_id, {}).get("areaName", "") or "(unnamed)"
        containment = "inside" if contained else "outside"
        jump_suffix = "  WARNING JUMP" if is_jump else ""
        ax.set_title(
            f"{player_name} ({side_label})  |  Round {rnd}  |  Frame {frame_idx}"
            f"  |  Area {area_id} '{area_name}'"
            f"  |  dist={dist:.1f}  |  {containment} tile{jump_suffix}",
            color="white", fontsize=5.5, pad=3,
        )
        ax.axis("off")

        # --- Save frame as PNG to temp staging directory ---
        # Zero-padded frame_idx in the filename guarantees correct alphabetical
        # sort order when stitching, even when frame indices span multiple digit
        # lengths (e.g. frames 9 and 10 sort correctly as f00009 and f00010).
        frame_path = os.path.join(
            GIF_TMP_DIR,
            f"{player_name.replace(' ', '_')}_r{rnd}_f{frame_idx:05d}.png",
        )
        fig.savefig(
            frame_path, dpi=GIF_DPI, facecolor="black",
        )
        plt.close(fig)   # essential — not closing causes memory to accumulate
                         # across hundreds of frames and will eventually OOM
        frame_png_paths.append(frame_path)

    # --- Stitch all frame PNGs for this track into one animated GIF ---
    if not frame_png_paths:
        skipped.append((player_name, rnd, "no frames rendered successfully"))
        note(f"  No frames rendered for {player_name} round {rnd} — skipping GIF")
        continue

    # Sanitise the player name for use as a filename component
    safe_name    = player_name.replace(" ", "_").replace("/", "_")
    gif_filename = f"{safe_name}_round{rnd}.gif"
    gif_path     = os.path.join(GIF_OUTPUT_DIR, gif_filename)

    try:
        # Sort by filename — zero-padded frame_idx guarantees correct order
        images = [imageio.imread(p) for p in sorted(frame_png_paths)]
        # duration: milliseconds per frame for imageio v3 GIF output.
        # Convention confirmed from awpy's own plot.py:
        #   imageio.imwrite(filename, images, duration=1000/fps)
        # i.e. duration=100 → 10fps, duration=200 → 5fps.
        # loop=0: repeat the GIF indefinitely when opened in a viewer.
        imageio.imwrite(gif_path, images, duration=GIF_FRAME_DURATION, loop=0)
        gif_count += 1
        note(f"  [{gif_count:>2}]  {gif_path}  ({len(frame_png_paths)} frames)")
    except Exception as exc:
        failed_gifs.append((player_name, rnd, str(exc)))
        note(f"  FAILED to write GIF for {player_name} round {rnd}: {exc}")

# --- Clean up temp PNG staging directory ---
# The per-frame PNGs are intermediate artefacts — only the final GIFs are kept.
if os.path.isdir(GIF_TMP_DIR):
    shutil.rmtree(GIF_TMP_DIR)

# --- Check 12 result ---
if failed_gifs:
    result("WARN", f"{len(failed_gifs)} GIF(s) failed to write: "
                   f"{[(p, r) for p, r, _ in failed_gifs]}")
if skipped:
    note(f"Skipped {len(skipped)} track(s) with insufficient frames: "
         f"{[(p, r) for p, r, _ in skipped]}")

if gif_count == 0 and not failed_gifs:
    result("WARN", "No GIFs were generated — check that the sample DataFrame "
                   "contains at least 2 frames per (player, round) combination")
elif gif_count > 0:
    result("PASS", f"{gif_count} GIF(s) written to '{GIF_OUTPUT_DIR}/'")
    note("Open each GIF and verify the four qualitative criteria:")
    note("  1. Player dot sits inside or immediately adjacent to the tile rectangle.")
    note("  2. Orange/red tiles make spatial sense — large tile, not a mapping to")
    note("     the wrong region of the map.")
    note("  3. WARNING JUMP frames show plausible fast movement, not teleportation.")
    note("  4. Full track traces a coherent path across frames — no random jumping.")
    note("  Start with these GIFs (flagged by earlier checks):")
    note(f"    shox_round*.gif      — largest mapping distances (Check 6)")
    note(f"    misutaaa_round*.gif  — second-largest mapping distances (Check 6)")
    note(f"    interz_round*.gif    — elevated z in late frames, z-overlap candidate")
    note(f"    Any GIF with WARNING JUMP — non-adjacent area transitions (Check 9)")
'''
# ---------------------------------------------------------------------------
# Cleanup temp column
# ---------------------------------------------------------------------------

df.drop(columns=["_area_id_int"], inplace=True, errors="ignore")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

header("VERIFICATION SUMMARY")

total = PASS_COUNT + WARN_COUNT + FAIL_COUNT
print(f"\n  {PASS_COUNT} PASS   {WARN_COUNT} WARN   {FAIL_COUNT} FAIL   "
      f"({total} checks total)\n")

if FAIL_COUNT > 0:
    print("  RESULT: PIPELINE BLOCKED — fix all FAIL items before proceeding to Task 4.")
    print("          The CSV should not be used as input to downstream stages until")
    print("          all FAILs are resolved.\n")
elif WARN_COUNT > 0:
    print("  RESULT: PROCEED WITH CAUTION — no hard failures, but review all WARN items.")
    print("          Pay particular attention to Check 9 (spatial jumps) and Check 6")
    print("          (large mapping distances) as these are the most likely signals of")
    print("          z-overlap misassignments that will propagate into Task 4 distances.")
    print(f"         Inspect GIFs in '{GIF_OUTPUT_DIR}/' for qualitative confirmation")
    print("          before building on this CSV in Task 4.\n")
else:
    print("  RESULT: ALL CHECKS PASSED — CSV is structurally sound and ready for Task 4.")
    print(f"         Inspect GIFs in '{GIF_OUTPUT_DIR}/' as a final qualitative check.\n")