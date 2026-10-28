"""
Metric 3 (T-side) — Control Margin

Direct T-side counterpart to metric3_control_margin.py. Read that file's
module docstring first — this one only documents what's different.

-------------------------------------------------------------------------
1. T control value
-------------------------------------------------------------------------
control_val (in outputs/control_fields/{match}.csv) is the CT-perspective
sigmoid control value c(v, f) = expit(K * (ct_mu - t_mu)), where 1 = total
CT control and 0 = total T control. The T-perspective value is simply the
complement:

    t_control_val(v, f) = 1 - control_val(v, f) = expit(K * (t_mu - ct_mu))

A_T(f) = (1/|V|) * sum_{v in V} t_control_val(v, f) is therefore, over the
SAME node set V, exactly 1 - A_CT(f). The full-map and (map-wide)
active-footprint T margins below are consequently not new information on
their own -- they are the mirror image of the CT script's results, and
mu_win/mu_loss for these two will simply be 1 minus the CT script's
mu_loss/mu_win. They are still computed here (a) for direct symmetry and
so the T-side hypothesis can be stated and tested in its own terms without
requiring the reader to mentally flip the CT numbers, and (b) because they
are the baseline against which the genuinely new bombsite-restricted
metrics below are compared.

-------------------------------------------------------------------------
2. The bombsite-restricted metrics (the new part)
-------------------------------------------------------------------------
Working hypothesis: unlike CT, which needs to sustain control across the
whole (active) map to win, T may only need control concentrated at the
bombsite to win a round. If true, this predicts that a T control-margin
metric restricted to bombsite nodes should separate T-win from T-loss
rounds at least as well as the whole-active-footprint metric does -- even
though it is computed over a much smaller, more concentrated set of
nodes.

Four bombsite-node-set margins are computed, using the static BombsiteA /
BombsiteB node lists in data/de_mirage_bombsite_nodes.json:

    V_site_all  = V_BombsiteA union V_BombsiteB   (both sites, unfiltered)
    V_siteA_all = V_BombsiteA                     (site A only, unfiltered)
    V_siteB_all = V_BombsiteB                     (site B only, unfiltered)

    A_T_bombsite_all(f)  = mean over V_site_all  of t_control_val(v, f)
    A_T_bombsiteA_all(f) = mean over V_siteA_all of t_control_val(v, f)
    A_T_bombsiteB_all(f) = mean over V_siteB_all of t_control_val(v, f)

IMPORTANT caveat with the "_all" (unfiltered) versions: in a round where
the round's action happens entirely at one site, the OTHER site's nodes
sit near neutral (0.5) all round -- nobody is near them, so ct_mu and
t_mu are both ~0 there, and control_val -> expit(0) = 0.5. Averaging both
sites together every round therefore reproduces exactly the same
dilution problem that motivated the CT script's active-footprint fix,
just at bombsite scale instead of whole-map scale: a round decided
entirely at site A gets its site-A signal diluted by site B's inert
0.5s (and vice versa).

The fix is the same idea applied locally: intersect the bombsite node set
with the frame's active footprint (ct_mu + t_mu > EPSILON, identical
EPSILON/SIGMA convention as the CT script), so an untouched site's nodes
are simply excluded from that frame's average rather than dragging it
toward 0.5:

    V_site_active(f)  = V_site_all  intersect V_active(f)
    V_siteA_active(f) = V_siteA_all intersect V_active(f)
    V_siteB_active(f) = V_siteB_all intersect V_active(f)

    A_T_bombsite_active(f)  = mean over V_site_active(f)  of t_control_val(v, f)
    A_T_bombsiteA_active(f) = mean over V_siteA_active(f) of t_control_val(v, f)
    A_T_bombsiteB_active(f) = mean over V_siteB_active(f) of t_control_val(v, f)

A_T_bombsite_active_bar(r) (the round-level mean of A_T_bombsite_active(f)
over the round's frames) is the PRIMARY metric for testing the hypothesis
above. If a frame has zero active bombsite nodes (neither site is being
contested that instant), that frame simply contributes no row to the
per-round average for this metric -- exactly analogous to how the CT
script's active-footprint version already treats a frame with zero active
nodes map-wide. This means rounds where a site is barely contested are
implicitly downweighted for that site's metric, which is the intended
behaviour, not a bug.

Three "active-node-fraction" diagnostics are also recorded per round,
analogous to the CT script's active_node_fraction but restricted to the
bombsite node sets: what fraction of BombsiteA+B / BombsiteA-only /
BombsiteB-only nodes were within the active footprint, on average, per
frame. A round where this stays near 0 the whole time means the site(s)
were essentially never approached.

Per your instruction: the "_all" (unfiltered) bombsite margins are
computed over ALL rounds (not conditioned on which site was actually
planted/contested), and the active-footprint intersection above is relied
on to prevent an untouched site from diluting a given round's result,
rather than trying to determine which site was "the" site for that round
from bomb_plants data.

-------------------------------------------------------------------------
3. Area-name breakdown (the "where" diagnostic)
-------------------------------------------------------------------------
Separately from the confirmatory margin/hypothesis-test machinery above,
this script also produces an exploratory per-named-area report: for every
named area on de_mirage (e.g. "Catwalk", "BombsiteA", "TicketBooth",
"ApartmentsRamp", ...), the mean t_control_val across T-win rounds vs
T-loss rounds, and their difference. This uses RAW (unfiltered)
t_control_val -- it is descriptive, not a formal test -- and exists to
show where on the map T's control differs most between rounds T wins and
rounds T loses, with BombsiteA/BombsiteB explicitly flagged so their rank
in that list is directly visible against every other named region.

Area names come from data/de_mirage_area_assignment.csv (loaded via
pipeline/area_assignment.load_assignment), NOT from awpy's raw NAV
areaName. That file gives every one of the 72 nodes with a blank awpy
areaName (the originally-orphaned tiles pre-patch; see
de_mirage_navmesh_findings.md) a real area, so all 903 nodes fall into
27 named areas (26 awpy names + ApartmentsRamp) and there is no "" bucket.
The per-area output records how many of each area's nodes were formerly
blank (n_formerly_blank_nodes) so reassigned-heavy areas can be flagged.

Area-level averaging is footprint-weighted (USE_FOOTPRINT_WEIGHTS = True):
within a frame, an area's value is the mean of its nodes' t_control_val
weighted by each tile's |width x height| from NAV, matching how the
assignment's own to_areas() aggregates node values to areas. Without
weights a 20-unit sliver counts the same as a 200-unit tile. Set the flag
to False to reproduce the earlier unweighted node mean. Frame -> round
averaging is unchanged.

NOTE: the BombsiteA / BombsiteB metrics above still use the static node
lists in data/de_mirage_bombsite_nodes.json. The "BombsiteA" / "BombsiteB"
AREAS in this breakdown are defined by the assignment file and may include
tiles (e.g. reassigned formerly-blank ones) that are not in that JSON, so
the two node sets are not guaranteed to be identical.

-------------------------------------------------------------------------
Inputs
-------------------------------------------------------------------------
    outputs/control_fields/{match_name}.csv  — round, frame_idx, area_id,
        control_val, [ct_mu, t_mu]  <- ct_mu/t_mu required for every
        "*_active_*" column; script degrades gracefully (NaN) if absent,
        same policy as metric3_control_margin.py
    outputs/round_metadata/{match_name}.csv  — match_id, round,
        winning_side, ...
    data/de_mirage_bombsite_nodes.json       — {"A": [area_id, ...],
        "B": [area_id, ...]}
    data/de_mirage_area_assignment.csv       — area_id -> assigned_area for
        all 903 nodes (read through pipeline/area_assignment.py's
        load_assignment, which must be importable, i.e. pipeline/ on
        sys.path, as for the project's other scripts)

-------------------------------------------------------------------------
Outputs
-------------------------------------------------------------------------
    outputs/metrics/control_margin_t_side/{match_name}.csv
        One row per round: match_id, round, control_margin_full_t,
        control_margin_active_t, active_node_fraction,
        control_margin_bombsite_all_t, control_margin_bombsiteA_all_t,
        control_margin_bombsiteB_all_t, control_margin_bombsite_active_t,
        control_margin_bombsiteA_active_t, control_margin_bombsiteB_active_t,
        bombsite_active_node_fraction, bombsiteA_active_node_fraction,
        bombsiteB_active_node_fraction, winning_side

    outputs/metrics/control_margin_t_side/_area_breakdown_by_match/{match_name}.csv
        Per-match intermediate cache (round, area_name, mean_t_control,
        winning_side, area_scheme) -- independently skip-if-exists cached
        from the round-margin CSV above, so either can be recomputed
        without the other. area_scheme records which area definition and
        weighting produced the file ("assignment_v1|footprint_weighted" or
        "assignment_v1|unweighted"); a cached file with a missing or
        different area_scheme (e.g. one written when areas were raw awpy
        names) is recomputed, not trusted.

    outputs/metrics/control_margin_t_side/_area_breakdown.csv
        Dataset-level pivot: one row per area_name, with
        mean_control_t_win, mean_control_t_loss, diff_win_minus_loss,
        n_nodes_in_area, n_formerly_blank_nodes, is_bombsite -- sorted by
        diff_win_minus_loss descending.

    outputs/metrics/control_margin_t_side/_summary.json
        mu_win/mu_loss, Welch's t-test, Cohen's d for: full_map,
        active_footprint, bombsite_all, bombsite_active, bombsiteA_active,
        bombsiteB_active -- plus an effect_size_comparison block lining up
        active_footprint's Cohen's d against bombsite_active's.

Note on terminology: throughout this script "win" / "T win" means
winning_side == "T", and "loss" / "T loss" means winning_side == "CT"
(i.e. T lost that round). This is the opposite grouping direction from
metric3_control_margin.py, where "win"/"loss" are from the CT perspective.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats

from awpy.data import NAV
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "pipeline"))
from area_assignment import load_assignment  # pipeline/area_assignment.py

# --/ Path consts
CONTROL_FIELD_PATH = "outputs/control_fields"
ROUND_METADATA_PATH = "outputs/round_metadata"
BOMBSITE_NODES_PATH = "data/de_mirage_bombsite_nodes.json"
AREA_ASSIGNMENT_PATH = "data/de_mirage_area_assignment.csv"
OUTPUT_DIR = "outputs/metrics/control_margin_t_side"
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "_summary.json")
AREA_BREAKDOWN_DIR = os.path.join(OUTPUT_DIR, "_area_breakdown_by_match")
AREA_BREAKDOWN_PATH = os.path.join(OUTPUT_DIR, "_area_breakdown.csv")

MAP_NAME = "de_mirage"

# --/ Active-footprint consts — MUST match team_agg.py / metric3_control_margin.py
SIGMA = 500.0
EPSILON = float(np.exp(-2))
REQUIRED_ACTIVE_COLS = {"ct_mu", "t_mu"}

# --/ Area-breakdown consts
# True: within a frame, an area's value is the footprint-weighted mean of its
# nodes (matches to_areas() in the area-assignment workflow). False: plain
# node mean (the earlier behaviour). Changing this changes AREA_SCHEME, which
# invalidates cached per-match area files so they are recomputed.
USE_FOOTPRINT_WEIGHTS = True
AREA_SCHEME = "assignment_v1|" + ("footprint_weighted" if USE_FOOTPRINT_WEIGHTS else "unweighted")


# ---------------------------------------------------------------------------
# Static lookups
# ---------------------------------------------------------------------------
def load_bombsite_nodes(path: str = BOMBSITE_NODES_PATH) -> tuple[set, set]:
    """Load the static BombsiteA / BombsiteB node-id sets."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    a_nodes = {int(a) for a in raw.get("A", [])}
    b_nodes = {int(b) for b in raw.get("B", [])}
    return a_nodes, b_nodes


def build_area_lookup(
    assignment_path: str = AREA_ASSIGNMENT_PATH, map_name: str = MAP_NAME
) -> tuple[dict, dict, dict, dict]:
    """Everything the area-name breakdown needs, from the area assignment.

    Returns:
        area_to_name:  {area_id: assigned_area} for all assigned nodes
        footprints:    {area_id: |width x height|} from NAV, the weights for
                       footprint-weighted area averaging (all 1.0 if
                       USE_FOOTPRINT_WEIGHTS is False)
        name_counts:   {assigned_area: number of nodes in that area}
        blank_counts:  {assigned_area: number of those nodes whose awpy
                       areaName was blank before assignment}
    """
    labels, formerly_blank = load_assignment(assignment_path)
    area_to_name = {int(a): str(name) for a, name in labels.items()}
    formerly_blank = {int(a) for a in formerly_blank}

    nav = NAV[map_name]
    footprints = {}
    for aid in area_to_name:
        if USE_FOOTPRINT_WEIGHTS:
            tile = nav[aid] if aid in nav else nav[str(aid)]
            footprints[aid] = abs(
                (tile["northWestX"] - tile["southEastX"])
                * (tile["northWestY"] - tile["southEastY"])
            )
        else:
            footprints[aid] = 1.0

    name_counts: dict = {}
    blank_counts: dict = {}
    for aid, name in area_to_name.items():
        name_counts[name] = name_counts.get(name, 0) + 1
        blank_counts[name] = blank_counts.get(name, 0) + (1 if aid in formerly_blank else 0)
    return area_to_name, footprints, name_counts, blank_counts


# ---------------------------------------------------------------------------
# Core per-match margin computation
# ---------------------------------------------------------------------------
def compute_t_control_margin(
    control_field: pd.DataFrame,
    round_metadata: pd.DataFrame,
    match_id: str,
    bombsite_a_nodes: set,
    bombsite_b_nodes: set,
) -> pd.DataFrame:
    """Compute every T-side margin column for one match. See module
    docstring for full definitions. Two-stage (per-frame mean, then
    per-round mean of frame means) throughout, for the same reason as
    metric3_control_margin.py: the node set a given metric averages over
    (V, V_active(f), V_site, V_site_active(f)) is not constant frame to
    frame for the "_active" variants, so a flat groupby-mean over all
    matching rows in a round would silently upweight frames with a larger
    matching node set.
    """
    control_field = control_field.copy()
    control_field["t_control_val"] = 1.0 - control_field["control_val"]

    control_field["is_bombsite_a"] = control_field["area_id"].isin(bombsite_a_nodes)
    control_field["is_bombsite_b"] = control_field["area_id"].isin(bombsite_b_nodes)
    control_field["is_bombsite_all"] = (
        control_field["is_bombsite_a"] | control_field["is_bombsite_b"]
    )

    has_active_cols = REQUIRED_ACTIVE_COLS.issubset(control_field.columns)
    if has_active_cols:
        control_field["summed_mu"] = control_field["ct_mu"] + control_field["t_mu"]
        control_field["is_active"] = control_field["summed_mu"] > EPSILON

    def _two_stage_mean(mask, out_name: str) -> pd.DataFrame:
        subset = control_field[mask] if mask is not None else control_field
        frame_means = (
            subset.groupby(["round", "frame_idx"], sort=False)["t_control_val"]
            .mean()
            .reset_index(name="_frame_mean")
        )
        return (
            frame_means.groupby("round", sort=False)["_frame_mean"]
            .mean()
            .reset_index(name=out_name)
        )

    # --- Full map (always computed — mirror image of the CT full-map margin) ---
    result = _two_stage_mean(None, "control_margin_full_t")

    # --- Bombsite, unfiltered by activity (always computed) ---
    for site_mask_col, colname in [
        ("is_bombsite_all", "control_margin_bombsite_all_t"),
        ("is_bombsite_a", "control_margin_bombsiteA_all_t"),
        ("is_bombsite_b", "control_margin_bombsiteB_all_t"),
    ]:
        piece = _two_stage_mean(control_field[site_mask_col], colname)
        result = result.merge(piece, on="round", how="left")

    if has_active_cols:
        # --- Active footprint, whole map ---
        piece = _two_stage_mean(control_field["is_active"], "control_margin_active_t")
        result = result.merge(piece, on="round", how="left")

        active_frac_per_frame = (
            control_field.groupby(["round", "frame_idx"], sort=False)["is_active"]
            .mean()
            .reset_index(name="_active_frac_frame")
        )
        active_frac_per_round = (
            active_frac_per_frame.groupby("round", sort=False)["_active_frac_frame"]
            .mean()
            .reset_index(name="active_node_fraction")
        )
        result = result.merge(active_frac_per_round, on="round", how="left")

        # --- Bombsite x active-footprint intersection (primary hypothesis metric) ---
        for site_mask_col, colname in [
            ("is_bombsite_all", "control_margin_bombsite_active_t"),
            ("is_bombsite_a", "control_margin_bombsiteA_active_t"),
            ("is_bombsite_b", "control_margin_bombsiteB_active_t"),
        ]:
            combined_mask = control_field[site_mask_col] & control_field["is_active"]
            piece = _two_stage_mean(combined_mask, colname)
            result = result.merge(piece, on="round", how="left")

        # --- Bombsite-specific active-node fractions (diagnostic) ---
        for site_mask_col, out_col in [
            ("is_bombsite_all", "bombsite_active_node_fraction"),
            ("is_bombsite_a", "bombsiteA_active_node_fraction"),
            ("is_bombsite_b", "bombsiteB_active_node_fraction"),
        ]:
            site_rows = control_field[control_field[site_mask_col]]
            frac_per_frame = (
                site_rows.groupby(["round", "frame_idx"], sort=False)["is_active"]
                .mean()
                .reset_index(name="_frac_frame")
            )
            frac_per_round = (
                frac_per_frame.groupby("round", sort=False)["_frac_frame"]
                .mean()
                .reset_index(name=out_col)
            )
            result = result.merge(frac_per_round, on="round", how="left")
    else:
        for col in [
            "control_margin_active_t",
            "active_node_fraction",
            "control_margin_bombsite_active_t",
            "control_margin_bombsiteA_active_t",
            "control_margin_bombsiteB_active_t",
            "bombsite_active_node_fraction",
            "bombsiteA_active_node_fraction",
            "bombsiteB_active_node_fraction",
        ]:
            result[col] = np.nan

    result["match_id"] = match_id
    merged = result.merge(round_metadata[["round", "winning_side"]], on="round", how="left")

    ordered_cols = [
        "match_id",
        "round",
        "control_margin_full_t",
        "control_margin_active_t",
        "active_node_fraction",
        "control_margin_bombsite_all_t",
        "control_margin_bombsiteA_all_t",
        "control_margin_bombsiteB_all_t",
        "control_margin_bombsite_active_t",
        "control_margin_bombsiteA_active_t",
        "control_margin_bombsiteB_active_t",
        "bombsite_active_node_fraction",
        "bombsiteA_active_node_fraction",
        "bombsiteB_active_node_fraction",
        "winning_side",
    ]
    return merged[ordered_cols]


def compute_area_breakdown_for_match(
    control_field: pd.DataFrame,
    round_metadata: pd.DataFrame,
    match_id: str,
    area_to_name: dict,
    footprints: dict,
) -> pd.DataFrame:
    """Per-round, per-named-area mean T control value for one match.

    Two-stage average (frame, then round), same discipline as everywhere
    else in this pipeline. Within a frame, the nodes of one assigned area
    are pooled into a single value BEFORE the round-level average, so an
    area with many tiles is not implicitly overweighted by the frame mean.
    The pooling is a footprint-weighted mean (weights = tile |width x
    height|, see USE_FOOTPRINT_WEIGHTS) so a small sliver tile does not
    count as much as a large one.

    Nodes whose area_id is not in the assignment are dropped from the area
    breakdown (and reported), not silently given a name.

    Uses raw (unfiltered) t_control_val -- this feeds the exploratory
    _area_breakdown.csv report, not the confirmatory Welch/Cohen's-d
    tests in compute_t_control_margin.
    """
    control_field = control_field[["round", "frame_idx", "area_id", "control_val"]].copy()
    control_field["t_control_val"] = 1.0 - control_field["control_val"]
    control_field["area_name"] = control_field["area_id"].map(area_to_name)
    control_field["w"] = control_field["area_id"].map(footprints)

    n_unmapped = int(control_field["area_name"].isna().sum())
    if n_unmapped:
        print(
            f"  WARNING: {n_unmapped} control-field rows have an area_id not in "
            f"the area assignment — excluded from the area breakdown"
        )
        control_field = control_field.dropna(subset=["area_name", "w"])

    control_field["wv"] = control_field["w"] * control_field["t_control_val"]
    frame_area = control_field.groupby(["round", "frame_idx", "area_name"], sort=False)[
        ["wv", "w"]
    ].sum()
    frame_area["_frame_area_mean"] = frame_area["wv"] / frame_area["w"]
    frame_area_means = frame_area["_frame_area_mean"].reset_index()

    round_area_means = (
        frame_area_means.groupby(["round", "area_name"], sort=False)["_frame_area_mean"]
        .mean()
        .reset_index(name="mean_t_control")
    )
    round_area_means["match_id"] = match_id

    merged = round_area_means.merge(
        round_metadata[["round", "winning_side"]], on="round", how="left"
    )
    merged["area_scheme"] = AREA_SCHEME
    return merged[["match_id", "round", "area_name", "mean_t_control", "winning_side", "area_scheme"]]


def _group_stats(win_group: pd.Series, loss_group: pd.Series) -> dict:
    """Welch's t-test + Cohen's d for one margin column. Same approach as
    metric3_control_margin.py's _group_stats, with field names adjusted
    (n_t_win/n_t_loss rather than n_ct_win/n_ct_loss) since here "win"
    always means the T-win group regardless of which margin is passed in.
    """
    win_group = win_group.dropna()
    loss_group = loss_group.dropna()

    mu_win = float(win_group.mean()) if len(win_group) > 0 else None
    mu_loss = float(loss_group.mean()) if len(loss_group) > 0 else None

    if len(win_group) > 1 and len(loss_group) > 1:
        t_stat, p_val = stats.ttest_ind(win_group, loss_group, equal_var=False)
        pooled_std = np.sqrt((win_group.var() + loss_group.var()) / 2)
        cohens_d = float((mu_win - mu_loss) / pooled_std) if pooled_std > 0 else None
        t_stat, p_val = float(t_stat), float(p_val)
    else:
        t_stat, p_val, cohens_d = None, None, None

    return {
        "n_t_win": int(len(win_group)),
        "n_t_loss": int(len(loss_group)),
        "mu_win": mu_win,
        "mu_loss": mu_loss,
        "welch_t_stat": t_stat,
        "welch_p_value": p_val,
        "cohens_d": cohens_d,
        "mu_win_exceeds_mu_loss": (
            mu_win is not None and mu_loss is not None and mu_win > mu_loss
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AREA_BREAKDOWN_DIR, exist_ok=True)

    bombsite_a_nodes, bombsite_b_nodes = load_bombsite_nodes()
    print(
        f"Loaded bombsite nodes: {len(bombsite_a_nodes)} BombsiteA node(s), "
        f"{len(bombsite_b_nodes)} BombsiteB node(s)\n"
    )

    area_to_name, footprints, name_counts, blank_counts = build_area_lookup()
    print(
        f"Loaded area assignment: {len(area_to_name)} nodes -> {len(name_counts)} areas "
        f"({sum(blank_counts.values())} formerly blank nodes reassigned); "
        f"area scheme = {AREA_SCHEME}\n"
    )

    control_field_files = sorted(glob.glob(os.path.join(CONTROL_FIELD_PATH, "*.csv")))
    print(f"Found {len(control_field_files)} control field files\n")

    all_rounds = []
    all_area_records = []
    any_match_missing_active_cols = False

    # Presence of these two columns is how a cached margin CSV is known to
    # already include the bombsite work — mirrors the CT script's
    # cache-validity check for its active-footprint columns.
    expected_margin_cols = {"control_margin_bombsite_all_t", "control_margin_bombsite_active_t"}

    for control_field_file in control_field_files:
        mname = os.path.splitext(os.path.basename(control_field_file))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{mname}.csv")
        area_output_path = os.path.join(AREA_BREAKDOWN_DIR, f"{mname}.csv")
        round_metadata_file = os.path.join(ROUND_METADATA_PATH, f"{mname}.csv")

        margin_cached = False
        if os.path.exists(output_path):
            cached = pd.read_csv(output_path)
            if expected_margin_cols.issubset(cached.columns):
                print(f"Skipping {mname} margins — already exists with bombsite columns")
                all_rounds.append(cached)
                margin_cached = True
            else:
                print(f"Recomputing {mname} margins — cached file predates bombsite columns")

        # Schema-aware: a cached area file is only trusted if it was written
        # under the current area definition + weighting (area_scheme).
        # Files from the raw-awpy-name version have no area_scheme column and
        # are recomputed, not silently reused.
        area_cached = False
        if os.path.exists(area_output_path):
            cached_area = pd.read_csv(area_output_path)
            if "area_scheme" in cached_area.columns and (cached_area["area_scheme"] == AREA_SCHEME).all():
                all_area_records.append(cached_area)
                area_cached = True
            else:
                print(f"Recomputing {mname} area breakdown — cached file has a different/missing area scheme")

        if margin_cached and area_cached:
            continue

        if not os.path.exists(round_metadata_file):
            print(f"Skipping {mname} — missing round_metadata file")
            continue

        try:
            control_field = pd.read_csv(control_field_file)
            round_metadata = pd.read_csv(round_metadata_file)

            if not REQUIRED_ACTIVE_COLS.issubset(control_field.columns):
                any_match_missing_active_cols = True
                print(
                    f"  NOTE: {mname} control_fields lacks ct_mu/t_mu — "
                    f"active_t, bombsite_active_t, and *_active_node_fraction "
                    f"columns will be NaN for this match"
                )

            if not margin_cached:
                print(f"Processing {mname} (margins)")
                round_scores = compute_t_control_margin(
                    control_field, round_metadata, mname, bombsite_a_nodes, bombsite_b_nodes
                )
                round_scores.to_csv(output_path, index=False)
                print(f"  Saved {len(round_scores)} round scores to {output_path}")
                all_rounds.append(round_scores)

            if not area_cached:
                print(f"Processing {mname} (area breakdown)")
                area_records = compute_area_breakdown_for_match(
                    control_field, round_metadata, mname, area_to_name, footprints
                )
                area_records.to_csv(area_output_path, index=False)
                print(f"  Saved {len(area_records)} area-round records to {area_output_path}")
                all_area_records.append(area_records)

            del control_field, round_metadata

        except Exception as e:
            print(f"\n FAILED: {mname} - {e}\n")
            if os.path.exists(output_path) and not margin_cached:
                os.remove(output_path)
            if os.path.exists(area_output_path) and not area_cached:
                os.remove(area_output_path)
            continue

    if not all_rounds:
        print("\nNo matches processed — nothing to summarise")
        return

    if any_match_missing_active_cols:
        print(
            "\nWARNING: at least one match's control_fields CSV lacked "
            "ct_mu/t_mu. active_t, bombsite_active_t, and the "
            "*_active_node_fraction columns are only as complete as their "
            "input — see metric3_control_margin.py's docstring for the "
            "required team_agg.py change."
        )

    # === Round-level margin summary ===
    combined = pd.concat(all_rounds, ignore_index=True)
    n_before_drop = len(combined)
    combined = combined.dropna(subset=["winning_side", "control_margin_full_t"])
    n_dropped = n_before_drop - len(combined)
    combined["winning_side"] = combined["winning_side"].str.upper()

    def _win_loss(col: str) -> tuple[pd.Series, pd.Series]:
        # T win  -> winning_side == "T"
        # T loss -> winning_side == "CT"
        win = combined.loc[combined["winning_side"] == "T", col]
        loss = combined.loc[combined["winning_side"] == "CT", col]
        return win, loss

    margin_specs = [
        (
            "full_map",
            "control_margin_full_t",
            "T control margin over ALL 903 nodes per frame — the T-side "
            "mirror of the CT full-map metric (mu_win/mu_loss here equal "
            "1 minus the CT script's mu_loss/mu_win over the same rounds).",
        ),
        (
            "active_footprint",
            "control_margin_active_t",
            "T control margin restricted to nodes with (ct_mu+t_mu) > "
            "EPSILON — the T-side mirror of the CT active-footprint metric.",
        ),
        (
            "bombsite_all",
            "control_margin_bombsite_all_t",
            "T control margin over BombsiteA + BombsiteB nodes, "
            "unfiltered by activity. Can be diluted toward 0.5 by an "
            "untouched site in a given round — see 'bombsite_active' for "
            "the de-diluted version.",
        ),
        (
            "bombsite_active",
            "control_margin_bombsite_active_t",
            "T control margin over BombsiteA + BombsiteB nodes, "
            "restricted to those also in the active footprint that "
            "frame. PRIMARY metric for testing whether T only needs "
            "bombsite control (not whole-map/whole-active-footprint "
            "control) to win a round.",
        ),
        (
            "bombsiteA_active",
            "control_margin_bombsiteA_active_t",
            "As bombsite_active, restricted to BombsiteA nodes only.",
        ),
        (
            "bombsiteB_active",
            "control_margin_bombsiteB_active_t",
            "As bombsite_active, restricted to BombsiteB nodes only.",
        ),
    ]

    stats_block = {}
    for key, col, desc in margin_specs:
        win, loss = _win_loss(col)
        stats_block[key] = {**_group_stats(win, loss), "description": desc}

    mean_active_fraction = (
        float(combined["active_node_fraction"].mean())
        if combined["active_node_fraction"].notna().any()
        else None
    )
    mean_bombsite_active_fraction = (
        float(combined["bombsite_active_node_fraction"].mean())
        if combined["bombsite_active_node_fraction"].notna().any()
        else None
    )

    d_active = stats_block["active_footprint"]["cohens_d"]
    d_bombsite_active = stats_block["bombsite_active"]["cohens_d"]
    effect_size_comparison = {
        "active_footprint_cohens_d": d_active,
        "bombsite_active_cohens_d": d_bombsite_active,
        "difference_bombsite_minus_active": (
            (d_bombsite_active - d_active)
            if (d_active is not None and d_bombsite_active is not None)
            else None
        ),
        "note": (
            "If bombsite_active's Cohen's d is comparable to or larger "
            "than active_footprint's, despite averaging over a much "
            "smaller set of nodes, that is consistent with the "
            "hypothesis that T's control advantage needs to be "
            "concentrated at the bombsite (rather than spread across the "
            "whole active footprint) to win a round. This is a "
            "descriptive comparison of two effect sizes computed here, "
            "not a formal statistical test of one hypothesis against "
            "the other, and it is not a comparison against the CT-side "
            "script's results."
        ),
    }

    summary = {
        "n_rounds_total": len(combined),
        "n_rounds_dropped_missing_outcome": n_dropped,
        "mean_active_node_fraction": mean_active_fraction,
        "mean_bombsite_active_node_fraction": mean_bombsite_active_fraction,
        "epsilon": EPSILON,
        "sigma": SIGMA,
        **stats_block,
        "effect_size_comparison": effect_size_comparison,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== T-side Control Margin summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {SUMMARY_PATH}")

    # === Area-name breakdown ===
    if all_area_records:
        area_combined = pd.concat(all_area_records, ignore_index=True)
        area_combined = area_combined.dropna(subset=["winning_side"])
        area_combined["winning_side"] = area_combined["winning_side"].str.upper()
        area_combined = area_combined[area_combined["winning_side"].isin(["CT", "T"])]

        grand_means = (
            area_combined.groupby(["area_name", "winning_side"], sort=False)["mean_t_control"]
            .mean()
            .reset_index()
        )
        pivoted = grand_means.pivot(
            index="area_name", columns="winning_side", values="mean_t_control"
        ).reset_index()
        pivoted = pivoted.rename(columns={"T": "mean_control_t_win", "CT": "mean_control_t_loss"})
        for c in ["mean_control_t_win", "mean_control_t_loss"]:
            if c not in pivoted.columns:
                pivoted[c] = np.nan

        pivoted["diff_win_minus_loss"] = pivoted["mean_control_t_win"] - pivoted["mean_control_t_loss"]
        pivoted["n_nodes_in_area"] = pivoted["area_name"].map(name_counts)
        pivoted["n_formerly_blank_nodes"] = pivoted["area_name"].map(blank_counts)
        pivoted["is_bombsite"] = pivoted["area_name"].isin(["BombsiteA", "BombsiteB"])

        pivoted = pivoted.sort_values("diff_win_minus_loss", ascending=False).reset_index(drop=True)
        pivoted.to_csv(AREA_BREAKDOWN_PATH, index=False)
        print(f"\nSaved area-name breakdown ({len(pivoted)} areas) to {AREA_BREAKDOWN_PATH}")

        bombsite_rows = pivoted[pivoted["is_bombsite"]]
        if not bombsite_rows.empty:
            n_areas = len(pivoted)
            print("\nBombsite rows in the area breakdown (rank / total, 1 = largest win-loss diff):")
            for _, row in bombsite_rows.iterrows():
                rank = int(pivoted.index[pivoted["area_name"] == row["area_name"]][0]) + 1
                print(
                    f"  {row['area_name']:<12} rank {rank}/{n_areas}  "
                    f"win={row['mean_control_t_win']:.4f}  "
                    f"loss={row['mean_control_t_loss']:.4f}  "
                    f"diff={row['diff_win_minus_loss']:+.4f}"
                )
    else:
        print("\nNo area-breakdown records produced — skipping area-name summary")


if __name__ == "__main__":
    main()
