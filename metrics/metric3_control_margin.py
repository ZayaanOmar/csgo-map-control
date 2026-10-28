"""
Metric 3 — Control Margin

Definition (Research_Proposal.pdf, §4.4.3, "Metric 3: Control Margin"):

    Let R be the set of all rounds and F_r the set of frames within round r
    (after buy-phase filtering). For a frame f, the CT control advantage is
    the mean control value over all nav mesh nodes:

        A_CT(f) = (1 / |V|) * sum_{v in V} c(v, f)

    c(v, f) close to 1 indicates CT control, so A_CT(f) in [0, 1] measures
    CT's mean spatial advantage across the map in frame f (0.5 = neutral).
    The Control Margin for round r is the mean of this quantity over all
    frames in the round:

        A_CT_bar(r) = (1 / |F_r|) * sum_{f in F_r} A_CT(f)

    Rounds are partitioned by outcome:

        R_win  = {r in R : CT wins round r}
        R_loss = {r in R : CT loses round r}

    and the group means are:

        mu_win  = (1 / |R_win|)  * sum_{r in R_win}  A_CT_bar(r)
        mu_loss = (1 / |R_loss|) * sum_{r in R_loss} A_CT_bar(r)

H3 / RQ3: mu_win > mu_loss — CT maintains a higher mean spatial control
advantage, sustained across the round, in rounds they win than in rounds
they lose.

-------------------------------------------------------------------------
ACTIVE-FOOTPRINT EXTENSION (added after empirical review)
-------------------------------------------------------------------------
A_CT(f) above averages c(v, f) over ALL |V| = 903 nodes. At any frame,
most nodes are far from every player: with ~903 nodes and typically 5-10
alive players, only a fraction sit within meaningful Gaussian-influence
range of anyone. For a node far from everyone, ct_mu(v,f) -> 0 and
t_mu(v,f) -> 0, so c(v,f) = expit(K * (ct_mu - t_mu)) -> expit(0) = 0.5
exactly — a hard-coded neutral value carrying no outcome-relevant
information. Because A_CT(f) averages over all 903 nodes regardless, both
mu_win and mu_loss get pulled toward 0.5 by this dead-weight majority,
compressing the raw group-mean difference to a few percentage points even
when the underlying effect (confirmed by a large Cohen's d) is real and
reliably detected.

The fix restricts the average to the "active footprint" — nodes with
non-negligible raw per-player influence reaching them, BEFORE the sigmoid
step:

    V_active(f) = { v in V : ct_mu(v, f) + t_mu(v, f) > EPSILON }

    A_CT_active(f) = (1 / |V_active(f)|) * sum_{v in V_active(f)} c(v, f)

EPSILON is defined relative to SIGMA (the Gaussian influence-decay
parameter in mu = exp(-d_eff^2 / 2*SIGMA^2)), NOT the sigmoid steepness
constant K. At d_eff = 3*SIGMA a single player's mu has already decayed to
exp(-4.5) ~= 0.011 — negligible, "map background," regardless of K.
EPSILON = exp(-4.5) is the default (use exp(-2) ~= 0.135 for a tighter
2*SIGMA cutoff by editing the EPSILON constant below).

IMPORTANT — upstream dependency: this filter requires the raw, pre-sigmoid
ct_mu and t_mu values per (round, frame_idx, area_id). The current
outputs/control_fields/{match_name}.csv schema does NOT carry these — it
only writes control_val, the post-sigmoid scalar. team_agg.py already
computes ct_mu and t_mu internally (the pivot step, immediately before the
sigmoid call) — it just does not currently persist them to the output CSV.
This script therefore DEGRADES GRACEFULLY: it always computes the
original full-map version (unchanged), and computes the active-footprint
version only for matches whose control_fields CSV already has ct_mu/t_mu
present, printing a NOTE for any match that doesn't. Until team_agg.py is
updated to also write ct_mu/t_mu, expect active_* columns to be NaN
everywhere. The required team_agg.py change is small — wherever it
currently writes only [round, frame_idx, area_id, control_val], add
ct_mu and t_mu to that same output selection.

Both the full-map version (A_CT_bar, original definition, always computed)
and the active-footprint version (A_CT_active_bar, new) are reported side
by side, plus the mean active-node fraction per round as a free
diagnostic — how much of the 903-node map is actually "in play" on
average, which is informative on its own.

Inputs:
    outputs/control_fields/{match_name}.csv  — round, frame_idx, area_id,
        control_val, [ct_mu, t_mu]  <- ct_mu/t_mu required for the active
        version; script falls back to full-map-only if absent
    outputs/round_metadata/{match_name}.csv  — match_id, round, winning_side, ...

Outputs:
    outputs/metrics/control_margin/{match_name}.csv — one row per round:
        match_id, round, control_margin_full, control_margin_active,
        active_node_fraction, winning_side
        (control_margin_active / active_node_fraction are NaN for matches
        whose control_fields CSV lacked ct_mu/t_mu)
    outputs/metrics/control_margin/_summary.json    — mu_win/mu_loss,
        Welch's t-test, and Cohen's d, computed separately for BOTH the
        full-map and active-footprint versions
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats

# --/ Path consts
CONTROL_FIELD_PATH = "outputs/control_fields"
ROUND_METADATA_PATH = "outputs/round_metadata"
OUTPUT_DIR = "outputs/metrics/control_margin"
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "_summary.json")

# --/ Active-footprint consts
SIGMA = 500.0                      # must match sigma used to build the influence field
EPSILON = float(np.exp(-2))      # d_eff = 2*SIGMA cutoff; see module docstring
REQUIRED_ACTIVE_COLS = {"ct_mu", "t_mu"}


def compute_control_margin(
    control_field: pd.DataFrame, round_metadata: pd.DataFrame, match_id: str
) -> pd.DataFrame:
    """Compute A_CT_bar(r) (full-map) and, if ct_mu/t_mu are present,
    A_CT_active_bar(r) (active-footprint) for every round in one match,
    with round outcome attached.

    Deliberately computed as two explicit stages per version — per-frame
    mean, then per-round mean of frame means — rather than one flat
    groupby-mean over all (round, area_id) rows. This matters more here
    than it did in the original full-map-only version: |V| is constant
    (903) every frame by construction, but |V_active(f)| genuinely varies
    frame to frame as players move. A flat mean over all active rows in a
    round would silently upweight frames with a larger active footprint
    relative to frames with a smaller one — the two-stage computation
    (mean per frame, then mean of those per-frame means) is the only way
    to keep every frame weighted equally, matching the proposal's formula.
    """
    has_active_cols = REQUIRED_ACTIVE_COLS.issubset(control_field.columns)

    if has_active_cols:
        control_field = control_field.copy()
        control_field["summed_mu"] = control_field["ct_mu"] + control_field["t_mu"]
        control_field["is_active"] = control_field["summed_mu"] > EPSILON

    # --- Full-map version (always computed, unchanged from the original) ---
    frame_means_full = (
        control_field.groupby(["round", "frame_idx"], sort=False)["control_val"]
        .mean()
        .reset_index(name="a_ct_frame_full")
    )
    round_means_full = (
        frame_means_full.groupby("round", sort=False)["a_ct_frame_full"]
        .mean()
        .reset_index(name="control_margin_full")
    )
    result = round_means_full

    # --- Active-footprint version (only if ct_mu/t_mu were provided) ---
    if has_active_cols:
        active_rows = control_field[control_field["is_active"]]

        frame_means_active = (
            active_rows.groupby(["round", "frame_idx"], sort=False)["control_val"]
            .mean()
            .reset_index(name="a_ct_frame_active")
        )
        round_means_active = (
            frame_means_active.groupby("round", sort=False)["a_ct_frame_active"]
            .mean()
            .reset_index(name="control_margin_active")
        )

        active_frac_per_frame = (
            control_field.groupby(["round", "frame_idx"], sort=False)["is_active"]
            .mean()
            .reset_index(name="active_frac_frame")
        )
        active_frac_per_round = (
            active_frac_per_frame.groupby("round", sort=False)["active_frac_frame"]
            .mean()
            .reset_index(name="active_node_fraction")
        )

        result = result.merge(round_means_active, on="round", how="left")
        result = result.merge(active_frac_per_round, on="round", how="left")
    else:
        result["control_margin_active"] = np.nan
        result["active_node_fraction"] = np.nan

    result["match_id"] = match_id

    merged = result.merge(
        round_metadata[["round", "winning_side"]], on="round", how="left"
    )
    return merged[
        [
            "match_id",
            "round",
            "control_margin_full",
            "control_margin_active",
            "active_node_fraction",
            "winning_side",
        ]
    ]


def _group_stats(win_group: pd.Series, loss_group: pd.Series) -> dict:
    """Welch's t-test + Cohen's d for one margin column — same approach as
    the original script (unequal-variance t-test as the safer default;
    pooled-SD Cohen's d), factored out so it can be applied identically to
    both the full-map and active-footprint columns.
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
        "n_ct_win": int(len(win_group)),
        "n_ct_loss": int(len(loss_group)),
        "mu_win": mu_win,
        "mu_loss": mu_loss,
        "welch_t_stat": t_stat,
        "welch_p_value": p_val,
        "cohens_d": cohens_d,
        "passes_h3": (mu_win is not None and mu_loss is not None and mu_win > mu_loss),
    }


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    control_field_files = sorted(glob.glob(os.path.join(CONTROL_FIELD_PATH, "*.csv")))
    print(f"Found {len(control_field_files)} control field files\n")

    all_rounds = []  # one row per round across the whole dataset — small, safe to hold in memory
    any_match_missing_active_cols = False

    for control_field_file in control_field_files:
        mname = os.path.splitext(os.path.basename(control_field_file))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{mname}.csv")

        round_metadata_file = os.path.join(ROUND_METADATA_PATH, f"{mname}.csv")

        # Skip-if-exists, BUT only if the cached file already has the new
        # active-footprint columns — a cache written before this change
        # would otherwise silently short-circuit recomputation and leave
        # active_* permanently NaN for that match even after team_agg.py
        # is patched to emit ct_mu/t_mu.
        # A cache that HAS the columns but only NaN values (written before
        # team_agg.py was patched) is also stale once the source CSV carries
        # ct_mu/t_mu, so it is recomputed too.
        if os.path.exists(output_path):
            cached = pd.read_csv(output_path)
            has_cols = {"control_margin_active", "active_node_fraction"}.issubset(cached.columns)
            empty_active = has_cols and cached["control_margin_active"].isna().all()
            source_has_active = REQUIRED_ACTIVE_COLS.issubset(
                pd.read_csv(control_field_file, nrows=0).columns
            )
            if has_cols and not (empty_active and source_has_active):
                print(f"Skipping {mname} - already exists with active-footprint columns")
                all_rounds.append(cached)
                continue
            print(f"Recomputing {mname} - cache is stale (missing or all-NaN active-footprint columns)")

        if not os.path.exists(round_metadata_file):
            print(f"Skipping {mname} - missing round_metadata file")
            continue

        try:
            print(f"Processing {mname}")

            control_field = pd.read_csv(control_field_file)
            round_metadata = pd.read_csv(round_metadata_file)

            if not REQUIRED_ACTIVE_COLS.issubset(control_field.columns):
                any_match_missing_active_cols = True
                print(
                    f"  NOTE: {mname} control_fields lacks ct_mu/t_mu — "
                    f"active-footprint columns will be NaN for this match"
                )

            round_scores = compute_control_margin(control_field, round_metadata, mname)
            round_scores.to_csv(output_path, index=False)
            print(f"  Saved {len(round_scores)} round scores to {output_path}")

            all_rounds.append(round_scores)

            del control_field, round_metadata, round_scores

        except Exception as e:
            print(f"\n FAILED: {mname} - {e}\n")
            if os.path.exists(output_path):
                os.remove(output_path)
            continue

    if not all_rounds:
        print("\nNo matches processed — nothing to summarise")
        return

    if any_match_missing_active_cols:
        print(
            "\nWARNING: at least one match's control_fields CSV lacked "
            "ct_mu/t_mu. The active-footprint metric is only as complete "
            "as its input — see module docstring for the required "
            "team_agg.py change (write ct_mu and t_mu alongside control_val)."
        )

    # --/ Dataset-level summary
    combined = pd.concat(all_rounds, ignore_index=True)

    n_before_drop = len(combined)
    combined = combined.dropna(subset=["winning_side", "control_margin_full"])
    n_dropped = n_before_drop - len(combined)
    combined["winning_side"] = combined["winning_side"].str.upper()

    win_full = combined[combined["winning_side"] == "CT"]["control_margin_full"]
    loss_full = combined[combined["winning_side"] == "T"]["control_margin_full"]
    win_active = combined[combined["winning_side"] == "CT"]["control_margin_active"]
    loss_active = combined[combined["winning_side"] == "T"]["control_margin_active"]

    stats_full = _group_stats(win_full, loss_full)
    stats_active = _group_stats(win_active, loss_active)

    mean_active_fraction = float(combined["active_node_fraction"].mean())

    summary = {
        "n_rounds_total": len(combined),
        "n_rounds_dropped_missing_outcome": n_dropped,
        "mean_active_node_fraction": mean_active_fraction,
        "epsilon": EPSILON,
        "sigma": SIGMA,
        "full_map": {
            **stats_full,
            "description": (
                "H3 / RQ3 tested over ALL 903 nodes per frame (original "
                "definition). Retained for continuity with earlier runs; "
                "diluted toward 0.5 by nodes far from every player — see "
                "'active_footprint' for the primary result."
            ),
        },
        "active_footprint": {
            **stats_active,
            "description": (
                "H3 / RQ3 tested over nodes with (ct_mu + t_mu) > EPSILON "
                "only, i.e. nodes within meaningful pre-sigmoid Gaussian "
                "influence range of at least one player. This is the "
                "primary Metric 3 result going forward. Will be all-None "
                "until team_agg.py is updated to persist ct_mu/t_mu."
            ),
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Control Margin summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
