"""
Metric 2 — Site Control Score

Definition (Research_Proposal.pdf, §4.4.3, "Metric 2: Site Control Score"):

    Let E be the set of all bomb-plant events. For each plant event e in E,
    let f_e denote the game frame immediately prior to e, and s in {A, B}
    the bomb site planted on. Let V_s be the set of nav mesh nodes
    belonging to bomb site s. The Site Control Score for plant event e is:

        C_e = (1 / |V_s|) * sum_{v in V_s} c(v, f_e)

    and the mean Site Control Score over all plant events is:

        C_bar = (1 / |E|) * sum_{e in E} C_e

    Since T-side control corresponds to c(v, f) < 0.5 in this model, a
    mean Site Control Score C_bar < 0.5 serves as a validity check: the
    T-side must physically occupy and survive on the bomb site to plant,
    so the model should show T-side spatial control at that moment.

Frame lookup ("the game frame immediately prior to the plant"): resolved
per round as the row in position_maps with the largest tick strictly less
than the plant's tick, using that row's frame_idx to key into
control_fields. This is the direct operationalisation of "immediately
prior" given that position_maps only has one row per (round, frame_idx)
per alive player, not one row per raw game tick.

Inputs:
    outputs/bomb_plants/{match_name}.csv    — match_id, round, bomb_site, tick, seconds, planter_name
    outputs/position_maps/{match_name}.csv  — round, frame_idx, tick, ... (one row per alive player per frame)
    outputs/control_fields/{match_name}.csv — round, frame_idx, area_id, control_val
    data/de_mirage_bombsite_nodes.json      — {"A": [area_ids...], "B": [area_ids...]}

Outputs:
    outputs/metrics/site_control_score/{match_name}.csv — one row per plant event
    outputs/metrics/site_control_score/_summary.json    — dataset-level C_bar and validity check
"""

import os
import glob
import json
import pandas as pd

# --/ Path consts
BOMB_PLANTS_PATH = "outputs/bomb_plants"
POSITION_MAPS_PATH = "outputs/position_maps"
CONTROL_FIELD_PATH = "outputs/control_fields"
BOMBSITE_NODES_PATH = "data/de_mirage_bombsite_nodes.json"
OUTPUT_DIR = "outputs/metrics/site_control_score"
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "_summary.json")

OUTPUT_COLUMNS = [
    "match_id", "round", "bomb_site", "plant_tick",
    "frame_idx", "site_control_score", "note",
]


def load_bombsite_nodes(path: str) -> dict:
    """Load V_A / V_B node-ID sets from the static bombsite-nodes file.

    Keys are normalised to bare 'A' / 'B' (see normalize_site_label) so
    lookups are consistent regardless of how the file's keys were written.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        normalize_site_label(site): set(int(a) for a in ids)
        for site, ids in raw.items()
    }


def normalize_site_label(raw) -> str:
    """Normalise a bomb-site label to a bare 'A' or 'B'.

    ESTA/awpy demo JSON has been observed to encode the bomb site either
    as a bare letter ('A' / 'B') or as a full named-place string
    ('BombsiteA' / 'BombsiteB') depending on parser/version. This
    normalises either form to the bare-letter form used as keys in
    de_mirage_bombsite_nodes.json, so a schema variation doesn't silently
    cause every plant event to fail the site lookup.
    """
    if raw is None:
        return raw
    raw = str(raw).strip()
    if raw in ("A", "B"):
        return raw
    if raw.endswith("A") or raw.endswith("B"):
        return raw[-1]
    return raw


def find_prior_frame(position_ticks: pd.DataFrame, round_num: int, plant_tick: int):
    """Return the frame_idx of the last frame in round_num with tick < plant_tick.

    position_ticks must contain distinct (round, frame_idx, tick) rows.
    Returns None if no such frame exists — this should not happen for a
    plant event inside a validly-filtered live round, but is handled
    explicitly rather than raising, so one malformed event doesn't kill
    the whole match's processing.
    """
    round_frames = position_ticks[position_ticks["round"] == round_num]
    prior = round_frames[round_frames["tick"] < plant_tick]
    if prior.empty:
        return None
    return int(prior.loc[prior["tick"].idxmax(), "frame_idx"])


def compute_site_control_scores(
    bomb_plants: pd.DataFrame,
    position_ticks: pd.DataFrame,
    control_field: pd.DataFrame,
    bombsite_nodes: dict,
) -> pd.DataFrame:
    """Compute C_e for every plant event in one match.

    Rows that cannot be scored (unknown site label, no prior frame found,
    or no control-field rows for V_s at that frame) are still emitted with
    site_control_score=None and a `note` explaining why, rather than being
    silently dropped — this keeps the per-event output a complete audit
    trail of every plant event in the match.
    """
    records = []
    for _, plant in bomb_plants.iterrows():
        site = normalize_site_label(plant["bomb_site"])
        base = {
            "match_id": plant["match_id"],
            "round": plant["round"],
            "bomb_site": site,
            "plant_tick": plant["tick"],
        }

        if site not in bombsite_nodes:
            records.append({**base, "frame_idx": None, "site_control_score": None,
                            "note": "unknown bomb_site label — no matching V_s"})
            continue

        frame_idx = find_prior_frame(position_ticks, plant["round"], plant["tick"])
        if frame_idx is None:
            records.append({**base, "frame_idx": None, "site_control_score": None,
                            "note": "no frame with tick < plant_tick found in this round"})
            continue

        v_s = bombsite_nodes[site]
        frame_rows = control_field[
            (control_field["round"] == plant["round"])
            & (control_field["frame_idx"] == frame_idx)
            & (control_field["area_id"].isin(v_s))
        ]
        if frame_rows.empty:
            records.append({**base, "frame_idx": frame_idx, "site_control_score": None,
                            "note": "no control field rows found for V_s at this frame"})
            continue

        c_e = float(frame_rows["control_val"].mean())
        records.append({**base, "frame_idx": frame_idx, "site_control_score": c_e, "note": ""})

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bombsite_nodes = load_bombsite_nodes(BOMBSITE_NODES_PATH)

    bomb_plant_files = sorted(glob.glob(os.path.join(BOMB_PLANTS_PATH, "*.csv")))
    print(f"Found {len(bomb_plant_files)} bomb plant files\n")

    all_scores = []  # one small DataFrame per match — one row per plant event, never per-node

    for bomb_plants_file in bomb_plant_files:
        mname = os.path.splitext(os.path.basename(bomb_plants_file))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{mname}.csv")

        position_maps_file = os.path.join(POSITION_MAPS_PATH, f"{mname}.csv")
        control_field_file = os.path.join(CONTROL_FIELD_PATH, f"{mname}.csv")

        if os.path.exists(output_path):
            print(f"Skipping {mname} - already exists")
            all_scores.append(pd.read_csv(output_path))
            continue

        if not (os.path.exists(position_maps_file) and os.path.exists(control_field_file)):
            print(f"Skipping {mname} - missing position_maps or control_fields file")
            continue

        try:
            print(f"Processing {mname}")

            bomb_plants = pd.read_csv(bomb_plants_file)
            if bomb_plants.empty:
                print(f"  No plant events in {mname} — writing empty output")
                pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_path, index=False)
                continue

            # Only the three columns needed for the frame lookup — this file
            # can be large (one row per alive player per frame), so column
            # projection at load time matters even though it's a CSV.
            position_ticks = (
                pd.read_csv(position_maps_file, usecols=["round", "frame_idx", "tick"])
                .drop_duplicates()
            )
            control_field = pd.read_csv(control_field_file)

            match_scores = compute_site_control_scores(
                bomb_plants, position_ticks, control_field, bombsite_nodes
            )
            match_scores.to_csv(output_path, index=False)
            print(f"  Saved {len(match_scores)} plant-event scores to {output_path}")

            all_scores.append(match_scores)

            del bomb_plants, position_ticks, control_field, match_scores

        except Exception as e:
            print(f"\n FAILED: {mname} - {e}\n")
            if os.path.exists(output_path):
                os.remove(output_path)
            continue

    if not all_scores:
        print("\nNo matches processed — nothing to summarise")
        return

    # --/ Dataset-level summary
    combined = pd.concat(all_scores, ignore_index=True)
    valid = combined[combined["site_control_score"].notna()]

    n_events_total = len(combined)
    n_events_valid = len(valid)
    n_events_dropped = n_events_total - n_events_valid

    c_bar = float(valid["site_control_score"].mean()) if n_events_valid > 0 else None
    c_std = float(valid["site_control_score"].std()) if n_events_valid > 0 else None

    per_site = {}
    if n_events_valid > 0:
        for site, group in valid.groupby("bomb_site"):
            per_site[site] = {
                "mean": float(group["site_control_score"].mean()),
                "std": float(group["site_control_score"].std()),
                "count": int(len(group)),
            }

    # breakdown of why any events were dropped, for diagnosing data-quality
    # issues rather than just seeing a lower n_events_valid with no context
    drop_reason_counts = (
        combined[combined["site_control_score"].isna()]["note"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "n_plant_events_total": n_events_total,
        "n_plant_events_scored": n_events_valid,
        "n_plant_events_dropped": n_events_dropped,
        "drop_reason_counts": drop_reason_counts,
        "mean_site_control_score": c_bar,
        "std_site_control_score": c_std,
        "per_site_breakdown": per_site,
        "hypothesis_check": {
            "description": (
                "H2 / RQ2: mean Site Control Score should be < 0.5 at the "
                "moment of planting, indicating T-side spatial control."
            ),
            "passes": bool(c_bar is not None and c_bar < 0.5),
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Site Control Score summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
