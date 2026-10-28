import os
import glob
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.special import expit

# --/ Path consts
INFLUENCE_FIELDS_PATH = "outputs/influence_fields"
CONTROL_FIELD_PATH = "outputs/control_fields"

# --/ Sigmoid steepness constant
K = 1.0

# --/ control field computation
def load_influence_columns(parquet_path: str) -> pd.DataFrame:
    """Read only the four columns needed for the control field computation using PyArrow projection
    Returns a DataFrame with columns:
        round, frame_idx, side, area_id, mu
    """
    dataset = ds.dataset(parquet_path, format="parquet")
    table = dataset.to_table(columns=["round", "frame_idx", "side", "area_id", "mu"])
    return table.to_pandas()

def team_agg_comp(df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level control field from per-player influence values.
    
    For each (round, frame_idx, area_id) group:\n
        1. Sum mu across all CT players
        2. Sum mu across all T players
        3. control_val = ct_sum - t_sum in range [0, 1]

    ct_mu/t_mu (the raw, pre-sigmoid summed influence) are kept in the
    output rather than dropped after computing control_val. Downstream
    metrics (metric3_control_margin.py's active-footprint filter) need to
    know whether (ct_mu + t_mu) at a node/frame is non-negligible BEFORE
    the sigmoid squashes everything toward [0, 1] — a node far from every
    player and a node contested by two nearby equal-strength players can
    both land near control_val = 0.5, and only the raw ct_mu/t_mu values
    can tell those two cases apart.

    Returns:
        pd.DataFrame("round", "frame_idx", "area_id", "ct_mu", "t_mu", "control_val")
    """
    # sum mu per (round, frame, side, area_id)
    grouped = (df.groupby(["round", "frame_idx", "side", "area_id"], sort=False)["mu"].sum().reset_index())

    # Pivot to split CT and T columns
    split = grouped.pivot(index=["round", "frame_idx", "area_id"], columns="side", values="mu").reset_index()

    # rename pivot columns cleanly
    split.columns.name = None
    for col in ["CT", "T"]:
        if col not in split.columns:
            split[col] = 0.0
    split[["CT", "T"]] = split[["CT", "T"]].fillna(0.0)

    # sigmoid normalised control val
    split["control_val"] = expit(K * (split["CT"] - split["T"])).astype(np.float32)

    return split.rename(columns={"CT": "ct_mu", "T": "t_mu"})

# Main loop over entire dataset
def main() -> None:
    os.makedirs(CONTROL_FIELD_PATH, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(INFLUENCE_FIELDS_PATH, "*.parquet")))
    print(f"Found {len(parquet_files)} influence parquets\n")

    for parquet_file in parquet_files:
        mname = os.path.splitext(os.path.basename(parquet_file))[0]
        output_path = os.path.join(CONTROL_FIELD_PATH, f"{mname}.csv")

        # skip check — schema-aware: a cached CSV written before ct_mu/t_mu
        # were added to the output would otherwise be trusted as-is forever,
        # permanently blocking the active-footprint metric for that match
        if os.path.exists(output_path):
            cached_cols = pd.read_csv(output_path, nrows=0).columns
            if {"ct_mu", "t_mu"}.issubset(cached_cols):
                print(f"Skipping {mname} - already exists with ct_mu/t_mu")
                continue
            print(f"Recomputing {mname} - cached file predates ct_mu/t_mu columns")

        try:
            print(f"Processing demo {mname}")

            df = load_influence_columns(parquet_file)
            res = team_agg_comp(df)
            del df

            res.to_csv(output_path, index=False)
            print(f" Control field saved to {output_path} ({len(res):,} rows)")

            del res

        except Exception as e:
            print(f"\n FAILED: {mname} - {e}\n")
            if os.path.exists(output_path):
                os.remove(output_path)
            continue

    print("\nTeam Aggregation complete")

if __name__ == "__main__":
    main()