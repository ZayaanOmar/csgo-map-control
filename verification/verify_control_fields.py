"""
verify_control_fields.py
------------------------
Verification script for the control field CSVs produced by team_agg.py.

Run from the project root:
    python verification/verify_control_fields.py

Optional: pass one or more CSV paths to verify specific files:
    python verification/verify_control_fields.py \
        outputs/control_fields/FaZe_VS_BIG_4bab9a63.csv

Memory design
-------------
The control field CSV (~44 MB for a full match) is held in memory throughout.
The paired influence parquet is read ONE ROUND AT A TIME using PyArrow filter
pushdown. Peak memory during cross-validation is:
    - CSV in memory:          ~50 MB
    - One round of parquet:   ~50-150 MB (1-2M rows at 5 cols)
    - Aggregated round result: negligible (~903 rows)
Total peak well under 500 MB per file, safe on a 10 GB VM.

Checks performed (15 total)
----------------------------
Structure:
    1.  Required columns present with correct names
    2.  No null / NaN values in any column
    3.  No infinite values in control_val
    4.  Correct dtypes (round/frame_idx/area_id integer, control_val float)
Geometry / completeness:
    5.  Every (round, frame_idx) has exactly 903 rows
    6.  area_id values are all valid patched-graph nodes
    7.  frame_idx is contiguous (0-based, no gaps) within each round
    8.  round numbers are positive integers
Value ranges:
    9.  control_val strictly in (0, 1) — sigmoid output bounds
    10. control_val distribution is not suspiciously degenerate
Cross-validation against influence parquet (round-by-round, low memory):
    11. Paired influence parquet exists
    12. Round sets match between CSV and parquet
    13. For each round: (round, frame_idx, area_id) key sets match
    14. For each round: control_val recomputes correctly end-to-end
        (groupby-sum CT/T mu → sigmoid(K*(ct-t))) within float tolerance
Global coverage:
    15. Every influence parquet has a matching control field CSV
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.special import expit

# ---------------------------------------------------------------------------
# Configuration — adjust paths if your layout differs
# ---------------------------------------------------------------------------
PROJECT_ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONTROL_FIELDS_DIR   = os.path.join(PROJECT_ROOT, "outputs", "control_fields")
INFLUENCE_FIELDS_DIR = os.path.join(PROJECT_ROOT, "outputs", "influence_fields")
PATCHED_GRAPH_PATH   = os.path.join(PROJECT_ROOT, "data", "de_mirage_patched_graph_v3.pkl")

# Must match K in team_agg.py
K = 1.0

# Expected nav node count
EXPECTED_N_NODES = 903

# Columns team_agg.py actually writes (after dropping CT/T intermediates)
EXPECTED_COLUMNS = ["round", "frame_idx", "area_id", "control_val"]

# Tolerance for floating-point recomputation comparison.
# float32 sigmoid output stored as float64 in CSV — some precision is lost
# at the CSV round-trip. 1e-5 is tight enough to catch formula errors while
# tolerating the float32→float64 CSV representation difference.
FLOAT_TOL = 1e-5

# Width used for check label alignment in output
LABEL_WIDTH = 55

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _header(text: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {text}")
    print(f"{'=' * 72}")

def _pass(label: str, detail: str = "") -> None:
    print(f"  PASS  {label:<{LABEL_WIDTH}}{detail}")

def _warn(label: str, detail: str = "") -> None:
    print(f"  WARN  {label:<{LABEL_WIDTH}}{detail}")

def _fail(label: str, detail: str = "") -> None:
    print(f"  FAIL  {label:<{LABEL_WIDTH}}{detail}")

# ---------------------------------------------------------------------------
# Load the patched graph node set (once, shared across all files)
# ---------------------------------------------------------------------------

def load_valid_area_ids() -> set:
    if not os.path.exists(PATCHED_GRAPH_PATH):
        print(f"  [NOTE] Patched graph not found at:\n         {PATCHED_GRAPH_PATH}")
        print(f"         Check 6 (area_id validity against graph) will be skipped.")
        return set()
    import pickle
    with open(PATCHED_GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    return set(G.nodes())

# ---------------------------------------------------------------------------
# Per-file verification
# ---------------------------------------------------------------------------

def verify_one(csv_path: str, valid_area_ids: set, results: dict) -> None:
    match_name = os.path.splitext(os.path.basename(csv_path))[0]
    _header(f"Verifying: {match_name}")

    passed = warned = failed = 0

    # ------------------------------------------------------------------
    # Load the control field CSV — held in memory for all checks
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        _fail("LOAD", f"Could not read CSV: {exc}")
        results[match_name] = {"passed": 0, "warned": 0, "failed": 1}
        return

    n_rows = len(df)
    print(f"\n  Rows loaded:  {n_rows:,}")
    print(f"  Columns:      {list(df.columns)}")

    # ==================================================================
    # CHECK 1 — Required columns present with correct names
    # ==================================================================
    actual_cols   = list(df.columns)
    missing_cols  = [c for c in EXPECTED_COLUMNS if c not in actual_cols]
    extra_cols    = [c for c in actual_cols if c not in EXPECTED_COLUMNS]
    if missing_cols:
        _fail("1 — Required columns", f"Missing: {missing_cols}")
        failed += 1
    elif extra_cols:
        _warn("1 — Required columns", f"Expected columns present; extra cols: {extra_cols}")
        warned += 1
    else:
        _pass("1 — Required columns", "All 4 expected columns present, no extras")
        passed += 1

    # Bail if core columns are absent — later checks would crash
    if missing_cols:
        results[match_name] = {"passed": passed, "warned": warned, "failed": failed}
        return

    # ==================================================================
    # CHECK 2 — No null / NaN values
    # ==================================================================
    null_totals = df.isnull().sum()
    n_nulls = null_totals.sum()
    if n_nulls > 0:
        detail = "  ".join(f"{c}:{n}" for c, n in null_totals.items() if n > 0)
        _fail("2 — No null values", f"{n_nulls} nulls: {detail}")
        failed += 1
    else:
        _pass("2 — No null values", "Zero nulls across all columns")
        passed += 1

    # ==================================================================
    # CHECK 3 — No infinite values in control_val
    # ==================================================================
    n_inf = np.isinf(df["control_val"]).sum()
    if n_inf > 0:
        _fail("3 — No infinite values", f"{n_inf} inf values in control_val")
        failed += 1
    else:
        _pass("3 — No infinite values", "Zero infinities in control_val")
        passed += 1

    # ==================================================================
    # CHECK 4 — Dtypes
    # Pandas reads int64 from CSV for integer columns — that is fine.
    # control_val should be float.
    # ==================================================================
    dtype_issues = []
    for col in ["round", "frame_idx", "area_id"]:
        if not pd.api.types.is_integer_dtype(df[col].dtype):
            dtype_issues.append(f"{col}: {df[col].dtype} (expected integer)")
    if not pd.api.types.is_float_dtype(df["control_val"].dtype):
        dtype_issues.append(f"control_val: {df['control_val'].dtype} (expected float)")
    if dtype_issues:
        _fail("4 — Column dtypes", "; ".join(dtype_issues))
        failed += 1
    else:
        _pass("4 — Column dtypes",
              f"round/frame_idx/area_id={df['round'].dtype}, "
              f"control_val={df['control_val'].dtype}")
        passed += 1

    # ==================================================================
    # CHECK 5 — Every (round, frame_idx) has exactly 903 rows
    # ==================================================================
    frame_counts  = df.groupby(["round", "frame_idx"], sort=False).size()
    wrong_frames  = frame_counts[frame_counts != EXPECTED_N_NODES]
    if len(wrong_frames) > 0:
        examples = wrong_frames.head(5).to_string()
        _fail(
            "5 — 903 rows per (round, frame_idx)",
            f"{len(wrong_frames)} frames with wrong count (expected {EXPECTED_N_NODES}):\n"
            f"    {examples}",
        )
        failed += 1
    else:
        n_frames = len(frame_counts)
        _pass("5 — 903 rows per (round, frame_idx)",
              f"All {n_frames:,} frames have exactly 903 rows")
        passed += 1

    # ==================================================================
    # CHECK 6 — area_id values are valid patched-graph nodes
    # ==================================================================
    if valid_area_ids:
        unique_areas = set(df["area_id"].unique())
        unknown      = unique_areas - valid_area_ids
        if unknown:
            _fail("6 — area_id validity",
                  f"{len(unknown)} unknown area IDs: {sorted(unknown)[:10]}"
                  + (" ..." if len(unknown) > 10 else ""))
            failed += 1
        else:
            _pass("6 — area_id validity",
                  f"{len(unique_areas)} unique area IDs — all valid patched-graph nodes")
            passed += 1
    else:
        _warn("6 — area_id validity", "SKIPPED — patched graph not found")
        warned += 1

    # ==================================================================
    # CHECK 7 — frame_idx is contiguous and 0-based within each round
    # ==================================================================
    bad_rounds = []
    for rnd, rnd_df in df.groupby("round", sort=True):
        frames = sorted(rnd_df["frame_idx"].unique())
        expected = list(range(0, len(frames)))  # must be 0, 1, 2, ...
        if frames != expected:
            bad_rounds.append(
                f"round {rnd}: starts={frames[0]}, "
                f"gaps={[i for i in expected if i not in frames][:5]}"
            )
    if bad_rounds:
        _fail("7 — frame_idx contiguous (0-based) per round",
              f"{len(bad_rounds)} problematic rounds: " + "; ".join(bad_rounds[:3]))
        failed += 1
    else:
        n_rounds = df["round"].nunique()
        _pass("7 — frame_idx contiguous (0-based) per round",
              f"All {n_rounds} rounds have contiguous 0-based frame_idx")
        passed += 1

    # ==================================================================
    # CHECK 8 — round numbers are positive integers
    # ==================================================================
    round_vals    = df["round"].unique()
    non_positive  = round_vals[round_vals <= 0]
    if len(non_positive) > 0:
        _fail("8 — Round numbers positive",
              f"Non-positive round values found: {sorted(non_positive)[:5]}")
        failed += 1
    else:
        _pass("8 — Round numbers positive",
              f"{df['round'].nunique()} rounds, "
              f"range [{round_vals.min()}, {round_vals.max()}]")
        passed += 1

    # ==================================================================
    # CHECK 9 — control_val strictly in (0, 1)
    # Sigmoid output for any finite input is always in (0, 1) open.
    # A value at exactly 0 or 1 means something went very wrong.
    # ==================================================================
    n_le_zero = (df["control_val"] <= 0.0).sum()
    n_ge_one  = (df["control_val"] >= 1.0).sum()
    if n_le_zero > 0 or n_ge_one > 0:
        _fail(
            "9 — control_val in (0, 1)",
            f"{n_le_zero} rows <= 0,  {n_ge_one} rows >= 1  "
            f"(range: [{df['control_val'].min():.8f}, {df['control_val'].max():.8f}])",
        )
        failed += 1
    else:
        _pass("9 — control_val in (0, 1)",
              f"range [{df['control_val'].min():.6f}, {df['control_val'].max():.6f}]")
        passed += 1

    # ==================================================================
    # CHECK 10 — control_val distribution is not suspiciously degenerate
    #
    # If almost every value is 0.5 the sigmoid received near-zero inputs,
    # which would mean mu values are all near zero — likely an inf-clamp bug
    # or wrong sigma. Flag if >95% of values are within 0.01 of 0.5.
    # ==================================================================
    near_half_frac = (np.abs(df["control_val"] - 0.5) < 0.01).mean()
    if near_half_frac > 0.95:
        _warn(
            "10 — control_val distribution",
            f"{near_half_frac:.1%} of values within 0.01 of 0.5 — "
            "suspiciously flat. Check sigma and mu magnitudes.",
        )
        warned += 1
    else:
        # Also report the fraction of CT-dominant (>0.5) vs T-dominant (<0.5) frames
        # as a quick sanity check on overall map control direction
        frac_ct = (df["control_val"] > 0.5).mean()
        _pass(
            "10 — control_val distribution",
            f"{near_half_frac:.1%} near 0.5;  "
            f"CT-dominant: {frac_ct:.1%},  T-dominant: {1-frac_ct:.1%}",
        )
        passed += 1

    # ==================================================================
    # CHECKS 11–14 — Cross-validation against the influence parquet
    # Read ONE ROUND AT A TIME to stay within memory budget.
    # ==================================================================
    parquet_path = os.path.join(INFLUENCE_FIELDS_DIR, f"{match_name}.parquet")

    # CHECK 11 — Paired influence parquet exists
    if not os.path.exists(parquet_path):
        _fail("11 — Paired influence parquet exists", f"Not found: {parquet_path}")
        failed += 1
        _warn("12 — Round sets match",         "SKIPPED — no parquet")
        _warn("13 — Per-round key sets match",  "SKIPPED — no parquet")
        _warn("14 — control_val recomputes",    "SKIPPED — no parquet")
        warned += 3
    else:
        _pass("11 — Paired influence parquet exists", os.path.basename(parquet_path))
        passed += 1

        try:
            dataset = ds.dataset(parquet_path, format="parquet")

            # -- CHECK 12 — Round sets match ---------------------------------
            # Read only the 'round' column to get the set of rounds in the
            # parquet without loading any mu values yet.
            round_table   = dataset.to_table(columns=["round"])
            parquet_rounds = set(round_table.column("round").to_pylist())
            del round_table
            csv_rounds    = set(df["round"].unique().tolist())

            only_parquet  = parquet_rounds - csv_rounds
            only_csv      = csv_rounds - parquet_rounds
            if only_parquet or only_csv:
                _fail(
                    "12 — Round sets match",
                    f"In parquet only: {sorted(only_parquet)[:5]}  "
                    f"In CSV only: {sorted(only_csv)[:5]}",
                )
                failed += 1
            else:
                _pass("12 — Round sets match",
                      f"{len(csv_rounds)} rounds agree between CSV and parquet")
                passed += 1

            # -- CHECKS 13 & 14 — Per-round key sets and value recomputation --
            # Process one round at a time: filter the parquet, aggregate,
            # then compare against the corresponding rows in the CSV.
            rounds_to_check   = sorted(csv_rounds & parquet_rounds)
            key_mismatches    = []   # round numbers with key-set problems
            value_mismatches  = []   # (round, max_diff, n_bad) tuples
            total_recomputed  = 0

            for rnd in rounds_to_check:
                # Read only this round's rows from the parquet
                rnd_table = dataset.to_table(
                    columns=["round", "frame_idx", "side", "area_id", "mu"],
                    filter=ds.field("round") == rnd,
                )
                rnd_inf = rnd_table.to_pandas()
                del rnd_table  # free arrow memory immediately

                # Aggregate: sum mu per (frame_idx, side, area_id)
                # (round is constant within this chunk so we drop it from groupby)
                grouped = (
                    rnd_inf
                    .groupby(["frame_idx", "side", "area_id"], sort=False)["mu"]
                    .sum()
                    .reset_index()
                )
                del rnd_inf

                # Pivot to CT / T columns
                pivot = (
                    grouped
                    .pivot(index=["frame_idx", "area_id"], columns="side", values="mu")
                    .reset_index()
                )
                del grouped
                pivot.columns.name = None
                for col in ["CT", "T"]:
                    if col not in pivot.columns:
                        pivot[col] = 0.0
                pivot[["CT", "T"]] = pivot[["CT", "T"]].fillna(0.0)

                # Recompute control_val using the same formula as team_agg.py
                pivot["control_val_ref"] = expit(K * (pivot["CT"] - pivot["T"])).astype(np.float32)
                pivot.drop(columns=["CT", "T"], inplace=True)

                # -- Check 13: key set ----------------------------------------
                ref_keys = set(zip(pivot["frame_idx"], pivot["area_id"]))
                csv_rnd  = df[df["round"] == rnd]
                csv_keys = set(zip(csv_rnd["frame_idx"], csv_rnd["area_id"]))
                if ref_keys != csv_keys:
                    key_mismatches.append(rnd)

                # -- Check 14: value recomputation ----------------------------
                merged = csv_rnd[["frame_idx", "area_id", "control_val"]].merge(
                    pivot[["frame_idx", "area_id", "control_val_ref"]],
                    on=["frame_idx", "area_id"],
                    how="inner",
                )
                if len(merged) > 0:
                    diff      = np.abs(merged["control_val"].values
                                       - merged["control_val_ref"].values)
                    max_diff  = diff.max()
                    n_bad     = (diff > FLOAT_TOL).sum()
                    total_recomputed += len(merged)
                    if n_bad > 0:
                        value_mismatches.append((rnd, max_diff, n_bad))

                del pivot, merged

            # Report check 13
            if key_mismatches:
                _fail(
                    "13 — Per-round (frame_idx, area_id) key sets match",
                    f"Key mismatches in rounds: {key_mismatches[:5]}"
                    + (" ..." if len(key_mismatches) > 5 else ""),
                )
                failed += 1
            else:
                _pass(
                    "13 — Per-round (frame_idx, area_id) key sets match",
                    f"All {len(rounds_to_check)} rounds agree",
                )
                passed += 1

            # Report check 14
            if value_mismatches:
                worst_rnd, worst_diff, worst_n = max(value_mismatches, key=lambda x: x[1])
                _fail(
                    "14 — control_val recomputes end-to-end",
                    f"{len(value_mismatches)} rounds with mismatches > {FLOAT_TOL:.0e}; "
                    f"worst: round {worst_rnd}, max_diff={worst_diff:.4e}, "
                    f"{worst_n} rows affected",
                )
                failed += 1
            else:
                _pass(
                    "14 — control_val recomputes end-to-end",
                    f"{total_recomputed:,} rows cross-validated across "
                    f"{len(rounds_to_check)} rounds; "
                    f"all within tol {FLOAT_TOL:.0e}",
                )
                passed += 1

        except Exception as exc:
            _fail("PARQUET_READ", f"Error reading parquet: {exc}")
            failed += 1

    # ------------------------------------------------------------------
    # Per-file summary
    # ------------------------------------------------------------------
    total = passed + warned + failed
    print(
        f"\n  ── {match_name}: "
        f"{passed}/{total} passed · {warned} warned · {failed} failed ──"
    )
    results[match_name] = {"passed": passed, "warned": warned, "failed": failed}


# ---------------------------------------------------------------------------
# Check 15 — Global coverage (run once in batch mode)
# ---------------------------------------------------------------------------

def check_coverage() -> tuple:
    _header("Check 15 — Global coverage (influence ↔ control_fields pairing)")
    parquet_stems = {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(INFLUENCE_FIELDS_DIR, "*.parquet"))
    }
    csv_stems = {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(CONTROL_FIELDS_DIR, "*.csv"))
    }

    if not parquet_stems:
        _warn("15 — Coverage", f"No parquets found in {INFLUENCE_FIELDS_DIR}")
        return 0, 0

    missing_csvs = parquet_stems - csv_stems
    extra_csvs   = csv_stems - parquet_stems

    if missing_csvs:
        _fail(
            "15 — Coverage",
            f"{len(missing_csvs)} parquets with no matching CSV:\n    "
            + "\n    ".join(sorted(missing_csvs)[:10])
            + (" ..." if len(missing_csvs) > 10 else ""),
        )
        return 0, 1

    if extra_csvs:
        _warn(
            "15 — Coverage",
            f"All {len(parquet_stems)} parquets matched. "
            f"{len(extra_csvs)} CSVs have no parquet (may be stale).",
        )
        return 1, 0

    _pass(
        "15 — Coverage",
        f"All {len(parquet_stems)} influence parquets have a matching control field CSV",
    )
    return 1, 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        target_csvs      = [os.path.abspath(a) for a in sys.argv[1:]]
        run_coverage     = False
        missing          = [p for p in target_csvs if not os.path.exists(p)]
        if missing:
            for p in missing:
                print(f"[ERROR] File not found: {p}")
            sys.exit(1)
    else:
        target_csvs  = sorted(glob.glob(os.path.join(CONTROL_FIELDS_DIR, "*.csv")))
        run_coverage = True

    if not target_csvs:
        print(f"[ERROR] No control field CSVs found in {CONTROL_FIELDS_DIR}")
        sys.exit(1)

    print(f"\nControl Field Verification")
    print(f"  Files to verify:      {len(target_csvs)}")
    print(f"  K (sigmoid steepness):{K}")
    print(f"  Float tolerance:      {FLOAT_TOL}")
    print(f"  Patched graph:        {PATCHED_GRAPH_PATH}")

    valid_area_ids = load_valid_area_ids()
    if valid_area_ids:
        print(f"  Valid area IDs:       {len(valid_area_ids)}")

    results = {}

    for csv_path in target_csvs:
        verify_one(csv_path, valid_area_ids, results)

    cov_passed = cov_failed = 0
    if run_coverage:
        cov_passed, cov_failed = check_coverage()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    _header("OVERALL SUMMARY")
    total_passed = sum(r["passed"] for r in results.values()) + cov_passed
    total_warned = sum(r["warned"] for r in results.values())
    total_failed = sum(r["failed"] for r in results.values()) + cov_failed

    print(f"\n  Files verified:  {len(results)}")
    print(f"  Checks passed:   {total_passed}")
    print(f"  Checks warned:   {total_warned}")
    print(f"  Checks failed:   {total_failed}")

    if total_failed > 0:
        print(f"\n  Files with failures:")
        for name, r in sorted(results.items()):
            if r["failed"] > 0:
                print(f"    {name}  ({r['failed']} failed)")
        sys.exit(1)
    elif total_warned > 0:
        print(f"\n  No blocking failures. Review WARN items above.")
    else:
        print(f"\n  All checks passed.")


if __name__ == "__main__":
    main()
