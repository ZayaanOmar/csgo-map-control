#!/usr/bin/env bash
# Run from the PROJECT ROOT.  Every step is skip-if-exists + schema-aware, so re-running resumes.
# Set Z_TOL to the value used in position_to_node_map.py BEFORE step 2 (default in the suite: 250).
# New dependency vs. the existing pipeline: scikit-learn  (pip install scikit-learn)
# ACTIVE_THR (active footprint) defaults to exp(-2) = 2 sigma. Cached frame_summaries / engagement tables carry an
# `active_thr` stamp and are REBUILT automatically if it changes (set ACTIVE_THR=0.0111 to reproduce the old 3-sigma value).
# N_BOOT (default 1000) sets the bootstrap resamples; the early-window and location scripts take their own --n-boot.
set -euo pipefail
S=analysis/validation

# ---- 0. ONE-TIME CHECKS (read the output before continuing) -------------------------------
# python $S/extract_supplementary_tables.py --inspect            # check the demo keys the suite relies on exist
# python $S/extract_supplementary_tables.py --check-mapper --limit 3   # must be ~100% (see Z_TOL note)

# ---- 1. extraction (per match) -------------------------------------------------------------
python $S/extract_supplementary_tables.py     # round_economy, kill_events, frame_state
python $S/build_frame_summaries.py            # frame_summaries (one row per frame)
python $S/build_engagement_features.py        # engagement_adv / _loc / _series

# ---- 2. analyses (each writes outputs/metrics/<name>/ and the results registry) -------------
python $S/analysis_kill_validity.py                     # Part 1: duel curve, kill-location contest, event study
python $S/analysis_plant_aligned_control.py             # Part 1E
python $S/analysis_metric3_stratified.py                # Part 2e  (also: reproduction check of Metric 3's d)
python $S/analysis_contention_definitions.py            # Metric 1 re-run: original vs mass-robust contested
python $S/analysis_early_window_prediction.py           # Part 2a-2c
python $S/analysis_win_probability_timeseries.py        # Part 2d  (use --frame-stride 2 if slow)
python $S/analysis_side_site_favourability.py           # Part 3a/3b (run AFTER plant_aligned)
python $S/analysis_location_importance.py               # Part 3c
