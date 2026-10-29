# Quantifying Map Control in CS:GO with Graph-Based Influence Fields

Code, static artefacts and dataset-level results for the honours project *Formalising Map Control as a
Continuous Spatial Quantity in Counter-Strike: Global Offensive*, Muhammed Zayaan Omar, University of the
Witwatersrand, 2026. Supervisors: Dr. Branden Ingram and Dr. Pravesh Ranchod.

Map control is modelled as a continuous, time-varying field over the de_mirage navigation mesh: each player
exerts Gaussian influence over geodesic (walking) distance, team influences are combined through a sigmoid
into a per-node control value in (0, 1), and round-level metrics are derived from the resulting fields.
Evaluation uses 278 professional de_mirage matches (7,393 rounds) from the ESTA dataset.

> The state of the repository that matches the submitted report is tagged **`v1.0-submission`**:
> <https://github.com/ZayaanOmar/csgo-map-control/tree/v1.0-submission>

---

## Repository layout

| Path | Contents |
|---|---|
| `pipeline/` | Nav-graph patching, APSP matrix, bombsite nodes, area-name assignment, demo filtering, position→node mapping, influence fields, team aggregation, Voronoi baseline |
| `metrics/` | Metrics 1–3 (contention rate, site control score, control margin CT and T side) and their figure scripts |
| `analysis/validation/` | Validation suite: kill validity, plant-aligned control, early-window prediction, win-probability time series, location-aware predictors (see `analysis/validation/README_validation_suite.md`) |
| `analysis/team/` | Team site preference, team control signatures, eco vs full-buy positioning, position heatmaps |
| `visualisation/` | Control/Voronoi heatmap animations, nav-mesh figures |
| `verification/` | Checks on the patched graph, APSP matrix, demo filtering, position mapping and `z_tol` sweep |
| `data/` | Static artefacts (patched graph and its bridge log, APSP matrix, area assignment, bombsite nodes, team aliases) and the demo manifest. Raw demos are **not** included; see `data/README.md` |
| `outputs/` | Dataset-level results: summary JSON/CSV tables and figures. Per-match outputs are in the release archive below |
| `docs/` | Development notes, nav-mesh findings, verification logs |

## Setup

Tested with Python 3.11 on Ubuntu 22.04.5 LTS.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt     # pins awpy==1.3.1 (CS:GO); awpy 2.x is NOT compatible
```

## Data

See [`data/README.md`](data/README.md) for obtaining ESTA and reproducing the 278-demo de_mirage subset
(checked against `data/filtered_demos_manifest.sha256`).

## Reproducing the results

Run everything from the repository root, in this order. Every script accepts `--help`.

```bash
# 1. Static graph artefacts (already committed in data/; only needed to rebuild them)
python pipeline/patch_de_mirage_graph_v3.py
python pipeline/all_pairs_shortest_path_gen.py
python pipeline/get_bombsite_nodes.py
python pipeline/area_assignment.py --patched-graph data/de_mirage_patched_graph_v3.pkl

# 2. Per-match pipeline
python pipeline/filter_demos.py
python pipeline/position_to_node_map.py
python pipeline/influence_function.py
python pipeline/team_agg.py
python pipeline/voronoi_baseline.py

# 3. Metrics 1-3 (Table I) and their figures
python metrics/metric1_contention_rate.py        && python metrics/visualize_metric1_contention_rate.py
python metrics/metric2_site_control_score.py     && python metrics/visualize_metric2_site_control_score.py
python metrics/metric3_control_margin.py         && python metrics/visualize_metric3_control_margin.py
python metrics/metric3_control_margin_t_side.py  && python metrics/visualize_metric3_control_margin_t_side.py
python metrics/metric3_control_margin_t_side_formal_tests.py

# 4. Validation suite: supplementary tables, kill validity, plant-aligned control (Fig. 3),
#    head-count strata (Table II), early-window and win-probability models
#    Set Z_TOL to the value used in pipeline/position_to_node_map.py first (see the script header).
bash run_validation_suite.sh

# 5. Location-aware predictors (Fig. 4)
S=analysis/validation
python $S/build_window_features.py
python $S/make_folds.py
python $S/analysis_location_predictor.py --presence
python $S/analysis_location_overlap.py
python $S/analysis_location_site.py --presence
python $S/analysis_location_site_effects.py --presence
python $S/build_site_timecourse_features.py
python $S/analysis_location_site_timecourse.py --presence
python $S/analysis_location_site_team.py --presence
#    Sensitivity variants (output folder names are added automatically where noted)
python $S/analysis_location_predictor.py --grouping teampair --presence         # -> location_predictor_teampair
python $S/analysis_location_predictor.py --econ-band 0.15 --no-pistol --presence  # -> location_predictor_econband0.15_nopistol
python $S/build_window_features.py --cap 115 --out-dir outputs/predictor_nocap
python $S/analysis_location_predictor.py --feat-dir outputs/predictor_nocap --folds outputs/predictor/folds.csv \
    --out-dir outputs/metrics/location_predictor_nocap
python $S/build_window_features.py --cap 60 --min-len 10 --out-dir outputs/predictor_cap60_min10
python $S/analysis_location_predictor.py --feat-dir outputs/predictor_cap60_min10 --folds outputs/predictor/folds.csv \
    --out-dir outputs/metrics/location_predictor_cap60_min10
# folds.csv covers every round, so it is shared by all window settings (it was copied into predictor_nocap originally).
# Earliest sensitivity run, min-len 5 s (originally run in place over outputs/predictor, then rebuilt at 12 s; results overwritten):
# python $S/build_window_features.py --min-len 5 --out-dir outputs/predictor_minlen5
# python $S/analysis_location_predictor.py --feat-dir outputs/predictor_minlen5 --folds outputs/predictor/folds.csv \
#     --grouping teampair --presence --out-dir outputs/metrics/location_predictor_minlen5
python $S/build_site_timecourse_features.py --t0 0 --caps 2,3,4,5,6,8,10 --out-dir outputs/predictor_timecourse_t0
python $S/analysis_location_site_timecourse.py --presence --tc-dir outputs/predictor_timecourse_t0 --out-dir outputs/metrics/location_site_timecourse_t0
python $S/analysis_location_site_team.py --tc-dir outputs/predictor_timecourse_t0 --cap 2 --presence   # -> location_site_team_cap2

# 6. Team analyses (Table III, Fig. 5)
python analysis/team/build_team_aliases.py          # data/team_aliases.json is already committed
python analysis/team/build_team_signatures.py          # optional smoke test first: add --limit 2
python analysis/team/extract_round_economy.py
python analysis/team/analysis_team_site_preference.py
python analysis/team/analysis_team_signatures.py
python analysis/team/analysis_eco_vs_full.py
python analysis/team/plot_position_heatmaps.py --team "G2 Esports" --times 60      # Fig. 5 (60 s after freeze time); add --labels for area names, --list-teams to list teams

# 7. Remaining figures
python pipeline/plot_area_names.py --assignment data/de_mirage_area_assignment.csv --light    # Fig. 1 (--labels optionally adds area names)
python metrics/visualize_active_footprint.py \
    --match x_kom_AGO_VS_Natus_Vincere_667970ee_833b_491f_b8c7_b91bcb8f2c04 --round 21 --frame 67   # Fig. 2(b); default epsilon is exp(-2)
python visualisation/generate_heatmaps.py x_kom_AGO_VS_Natus_Vincere_667970ee_833b_491f_b8c7_b91bcb8f2c04  # control-field GIFs; Fig. 2(a) is frame 67 of round 21
python visualisation/plot_patched_nav_mesh.py                                                  # nav mesh
```

The scripts in `verification/` are independent checks of the graph patch, APSP matrix, demo filtering,
position mapping and `z_tol` sweep; their logs from the original runs are in `docs/verification_logs/`.
`verify_positions_projection_map.py` and `verify_ztol_sweep.py` checked intermediate tables from an earlier
development stage (`player_positions_projected.csv`, `player_df_all_rounds_fca_baseline.csv`) that are not part
of this repository; they are kept as a record and their results are in the logs.

## Key parameters

| Parameter | Value | Meaning |
|---|---|---|
| σ | 500 game units | Gaussian influence decay over geodesic distance |
| K | 1.0 | Sigmoid steepness for `control_val = expit(K·(ct_mu − t_mu))` |
| Projection clamp | 350 units | Projected tile replaced by current tile beyond this graph distance |
| Player-to-tile height tolerance | 250 units | Nearest of the 5 closest tile centroids within this height; else nearest overall |
| Projection horizon Δt | 1 s | Position projected ahead along the player's velocity |
| Graph repair gap | 2 units | Disconnected tiles linked to tiles whose footprint lies within this distance |
| Contested band | [0.4, 0.6] | Metric 1 contention definition |
| Active footprint ε | exp(−2) ≈ 0.135 | Threshold on `ct_mu + t_mu` |
| Frame spacing | 64 ticks (≈0.5 s) | ESTA parse rate |
| Early window (RQ5) | 5 s after freeze time to first kill, plant or 30 s; min 12 s | Defaults of `build_window_features.py` |
| Heatmap smoothing | diffusion, bandwidth 100 units, cut 1.6 | Defaults of `plot_position_heatmaps.py` (`--bandwidth`, `--cut`) |

## Where each result in the report comes from

| Report item | Script | Output |
|---|---|---|
| Fig. 1, the 27 named areas of de_mirage | `pipeline/plot_area_names.py` | `outputs/area_name_map/de_mirage_area_names_assigned.png` |
| Fig. 2, one frame (round 21, frame 67 of x-kom vs Natus Vincere): (a) control field, (b) with the active footprint | (a) `visualisation/generate_heatmaps.py`, (b) `metrics/visualize_active_footprint.py` | (b) `outputs/active_footprint_viz/x_kom_AGO_VS_Natus_Vincere_667970ee_833b_491f_b8c7_b91bcb8f2c04/round_21/frame_0067.png` (in the release archive) |
| Fig. 3, T control of the planted site, plant-aligned | `analysis/validation/analysis_plant_aligned_control.py` | `outputs/metrics/plant_aligned/figures/plant_aligned_site_control.png` |
| Fig. 4, effect of early-round T control by area on B-plant log-odds | `analysis/validation/analysis_location_site_effects.py` | `outputs/metrics/location_site/effects/figures/effect_map.png` |
| Fig. 5, G2 Esports positions at 60 s, eco vs full buy | `analysis/team/plot_position_heatmaps.py` | `outputs/analysis/figures/position_heatmaps/G2_Esports/G2_Esports_{CT,T}_{eco,full}_t60s.png` |
| Table I, the three proposed metrics | `metrics/metric1_contention_rate.py`, `metrics/metric2_site_control_score.py`, `metrics/metric3_control_margin.py`, `metrics/metric3_control_margin_t_side.py` | `_summary.json` in `outputs/metrics/{contention_rate,site_control_score,control_margin,control_margin_t_side}/` |
| Table II, active-footprint margin by head-count stratum | `analysis/validation/analysis_metric3_stratified.py` | `outputs/metrics/metric3_stratified/metric3_stratified.csv` |
| Table III, eco vs full-buy positioning (Jensen-Shannon distance and permutation null) | `analysis/team/analysis_eco_vs_full.py` | `outputs/analysis/eco_vs_full_omnibus.csv` |
| Patched nav mesh (not in the report) | `visualisation/plot_patched_nav_mesh.py` | `outputs/nav_mesh/mirage_navmesh_patched.png` |

### Results quoted in the text

| Result in the report | Script | Output |
|---|---|---|
| Control at the plant: adjusted association with T winning, ΔAUC (Section V-B) | `analysis/validation/analysis_plant_aligned_control.py` | `outputs/metrics/plant_aligned/` |
| Pre-engagement prediction of the winner and the fixed-window comparison (Section V-C) | `analysis/validation/analysis_early_window_prediction.py` | `outputs/metrics/early_window/` |
| Area-level and node-level prediction of the winner; balanced-economy, team-pair and window sensitivities (Section V-D) | `analysis/validation/analysis_location_predictor.py` (variants in step 5) | `outputs/metrics/location_predictor*/` |
| Planted-site prediction; per-area effects (Fig. 4) | `analysis/validation/analysis_location_site.py`, `analysis_location_site_effects.py` | `outputs/metrics/location_site/` |
| Site readable from the first 2 s (AUC 0.726) | `analysis/validation/analysis_location_site_timecourse.py` (t0 variant in step 5) | `outputs/metrics/location_site_timecourse_t0/` |
| Team-habit control for site prediction | `analysis/validation/analysis_location_site_team.py` | `outputs/metrics/location_site_team*/` |
| Eco vs full-buy area shifts, team counts, field vs occupancy correlation | `analysis/team/analysis_eco_vs_full.py` | `outputs/analysis/eco_vs_full_area_shifts.csv`, `eco_vs_full_omnibus.csv` |
| Team identification accuracy (field and occupancy signatures) | `analysis/team/analysis_team_signatures.py` | `outputs/analysis/signature_identification.csv` |

All headline estimates from the validation suite, with confidence intervals, are also collected in
`outputs/metrics/results_registry.json`.

## Full outputs archive

Per-match tables (frame summaries, engagement features, kills, round metadata, per-round metric values,
Voronoi metrics, predictor feature files) and the per-frame active-footprint images are published as a release
archive:
<https://github.com/ZayaanOmar/csgo-map-control/releases/download/v1.0-submission/csgo-map-control-outputs.tar.xz>

Extract it into the repository root to skip re-running the per-match steps:

```bash
tar -xJf csgo-map-control-outputs.tar.xz
```

The per-match influence fields, control fields, Voronoi fields and position maps (about 300 GB in total) are
not published. They are regenerated from the ESTA demos by `pipeline/`.

## Acknowledgements and citation

- Data: ESTA, the Esports Trajectory and Action dataset (Xenopoulos et al.), <https://github.com/pnxenopoulos/esta>.
- Demo parsing and nav-mesh data: awpy 1.3.1 (Xenopoulos et al.), <https://github.com/pnxenopoulos/awpy>.

Full references are in the report's bibliography.
