# Quantifying Map Control in CS:GO with Graph-Based Influence Fields

Code, static artefacts and dataset-level results for the honours project
Formalising Map Control as a Continuous Spatial Quantity in Counter-Strike: Global Offensive, Zayaan Omar, University of the Witwatersrand, 2026.
Supervisors: Dr. Branden Ingram and Dr. Pravesh Ranchod.

Map control is modelled as a continuous, time-varying field over the de_mirage
navigation mesh: each player exerts Gaussian influence over geodesic (walking)
distance, team influences are combined through a sigmoid into a per-node control
value in (0, 1), and round-level metrics are derived from the resulting fields.
Evaluation uses 278 professional de_mirage matches (7,393 rounds) from the ESTA dataset.

> The state of the repository that matches the submitted report is tagged
> **`v1.0-submission`**. https://github.com/ZayaanOmar/csgo-map-control/tree/v1.0-submission

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
| `data/` | Static artefacts (patched graph, APSP matrix, area assignment, bombsite nodes, team aliases) and the demo manifest. Raw demos are **not** included — see `data/README.md` |
| `outputs/` | Dataset-level results: summary JSON/CSV tables and every figure. Per-match outputs are in the full archive below |
| `docs/` | Method notes, nav-mesh findings, verification logs |

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

Run from the repository root.

```bash
# 1. Static graph artefacts (already committed in data/; only needed to rebuild)
python pipeline/patch_de_mirage_graph_v3.py
python pipeline/all_pairs_shortest_path_gen.py
python pipeline/get_bombsite_nodes.py
python pipeline/area_assignment.py

# 2. Per-match pipeline
python pipeline/filter_demos.py
python pipeline/position_to_node_map.py
python pipeline/influence_function.py
python pipeline/team_agg.py
python pipeline/voronoi_baseline.py

# 3. Metrics and figures
python metrics/metric1_contention_rate.py      && python metrics/visualize_metric1_contention_rate.py
python metrics/metric2_site_control_score.py   && python metrics/visualize_metric2_site_control_score.py
python metrics/metric3_control_margin.py       && python metrics/visualize_metric3_control_margin.py
python metrics/metric3_control_margin_t_side.py && python metrics/visualize_metric3_control_margin_t_side.py

# 4. Validation suite and team analyses
bash run_validation_suite.sh
python analysis/team/extract_round_economy.py
python analysis/team/build_team_signatures.py
python analysis/team/analysis_team_site_preference.py
python analysis/team/analysis_team_signatures.py
python analysis/team/analysis_eco_vs_full.py
```

## Key parameters

| Parameter | Value | Meaning |
|---|---|---|
| σ | 500 game units | Gaussian influence decay over geodesic distance |
| K | 1.0 | Sigmoid steepness for `control_val = expit(K·(ct_mu − t_mu))` |
| Projection clamp | 350 units | Projected tile replaced by current tile beyond this graph distance |
| Contested band | [0.4, 0.6] | Metric 1 contention definition |
| Active footprint ε | exp(−2)  | Threshold on `ct_mu + t_mu` |
| Frame spacing | 64 ticks (≈0.5 s) | ESTA parse rate |

## Where each result in the report comes from

| Report item | Script | Output |
|---|---|---|
| Fig. 1, the 27 named areas of de_mirage | `pipeline/plot_area_names.py` | `outputs/area_name_map/de_mirage_area_names_assigned.png` |
| Fig. 2, control field and active footprint for one frame | `metrics/visualize_active_footprint.py`, `visualisation/generate_heatmaps.py` | outputs/active_footprint_viz/x_kom_AGO_VS_Natus_Vincere_667970ee_833b_491f_b8c7_b91bcb8f2c04/round_21/ |
| Fig. 3, T control of the planted site, plant-aligned | `analysis/validation/analysis_plant_aligned_control.py` | `outputs/metrics/plant_aligned/figures/` |
| Fig. 4, early-round T control by area vs B-plant log-odds | `analysis/validation/analysis_location_site_effects.py` | `outputs/metrics/location_site/effects/figures/` |
| Fig. 5, G2 Esports positions, eco vs full buy | `analysis/team/plot_position_heatmaps.py` | `outputs/analysis/figures/position_heatmaps/G2_Esports/` |
| Table I, the three proposed metrics | `metrics/metric1_contention_rate.py`, `metrics/metric2_site_control_score.py`, `metrics/metric3_control_margin.py`, `metrics/metric3_control_margin_t_side.py` | `_summary.json` in `outputs/metrics/{contention_rate,site_control_score,control_margin,control_margin_t_side}/` |
| Table II, Metric 3 by head-count stratum | `analysis/validation/analysis_metric3_stratified.py` | `outputs/metrics/metric3_stratified/metric3_stratified.csv` |
| Table III, team positional signatures | `analysis/team/analysis_team_signatures.py` | `outputs/analysis/signature_identification.csv` |
| Patched nav mesh figure (supplementary) | `visualisation/plot_patched_nav_mesh.py` | `outputs/nav_mesh/mirage_navmesh_patched.png` |

All headline estimates from the validation suite, with confidence intervals, are also collected in
`outputs/metrics/results_registry.json`.

## Full outputs archive

Per-match tables (frame summaries, engagement features, kills, round metadata, per-round metric values,
Voronoi metrics, predictor feature files) are regenerable with the pipeline above and are published as a
release archive: https://github.com/ZayaanOmar/csgo-map-control/releases/download/v1.0-submission/csgo-map-control-outputs.tar.xz. Extract it into the repository root to skip
re-running the per-match steps.

The per-match influence fields, control fields, Voronoi fields and position maps (about 300 GB in total) are
not published. They are regenerated from the ESTA demos by `pipeline/`.

## Acknowledgements and citation

Data: ESTA (Xenopoulos et al.). Demo parsing: awpy 1.3.1. (Xenopoulos et al.).
