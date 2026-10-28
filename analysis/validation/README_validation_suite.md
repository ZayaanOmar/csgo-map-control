# Validation & Extension Suite

Additive to the main pipeline. Nothing here modifies or replaces a pipeline script, table, or metric.
Run from the project root (`CSGO_ROOT` env var overrides). Order and checks: `run_validation_suite.sh`.

The suite was first developed against a small synthetic project (grid nav mesh, awpy-schema demos,
self-consistent position/control tables) and then run on the full 278-demo de_mirage dataset; the results in
`outputs/metrics/` come from that real-data run. The checks listed below are the ones that must pass before the
outputs are trusted, and should be repeated if any upstream table is regenerated.

New dependency vs the existing pipeline: **scikit-learn**. (No awpy import anywhere in the suite: area names,
tile rectangles and centroids come from the patched-graph pickle's node attributes.)

## Data flow

```
data/filtered_demos/*.json.xz ──► extract_supplementary_tables.py ──► round_economy/  kill_events/  frame_state/
position_maps + control_fields + frame_state ──► build_frame_summaries.py ──► frame_summaries/ (1 row per frame, parquet)
kill_events + frame_state + position_maps ─────► build_engagement_features.py ──► engagement_adv/ _loc/ _series/
                                   analysis_*.py ──► outputs/metrics/<name>/ (+ figures/) and the results registry
```

| Script | Part | Question answered |
|---|---|---|
| `extract_supplementary_tables.py` | support | economy, kill events (+trade/chain flags, nav nodes), per-frame game state |
| `build_frame_summaries.py` | support | per-frame control aggregates, per-area control/contest/occupancy, site control, Voronoi margin |
| `build_engagement_features.py` | support | local advantage before each kill, kill-location contest vs null, kill-aligned series |
| `analysis_kill_validity.py` | 1 (C, D) | duel win-rate vs local advantage; kill-location contest; kill event study |
| `analysis_plant_aligned_control.py` | 1 (E) | site control −30…+15 s around plants, by site and outcome; Metric-2-style snapshot |
| `analysis_contention_definitions.py` | Metric 1 | contention under the original vs mass-robust definitions |
| `analysis_metric3_stratified.py` | 2e | does Metric 3's win/loss gap survive equal head-counts? (+ reproduction check) |
| `analysis_early_window_prediction.py` | 2a–2c | pre-engagement / fixed / %-timer window models; AUC-vs-time sweep |
| `analysis_win_probability_timeseries.py` | 2d | baseline vs control-augmented WP, aggregates, lead–lag, case studies |
| `analysis_side_site_favourability.py` | 3a, 3b | side win rates, territory vs winning, site stats, structural distances |
| `analysis_location_importance.py` | 3c | six importance measures per area + structural-vs-behavioural comparison |

## Checks to run before trusting the outputs (in this order)

1. `extract_supplementary_tables.py --inspect` — confirms the demo keys relied on (`kills[*]` fields,
   `ctBuyType`, `ctFreezeTimeEndEqVal`, `tScore`/`ctScore`, `tickRate`, frame `bombPlanted`/`bombsite`) and prints
   the distinct buy-type strings so you can see the real vocabulary.
2. `--check-mapper` — the kill-coordinate mapper must reproduce `position_maps.current_areaId`. **Set `Z_TOL`
   (env var) to the value used in `pipeline/position_to_node_map.py`** (the suite defaults to 250; the two must
   match). Exact-tie disagreements on
   boundaries are harmless; a systematic <99.9% is not.
3. `position_maps` columns: `round, frame_idx, player_name, side, current_areaId, projected_areaId`
   (resolver also accepts `curr_areaId`/`proj_areaId`). `control_fields`: `round, frame_idx, area_id, ct_mu, t_mu`,
   `control_val` or `control_value`. `round_metadata`: `round, winning_side, freeze_time_end_tick`.
   `bomb_plants`: `round, bomb_site, tick`. A mismatch raises a KeyError listing the columns that *are* present.
4. `ct_mu`/`t_mu` are the **raw summed** influences (the active-footprint rule needs `ct_mu + t_mu > EPSILON`).
5. `build_frame_summaries.py` re-implements the Voronoi baseline (symmetrised geodesic, ties excluded, tile-area
   weighted). Compare `vor_margin` against the pipeline's `outputs/voronoi_metrics/` for one match before relying on it.
6. `analysis_metric3_stratified.py` prints a **reproduction check** of Metric 3's Cohen's d against reference
   values set in the script (update them if Metric 3 is re-run). If it differs by >0.15 the definitions differ (warmup handling,
   window, sign) and the stratified numbers should not be compared with the originals until reconciled.
7. Spawn areas are assumed to be named `TSpawn` / `CTSpawn` (structural distances only).

## Design decisions

* **Trade filtering.** The default `fresh` preset drops the *avenging* kill (`is_trade_kill`) and same-killer chain
  kills but keeps the first kill of a traded pair. Dropping kills because they were later avenged (`strict`)
  conditions on a future event and can bias the sample; it is reported as a robustness row only. Trade flags are
  computed from kill sequences (victim/attacker identity, side, 5 s window), not from the parser's `isTrade`;
  agreement with the parser flag is printed per match as a diagnostic.
* **Lead frames.** Frames are 64 ticks (~0.5 s) apart; `frame_dt` stores the exact gap between
  the lead frame and the kill. Results are reported for leads 1–5 s and radii 250/400/600.
* **Local advantage** uses the union of two graph balls around the victim's and attacker's kill-time nodes, from the
  lead frame's positions via the *exact* `inf_comp()` formula. `_all` includes the duelists; `_support` excludes
  them (who else was around). Local-numbers stratification (`n_att_near` vs `n_vic_near`) is reported.
* **Pre-engagement window** (headline for Part 2): alive count is 5v5 by construction, so any predictive signal
  is positioning, not head-count. Fixed windows drop rounds shorter than the window (otherwise end-of-round
  frames leak the outcome); n(t) is plotted so survivorship is visible.
* **Grouped cross-validation by match everywhere**, cluster (match) bootstrap for every CI, paired bootstrap for
  AUC differences.
* **Baseline WP model has no nav-mesh features**, so plotting it against control margin is independent evidence.

## Known limitations (state these, don't hide them)

* **Kill-location contest is partly mechanical** (kills need nearby opposing players; the field is contested where
  opposing players are near). It is a supporting panel; the duel win-rate curve is the real test because the
  *direction* of the advantage is not mechanically tied to who wins. The matched null (occupied nodes in the same
  lead frame) and the lead-time sweep are there to show how much of the effect is mechanical.
* The field has no line of sight, so long-range duels (AWP across Mid) will show weaker local-advantage effects.
  Distance terciles are reported.
* The event-study drop at t = 0 is partly mechanical (the victim leaves the field). Read the pre-kill trend and
  recovery, not the step.
* Economy is the dominant confounder of any "control predicts the round" claim. `round_economy` supplies
  equipment-value and pistol features; buy-type strings are kept raw for descriptive stratification.
* Betweenness depends on how finely the nav mesh subdivides an area — treat it as a structural prior.
* Site statistics are conditional on a plant (a T-side choice); plant frequency is a preference, not a strength.
* Not yet done: the per-round `plant_t` uses the first plant per round; double-plant demos are not modelled.

## Key outputs

`outputs/metrics/kill_validity/` (duel_curve.csv, duel_stratified.csv, duel_robustness_grid.csv,
kill_location_contest.csv, kill_event_study.csv) · `plant_aligned/` · `metric3_stratified/` ·
`early_window/` (window_models.csv, window_deltas.csv, sweep_auc.csv, headline_coefficients.csv) ·
`win_probability/` · `side_site/` · `location_importance/` · `results_registry.json` (every headline estimate with its CI, written by `validation_common.record_result`).

## Changelog — revision 2

* **Active footprint is now `exp(-2)` = 0.135 (2 sigma)** (`ACTIVE_THR` in `validation_common.py`; env var overrides).
  The original `exp(-4.5)` survives only as `ctrl_active_legacy` for the Metric 3 reproduction check. Cached
  `frame_summaries` / `engagement_*` tables carry an `active_thr` stamp and rebuild automatically when it changes.
* **S1 kill presets**: added `opener_cluster` (first kill of each engagement cluster; **new default**) and `opener_round`
  (first kill of the round, always 5v5). `duel_by_preset.csv` puts all five side by side: `fresh` is biased down,
  `strict` is biased up, the openers use only past information.
* **S2 rewritten** with three nulls (naive occupied nodes; robustly contested = both teams >= m mass; mass-matched).
  The naive null is unfair because occupied nodes always carry a full player's mass while kill nodes often carry little,
  and faint-both-sides nodes score ~0.5 by symmetry. Figure title corrected.
* **Plant analysis conditioned on CT alive at the plant** (per-stratum d, d after head-count fixed effects, adjusted
  logit coefficient, grouped-CV delta AUC given head-count + site + economy).
* **Early-window**: head-count baseline now includes history features; all paired deltas are printed;
  sweep figure fixed (one colour per model, legend on every panel) and a paired-delta row added. **Caveat:** for windows
  that contain kills, a gain of (econ+alive+control) over (econ+alive) is an UPPER BOUND on positioning, because window-mean
  control is a nonlinear time-average of the head-count. Clean tests: pre-engagement windows, the still-5v5 panel, 2e.
* **New** `analysis_contention_definitions.py` (Metric 1 re-run).
* **Location importance**: betweenness reported per node (the sum scales with tile subdivision), per-node occupancy,
  robust-contest measure, 1500 bootstrap resamples by default (smallest resolvable p = 1/(n+1)).
* Bootstrap counts raised throughout (default 1000, `N_BOOT`); logistic bootstrap uses a fast Newton fit.
