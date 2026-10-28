# Active-Footprint Extension: Methods, Implementation and Results

**Scope of this file.** Everything done in this working session on (a) diagnosing why whole-map averages of the control field are diluted, (b) the "active footprint" correction and its application to Metric 3 (Control Margin) and Metric 1 (Contention Rate), (c) the diagnostic visualisations built around it, and (d) a proposed set of additional validity metrics. Other analyses are not covered here.

**Status key used throughout:** ✅ run, with results available · 🟡 written/edited but not yet run · 📝 proposed only.

---

## 1. Summary

| Item | Status | One-line outcome |
|---|---|---|
| Diagnosis of dilution in whole-map averages | ✅ | Nodes far from every player sit at `control_val = 0.5` exactly and dominate the average |
| Active-footprint definition (threshold on pre-sigmoid `ct_mu + t_mu`) | ✅ | ε = exp(−4.5) ≈ 0.0111, i.e. the 3σ cutoff, σ = 500 |
| `team_agg.py` change to persist `ct_mu`, `t_mu` | ✅ (control fields were regenerated; Metric 3 active results exist) | Prerequisite for every active-footprint calculation |
| Metric 3 (CT side), full-map vs active-footprint | ✅ | Cohen's d 1.62 → 1.84; raw group-mean gap 2.4 pp → 4.2 pp |
| Metric 1, full-map vs active-footprint | 🟡 | Scripts updated; **no results yet** (needs rerun) |
| Active-footprint overlay heatmap (per match/round) | 🟡 | Script written and iterated; PNG stage ran for the user, GIF stage fixed after a crash |
| Proposed additional validity metrics (incl. kill-location alignment) | 📝 | Draft script for kill-location metrics exists but is untested and uses guessed column names |

Mean active node fraction on de_mirage (σ = 500, ε = exp(−4.5)): **0.678**, i.e. about **612 of 903** nodes are inside some player's effective influence range in an average frame.

---

## 2. Background: the model being corrected

Per-player influence is a Gaussian decay over effective graph distance:

```
mu_i(v, f) = exp( -d_eff(i, v, f)^2 / (2 * sigma^2) )
```

Team aggregation and normalisation (`team_agg.py`):

```
ct_mu(v, f) = sum of mu_i over alive CT players
t_mu(v, f)  = sum of mu_i over alive T players
control_val(v, f) = expit( K * (ct_mu - t_mu) )      # K = 1.0, in (0, 1); 0.5 = neutral
```

Two parameters that are easy to conflate and are **fully independent**:

| Parameter | Role | Value | Where it acts |
|---|---|---|---|
| σ (sigma) | Spatial reach of one player's influence (Gaussian width, game units of geodesic distance) | 500 (sensitivity set {300, 500, 750}) | Inside `mu_i`, before any team aggregation |
| K | Steepness of the sigmoid mapping net influence to [0, 1] | 1.0 | After aggregation, only affects `control_val` |

Useful geometric reference points for σ = 500 (influence of a single player at distance d_eff):

| d_eff | Multiple of σ | mu | Comment |
|---|---|---|---|
| 500 | 1σ | 0.607 | |
| 589 | σ√(2 ln 2), half-maximum radius | 0.500 | |
| 1000 | 2σ | 0.135 | |
| 1500 | 3σ | 0.0111 | the default ε |

---

## 3. The problem: whole-map averages are dominated by empty map

For a node far from every player, `ct_mu → 0` and `t_mu → 0`, so the sigmoid argument is 0 and

```
control_val = expit(0) = 0.5   (exactly, up to float precision)
```

This 0.5 does **not** mean "contested". It means "nobody is near this node". The sigmoid cannot distinguish that case from "two near-equal forces are both close", which also produces ≈ 0.5. Consequences for the two metrics:

**Metric 3 (Control Margin).** Averaging `control_val` over all 903 nodes mixes a small amount of informative signal with a large mass of structurally neutral nodes. Both group means (CT-won, CT-lost rounds) are pulled toward 0.5. Observed symptom in the original full-map run: group means of about 52–53 % vs 50–51 %, which look almost identical, alongside a very large Welch t and Cohen's d. These are not contradictory: the dilution shrinks the numerator (mean difference) and also shrinks the within-group spread, so the standardised effect stays large while the raw difference looks tiny.

**Metric 1 (Contention Rate).** The problem is worse, because 0.5 lies inside the contested band [0.4, 0.6]. Every empty node is therefore counted as contested.

Per-frame identity (derived, exact, given the definitions): let `a_f = |V_active(f)| / |V|` and `rho_active(f)` the contested fraction within the active set. Every inactive node has `|ct_mu − t_mu| ≤ ct_mu + t_mu ≤ ε`, so its `control_val` lies within about 0.5 ± 0.0028, safely inside [0.4, 0.6]. Hence

```
rho_full(f) = (1 - a_f) + a_f * rho_active(f)   >=   rho_active(f)
```

so the full-map contention rate can never be lower than the active-footprint rate. This is a useful sanity check on the Metric 1 rerun: if `rho_bar_active > rho_bar_full` appears in the output, something is wrong.

---

## 4. Method: the active footprint

### 4.1 Definition

```
V_active(f) = { v in V : ct_mu(v, f) + t_mu(v, f) > EPSILON }
```

- `EPSILON = exp(-4.5) ≈ 0.011109` (the `epsilon` value in `_summary.json`), the influence a single player contributes at 3σ.
- The threshold is applied to the **raw summed influence, before the sigmoid**, never to `control_val`. Thresholding `control_val` would conflate "far from everyone" (should be excluded) with "contested between two nearby, roughly equal players" (must be included).
- ε is defined relative to σ, not K. K does not appear anywhere in the activity test.
- The test is on the *sum* over both teams and all players, so several players each contributing a little can jointly push a node over ε. The effective reach is therefore slightly larger than a strict "within 3σ of one player" rule.
- Tightening ε to exp(−2) ≈ 0.135 (2σ) shrinks the footprint; loosening toward 0 recovers the full map. The footprint size also scales with σ.

### 4.2 Active-footprint versions of the metrics

**Metric 3.** Per frame, `A_CT_active(f)` = mean of `control_val` over `V_active(f)`. Per round, `A_CT_active_bar(r)` = mean of those per-frame values.

Implementation detail that matters: the average is computed in **two explicit stages** (frame mean, then round mean of frame means). `|V|` is constant (903) but `|V_active(f)|` varies as players move; a flat mean over all active rows in a round would silently up-weight frames with larger footprints.

**Metric 1.**

```
rho_active(f) = |{ v in V_active(f) : control_val(v, f) in [0.4, 0.6] }| / |V_active(f)|
```

Frames with zero active nodes return NaN (excluded from the active mean). Dataset-level `rho_bar` is the frame-count-weighted mean, as in the original definition.

**Voronoi baseline for RQ1** stays exactly 0.0 under both restrictions. A hard nearest-player assignment gives `c(v, f) ∈ {0, 1}` for every node on any subset of V, so it can never fall in [0.4, 0.6].

### 4.3 Per-area breakdown of Metric 1 (where the contention is)

Each node is assigned to one of 27 named areas (`data/de_mirage_area_assignment.csv`, see 8.1). For area `a` and frame `f`:

```
rate_full(a, f)   = contested tiles in a / tiles in a
active_frac(a, f) = active tiles in a / tiles in a
rate_active(a, f) = active contested tiles in a / active tiles in a     (NaN if no active tile)
```

The dataset-level value for an area is the mean over frames (each frame weighted equally; `rate_active` averages only over frames where the area had at least one active tile). Per-match files store the sums and frame counts, so the dataset mean is exact rather than a mean of per-match means.

Columns in `by_area/_area_summary.csv` and how to read them:

| Column | Meaning |
|---|---|
| `mean_active_fraction` | How often the area is in play at all |
| `mean_contention_rate_active` | When in play, how often a tile is balanced |
| `mean_contention_rate_full` | Same, over all tiles; inflated for rarely occupied areas |
| `inflation_full_minus_active` | The inflation itself |
| `share_of_active_tile_frames` | Area's share of all active tile-frames (presence) |
| `share_of_active_contested` | Area's share of all active contested tile-frames |
| `contested_to_presence_ratio` | Second share over first; > 1 means more contested than its presence would suggest, < 1 means usually held decisively when occupied |
| `n_tiles`, `n_formerly_blank_tiles` | Area size and how many tiles were reassigned from a blank name |

A never-occupied area has `mean_contention_rate_active = NaN` (not 0.5) and a full-map rate near 1.0, which is the Metric 1 inflation in its purest form. Areas with 1–2 tiles are noisy. The breakdown is descriptive; there is no significance test. **No real results yet**; the figures in the report should come from the rerun.

### 4.4 Statistical procedure (Metric 3)

- Unit of analysis: the round (7,393 rounds from 278 matches in the dataset).
- Groups: CT-won rounds (n = 4,012) vs CT-lost rounds, i.e. T-won (n = 3,381). CT win rate 54.3 %.
- Test: Welch's unequal-variance t-test (`scipy.stats.ttest_ind(..., equal_var=False)`).
- Effect size: Cohen's d with the **equal-weight** pooled SD, `sqrt((var_win + var_loss) / 2)`, as in the original script (not the n-weighted pooled SD; the two differ slightly when group sizes differ).
- Hypothesis (H3 / RQ3): `mu_win > mu_loss`.
- p-values: the reported `welch_p_value = 0.0` is floating-point underflow, not exactly zero. Report as p < 0.001 (or p < 1e-300).

---

## 5. Results

### 5.1 Metric 3, CT side (σ = 500, ε = exp(−4.5), n_win = 4,012, n_loss = 3,381)

Values straight from `_summary.json`:

| | Full map (903 nodes) | Active footprint |
|---|---|---|
| `mu_win` | 0.529399 | 0.544510 |
| `mu_loss` | 0.505312 | 0.502927 |
| Difference | 0.024088 (2.4 pp) | 0.041583 (4.2 pp) |
| Welch t | 68.52 | 77.76 |
| Cohen's d | 1.618 | 1.839 |
| H3 (`mu_win > mu_loss`) | passes | passes |

Derived quantities (computed from the table, useful for the write-up):

| | Full map | Active footprint |
|---|---|---|
| Signed advantage `2c − 1`, CT-won rounds | +0.0588 | +0.0890 |
| Signed advantage `2c − 1`, CT-lost rounds | +0.0106 | +0.0059 |
| Implied pooled SD (difference / d) | 0.0149 | 0.0226 |
| Gap relative to full map | 1.00× | 1.73× (+72.6 %) |
| d relative to full map | 1.00× | 1.14× (+13.6 %) |

Dataset diagnostics: `mean_active_node_fraction = 0.678`, `n_rounds_dropped_missing_outcome = 0`.

### 5.2 Interpretation points

1. **The active-footprint restriction mainly changes interpretability, and only moderately changes the standardised effect.** The raw gap grows by about 73 %, but Cohen's d grows by about 14 %. The standardised effect was already large under dilution because dilution shrinks signal and spread together. Avoid claiming the correction "uncovered" a much larger effect; it removed an artefact that made a large effect look small.
2. **CT-lost rounds are essentially neutral in the active region** (`mu_loss = 0.5029`, signed +0.006), whereas the full-map value (0.5053) sits slightly above neutral. In won rounds CT control over the contested region is clearly CT-favoured (0.5445, signed +0.089). This is a cleaner statement than the full-map one.
3. **Effect-size scale.** d ≈ 1.8 is very large by conventional benchmarks (0.8 is "large"). See Section 7 for why this should be read as a concurrent association, not predictive strength.
4. **Group imbalance.** 54.3 % CT win rate; Welch's test handles unequal group sizes, so this does not invalidate the test, but mention it in the methods.

### 5.3 Metric 1: not yet run

Rerun `metric1_contention_rate.py`. Expected outputs and checks:

- `_summary.json` will contain `full_map.rho_bar_contention_rate`, `active_footprint.rho_bar_contention_rate`, `mean_active_node_fraction`, `voronoi_baseline_contention_rate = 0.0`.
- Required relation: `rho_bar_active ≤ rho_bar_full` (Section 3 identity).
- Back-of-envelope link: `rho_bar_full ≈ (1 − ā) + ā · rho_bar_active`, with ā ≈ 0.678 from Metric 3. The relation is only approximate at dataset level because `a_f` varies by frame.
- RQ1 conclusion is unaffected in direction (Voronoi = 0 under both), but the active-footprint value is the defensible magnitude to report.

---

## 6. Implementation record

### 6.1 Code changes

| File | Change | Status |
|---|---|---|
| `team_agg.py` | `team_agg_comp` now returns `ct_mu`, `t_mu` (renamed from the pivoted `CT`, `T` columns) alongside `control_val` instead of dropping them. Skip-if-exists is schema-aware: a cached CSV missing `ct_mu`/`t_mu` is recomputed, not trusted. | ✅ |
| `metric3_control_margin.py` | Computes `control_margin_full` and `control_margin_active` plus `active_node_fraction` per round; two-stage averaging; shared Welch/Cohen helper; nested `full_map` / `active_footprint` blocks in `_summary.json`; schema-aware cache check; degrades to NaN (with a printed NOTE) if `ct_mu`/`t_mu` are absent. Constants: `SIGMA = 500.0`, `EPSILON = exp(-4.5)`. A cache whose active columns are all NaN is now also recomputed once the source CSV carries `ct_mu`/`t_mu` (see section 9). | ✅ |
| `visualize_metric3_control_margin.py` | Reads new column names and nested JSON. Figures 01–04 become two-panel (full-map vs active); new figure 05 (active node fraction); placeholder panel if active data is missing. | 🟡 |
| `metric1_contention_rate.py` | Adds `num_active_nodes`, `num_active_contested`, `contention_rate_active`, `active_node_fraction` per frame; nested summary matching Metric 3; schema-aware cache; same constants. **Per-area breakdown** (4.3): per-match `by_area/{match}.csv` and a dataset-level `by_area/_area_summary.csv`, using `data/de_mirage_area_assignment.csv`; skipped with a warning if that file is missing. Tested on synthetic data only (fresh run, cached rerun, stale cache, missing assignment). | 🟡 |
| `visualize_metric1_contention_rate.py` | Figures 01, 02, 04 two-panel; figure 03 becomes a three-bar chart (full-map field, active-footprint field, Voronoi = 0); new figure 05 (active fraction) and figure 06 (per-area). | 🟡 |
| `visualize_active_footprint.py` (new) | Per-frame overlay heatmap for a chosen match and round (see 6.3). | 🟡 |

### 6.2 Output schemas (for reference when describing the pipeline)

`outputs/control_fields/{match}.csv`: `round, frame_idx, area_id, ct_mu, t_mu, control_val`

`outputs/metrics/control_margin/{match}.csv`: `match_id, round, control_margin_full, control_margin_active, active_node_fraction, winning_side`

`outputs/metrics/contention_rate/{match}.csv`: `round, frame_idx, num_nodes, num_contested, contention_rate, num_active_nodes, num_active_contested, contention_rate_active, active_node_fraction`

`outputs/metrics/contention_rate/by_area/{match}.csv` (per-area sums and frame counts): `assigned_area, n_tiles, n_formerly_blank_tiles, n_frames, sum_rate_full, sum_contested_tiles, sum_active_fraction, sum_rate_active, n_frames_rate_active, sum_active_tiles, sum_active_contested_tiles`; `by_area/_area_summary.csv` is the dataset-level table (4.3). The `by_area/` sub-folder keeps these out of the visualisation script's `*.csv` glob over per-match frame files.

Both `_summary.json` files have top-level `epsilon`, `sigma`, `mean_active_node_fraction`, and separate `full_map` and `active_footprint` blocks.

### 6.3 Overlay visualisation (`visualize_active_footprint.py`)

Usage:

```bash
python visualize_active_footprint.py --match <match_name> --round <n>
python visualize_active_footprint.py --match <match_name> --round <n> --frame <idx> --no-gif
python visualize_active_footprint.py --match <match_name> --round <n> --epsilon 0.135   # 2-sigma footprint
```

Design:

- Base layer: each nav tile coloured by `control_val`, using a two-segment linear interpolation: T orange `(1.0, 0.5, 0.0)` ← neutral → CT cyan `(0.0, 0.9, 0.9)`.
- **Neutral colour = the radar floor colour `(1/255, 19/255, 48/255)`**, matching the project's map-control heatmap, so uncontested tiles blend into the map instead of appearing as a bright wash.
- Overlay: tiles with `ct_mu + t_mu > ε` are drawn in semi-transparent purple (`#9B30FF`, alpha 0.35), chosen for contrast against both cyan and orange.
- Player markers coloured by side (CT `#00AAAA`, T `#CC6600`, size 20) and a red square bomb marker, using the same constants as the existing heatmap scripts; positions come from `position_maps` (`current_x`, `current_y`, optional `bomb_x`, `bomb_y`) via `position_transform`.
- Per-frame info box: full-map mean, active mean, active node count and fraction, ε and σ.
- Output: `outputs/active_footprint_viz/{match}/round_{n}/frame_XXXX.png` plus an optional GIF.
- The script needs `ct_mu`/`t_mu` in the control field; without them it still draws the heatmap and warns that the overlay is skipped.

Debugging record: the first GIF build failed with `ValueError: all input arrays must have the same shape` because `savefig(..., bbox_inches="tight")` trims each frame to its content, and the content size changes frame to frame. Fixed with a fixed `DEFAULT_FIG_SIZE = (9.5, 8.5)` and no `bbox_inches` (the same fix the existing GIF heatmap script uses).

**Qualitative use in the report:** an overlay frame, especially one early in a round and one mid-round, is a direct visual answer to "what is the active footprint". It also visually explains the Metric 1 inflation (the non-purple tiles are exactly the ones counted as contested under the full-map definition).

### 6.4 Run order

```
1. python team_agg.py                              # recomputes any control_fields CSV lacking ct_mu/t_mu
2. python metric3_control_margin.py                # recomputes stale per-match CSVs automatically
3. python visualize_metric3_control_margin.py
4. python metric1_contention_rate.py
5. python visualize_metric1_contention_rate.py
6. python visualize_active_footprint.py --match <match_name> --round <n>
```

Example match used while developing the overlay: `x_kom_AGO_VS_Natus_Vincere_667970ee_833b_491f_b8c7_b91bcb8f2c04`, round 21.

### 6.5 Figure inventory

| Figure | What it shows | Use in report |
|---|---|---|
| Metric 3 fig 01 | Control margin box plot by outcome, full-map vs active panels | Primary RQ3 figure |
| Metric 3 fig 02 | Overlaid histograms by outcome | Shows overlap alongside Cohen's d |
| Metric 3 fig 03 | Group means with 95 % CI (normal approximation) | Compact headline figure |
| Metric 3 fig 04 | Margin vs round number (QA/trend) | Probably appendix or omit |
| Metric 3 fig 05 / Metric 1 fig 05 | Distribution of active node fraction | Justifies ε; shows how much map is in play |
| Metric 1 fig 01 | Contention rate vs round progress (median + IQR), two panels | Time structure of contention |
| Metric 1 fig 02 | Per-frame contention distribution, two panels | |
| Metric 1 fig 03 | Full-map field vs active field vs Voronoi (0) | RQ1 headline |
| Metric 1 fig 04 | Per-match box plots | QA/outlier check |
| Metric 1 fig 06 | Per-area bars: how often each area is in play, and how contested it is when it is (full-map rate as hollow marker) | Where on the map contention happens |
| Overlay frames/GIF | Heatmap with purple active footprint, players, bomb | Qualitative illustration of the method |

With an 8-page limit, the realistic selection is: the Metric 3 fig 01 (or 03), the Metric 1 fig 03, one overlay frame, and possibly the active-fraction histogram.

---

## 7. Limitations and caveats to state in the report

1. **Dependence on σ and ε.** The footprint size (0.678) is a consequence of σ = 500 on this map. Results at other σ in the sensitivity set {300, 500, 750} and at other ε (e.g. the 2σ cutoff exp(−2)) will differ, and the footprint will change. State the ε used for every reported number; if other analyses in the report use a different ε, say so explicitly to avoid apparent inconsistency.
2. **Contested ≠ both teams present.** With K = 1, the band [0.4, 0.6] corresponds to `|ct_mu − t_mu| ≤ ln(1.5) ≈ 0.405`. A node reached by only one distant player (influence below 0.405) therefore counts as "contested" even with no opposing influence. The active-footprint filter removes empty nodes but not these weakly one-sided ones. A stricter definition (for example requiring both `ct_mu` and `t_mu` above some small δ) is a possible refinement; it has **not** been implemented. Worth acknowledging as a limitation of Metric 1's operationalisation.
3. **Metric 3 is a concurrent association, not a predictor.** The round-mean margin includes frames after the decisive events of the round. A team that has already won fights has more alive players and more influence, so control margin is partly mechanically coupled to outcome. Metric 3 therefore shows that the field is consistent with round outcomes, not that early control forecasts them. An early-window version is the appropriate predictive test.
4. **Alive-player count confound.** Team influence sums over alive players, so margin scales with man-advantage. The effect is not separated from "map control" in Metric 3.
5. **Non-independence.** Rounds are nested in matches (and teams), so the Welch p-value is anti-conservative. Effect sizes are not affected, but consider a match-clustered bootstrap CI for d or a match-level permutation test.
6. **No unit of causal inference.** Neither metric tests whether control causes wins.
7. **Cohen's d definition.** Equal-weight pooled SD; state it.
8. **Visualisation is illustrative.** A single frame/round is an example, not evidence.
9. **Bombsite node sets may not match the assigned areas.** `V_A` and `V_B` (Site Control Score, Metric 2) were built from raw awpy `areaName == "BombsiteA"/"BombsiteB"` (`get_bombsite_nodes.py`, `data/de_mirage_bombsite_nodes.json`), and the T-side bombsite margins use `areaName` the same way. If any of the 72 blank-named nodes were assigned to a bombsite in `data/de_mirage_area_assignment.csv`, they are missing from those node sets. Not checked in this session; compare the two sets and, if they differ, state which set each result used (or regenerate Metric 2 and the T-side bombsite results).

---

## 8. Proposed additional validity metrics (📝 not run)

Motivation: Metrics 1–3 show the field is *useful* but do not directly test whether it is *correct* in cases where the right answer is known. Metrics grouped by what they establish:

**Construct validity (uses data already stored)**
- *Distance–advantage monotonicity.* Spearman correlation between `Δd = min_T_dist − min_CT_dist` (geodesic) at a node and its `control_val`. A well-behaved field should be strongly monotonic; reversals indicate a bug in the aggregation or distance computation.
- *Saturation at isolation.* Restrict to (node, frame) pairs where the nearest opponent is very far away; mean `|2c − 1|` there should approach 1. A limiting-case check.

**Criterion validity (needs a `kills` table)**
- *Kill-location contention alignment.* Compare `|2c − 1|` at the victim's nearest node in the frame before each kill with a random active-footprint baseline from the same rounds (one-sided Mann–Whitney U, rank-biserial effect size). Claim: kills occur in more contested space than average.
- *Pre-kill local advantage rate.* Fraction of kills where the killer's side held the higher local control just before the kill, tested against 50 % (exact binomial).

**Cross-method agreement**
- Correlation with awpy's built-in BFS map-control metric (`awpy.analytics.map_control`; confirm the exact function name in the installed version).
- Sign agreement with the Voronoi baseline per node, with disagreements broken down by **assigned area** (`data/de_mirage_area_assignment.csv`, see 8.1) to show whether they cluster at chokepoints. Not implemented.

**Internal consistency (debugging, not reportable findings)**
- Frame-to-frame smoothness against actual player displacement.
- σ-to-geometry plausibility: half-maximum radius `σ·√(2 ln 2)` = 353 / 589 / 883 units for σ = 300 / 500 / 750, compared with known de_mirage distances.

**Status.** These were not implemented as a separate Metric 4. An early kill-location draft was dropped and is not in the final repository; kill-based validity is covered by the validation suite (`analysis/validation/analysis_kill_validity.py`: duel win rate against local advantage, kill-location contest, kill event study).

### 8.1 Named areas (`data/de_mirage_area_assignment.csv`)

awpy's `areaName` is blank for 72 of the 903 nodes (including the CT-stairs tiles up to A and the Apartments ramp). The project resolves every node to one of 27 named areas (awpy's 26 plus `ApartmentsRamp`) in `data/de_mirage_area_assignment.csv`, with provenance columns (`original_areaName`, `method`, `vote_share`, ...). Any per-area result in the report must use `assigned_area`, not raw `areaName`; otherwise kills and control on those 72 tiles are silently dropped or lumped into a blank category.

How it is used in the active-footprint work:

- Metric 3, `team_agg.py` and the overlay are **node-level** and are unaffected by the assignment. Metric 1's headline value is also node-level; the assignment is used only for its per-area breakdown (4.3).
- If a per-area control value is ever computed by averaging tile values (footprint-weighted mean over an area's tiles), average over the area's **active** tiles only and return NaN where none are active. Averaging over all tiles dilutes toward 0.5 for the same reason as the whole-map mean, and an area with no active tiles is "unobserved", not "neutral".

---

## 9. Corrections and lessons from this session

- The first Metric 3 rewrite was based on guessed column names (`control_value`, `ct_mu`/`t_mu` assumed present). The real files use `control_val` and did not carry the raw sums; the final version follows the real schema and degrades gracefully. Lesson: read the actual script before editing.
- Earlier proposals to break results down by "`areaName`" would have lost the 72 blank-named nodes; the assignment file (8.1) is the correct source of area labels.
- The bombsite node sets behind Metric 2 and the T-side bombsite results use raw `areaName` (limitation 9).
- Cache staleness had a second case I had missed: a cache written *before* `team_agg.py` was patched has the active columns but only NaN values, so the column-presence check trusted it forever. Metric 1 and Metric 3 now also recompute when the active columns are all NaN and the source CSV now carries `ct_mu`/`t_mu`. (Matters only if a metric script was run before the patch.)
- `control_fields` originally dropped `ct_mu`/`t_mu` after computing `control_val`. The active-footprint filter cannot be reconstructed from `control_val` alone.
- Skip-if-exists caches would silently preserve pre-change outputs. All modified scripts now check for the new columns before trusting a cache.
- A figure quoted in an earlier chat message said the raw gap grew by "75 %". The exact value is 0.041583 / 0.024088 = 1.726, i.e. **+72.6 %**; use the corrected figure.
- GIF assembly requires identical pixel dimensions across frames: fixed figure size, no `bbox_inches="tight"`.
- Earlier chat text called the `p ≈ 60+` figure a p-value; it is the Welch t-statistic (68.5 full-map, 77.8 active).

---

## 10. Draft wording for the report

*(Plain-text drafts; adapt to the template and citation keys. The two works cited are the ones the methodology extends; check them against your bibliography file.)*

**Methods: active footprint.**
> Our control field extends the influence-field approach of Fernández & Bornn (2018) and Xenopoulos et al. (2020). For each node and frame, per-player Gaussian influence (σ = 500 game units of graph distance) is summed by team and mapped to a control value in (0, 1) with a logistic function (K = 1), so that 0.5 denotes neutrality. Because a node that no player is near also receives a control value of exactly 0.5, averaging control over the whole navigation mesh mixes informative nodes with nodes that carry no information. We therefore define the *active footprint* of a frame as the set of nodes whose summed pre-sigmoid influence exceeds ε = exp(−4.5) ≈ 0.011, the influence contributed by a single player at three Gaussian widths. All metrics are reported both over the full mesh and over the active footprint; the footprint covers 67.8 % of nodes on average.

**Results: Metric 3.**
> CT-won rounds had a higher mean control value than CT-lost rounds under both definitions (full mesh: 0.529 vs 0.505, Welch t = 68.5, d = 1.62; active footprint: 0.545 vs 0.503, t = 77.8, d = 1.84; n = 4,012 and 3,381 rounds, p < 0.001). Restricting to the active footprint increased the raw difference from 2.4 to 4.2 percentage points; the standardised effect increased by 14 % because dilution had compressed the within-group spread as well as the mean difference. In CT-lost rounds mean control over the active footprint was effectively neutral (0.503).

**Methods/Results note: Metric 1 (once run).**
> Because neutral nodes with no nearby player fall inside the contested band [0.4, 0.6] by construction, the full-mesh contention rate is an upper bound on the active-footprint rate (ρ_full = (1 − a) + a·ρ_active for active fraction a). We therefore report the active-footprint rate as the primary contention measure. The Voronoi baseline is 0 under both definitions because hard assignment places every node at 0 or 1.

**Limitations.**
> Control margin is evaluated concurrently with round outcome and scales with the number of surviving players, so Metric 3 demonstrates consistency between the field and outcomes rather than predictive power. Rounds are nested within matches, so p-values are likely optimistic; effect sizes are the more reliable summary. The contested band is defined on net influence and therefore also includes nodes weakly influenced by a single team.

---

## 11. Open items

1. Rerun Metric 1 (and its visualisation) and record `rho_bar_full`, `rho_bar_active`, and the check that active ≤ full.
2. Rerun/verify the Metric 3 visualisations against the new schema and select the figures to include.
3. Decide whether to run Metric 3 and Metric 1 at σ ∈ {300, 750} and at ε = exp(−2) as a sensitivity table (footprint size and effect sizes).
4. Add a match-clustered bootstrap CI or match-level permutation test for the Metric 3 effect size.
5. Run the construct-validity checks (monotonicity, saturation), which need no new extraction. (The Metric 4 draft was retired; kill-based validity is covered by the validation suite.)
6. Optional: implement a "both teams present" contested definition (Section 7, point 2) and compare.
6a. Compare the bombsite node sets in `data/de_mirage_bombsite_nodes.json` with nodes whose `assigned_area` is BombsiteA/BombsiteB; if they differ, decide whether to regenerate Metric 2 and the T-side bombsite results (limitation 9).
6b. After rerunning Metric 1, read `by_area/_area_summary.csv` and pick which areas to discuss (likely those with the highest `share_of_active_contested` and `contested_to_presence_ratio`, excluding 1–2 tile areas).
7. Run the overlay script on a few illustrative rounds and pick one or two frames for the paper.
