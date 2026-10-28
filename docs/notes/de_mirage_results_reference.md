# de_mirage Influence-Field Metrics — Results Reference

Purpose: a lookup file for pulling numbers into the final write-up. Every result is
tagged with the **metric** it belongs to and **how it was found**. Where the output
files do not say how something was computed, the entry says so and gives only the
result. Nothing here is interpretation beyond what the numbers show, apart from the
clearly marked "Observations" lines.

---

## 0. Provenance, scale, and conventions

### Source files these results come from

| File | Dated | Contains |
|---|---|---|
| `_summary.json` | 20 Sep | Metric 1 (full-map + active-footprint + Voronoi baseline), per-match block (278 matches) |
| `metric_summaries.txt` | 17 Sep | Metric 2; Metric 3 CT-side; Metric 3 T-side (incl. bombsite variants, effect-size comparison). Its Metric 1 block is **superseded** by `_summary.json` |
| `_formal_comparison.json` | 17 Sep | Nested logistic regression, likelihood-ratio test, DeLong AUC test (T-side margins) |
| `_area_breakdown.csv` | 17 Sep | Per-area T control, T-win vs T-loss rounds |
| `01_`–`07_*.png` | 17 Sep (RQ1 figures 20 Sep) | Figures (index in section 9) |
| `frame_0067.png`, `frame_0067_af.png` | 21 Sep | Single-frame illustration (Round 21, Frame 67) |

### Dataset scale

| Quantity | Value | Where reported |
|---|---|---|
| Matches | 278 | `_summary.json` |
| Frames (Metric 1 run) | 1,258,403 | `_summary.json` |
| Frames (older run, `metric_summaries.txt`) | 1,255,992 | `metric_summaries.txt` — see flag F1 |
| Rounds with outcome | 7,393 (0 dropped for missing outcome) | `metric_summaries.txt` |
| CT-win / CT-loss rounds | 4,012 / 3,381 | `metric_summaries.txt` |
| T-win / T-loss rounds | 3,381 / 4,012 | `metric_summaries.txt` |
| Plant events (total / scored / dropped) | 3,794 / 3,785 / 9 | `metric_summaries.txt` |
| Nav-mesh nodes | 903 | `metric_summaries.txt` |

### Model parameters reported in the outputs

- σ (Gaussian decay) = **500**
- ε (active-footprint threshold) = **0.1353352832…** (numerically equal to exp(−2)); applied to `ct_mu + t_mu` (the pre-sigmoid sum)
- Active node = a node with `ct_mu + t_mu > ε` in that frame (definition stated in the output descriptions)

### Score-direction conventions — these differ between metrics

| Metric | Quantity | Direction |
|---|---|---|
| Metric 2 | Site Control Score `C_e` (a mean of `control_val`) | **0 = full T control, 1 = full CT control, 0.5 = neutral.** Lower = more T |
| Metric 3, CT-side | Control margin | 0.5 neutral; **higher = more CT advantage** |
| Metric 3, T-side | T control margin | 0.5 neutral; **higher = more T advantage** (the output states full-map T values equal 1 − the CT script's values over the same rounds) |
| Area breakdown | `mean_control_t_*` | T-oriented: higher = more T control (T-win > T-loss for most areas) |

---

## 1. Metric 1 — Contention Rate (RQ1 / H1)

**Metric definition (as stated in the output):** proportion of nodes classified as
contested, where contested = `control_val` in the [0.4, 0.6] band. Computed two ways:
over all 903 nodes (full map) and over active-footprint nodes only. Voronoi baseline =
hard nearest-player assignment, which cannot place a node in the contested band.

| Result | Value | How found |
|---|---|---|
| Full-map mean contention rate ρ̄ | **0.6427** (0.642725) | Pooled over 1,258,403 frames, all 903 nodes. The frame-weighted average of the 278 per-match values reproduces this to 4 d.p. |
| Active-footprint mean contention rate ρ̄ | **0.3144** (0.314386) | Same frames, restricted to nodes with `ct_mu + t_mu > ε`. Frame-weighted average of per-match values reproduces this to 4 d.p. |
| Voronoi baseline contention rate | **0.0000** | Analytic: stated as 0 by construction (every node hard-assigned to one team); the output states it stays 0 on any subset of nodes, including the active footprint |
| H1 (continuous field > Voronoi contention) | **Passes** under both full-map and active-footprint definitions | `hypothesis_check.passes = true` in both blocks. No test statistic reported; the comparison is against the analytic zero |
| Why full-map ρ̄ is inflated | Nodes far from every player sit at exactly 0.5, which falls inside [0.4, 0.6] | Stated in the output's own description of the full-map check. The active-footprint figure is labelled the primary Metric 1 result |
| Mean active node fraction | **0.5158** (0.515796) | `_summary.json`; mean over the run (see flag F1 for the older 0.5137) |

**Per-match spread** (computed from the `per_match` block in `_summary.json`, 278 matches, unweighted across matches):

| Quantity | Min | Max | SD | Unweighted mean |
|---|---|---|---|---|
| Full-map contention rate | 0.5918 | 0.6906 | 0.0158 | 0.6430 |
| Active-footprint contention rate | 0.2870 | 0.3495 | 0.0108 | 0.3143 |
| Active node fraction | 0.4543 | 0.5771 | 0.0196 | 0.5153 |

**Distribution and time-course results (read from figures, so approximate; no underlying table was provided):**

| Result | Approximate reading | Figure | How found |
|---|---|---|---|
| Per-frame distribution, full map | Spans roughly 0.3–1.0, peak around 0.6, right-skewed tail to 1.0 | `02_contention_rate_distribution.png` (left panel) | Histogram of per-frame contention rate over all frames; dashed line at ρ̄ |
| Per-frame distribution, active footprint | Spans roughly 0.1–0.8, peak around 0.3, right-skewed | `02_contention_rate_distribution.png` (right panel) | Same, active-footprint variant |
| Full-map contention vs round progress | Median starts ≈0.64 at round start, dips to ≈0.55 by ≈0.1 progress, then rises steadily to ≈0.81 at round end | `01_contention_over_round_progress.png` (left) | Median and IQR (25th–75th pct) of per-frame contention vs normalised round progress (0 = start, 1 = end). Bin width not stated in the files |
| Active-footprint contention vs round progress | Median starts ≈0.17, rises to ≈0.31 by ≈0.2 progress, stays ≈0.30–0.32 through most of the round, rises to ≈0.37 at the end | `01_contention_over_round_progress.png` (right) | Same method, active-footprint variant |

---

## 2. Metric 2 — Site Control Score (RQ2 / H2)

**Metric definition:** mean `control_val` over the bombsite's nodes at the frame
immediately prior to the plant event. Scale: 0 = full T control, 1 = full CT control.

| Result | Value | How found |
|---|---|---|
| Plant events scored | **3,785** of 3,794 | Output counts |
| Plant events dropped | **9** | Reason stated in output: "no control field rows found for V_s at this frame" |
| Mean Site Control Score (all plants) | **0.2717** (0.271716) | Mean over the 3,785 scored events |
| Standard deviation | **0.0827** (0.082686) | Over the same events |
| Site A | mean **0.2572**, SD 0.0853, n = 2,309 | Per-site mean/SD/count from the output |
| Site B | mean **0.2944**, SD 0.0729, n = 1,476 | Same |
| H2 (mean < 0.5) | **Passes** | `hypothesis_check.passes = true`. No formal test statistic or p-value is reported in the output |
| Distribution shape | Roughly unimodal, peak around 0.22–0.26, range ≈0.08–0.63, small tail above 0.5 (count not reported) | `01_score_distribution.png` — read from figure |
| A vs B boxplot | Medians ≈0.25 (A) and ≈0.29 (B); A box ≈0.20–0.31, B box ≈0.24–0.34 | `02_score_by_site_violin.png` — read from figure |

**Not reported:** no significance test comparing Site A and Site B appears in the provided outputs, so the A vs B gap is descriptive only.

---

## 3. Metric 3 (CT-side) — Control Margin vs round outcome (RQ3 / H3)

**Metric definition (as stated in the output):** per-round mean control margin, split by
round outcome (CT win vs CT loss), compared with Welch's t-test and Cohen's d. Two
variants: full-map (all 903 nodes) and active-footprint (nodes with `ct_mu + t_mu > ε`).
Rounds: 7,393 (4,012 CT wins, 3,381 CT losses).

| Variant | μ (CT win) | μ (CT loss) | Welch t | p (as reported) | Cohen's d | H3 |
|---|---|---|---|---|---|---|
| Full map | 0.5294 | 0.5053 | 68.52 | 0.0 | **1.618** | passes |
| Active footprint | 0.5566 | 0.5012 | 80.03 | 0.0 | **1.894** | passes |

- "p = 0.0" is exactly what the output reports; the exact value is not recoverable from the files.
- The output labels the active-footprint result as the primary Metric 3 result; the full-map result is "retained for continuity" and described as diluted toward 0.5.
- Mean active node fraction reported in this run: **0.5137** (see flag F1).
- Figures: `03_mean_bar_with_ci.png` (means with 95% CI, both variants). Underlying CIs are drawn on the figure but the numeric bounds are not in the summary files.

**Observation from the numbers:** CT-loss rounds in the active-footprint variant average 0.5012, essentially neutral, while CT-win rounds average 0.5566.

---

## 4. Metric 3 (T-side) — T Control Margin vs round outcome

**Metric definition (as stated in the output):** T-side mirror of Metric 3, with four
margin definitions: full map, active footprint, bombsite (all bombsite nodes, unfiltered),
and bombsite-active (bombsite nodes that are also in the active footprint that frame).
Same test: Welch's t-test and Cohen's d, T win (n = 3,381) vs T loss (n = 4,012).

| Variant | μ (T win) | μ (T loss) | Welch t | p (as reported) | Cohen's d | n win / loss |
|---|---|---|---|---|---|---|
| Full map | 0.4947 | 0.4706 | 68.52 | 0.0 | **1.618** | 3,381 / 4,012 |
| Active footprint | 0.4988 | 0.4434 | 80.03 | 0.0 | **1.894** | 3,381 / 4,012 |
| Bombsite (unfiltered) | 0.4724 | 0.4185 | 47.52 | 0.0 | **1.113** | 3,381 / 4,012 |
| Bombsite-active (A + B) | 0.4797 | 0.3993 | 53.43 | 0.0 | **1.259** | 3,381 / 4,012 |
| Site A, active | 0.4611 | 0.3997 | 29.60 | 2.57e-179 | **0.702** | 3,381 / 4,012 |
| Site B, active | 0.4353 | 0.3865 | 29.74 | 5.84e-178 | **0.715** | 3,352 / 4,008 |

- Full-map and active-footprint t-statistics and d values are identical to the CT-side ones. The output explains why for the full-map case (T margin = 1 − CT margin over the same rounds).
- Site B active has fewer rounds (3,352 / 4,008). The output does not state the reason.
- Mean bombsite active-node fraction: **0.7058**.
- `mu_win_exceeds_mu_loss = true` for every variant.
- The output labels bombsite-active as the primary metric for testing whether T needs concentrated bombsite control rather than whole-footprint control.

**Effect-size comparison (`effect_size_comparison` block):**

| Quantity | Value |
|---|---|
| Cohen's d, active footprint | 1.8944 |
| Cohen's d, bombsite-active | 1.2587 |
| Difference (bombsite − active) | −0.6357 |

The output states this is a descriptive comparison of two effect sizes, **not a formal statistical test** of one hypothesis against the other, and not a comparison against the CT-side script. Figure: `06_effect_size_comparison.png` (all six d values).

**Observation from the numbers:** the mean T margin is below 0.5 in both T-win and T-loss rounds for every variant; the outcome difference is in how far below 0.5 it sits.

---

## 5. Metric 3 (T-side) — Formal model comparison

Run on the T-side margins (`margin_a = control_margin_active_t`, `margin_b = control_margin_bombsite_active_t`). Outcome variable: `y_t_win`. Rounds: 7,393, clustered in 278 matches.

### 5.1 Nested logistic regression

| | Model A: `y_t_win ~ margin_active_z` | Model B: `y_t_win ~ margin_active_z + margin_bombsite_active_z` |
|---|---|---|
| Intercept | −0.0910 (SE 0.0346, p = 0.0085) | −0.0871 (SE 0.0351, p = 0.0131) |
| `margin_active_z` | **2.8823** (SE 0.0665, z = 43.34, p = 0) | **2.5746** (SE 0.0697, z = 36.92, p = 2.4e-298) |
| `margin_bombsite_active_z` | — | **0.5341** (SE 0.0472, z = 11.31, p = 1.1e-29) |
| Log-likelihood | −2681.89 | −2616.00 |
| AIC | 5367.77 | **5238.00** |
| BIC | 5381.59 | **5258.73** |

How found / what the output states: two nested logistic regressions fitted on the same rounds; the `_z` suffix is the variable naming used in the output (how the standardisation was done is not stated in the files).

**Cluster-robust version of Model B** (sandwich standard errors clustered on `match_id`, 278 clusters; point estimates unchanged):

| Term | Coef | Robust SE | z | p |
|---|---|---|---|---|
| Intercept | −0.0871 | 0.0400 | −2.18 | 0.0294 |
| `margin_active_z` | 2.5746 | 0.0677 | 38.02 | 0 |
| `margin_bombsite_active_z` | 0.5341 | 0.0486 | 11.00 | 3.9e-28 |

### 5.2 Likelihood-ratio test (Model B vs Model A)

| Result | Value |
|---|---|
| LR statistic | **131.77** |
| df | 1 |
| p-value | **1.68e-30** |
| Conclusion stated in output | `bombsite_adds_significant_information = true`, i.e. bombsite margin carries information about T round outcome beyond whole-active-footprint margin |

### 5.3 DeLong AUC comparison

| Result | Value |
|---|---|
| AUC, `margin_active` | **0.9168** |
| AUC, `margin_bombsite_active` | **0.8210** |
| AUC difference | **0.0958** |
| Variance of the difference | 2.16e-5 |
| z-statistic | **20.61** |
| p-value | 0.0 (as reported) |
| Class counts | 3,381 T-win (positive), 4,012 T-loss (negative) |
| Conclusion stated in output | `margin_active_significantly_better = true` |

How found: DeLong test comparing the AUCs of the two margin variables as scores for T win vs T loss.

**Not reported in the output files:** any held-out or cross-validated evaluation. The regression and AUC figures above are computed on the full set of 7,393 rounds as given.

---

## 6. Metric 3 (T-side) — Area breakdown

**Metric:** mean T control per named nav-mesh area, split by T-win vs T-loss rounds, with the difference. How found: per-area aggregation using the area-name field; `is_bombsite` flags BombsiteA/BombsiteB. The 72 nodes with a blank area name appear as an empty `area_name` in the CSV and as "nan" in the figure. The files do not state whether the computation was restricted to plant rounds.

Sorted by difference (T-win minus T-loss), from `_area_breakdown.csv`:

| Area | Mean T control, T-win | Mean T control, T-loss | Diff | Nodes | Bombsite |
|---|---|---|---|---|---|
| BombsiteA | 0.4761 | 0.4161 | +0.0600 | 111 | yes |
| TicketBooth | 0.4211 | 0.3662 | +0.0549 | 32 | |
| Jungle | 0.4435 | 0.3926 | +0.0509 | 13 | |
| BombsiteB | 0.4666 | 0.4223 | +0.0444 | 71 | yes |
| Stairs | 0.4720 | 0.4277 | +0.0443 | 10 | |
| Connector | 0.4912 | 0.4482 | +0.0430 | 24 | |
| SnipersNest | 0.4254 | 0.3895 | +0.0359 | 9 | |
| (blank name) | 0.4566 | 0.4215 | +0.0352 | 72 | |
| CTSpawn | 0.4304 | 0.4016 | +0.0288 | 110 | |
| Scaffolding | 0.5085 | 0.4798 | +0.0287 | 7 | |
| Catwalk | 0.4916 | 0.4635 | +0.0280 | 33 | |
| Truck | 0.4835 | 0.4566 | +0.0269 | 8 | |
| Middle | 0.5281 | 0.5098 | +0.0183 | 39 | |
| Shop | 0.4482 | 0.4326 | +0.0156 | 22 | |
| Ladder | 0.4871 | 0.4731 | +0.0139 | 10 | |
| TRamp | 0.5549 | 0.5425 | +0.0124 | 26 | |
| Balcony | 0.5061 | 0.4957 | +0.0104 | 4 | |
| Apartments | 0.5284 | 0.5240 | +0.0043 | 23 | |
| TopofMid | 0.5641 | 0.5608 | +0.0033 | 50 | |
| PalaceInterior | 0.5310 | 0.5277 | +0.0033 | 31 | |
| Tunnel | 0.5241 | 0.5221 | +0.0020 | 13 | |
| PalaceAlley | 0.5510 | 0.5552 | −0.0041 | 37 | |
| BackAlley | 0.5564 | 0.5617 | −0.0053 | 43 | |
| TunnelStairs | 0.5330 | 0.5390 | −0.0060 | 7 | |
| TSpawn | 0.5432 | 0.5493 | −0.0062 | 49 | |
| House | 0.5450 | 0.5533 | −0.0084 | 16 | |
| SideAlley | 0.5602 | 0.5728 | −0.0126 | 33 | |

Totals: 27 rows, 903 nodes. Figure `07_area_breakdown.png` shows the top 20 by difference plus any bombsite row outside the top.

**Not reported:** no significance test or confidence interval per area.

---

## 7. Diagnostic — active node fraction

| Result | Value | How found |
|---|---|---|
| Mean active node fraction per round | **0.514** (0.5137, T/CT-side run) / 0.5158 (Metric 1 run) | Proportion of the 903 nodes with `ct_mu + t_mu > ε`, per round, then averaged (`|V_active| / |V|` is the figure's x-axis) |
| Distribution across rounds | Unimodal, roughly 0.22–0.72, peak near 0.5–0.55 | `05_active_node_fraction.png` — read from figure; dashed line at the mean |
| Bombsite active fraction | 0.7058 | `metric_summaries.txt` T-side block |

---

## 8. Single-frame illustration (Round 21, Frame 67)

Match: x_kom_AGO vs Natus Vincere. Figures: `frame_0067.png` (raw control field, colour scale CT = cyan, T = orange, 0.5 = neutral navy) and `frame_0067_af.png` (same frame with active-footprint nodes in purple).

| Result | Value | How found |
|---|---|---|
| Full-map mean control | 0.5433 | Printed on the figure |
| Active-footprint mean control | 0.5677 | Printed on the figure |
| Active nodes | 580 / 903 (64.2%) | Printed on the figure |
| ε, σ | 0.1353, 500 | Printed on the figure |
| Purple definition | `ct_mu + t_mu > ε` | Printed in the figure legend |

Relates to: active-footprint definition used by Metrics 1 and 3. This is one frame, so it illustrates the filter rather than measuring anything.

---

## 9. Figure index

| File | Shows | Metric |
|---|---|---|
| `03_continuous_vs_voronoi.png` | Full-map 0.6427, active-footprint 0.3144, Voronoi 0.0000 | 1 |
| `02_contention_rate_distribution.png` | Per-frame distributions, full map and active footprint | 1 |
| `01_contention_over_round_progress.png` | Median/IQR vs round progress, full map and active footprint | 1 |
| `01_score_distribution.png` | Histogram of Site Control Score at plant, mean line and 0.5 line | 2 |
| `02_score_by_site_violin.png` | Site Control Score by site A vs B (box plus points) | 2 |
| `03_mean_bar_with_ci.png` | CT win vs loss mean margin with 95% CI, full map and active footprint | 3 (CT) |
| `05_active_node_fraction.png` | Distribution of active node fraction per round | Diagnostic |
| `06_effect_size_comparison.png` | Cohen's d for six T-side margin definitions | 3 (T) |
| `07_area_breakdown.png` | T control difference by area (top 20 plus bombsites) | 3 (T) |
| `01_win_loss_boxplot.png` | T margin by outcome, three panels (full map, active footprint, bombsite-active) | 3 (T) |
| `02_win_loss_histograms.png` | T margin histograms by outcome, three panels | 3 (T) |
| `frame_0067.png`, `frame_0067_af.png` | Single-frame control field and active-footprint mask | Illustration |

Note: the CT-side versions of the win/loss boxplot and histogram figures were generated under the same filenames as the T-side ones, so only the T-side versions were available when this file was written. Their CT-side numbers are in section 3.

---

## 10. Flags to resolve before citing

- **F1 — two slightly different run totals.** `_summary.json` (20 Sep) reports 1,258,403 frames and mean active node fraction 0.5158. `metric_summaries.txt` (17 Sep) reports 1,255,992 frames and 0.5137. The full-map ρ̄ is 0.6427 in both. The Metric 2 and Metric 3 results come from the 17 Sep run; the Metric 1 results come from the 20 Sep run. Worth confirming whether Metrics 2/3 need re-running on the newer frame set, or stating the frame count per metric.
- **F2 — threshold change between runs.** If any earlier notes or drafts quote ε ≈ 0.011 (exp(−4.5)), an active fraction ≈ 0.678, or an active-footprint d of 1.84, those belong to an earlier run. The results in this file use ε = 0.1353, active fraction ≈ 0.514–0.516, and d = 1.894.
- **F3 — score direction differs by metric** (section 0). Metric 2 is "lower = more T"; the T-side margin and area breakdown are "higher = more T". State the direction wherever a number appears.
- **F4 — round-averaged aggregation.** The Metric 3 margins are per-round means as described in the outputs, so these results do not by themselves show *when* in the round the control advantage appears.
- **F5 — in-sample statistics.** The logistic regression and AUC results are computed on all 7,393 rounds with no held-out evaluation reported (section 5).
- **F6 — hypothesis labels.** The output files give `passes` flags for H1, H2, H3 but do not restate the full hypothesis wording for H3, so quote the hypothesis text from your own proposal, not from these files.
