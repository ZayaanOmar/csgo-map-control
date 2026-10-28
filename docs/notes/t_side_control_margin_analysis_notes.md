# T-Side Control Margin and Bombsite-Concentration Analysis: Working Notes

Date written: 26 Oct 2026
Project: CS:GO map control on de_mirage (graph-based nav mesh influence field, ESTA dataset)
Purpose of this file: a single reference to pull methods, results and discussion points from for the final write-up. Numbers in the results section are copied from the run output you shared (`_summary.json` printout, area-breakdown bombsite rows, and the formal-test printout). Anything derived by hand or interpretive is labelled as such.

---

## 0. Key numbers at a glance

| Quantity | Value |
|---|---|
| Rounds analysed | 7,393 (3,381 T wins, 4,012 T losses; T win rate 45.7%) |
| Active-footprint threshold | EPSILON = exp(-2) = 0.1353, SIGMA = 500 |
| Mean active node fraction (whole map) | 0.514 |
| Mean active node fraction (bombsite A+B nodes) | 0.706 |
| Cohen's d, full map | 1.618 |
| Cohen's d, active footprint | 1.894 |
| Cohen's d, bombsite (unfiltered) | 1.113 |
| Cohen's d, bombsite (active-filtered) | 1.259 |
| Cohen's d, BombsiteA / BombsiteB (active-filtered) | 0.702 / 0.715 |
| AUC, active footprint vs bombsite-active | 0.917 vs 0.821 (DeLong z = 20.61, p underflows to 0) |
| LR test, adding bombsite term to active-footprint model | LR = 131.77, df = 1, p = 1.7e-30 |
| Area-name rank by T-win minus T-loss control (of 27), FIRST run only (raw awpy names, unweighted; superseded, rerun pending) | BombsiteA #1 (+0.0600), BombsiteB #4 (+0.0444) |

One-line conclusion: whole-active-footprint control is a clearly stronger single predictor of T round outcome than bombsite-only control, so the strict form of the hypothesis (T needs only bombsite control) is not supported. Bombsite control nonetheless adds significant information beyond the active footprint, and the two bombsites are among the most outcome-differentiating named areas on the map.

---

## 1. Research question and hypothesis

The CT-side Metric 3 (Control Margin) shows that CTs hold more spatial control, averaged over the round, in rounds they win than in rounds they lose, and that the effect is larger when averaging over the active footprint than over the whole map.

The question for the T side: does T need broad control (the whole active footprint), as CT appears to, or is control concentrated at the bombsites sufficient?

Working hypothesis (stated before running anything): contrary to CT, T only needs more control over the bombsite, rather than over the entire active footprint, to win rounds.

Operationalisation used here: if the hypothesis holds, a T control margin restricted to bombsite nodes should separate T-win from T-loss rounds at least as well as the active-footprint margin, despite being computed over far fewer nodes. This is a statement about relative discriminative strength. It is not a direct test of necessity or sufficiency (see Limitations and Section 7).

---

## 2. Methods

### 2.1 Inputs

- `outputs/control_fields/{match}.csv`: per frame, per nav node (`area_id`): `control_val` (CT-perspective sigmoid control), `ct_mu`, `t_mu` (team-level summed Gaussian influence).
- `outputs/round_metadata/{match}.csv`: `round`, `winning_side`.
- Static bombsite node sets for BombsiteA and BombsiteB (`de_mirage_bombsite_nodes.json`, keys "A" and "B"; path is the `BOMBSITE_NODES_PATH` constant in the script).
- `data/de_mirage_area_assignment.csv` (via `scripts/area_assignment.load_assignment`) for `area_id -> assigned_area`: all 903 nodes in 27 areas (26 awpy names plus ApartmentsRamp), with the 72 formerly blank nodes reassigned. `awpy.data.NAV["de_mirage"]` supplies tile footprints for area weighting.

Dataset: 278 matches, 7,393 rounds (from project inspection notes; the 7,393 matches the `n_rounds_total` in the T-side summary).

### 2.2 T-perspective control value

`control_val` is the CT-perspective value `c(v,f) = expit(K * (ct_mu - t_mu))` with K = 1, so 1 means full CT control and 0 means full T control. The T-perspective value is the complement:

    t_control_val(v,f) = 1 - control_val(v,f) = expit(K * (t_mu - ct_mu))

Over the same node set, `A_T(f) = 1 - A_CT(f)` exactly. The full-map and active-footprint T margins are therefore mirror images of the CT results, not independent information. They are computed anyway so the T hypothesis can be stated and tested in T's own terms, and as the baseline for the bombsite-restricted metrics, which are the genuinely new content.

### 2.3 Margin definitions

Node sets: V = all 903 nodes; V_active(f) = nodes with `ct_mu + t_mu > EPSILON` in frame f (threshold on the raw pre-sigmoid sum, not on `control_val`, because the sigmoid maps both "nobody nearby" and "equal players nearby" to 0.5); V_A and V_B = BombsiteA and BombsiteB nodes; V_site = V_A union V_B.

| Margin (column) | Nodes averaged per frame | Needs `ct_mu`/`t_mu` |
|---|---|---|
| `control_margin_full_t` | V | no |
| `control_margin_active_t` | V_active(f) | yes |
| `control_margin_bombsite_all_t` | V_site (also `..bombsiteA_all_t`, `..bombsiteB_all_t`) | no |
| `control_margin_bombsite_active_t` | V_site intersect V_active(f) (also `..bombsiteA_active_t`, `..bombsiteB_active_t`) | yes |

Why the intersection with the active footprint: in a round decided at one site, the other site's nodes sit near 0.5 all round because nobody is near them. Averaging both sites unfiltered therefore dilutes the signal toward 0.5, which is the same dilution problem the CT-side active-footprint fix addresses, at bombsite scale. Intersecting with the active footprint drops an untouched site's nodes from that frame's average. `control_margin_bombsite_active_t` is the primary metric for the hypothesis.

Consequence: if a frame has zero active bombsite nodes it contributes nothing to that round's bombsite average. A round with no active frames at a given site for the whole round gets NaN for that site's metric and drops out of that site's test (this happened for 33 rounds at BombsiteB, see Results).

Scope decision (confirmed): computed over all rounds, active-footprint filtered, rather than conditioning each round on the site where a plant occurred.

Per-round diagnostics also recorded: `active_node_fraction` (whole map) and `bombsite_active_node_fraction`, `bombsiteA_active_node_fraction`, `bombsiteB_active_node_fraction` (fraction of those nodes that were active, per frame, averaged over the round).

### 2.4 Two-stage averaging

Every margin is the mean over frames of the per-frame mean over the relevant nodes (frame mean first, then round mean of frame means). A flat mean over all matching rows would silently upweight frames with a larger matching node set, which matters for the "active" variants because the node set changes frame to frame.

### 2.5 Round grouping and univariate tests

Grouping is from T's perspective: "T win" = `winning_side == "T"`, "T loss" = `winning_side == "CT"`. This is the opposite direction from the CT script, where win/loss refers to CT.

For each margin: mean in each group (mu_win, mu_loss), Welch's t-test (unequal variances), and Cohen's d using `(mu_win - mu_loss) / sqrt((var_win + var_loss) / 2)`. The one-sided direction of interest is mu_win > mu_loss. The comparison of d(active_footprint) with d(bombsite_active) written to `_summary.json` is descriptive only.

### 2.6 Area-name breakdown (the "where" diagnostic)

For every assigned area (from `de_mirage_area_assignment.csv`), mean `t_control_val` over rounds, separately for T-win and T-loss rounds, and their difference. Averaging is again two-stage (frame, then round). Within a frame, an area's nodes are pooled with a footprint-weighted mean (weight = tile |width x height| from NAV), matching the assignment workflow's `to_areas()`, so a small tile does not count as much as a large one and multi-tile areas are not overweighted by the frame mean. A flag (`USE_FOOTPRINT_WEIGHTS`) switches back to the plain node mean. The output also records `n_formerly_blank_nodes` per area, and per-match cache files carry an `area_scheme` tag so caches written under a different area definition or weighting are recomputed. Raw (unfiltered) `t_control_val` is used because this table is exploratory and descriptive, not a confirmatory test. Bombsite rows are flagged and ranked by the win-minus-loss difference.

### 2.7 Formal comparison of the two margins

The informal comparison of two Cohen's d values has no sampling distribution for their difference, and the two margins are measured on the same rounds (paired, not independent samples). Two formal procedures were therefore implemented in `metric3_control_margin_t_side_formal_tests.py`, both restricted to rounds with non-null values in both `control_margin_active_t` and `control_margin_bombsite_active_t`:

**Test 1: nested logistic regression with a likelihood-ratio test.**

- Model A: `logit P(T win) = b0 + b1 * active_z`
- Model B: `logit P(T win) = b0 + b1 * active_z + b2 * bombsite_active_z`
- Predictors are z-scored (this does not change the LR statistic; it makes the coefficients comparable). Both models are fit on identical rows.
- LR = 2 * (llf_B - llf_A), compared to chi-square with 1 df. This asks whether bombsite control explains anything beyond active-footprint control.
- Model B is also refit with cluster-robust (sandwich) standard errors clustered on `match_id`, because rounds within a match are not independent. This corrects the per-coefficient p-values only. It does not correct the LR test itself (that would need a mixed-effects model).

**Test 2: DeLong's test for two correlated AUCs.**

Each margin is treated as a score for classifying T win vs T loss. DeLong et al.'s (1988) nonparametric method gives the variance of the AUC difference while accounting for the correlation between the two scores on the same rounds. The implementation is the direct O(n_win x n_loss) version using numpy broadcasting.

### 2.8 Implementation and validation notes

- Scripts: `metric3_control_margin_t_side.py` (metrics, area breakdown, summary), `visualize_metric3_control_margin_t_side.py` (7 figures), `metric3_control_margin_t_side_formal_tests.py` (Tests 1 and 2).
- Skip-if-exists caching is schema-aware (a cached per-match CSV is reused only if it already contains the bombsite columns). The area breakdown has its own per-match cache so either output can be recomputed independently.
- Active-footprint columns degrade to NaN with a printed note if `ct_mu`/`t_mu` are absent from a match's control field CSV.
- `_area_breakdown.csv` sits in the same directory as the per-match CSVs, so loaders skip any file whose name starts with an underscore.
- The metric and visualization scripts were syntax-checked in the sandbox only; the metric script was run by you on the real data. I have not seen output from the visualization script.
- The formal-tests script was validated on synthetic data with known effect sizes (the more separable margin was correctly detected as better by both tests), and its AUC calculation matched `sklearn.metrics.roc_auc_score` to machine precision. You then ran it on the real data. The DeLong variance/z-test part was not independently cross-checked against another implementation.
- New dependency for the formal tests: `statsmodels` (`pip install statsmodels --break-system-packages`).
- EPSILON note: the CT script's docstring describes exp(-4.5) as the default, but its constant is set to exp(-2) and the T-side run used exp(-2) (confirmed by `epsilon: 0.1353` in the summary). Any CT-side figure quoted alongside these T-side numbers should be from a run with the same EPSILON. Project notes record an earlier CT active-footprint d of 1.84 and mean active fraction of 0.678, which probably came from a different threshold, so they are not directly comparable to the T-side 1.894 and 0.514 here.

---

## 3. Results

### 3.1 Univariate win/loss comparison (T win vs T loss)

| Margin | n win | n loss | mu_win | mu_loss | Gap (win - loss) | Welch t | Welch p | Cohen's d |
|---|---|---|---|---|---|---|---|---|
| Full map | 3381 | 4012 | 0.4947 | 0.4706 | 0.0241 | 68.52 | 0 (underflow) | 1.618 |
| Active footprint | 3381 | 4012 | 0.4988 | 0.4434 | 0.0554 | 80.03 | 0 (underflow) | 1.894 |
| Bombsite A+B, unfiltered | 3381 | 4012 | 0.4724 | 0.4185 | 0.0539 | 47.52 | 0 (underflow) | 1.113 |
| Bombsite A+B, active-filtered | 3381 | 4012 | 0.4797 | 0.3993 | 0.0804 | 53.43 | 0 (underflow) | 1.259 |
| BombsiteA, active-filtered | 3381 | 4012 | 0.4611 | 0.3997 | 0.0613 | 29.60 | 2.6e-179 | 0.702 |
| BombsiteB, active-filtered | 3352 | 4008 | 0.4353 | 0.3865 | 0.0489 | 29.74 | 5.8e-178 | 0.715 |

All six comparisons have mu_win > mu_loss. "p = 0" means the value underflowed below floating-point range, not that it is exactly zero. With n around 7,400 the p-values are uninformative on their own; the effect sizes are the useful quantity.

Derived by hand (gap divided by d, approximate pooled SD of the round-level margin): full map about 0.015, active footprint about 0.029, bombsite active about 0.064, BombsiteA about 0.087, BombsiteB about 0.068.

BombsiteB has 33 fewer rounds than BombsiteA (29 T-win, 4 T-loss): rounds in which BombsiteB had no active frame at all, so the metric is NaN.

Level observation: T's mean active-footprint margin in T-win rounds is about 0.50 (neutral) and in T-loss rounds about 0.44 (CT-leaning). T winning rounds are, on average, roughly balanced rather than T-dominated.

### 3.2 Effect-size comparison (from `_summary.json`)

| | Cohen's d |
|---|---|
| Active footprint | 1.894 |
| Bombsite, active-filtered | 1.259 |
| Difference (bombsite - active) | -0.636 |

The bombsite-restricted margin has the larger raw win-loss gap (0.080 vs 0.055) but the smaller d, meaning its round-to-round variability is much larger (about 0.064 vs 0.029 in SD terms, derived above).

### 3.3 Activity

Mean active node fraction: 0.514 whole map; 0.706 over bombsite A+B nodes. Bombsite nodes are more often within the active footprint than a typical map node.

### 3.4 Area-name breakdown (bombsite rows only; the full ranking was not shared with me)

| Area | Rank (of 27) | Mean T control, T-win rounds | Mean T control, T-loss rounds | Difference |
|---|---|---|---|---|
| BombsiteA | 1 | 0.4761 | 0.4161 | +0.0600 |
| BombsiteB | 4 | 0.4666 | 0.4223 | +0.0444 |

These bombsite rows are from the FIRST run, which used raw awpy names (the 72 blank-name tiles collapsed into one `""` bucket, so the 27 was probably 26 named areas plus that bucket) and an unweighted node mean. The script now uses `de_mirage_area_assignment.csv` with footprint-weighted averaging; the area count is still 27 but the areas are different (ApartmentsRamp replaces the blank bucket, and the 72 tiles are spread across areas, including possibly the bombsites). The numbers and ranks above will change after a rerun and should not be quoted until then. Also note that the BombsiteA/BombsiteB *areas* in this table come from the assignment file, while the bombsite *margins* use `de_mirage_bombsite_nodes.json`; the two node sets are not guaranteed identical.

### 3.5 Formal comparison of active footprint vs bombsite-active

**Test 1: nested logistic regression (T win ~ margins, standardised predictors)**

| | Log-likelihood | AIC |
|---|---|---|
| Model A (active only) | -2681.89 | 5367.77 |
| Model B (active + bombsite active) | -2616.00 | 5238.00 |

Likelihood-ratio test: LR = 131.77, df = 1, p = 1.68e-30. Delta AIC = -129.77 in favour of Model B.

| Model B coefficient (per 1 SD) | Estimate | Odds ratio (derived) | p, naive SE | p, cluster-robust by match |
|---|---|---|---|---|
| `active_z` | +2.575 | about 13.1 | 2.4e-298 | 0 (underflow) |
| `bombsite_active_z` | +0.534 | about 1.71 | 1.1e-29 | 3.9e-28 |

The active-footprint coefficient is about 4.8 times the bombsite coefficient. Clustering on match does not change the conclusion for either coefficient.

**Test 2: DeLong's test for correlated AUCs**

| | AUC |
|---|---|
| Active footprint | 0.9168 |
| Bombsite, active-filtered | 0.8210 |
| Difference | +0.0958 (z = 20.61, p underflows to 0) |

The header line reporting the number of rounds used by the two tests was not in the output you shared. Since `bombsite_active` and `active_footprint` both had all 7,393 rounds in the univariate tests, the expectation is that all 7,393 were used; confirm in `_formal_comparison.json`.

---

## 4. Discussion

**The strict hypothesis is not supported.** If T needed only bombsite control, the bombsite-restricted margin should have separated wins from losses about as well as the active-footprint margin. It separated them clearly worse on every measure: Cohen's d (1.26 vs 1.89), AUC (0.82 vs 0.92) and the DeLong test of the AUC difference. The same holds when each site is taken alone (d around 0.70 each), so combining both sites is better than either one, which fits a round being decided at either site.

**A weaker version is supported.** Adding bombsite control to a model that already contains active-footprint control improves fit substantially (LR = 131.8, delta AIC about -130), so bombsite control is not just a noisier copy of overall footprint control. It carries independent information about the round outcome. It is also a much smaller contributor (standardised coefficient about 0.53 vs 2.57). The area breakdown agrees in spirit: BombsiteA and BombsiteB ranked first and fourth of 27 areas by the T-win minus T-loss control difference in the first (raw-name, unweighted) run, which suggests they are unusually outcome-relevant locations even though a bombsite-only metric is not enough. This needs re-confirming with the area assignment applied.

**Reading the two tests together.** DeLong answers "which single metric would you choose to predict the outcome" (active footprint). The LR test answers "does the second metric add anything" (yes). These are different questions and the results are consistent.

**Interpretation, not tested here:** the bombsite metric's larger variance relative to its gap is plausibly because whether and when T reaches and holds a site varies a lot between rounds, while overall footprint dominance is steadier. Bombsite nodes being active 70.6% of the time versus 51.4% for the whole map is plausibly because bombsites are focal points. Neither explanation was examined directly.

---

## 5. Limitations and caveats

1. **Outcome contamination from whole-round averaging.** Each margin averages over all frames, including late-round states after kills and after the plant. Winners tend to hold more control late in the round partly because they are winning, and T control of a bombsite after a plant is partly a consequence of having planted. The high AUC (0.92) probably reflects this and should be described as an association with round outcome, not as early-round prediction. This is the same concern that motivates the early-window analyses already planned for the project.
2. **Overlapping node sets.** The active footprint includes the active bombsite nodes, so the two metrics are not disjoint. A cleaner contrast would use bombsite nodes against the active footprint excluding bombsite nodes (not run).
3. **Non-independence of rounds.** Rounds within a match share teams and context. The Welch tests and the LR test treat rounds as independent. Only the per-coefficient p-values in Test 1 were cluster-corrected. DeLong's test as implemented also assumes independent cases. The conclusions are unlikely to flip given the size of the effects (z = 20.6, LR = 131.8), but the p-values themselves are optimistic.
4. **Direction of the nesting.** Test 1 asks whether bombsite adds to active footprint. The reverse (does the active footprint add to a bombsite-only model) is arguably the more direct test of bombsite sufficiency and was not run.
5. **Descriptive, not causal.** Nothing here shows that holding control causes a win.
6. **Single map, one value each of SIGMA (500) and EPSILON (exp(-2)).** The sensitivity analysis over sigma {300, 500, 750} and over EPSILON has not been done for these metrics.
7. **Area breakdown results are stale.** The reported area ranks come from the first run (raw awpy names, unweighted). The script now applies the area assignment with footprint weighting; rerun before quoting. Reassigned tiles that are ambiguous (low `vote_share` under the `touch_vote` method) could still affect specific areas; `n_formerly_blank_nodes` shows which areas are most exposed.
8. **Multiple comparisons.** Six univariate tests and two formal tests were run with no correction. Given the effect sizes this does not affect any conclusion.

---

## 6. Suggested next steps

1. Run the reverse-direction LR test (add active footprint to a bombsite-only model) and a disjoint-node-set version (bombsite vs active footprint excluding bombsite).
2. Report the correlation between the two margins (not computed) to put the Model B coefficients in context.
3. Repeat Tests 1 and 2 on an early window of each round to remove outcome contamination.
4. Run the same pair of tests on the CT side for a direct statement of the CT/T asymmetry (information gain from a bombsite term for each side). Not yet implemented.
5. Rerun `metric3_control_margin_t_side.py` (area cache recomputes automatically) and update Sections 0, 3.4 and 4 with the new ranks; optionally rerun with `USE_FOOTPRINT_WEIGHTS = False` to see whether the weighting changes the ranking.
6. Optional: a mixed-effects logistic model (random intercept per match) for a fully cluster-corrected LR-style test.
7. Confirm the visualization script's seven figures render as intended.

---

## 7. Files and reproduction

Order of execution:

    python metric3_control_margin_t_side.py
    python visualize_metric3_control_margin_t_side.py
    pip install statsmodels --break-system-packages
    python metric3_control_margin_t_side_formal_tests.py

Outputs under `outputs/metrics/control_margin_t_side/`:

- `{match}.csv`: per-round margins and diagnostics
- `_summary.json`: univariate statistics and the effect-size comparison block
- `_area_breakdown.csv` and `_area_breakdown_by_match/{match}.csv`: named-area differential table and its per-match cache
- `_formal_comparison.json`: Tests 1 and 2
- `figures/01..07_*.png`: boxplots, histograms, mean bars with CI, margin vs round, active node fractions, effect-size comparison, area breakdown

---

## 8. Citations to verify

I do not have search or a database available, and I may misremember details, so please check each of these against the original source before citing:

- DeLong, E. R., DeLong, D. M., and Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics, 44(3), 837-845.
- Welch, B. L. (1947). The generalization of "Student's" problem when several different population variances are involved. Biometrika, 34, 28-35.
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum Associates.
- Fernandez and Bornn (2018) and Xenopoulos et al. (2020), the methodological foundations named in the project notes; exact titles and venues are not recorded here.
