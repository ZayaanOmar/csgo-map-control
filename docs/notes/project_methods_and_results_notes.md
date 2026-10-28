# de_mirage map control: analyses, methods, results and caveats

Working notes for the final report (IEEEtran conference template, maximum 8 pages, author-year citations).

**How to use this file.** It is a source document, not report text. Every number comes from your real runs unless marked *(synthetic)*, which means it comes from a test on generated data and says nothing about the real dataset. Section 8 lists limitations to state; section 9 lists claims from earlier in the sessions that were wrong or too strong, so they are not reused by accident. No literature is cited here. Citations need to be added from sources you have checked yourself.

---

## 1. What was done and why

After finding that aggregate map control predicts little about round outcomes, the project moved to **descriptive and diagnostic analyses** using the same pipeline outputs. A weak predictor is not a weak framework: round outcomes are dominated by duels, utility and economy, and control is partly an outcome of the same positioning.

| # | Analysis | Question | Headline result | Report priority |
|---|---|---|---|---|
| A | Team site preference | Do teams differ in which site they plant on? | Yes, but the effect is small (between-team SD of P(A) ≈ 0.033); individual team estimates are not shown to be stable | Medium: one short paragraph plus a figure or table |
| B | Team control signatures | Do teams have distinguishable early-round positioning? Does the influence field add anything over raw occupancy? | Teams are identifiable well above chance; the field is **not** better than occupancy at this | High: the field-vs-occupancy finding matters for the thesis |
| C | Eco vs full-buy positioning | Does a team position differently when on eco, compared with itself in the same match? | Yes, in every cell tested; the shifts are located and interpretable | High: strongest and cleanest result |
| D | Position heatmaps | Visual illustration of C | Tool built and validated; G2 used as a worked example | Low-medium: one or two illustrative figures |

Key design principle across B and C: **resample by match, never by round**, because rounds in a match share an opponent and game plan.

---

## 2. Dataset and pipeline facts established

### Scale
- 278 de_mirage matches, 7,393 rounds, 3,794 bomb plants (A 2,314, B 1,480).
- T plants in 51.3% of T-side rounds (3,599 rounds have no plant). Pooled **P(A | plant) = 0.610**.
- Each team's T rounds per match: median 15 (min 1, max 36 with overtime).

### Team names
- 92 raw names merge to **60 canonical teams** through `data/team_aliases.json`. Case and punctuation variants merge automatically; the rest were merged by an explicit seed list (e.g. NAVI/NAVI GGBET/Natus Vincere, Heroic/Heroic GGBET, VP Parimatch/Virtus.pro, EIQ PRG/Entropiq, CPH Flames/Copenhagen Flames).
- "Team" means an **organisation**, not a fixed roster. Rosters changed over the period.
- Median team has 49 T-side rounds (mean 123); 36 teams have at least 40.

### Time axis (a correction worth stating in the methods)
- Frames are **64 ticks apart at 127 or 128 ticks per second, about 0.5 s**. Earlier project notes assumed 1 s per frame; that was wrong.
- All times are `(tick − freeze_time_end_tick) / 128`. The 127/128 difference is under 1%.
- Mean round: about 169 frames (about 85 s) in an 8-demo sample.

### Area resolution
- `data/de_mirage_area_assignment.csv` resolves all 903 nav tiles into **27 named areas** (`assigned_area`). The 72 blank tiles were filled by `split_nearest` (42), `touch_vote` (20), `hand_cluster` (8) and `expanded_25` (2). A new label, **ApartmentsRamp** (8 tiles), is separate from Apartments (23). Only 3 tiles have a vote share below 0.6.
- Areas differ greatly in node count (CTSpawn 133, BombsiteA 117, BombsiteB 72, ... Balcony 4), so area mass shares are node-weighted and absolute shares are not physical-area shares. Team comparisons are unaffected because every team uses the same nodes.

### Economy fields
- Buy labels (`Full Eco`, `Semi Eco`, `Semi Buy`, `Full Buy`) are non-overlapping bands of team equipment value at freeze end, at **$5k / $10k / $20k**.
- **Pistol rounds (1 and 16) are labelled "Full Eco"**: 1,106 of 1,112 pistol team-rounds. They must be excluded from any eco class. Overtime (round > 30, 203 rounds) is also excluded, because it starts every player at a fixed $10k.
- Non-pistol share of team-rounds by class (CT / T): eco 6.4% / 8.3%; semi-eco 7.3% / 7.0%; semi-buy 9.8% / 13.9%; full 76.5% / 70.8%.
- Join to `round_metadata`: 7,393 of 7,393 rounds matched with zero mismatches in winner or team names.
- Sanity check on outcomes (T win rate, non-pistol): eco vs full-buy CT 7%; full-buy vs eco CT 92%; full vs full 45%. Eco-vs-eco never occurs.

### Radar orientation (verified from nav-mesh centroids)
T spawn is the right-middle green box; CT spawn the bottom-left box; **BombsiteA is the bottom-centre red box; BombsiteB the top-left red box**. Mid areas (TopofMid, Middle, Catwalk) sit in the centre; BackAlley, House and SideAlley run along the top; PalaceAlley, PalaceInterior and TRamp along the right and bottom right.

---

## 3. Analysis A: team site preference

**Question.** Do teams differ in P(A | plant) on the T side beyond chance and match-to-match variation?

**Unit and data.** One unit is one team's T-side plants in one match. `round_metadata` + `bomb_plants`; teams need at least 20 plants (33 teams, 507 units, 3,560 of 3,794 plants).

**Method.**
1. **Heterogeneity test.** Chi-square-type statistic of team P(A) against the pooled rate. The null shuffles team labels across *units* (10,000 permutations), which keeps match-level clustering in the null.
2. **Shrinkage.** Beta-binomial empirical Bayes prior fitted by maximum likelihood to effective counts (counts divided by a quasi-binomial dispersion factor φ, floored at 1). 90% credible intervals per team.
3. **Stability.** Random match-level split-half correlation of team P(A), compared with a label-permutation null (2,000 permutations), with Spearman-Brown projection.

**Validation.** The script asserts unique keys, a complete team-name mapping and a lossless join of plants to rounds, and prints the T rounds per team-match (median 15, as expected) as a side-swap check. Unlike the later analyses, it was **not** run on synthetic data with known truth before the real run; its output was checked for plausibility afterwards. Say so if the report describes validation.

**Results.**

| Quantity | Value |
|---|---|
| Heterogeneity statistic / null mean (95th pct) | 49.4 / 31.4 (44.2) |
| Permutation p | **0.0151** (0.0113 excluding pistol rounds) |
| Quasi-binomial dispersion φ | 1.00 (floored; see section 9) |
| Prior mean / concentration | 0.611 / 223.1 |
| **Between-team SD of P(A)** | **≈ 0.033** |
| Split-half r (median over 200 splits) / null 95th pct | 0.253 / 0.364 |
| Split-half p / Spearman-Brown r | 0.128 / 0.403 |
| Shrunk team P(A) range | 0.566 to 0.653 (raw range 0.455 to 0.773) |
| Teams whose 90% interval excludes the pooled rate | **2**: Ninjas in Pyjamas (0.566, CI 0.524 to 0.608; raw 0.500) and Team Vitality (0.653, CI 0.616 to 0.688) |

Shrinkage is heavy because the between-team SD is small: the implied weight on a team's own data is about 0.11 below 40 plants, 0.29 for 40 to 100, and 0.41 for 100 to 400. The forest plot therefore understates the spread of the raw proportions.

**Safe claims.** Teams differ in plant-site preference more than chance and match-level clustering explain (p = 0.015), but the effect is modest (SD ≈ 0.033 around 0.61). Only two teams are individually distinguishable from the pool. The split-half test is **inconclusive** (p = 0.13), so individual teams should not be described as having a stable site preference.

**Limitations specific to A.**
- P(A | plant) records the site where T *succeeded* in planting, which mixes intent and execution. A team that tries B often but loses those rounds looks A-leaning.
- Plant rate is a separate quantity and was not modelled (it mixes buy type, round outcome and style).
- The split-half test is random, not chronological, because no match dates were extracted.

---

## 4. Analysis B: team control signatures

**Question.** Do teams have distinguishable positioning in the early round, and does the influence field add information over raw occupancy?

### 4.1 Signature definition
- **Signature** = a side's *share* of mass over the 27 areas, per round, averaged over the frames in a window; it sums to 1.
- Two versions: **field** (`ct_mu` / `t_mu` from `control_fields`, summed over the nodes in each area and normalised per frame) and **occupancy** (players per area from `position_maps`, normalised by alive count). Occupancy is the model-free baseline.
- `control_val` was deliberately not used: it gives 0.5 both for "nobody there" and "evenly contested".
- **Windows.** `pre_first_loss`: from 5 s to the first death (first frame where either side's alive count drops), kept only if at least 12 s long. `early_5_20` and `mid_20_40`: fixed windows. First death is inferred from `position_maps` (alive players only) because there is no kills table.

### 4.2 Builder output (`build_team_signatures.py`)
- 80,594 signature rows from 278 matches, 0 failures.
- 5,435 of 7,393 rounds (**73.5%**) have a usable `pre_first_loss` window. Window length: median 28.2 s, 10th/25th/75th/90th percentiles 14.4/18.5/49.4/67.1 s.
- Teams with at least 30 usable rounds: 33 (T side), 32 (CT side). `MIN_ROUNDS = 30`.
- Per-round Jensen-Shannon distance, field vs occupancy: median 0.358. This is mostly smoothing, not different information, so it is **not** evidence that the field adds value.
- Pooled field shares (pre_first_loss): CT: BombsiteA 0.282, CTSpawn 0.177, BombsiteB 0.140, TicketBooth 0.109. T: TopofMid 0.159, BackAlley 0.155, SideAlley 0.097, PalaceAlley 0.094.

### 4.3 Identification test (`analysis_team_signatures.py`)
- Each team's matches are split at random into two halves (by match). A team is correctly identified if its half-A signature is nearest (Jensen-Shannon distance) to its own half-B among all teams.
- Test statistic = mean over 20 random splits. The null permutes team labels across team-match units 1,000 times and recomputes the *same* statistic. Holm adjustment over the four primary cells. Minimum 8 rounds per half.
- **Design lesson.** The first version compared the median of 200 dependent observed splits against single-split nulls, which gave p ≈ 0.67 on null data (miscalibrated). The final version uses an identical statistic for observed and null; on null data p-values spread across 0.04 to 0.97 *(synthetic)*, and on planted differences every cell reached the permutation floor *(synthetic)*.

**Results, pre_first_loss window (primary).**

| Side / method | Teams | Chance | Accuracy | Null 95th pct | Between / within distance | Holm p |
|---|---|---|---|---|---|---|
| T / field | 33 | 0.030 | 0.233 | 0.064 | 0.125 / 0.100 | 0.004 |
| T / occupancy | 33 | 0.030 | 0.285 | 0.060 | 0.162 / 0.131 | 0.004 |
| CT / field | 32 | 0.032 | 0.230 | 0.066 | 0.085 / 0.066 | 0.004 |
| CT / occupancy | 32 | 0.032 | **0.436** | 0.060 | 0.160 / 0.121 | 0.004 |

All primary p-values are at the permutation floor (0.001; the table shows Holm-adjusted). Accuracy is 7 to 14 times chance.

Exploratory windows (accuracy, field vs occupancy): early_5_20: T 0.281 vs 0.354, CT 0.299 vs 0.459; mid_20_40: T 0.246 vs 0.291, CT 0.235 vs 0.309. **Occupancy is at least as accurate as the field in all six window-by-side comparisons**, by a large margin on the CT side. This difference was not tested formally.

Other results:
- **Field vs occupancy agreement:** Mantel Spearman ρ between team-distance matrices: T 0.929, CT 0.792 (both p = 0.0002). Mean team-to-team distance: field 0.099 vs occupancy 0.130 (T); 0.070 vs 0.132 (CT).
- **Nearest-neighbour stability** (bootstrap over team-matches): 0.30 (T field), 0.33 (T occupancy), 0.40 (CT field), 0.41 (CT occupancy). Fine-grained "who is most similar to whom" structure is unreliable; do not interpret specific dendrogram branches.

**Interpretation.** Teams have distinguishable early positioning, and CT setups appear more team-specific than T movement (not tested). The influence field does not add team-identifying information over occupancy. A likely reason, not isolated: σ = 500 smooths away small deliberate positions such as SnipersNest, which is strongly team-specific in occupancy and nearly flat in the field.

**Required limitation (your decision, 24 Oct).** The era-controlled check was **not run**. Teams that played in the same period or tournament share a meta, so some identifiability may reflect period or tournament instead of team style. `build_match_info.py` and an era-controlled null exist in the code (tested *(synthetic)*: an era-only world is rejected by the plain test but not by the era-controlled one) but were not applied to the real data. The report must state this.

**Team-level observations from the heatmaps (by eye, not tested).** The two teams from analysis A with intervals excluding the pooled rate look consistent with their T-side signatures: NiP is heavy on TopofMid and light on PalaceAlley and BombsiteA; Team Vitality is heavier on PalaceAlley, PalaceInterior and BombsiteA. A group of lower-sample teams with a BackAlley-heavy T signature averaged a raw P(A) of 0.57 against 0.69 for a Palace/TRamp-heavy group (9 of 13 teams in the expected direction). This is post-hoc, small and suggestive only.

---

## 5. Analysis C: eco vs full-buy positioning

**Question.** Does a team position differently on eco than on full-buy, controlling for team, match and opponent?

### 5.1 Design (`analysis_eco_vs_full.py`)
- **Conditions** (non-pistol, non-overtime), from one team's side: **E** = own eco vs a full-buy opponent; **F** = own full-buy vs a full-buy opponent; **A** = own full-buy vs an eco opponent. Primary contrast **E vs F**; secondary A vs F. E on the T side and A on the CT side are the same 507 rounds seen from both sides (and CT-E / T-A the same 379), so the four primary cells are not independent.
- **Unit** = one team's rounds on one side in one match. Only units containing *both* conditions are used, so team style, opponent, tournament and era cancel by construction.
- **Pooling.** Unit means combined with weights equal to the harmonic mean of the unit's two round counts.
- **Omnibus statistic** D = Jensen-Shannon distance between pooled signatures. **Null:** condition labels shuffled among the rounds of the same unit (2,000 permutations; smallest possible p = 0.0005). Holm over the four primary cells.
- **Area shifts** = pooled share (condition) − pooled share (F), with 95% bootstrap intervals over units (500 resamples; one unit per match per side, so units are independent) and **max-T adjusted p-values** that control the family-wise error rate across the 27 areas. The "same-sign" fraction is the share of teams (with at least 2 paired units) whose own shift matches the pooled sign.
- **Primary window `early_5_20`** (fixed 15 s). `pre_first_loss` and `mid_20_40` are sensitivity checks. Reason: `pre_first_loss` windows are about 11% shorter on eco rounds (eco-side 33 to 34 s vs 37.5 s for full-buy), which mechanically changes average positions. Usable-window rates by buy class were similar (72% to 77%), so there is no exclusion bias, and `early_5_20` is usable for 100% of rounds.
- **Sensitivity variant:** drop rounds 2 to 4 of each half. 43.4% of T-eco rounds (37.2% CT) are rounds 2 to 4 of their half, against 13.0% of full-buy rounds, so round position is a real confound.

### 5.2 Validation *(all synthetic)*
- A planted +0.03/−0.03 shift on two areas: max-T flagged exactly those two areas in all four primary cells; estimates +0.031 and −0.028; intervals covered the truth; the anti-eco contrast stayed null.
- No effect: no primary cell rejected.
- A world where eco-prone teams also have very different baselines (a naive pooled comparison would be badly confounded): no area flagged. Over 16 such worlds (64 cells): omnibus rejection rate 4.7% at 0.05 and 1.6% at 0.01; max-T family-wise rate 4.7%. Naive unpaired D = 0.079 against paired D = 0.023, so pairing removes roughly 70% of an apparent difference that is purely team baseline.
- A figure-code bug (wrong column names) was found and fixed during these tests.

### 5.3 Feasibility on the real data
| Window | Side | Eco rounds in paired units | Paired team-matches | Teams with ≥ 12 rounds in both |
|---|---|---|---|---|
| early_5_20 | T | 507 | 369 | 17 |
| early_5_20 | CT | 378 | 289 | 11 |
| pre_first_loss | T | 377 | 286 | 11 |
| pre_first_loss | CT | 270 | 224 | 9 |

(The feasibility counts excluded only pistol rounds, whereas the final analysis also excludes overtime, which accounts for its slightly smaller full-round totals: 507 eco and 2,366 full rounds on the T side, 378 and 1,902 on the CT side.)

### 5.4 Results (E vs F, early_5_20, all rounds)

| Cell | Units / teams | Eco / full rounds | D | Null mean (95th) | p (Holm) |
|---|---|---|---|---|---|
| T / field | 369 / 52 | 507 / 2,366 | 0.1078 | 0.0226 (0.0347) | 0.0005 (0.002) |
| T / occupancy | 369 / 52 | 507 / 2,366 | 0.1298 | 0.0326 (0.0443) | 0.0005 (0.002) |
| CT / field | 289 / 54 | 378 / 1,902 | 0.0650 | 0.0181 (0.0256) | 0.0005 (0.002) |
| CT / occupancy | 289 / 54 | 378 / 1,902 | 0.1279 | 0.0347 (0.0445) | 0.0005 (0.002) |

- **Robustness.** Dropping rounds 2 to 4 of each half does not weaken the omnibus result: D = 0.126, 0.150, 0.071 and 0.128 in the same four cells, all p = 0.0005. `mid_20_40` and `pre_first_loss` windows and the anti-eco contrast are also at the floor in every all-rounds cell. Because every cell sits at the floor, effect sizes, not p-values, carry the information.
- **For scale only** (different estimators, so not a formal comparison): the eco shift above its noise floor (D minus null mean) is about 2.5 to 3.4 times the between-team gap in the identification test.
- **Area-level shifts (percentage points of mass share; max-T adjusted p in brackets).**
  - **T side:** TopofMid −5.7 (field; 19.3% to 13.6%; CI −7.2 to −4.4; p = 0.0005) and −5.5 (occupancy; 15.3% to 9.8%), consistent in about 36 of 42 teams. BackAlley +3.3 (field), SideAlley +1.7 (field) / +3.5 (occupancy), House +2.7 (occupancy), BombsiteA +1.5 (field), Middle −1.4 (field). PalaceAlley (−1.4 field, −2.5 occupancy) and, on occupancy, BackAlley (+1.9) do **not** survive adjustment.
  - **CT side:** TicketBooth −2.6 (field; 10.7% to 8.1%; CI −3.3 to −1.9) and −2.8 (occupancy; 7.6% to 4.8%), consistent in about 33 of 37 teams. Connector −2.3 and Truck −1.9 (occupancy; consistent in about 84% and 92% of teams). BombsiteA +3.6 on occupancy (CI 1.4 to 5.6) but consistent in only 57% of teams and not significant on the field (+1.5, CI −0.3 to 3.5).
  - Team counts behind the same-sign fractions were read off the fractions (42 T-side, 37 CT-side teams); the exact counts are in the `n_teams_sign` column of `eco_vs_full_area_shifts.csv`.
- **Field vs occupancy agreement on area shifts:** T r = 0.892; CT r = **0.439**, with sign flips (CTSpawn −2.0 on the field vs +1.95 on occupancy; SnipersNest and Shop rise on the field only; BombsiteB falls on occupancy only). This is unexplained.

### 5.5 Interpretation and caveats
- **Safe claims.** Early (5 to 20 s) positioning on full eco differs from the same team's full-buy positioning in the same match against the same full-buy opposition, and the result is robust to removing post-pistol rounds. On the T side the clearest shift is away from mid (TopofMid) toward the alley routes; on the CT side the clearest shift is away from TicketBooth.
- **Money, loadout and tactics are confounded.** Eco players lack rifles and AWPs, which changes playable angles, and teams may also choose different tactics. The data cannot separate these. Say "positioning differs on eco", not "teams choose X because of economy".
- **Matching is incomplete.** Pairs are matched on team, match and opponent buy, but not on score, loss streak, or round position beyond the post-pistol check.
- **CT field results are unresolved** (see above). Prefer occupancy for CT area-level claims. Untested hypothesis: eco loadouts move faster, which changes the one-second projected position used by the field.
- `early_5_20` includes frames after the first death in rounds where someone dies before 20 s, so the alive-player composition shifts. `pre_first_loss` avoids that but has the window-length issue. Both agree on the omnibus result.
- Eco means "Full Eco" only (under $5k, pistol rounds excluded); semi-eco and semi-buy are excluded from the primary contrast.

---

## 6. Position heatmaps (visualisation tool)

`plot_position_heatmaps.py` draws, for a chosen time *t* after freeze end, one PNG each for **T eco, T full-buy, CT eco, CT full-buy**, optionally for a single team (`--team "G2 Esports"`). Each PNG shows the mean number of alive players per round per 100×100 game units, over the awpy radar.

### 6.1 Definitions
- **Time and round selection.** Each round contributes the frame nearest *t* (within ±0.3 s); rounds that ended before *t* are dropped. Eco/full definitions are those of analysis C (against a full-buy opponent by default). Pooled mode keeps only team-matches with both conditions; team mode uses all of the team's rounds (`--paired` to restrict).
- **Subtitle fields:** round count; "(x of y lasted to t)" (x of the team's y candidate rounds lasted to *t*); alive players per round; opponent buy; scope; smoothing method; `LOW N` when a panel has under 15 rounds.
- **Colour scale.** Units are alive players per round per 100×100 units. A tight cluster with one player in it for one round adds about **0.16 / n** at its centre (n = rounds in the panel), so a hotspot value D corresponds to roughly D / 0.16 players per round standing there (exact for open ground; higher in corridors). Eco and full panels of one side and time share one scale (99.5th percentile; values below 4% of the maximum are transparent). The scale changes between times (G2 CT: 0.066, 0.042, 0.027 at 10/20/30 s) and between sides, so the same colour at different times means different densities. Zero density is drawn as the radar floor colour (1, 19, 48)/255 (the light radar's floor is (38, 50, 70)).
- **Noise.** With 27 rounds, one player-round adds about 0.006, so a 0.066 peak is about 11 player-rounds and carries roughly ±30% Poisson noise.

### 6.2 How the smoothing was settled (and why)
1. **Plain Gaussian** bled over walls and off the map.
2. **Clipped Gaussian** (`--geometry mask`): clipped to the nav-mesh walkable area with boundary-corrected smoothing. Fixed the map edge; still leaked across thin walls between parallel corridors.
3. **Graph kernel** (`--geometry graph`): mass spread along the nav graph with geodesic distance from the project's APSP matrix (symmetrised min of the two directions, since 33.27% of finite pairs are directionally asymmetric), normalised so each player integrates to one. Wall-proof but painted flat per nav tile (median tile 43×43 units), so adjacent tiles showed hard steps. On real data (G2 CT eco) the result looked blocky.
4. **Wall-aware diffusion** (`--geometry diffuse`, the default when `data/de_mirage_apsp_dict.pkl` exists): heat flows over the walkable area at 10-unit cells for the number of steps matching a Gaussian of width `--bandwidth` (default 100 units), with no flux into walls, **and** flow between touching tiles is blocked when their walking distance exceeds `--cut` (default 1.6) × straight-line distance + 40 units. Players are placed at their real positions, clamped into their own tile. Mass is conserved. Approved as the final style.

**Validation of the smoothing *(synthetic, using awpy's bundled unpatched graph, so percentages differ from the real patched graph)*.**
- Graph kernel: every source integrates to 1; a 4,000-player-round test map integrated to 1.0000 players per round.
- **Wall test** on 3,210 tile pairs 60 to 150 units apart in a straight line but more than 1,000 apart by walking (or unreachable): a plain Gaussian gives the far side 56% of the source tile's density; the graph kernel gives at most 1e-24.
- **Raster walls alone do not block leakage** (0.56 across the wall with `--cut 0`, the same as a plain blur) because nav tiles on either side of many walls abut or stack on the 2D radar; the graph test brings it to 0.011 (1% of pairs above 0.05). The graph test is essential.
- **Injection bug found and fixed:** rounding tile rectangles outward made neighbouring tiles share a border cell, so 18.7% of players started their heat in a different tile's cells and cell-level leakage was 5.75%. With cell-centre ownership and injection onto the player's own tile: 100% start in their own tile and cell-level leakage is 0.00%.
- True multi-level stacking is rare: only 28 of 903 tiles overlap another by more than 5%, nearly all at a different height; in those spots the 2D picture cannot separate levels.
- Hard steps between adjacent pixels (99.9th percentile as a share of peak): tile-flat 0.079, diffusion 0.021. The largest single jump in diffusion maps (0.50) occurs at blocked boundaries, by design.
- Colours are not comparable across geometry modes; each figure's second subtitle line names the smoothing used.

**One number to check on real runs.** The script prints the percentage of touching tile boundaries blocked by the graph test (9.1% on the unpatched test graph). Visible dark seams through corridors mean the cut is too aggressive (raise `--cut`); heat across a wall means too loose.

### 6.3 Worked example: G2 Esports (real data, readings by eye with approximate hotspot positions converted to radar coordinates and named from nav-tile lookups)
These figures were made with the earlier unmasked Gaussian version, so regenerate before using them.

| Panel | 10 s | 20 s | 30 s |
|---|---|---|---|
| T eco rounds (alive/round) | 49 (4.9) | 47 (4.3) | 40 (4.0) |
| T full rounds (alive/round) | 315 (4.9) | 313 (4.6) | 309 (4.2) |
| CT eco rounds (alive/round) | 27 (5.0) | 27 (4.6) | 26 (4.3) |
| CT full rounds (alive/round) | 251 (5.0) | 251 (4.8) | 251 (4.5) |

- **Survivorship:** by 30 s, 18% of G2's T eco rounds have ended against 2% of full-buy rounds, so the 30 s T eco panel describes the rounds that survived. Eco sides lose players slightly faster by 20 s.
- **10 s, both conditions, CT:** two spots inside BombsiteB, a SnipersNest spot, and a looser BombsiteA/Scaffolding glow; CT spawn empty. **T:** clusters at BackAlley, TopofMid (strongest, about 0.06, roughly 0.4 players per round), PalaceAlley and PalaceInterior; T spawn already empty.
- **20 to 30 s:** CT full-buy spreads across TicketBooth, Connector, Scaffolding/BombsiteA and Truck/Balcony; CT eco concentrates more on B site and the Catwalk/Tunnel/Ladder corridor, with only a dim TicketBooth, which matches the pooled finding. G2's T eco keeps TopofMid as its main hotspot, unlike the pooled shift; this is one team with 49 rounds and the pooled result had a minority of teams going the other way, so it is not evidence against the pooled finding.

---

## 7. Cross-cutting methodology

- **Resample by match, not round.** All team comparisons permute or split at the team-match level. Rounds in a match share an opponent and game plan.
- **Identical statistic under observed and null.** A permutation p-value is only valid when the observed data and every permutation use the same statistic; the first identification test violated this and was rebuilt.
- **Pairing removes confounds by construction.** Eco vs full-buy is tested within team and match; the calibration test shows the false-positive rate stays near 5% even when a naive comparison would be badly confounded.
- **Effect sizes over p-values.** With 1,000 to 2,000 permutations, p-values reach a floor and cannot discriminate cells; report D, area shifts with intervals, and null scales.
- **Multiple testing.** Holm over the four confirmatory cells; max-T over the 27 areas; all other windows and variants are exploratory and labelled so.
- **Field and occupancy are compared everywhere.** The occupancy baseline was kept in every analysis because the field did not beat it on identification.
- **Rule of thumb for small panels.** Treat any map or per-team result built from under about 40 rounds as noise-limited.

---

## 8. Limitations to state in the report

1. **Era and tournament confound (identification result):** not tested; some identifiability may come from period or tournament, not team style.
2. **"Team" is an organisation, not a roster;** rosters and in-game leaders changed. The dataset's full date range was not extracted (a sample of six demos was all from 2021).
3. **Site preference measures planting success, not intent;** it mixes attempts and executions. The split-half stability test is inconclusive.
4. **Eco analysis:** money, loadout and tactics are confounded; matching omits score state and loss streak; 43% of T-eco rounds are post-pistol rounds 2 to 4 (robust to dropping them at the omnibus level; check area level with the provided snippet).
5. **Field vs occupancy:** the field adds no demonstrated value in area-share analyses; CT field shifts disagree with occupancy (r = 0.44) for an untested reason. The field's expected strength is where graph reach matters, which these analyses do not test.
6. **Time axis:** fixed 128 ticks per second, whereas demos report 127 or 128 (under 1% effect).
7. **Windows:** `early_5_20` includes post-death frames when a death occurs before 20 s; `pre_first_loss` has an inferred (from alive counts) first death.
8. **Heatmaps are descriptive.** Small-n panels are noisy, late-time panels suffer survivorship, and different smoothing modes give non-comparable colours. Significance claims come from analysis C, not from the maps.
9. **Level stacking:** 28 of 903 tiles stack over another level on the 2D radar; the heatmaps cannot separate levels there.

---

## 9. Corrections log: statements that were wrong or too strong (do not reuse)

| Earlier statement | Status |
|---|---|
| Frames are about 1 s apart | **Wrong.** About 0.5 s. |
| φ = 1.00 means within-match variance is exactly binomial and teams do not adapt within matches | **Too strong.** φ was floored at 1.0; the supportable statement is "no detectable match-level clustering"; power is low (about 7 plants per unit). |
| The Spearman-Brown r = 0.40 is an upper bound; the split-half is a null result | **Wrong.** It is a projection; the result is inconclusive. |
| Site choice is "primarily driven by map structure" (A-site geometry, rotation times, CT stacking) | **Untested speculation.** Remove. |
| B-leaning teams are "Northern European / CIS" | **Retracted.** Geographically sloppy and unsupported. |
| Fiend has the highest raw P(A) with a wide interval | **Wrong.** Highest raw values are x-kom AGO 0.773 (n = 22), Movistar Riders 0.750, Fiend 0.744 (n = 90), Dynamo Eclot 0.741. |
| NiP's shrunk estimate is 0.54; all but a handful of estimates lie in 0.55 to 0.70; Heroic, mousesports, OG, BIG below the pool | **Wrong.** NiP 0.566; all 33 lie in 0.566 to 0.653; only NiP and Vitality are distinguishable. |
| A plant-rate panel would show whether B-leaning teams "simply plant less" | **Muddled.** P(A \| plant) is conditional on planting; the real confound is intent vs execution. |
| Site-preference result is a "consistent story" with weak control margin prediction | **Unsupported.** Different questions. |
| Most eco rounds end sooner, so `pre_first_loss` would exclude them more | **Not borne out.** Usable rates were 72% to 77% across classes; the window-length difference is real but modest (about 4 s). |
| A per-round field-vs-occupancy distance of 0.36 shows the field adds information | **Wrong.** It mostly reflects smoothing. |
| Raster walls are enough to stop heat crossing walls | **Wrong.** The nav graph test is required. |

---

## 10. Cautious draft wording

**Site preference.** "Plant-site preference differed between teams more than chance and match-level clustering explain (match-level permutation test, p = 0.015; 33 teams with at least 20 plants), but the effect was modest (between-team SD of P(A | plant) ≈ 0.033 around a pooled rate of 0.61). Only two teams (Ninjas in Pyjamas, Team Vitality) had 90% credible intervals excluding the pooled rate. A split-half analysis of team estimates was inconclusive (median r = 0.25, p = 0.13), so we do not claim a stable site preference for any individual team."

**Signature identification.** "Teams were identifiable from their early-round control signatures at 7 to 14 times chance (accuracy 0.23 to 0.44 against chance of about 0.03; Holm-adjusted p = 0.004 in all four side-by-method cells). Raw occupancy identified teams at least as accurately as the influence-field signature in every window and side tested, so the field did not add team-identifying information. We did not control for period or tournament, so some of this may reflect shared meta rather than team-specific style."

**Eco vs full-buy.** "Within matched team-match units (369 T-side and 289 CT-side), early-round (5 to 20 s) positioning on full-eco rounds differed from the same team's positioning on full-buy rounds against the same full-buy opposition (within-unit permutation tests, p < 0.001 in all four side-by-method cells, Holm-adjusted p = 0.002; Jensen-Shannon distance 0.065 to 0.130 against null means of 0.018 to 0.033), and the result was unchanged when post-pistol rounds were removed. On the T side, eco teams placed less presence at TopofMid (−5.7 percentage points of mass share, 95% CI −7.2 to −4.4; about 36 of 42 teams in the same direction); on the CT side, less presence at TicketBooth (−2.6 points, 95% CI −3.3 to −1.9). Field and occupancy measures agreed on the T side but diverged on several CT areas. Rounds were matched on team, match and opponent buy but not on score or loss streak, and loadout and tactical choice cannot be separated."

**Heatmap smoothing.** "Positions were smoothed by diffusion over the walkable navigation mesh (Gaussian-equivalent width 100 game units), with flow blocked between tiles whose graph walking distance exceeds 1.6 times their straight-line distance plus 40 units, so density cannot leak across walls, between levels, or off the map; the map integrates to the mean number of alive players per round."

---

## 11. Reproducibility: scripts, commands and outputs

All scripts live in `analysis/team/` and are run from the project root.

| Step | Script | Key outputs |
|---|---|---|
| Team aliases | `build_team_aliases.py` | `data/team_aliases.json` |
| Site preference | `analysis_team_site_preference.py` | `outputs/analysis/team_site_preference.csv`, `team_site_results.json`, `figures/team_site_preference_forest.png` |
| Signature builder | `build_team_signatures.py` | `outputs/analysis/round_signatures/*.parquet`, `round_signatures.parquet` |
| Identification | `analysis_team_signatures.py` | `signature_identification.csv`, `team_signatures/*.csv`, `team_neighbours_*.csv`, `figures/signature_heatmap_*.png` |
| Economy extraction | `extract_round_economy.py` | `outputs/analysis/round_economy.csv` |
| Eco vs full-buy | `analysis_eco_vs_full.py` | `eco_vs_full_omnibus.csv`, `eco_vs_full_area_shifts.csv`, `figures/eco_vs_full_shifts.png` |
| Heatmaps | `plot_position_heatmaps.py` | `outputs/analysis/figures/position_heatmaps/…` (per-team subfolders) |
| Not run | `build_match_info.py` (dates, competitions, tick rates) | `data/match_info.csv` |

Key parameters: site preference `MIN_PLANTS = 20`, 10,000 permutations; signatures `MIN_ROUNDS = 30`, `MIN_HALF_ROUNDS = 8`, 20 splits × 1,000 permutations; eco analysis 2,000 permutations, 500 bootstrap resamples, primary window `early_5_20`; heatmaps bandwidth 100 units, `--cut 1.6`, tolerance 0.3 s. Random seed 42 throughout.

Typical heatmap commands:
```
python3 plot_position_heatmaps.py --list-teams
python3 plot_position_heatmaps.py --team "G2 Esports" --times 10 20 30 --labels
python3 plot_position_heatmaps.py --times 10          # all teams pooled
```

Snippet to compare area-level eco shifts with and without post-pistol rounds:
```python
import pandas as pd, numpy as np
s = pd.read_csv("outputs/analysis/eco_vs_full_area_shifts.csv")
k = ["contrast", "window", "side", "method", "area"]
a = s[(s.variant == "all") & (s.window == "early_5_20")].set_index(k)
b = s[s.variant == "excl_early_half"].set_index(k)
m = a[["delta", "p_maxt"]].join(b[["delta", "p_maxt"]], lsuffix="_all", rsuffix="_excl", how="inner")
for (c, side, meth), g in m.groupby(level=["contrast", "side", "method"]):
    sig = g[(g.p_maxt_all < 0.05) & (g.delta_all.abs() >= 0.01)]
    print(c, side, meth, "r =", round(np.corrcoef(g.delta_all, g.delta_excl)[0, 1], 3),
          "| same sign:", int((np.sign(sig.delta_all) == np.sign(sig.delta_excl)).sum()), "of", len(sig))
```

---

## 12. Open items (not done)

- **Era-controlled identification** (code exists; deliberately not run). Also a chronological split-half for analysis A, which needs match dates from `build_match_info.py`.
- **Per-round behavioural features** for eco vs full-buy: distance advanced from spawn, teammate spread, area footprint, mean speed. Mean speed would also test the weapon-speed hypothesis for the CT field/occupancy disagreement.
- **Plant rate and time to plant by buy class** (`bomb_plant_tick` is already in `round_economy.csv`).
- **Signature-to-site link** (round-level classification of the planted site from early signatures, field vs occupancy), if the site-preference story is developed further.
- **Post-plant retake path exposure** (T influence along each CT's shortest path to the bomb): where graph reach matters and occupancy cannot substitute, so the most likely place for the influence field to show added value.
- **σ sensitivity** (300, 500, 750) for the field-based results; expected to move the field toward occupancy as σ falls.
- **Regenerate all heatmaps** in diffusion mode before using them in the report; the G2 examples above were produced with the earlier unmasked version.

---

*Report constraints on file: IEEEtran conference template, maximum 8 pages, author-year citations. Suggested allocation if useful: methods and data about 2 pages, results about 3 pages (one site-preference figure or table, one identification table, the eco omnibus table plus the 2×2 area-shift figure, optionally two heatmap panels), limitations and discussion about 1.5 pages. The corrections log (section 9) and the limitations (section 8) are the parts most worth keeping when cutting.*
