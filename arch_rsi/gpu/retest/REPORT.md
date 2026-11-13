# Retest: do Stage 1's picks still beat the start under Stage G's recipe? Result

The retest ran from 16:18:39 to 18:18:11 UTC on 9 November 2026 (2 h 0 min), under `PROTOCOL_R.md` (SHA-256 `6240851f…`, recorded in `runs/manifest.json` before the first run).

**Sources:**
- every number below comes from `runs/results.json`, written by `run_retest.py`;
- the two extra checks are in `seedcheck/` and are described under Departures.

**What ran:**
- 224 training runs (27 networks), all finished, with none failed or timed out. The longest took 704 s, against a 1,800 s limit.
- No Claude calls were made, and nothing was selected from these results.
- Every score is exact-build success on Q's 512 dev designs. The final split was never read.

**One thing differs from Stage 1 besides the recipe.** Stage 1 trained at 131,072 steps with the frozen recipe; the retest trains at 262,144 steps with Stage G's recipe (256 worlds per update, 32 minibatches, learning rate 0.001). So "under the new recipe" below means Stage G's recipe and budget together.

## Outcome: the lead survives, smaller

| Contrast (primary, Holm over 2) | Retest difference | Bootstrap 95% interval | Holm-adjusted p | Stage 1 difference |
|---|---|---|---|---|
| **R1:** g0 − start | +37.1 points | +27.0 to +47.1 | below 0.001 | +52.5 |
| **R2:** no feedback − start | +38.0 points | +27.9 to +47.9 | below 0.001 | +50.9 |

Both pass, so by the protocol's rule the reading is **"the lead survives"**. Claude's picks still beat the starting network by about 37 points on the same seeds. That's 13 to 15 points less than under the old recipe.

## Per arm

| Arm | Retest mean | SD between searches | Seeds that learned the hidden-cell skill | Stage 1 mean (old recipe) |
|---|---|---|---|---|
| g0 (sees its results) | 88.1% | 0.045 | 80 of 80 | 75.2% |
| No feedback (scores hidden) | 88.9% | 0.046 | 80 of 80 | 73.6% |
| Random menu | 72.1% | 0.178 | 76 of 80 | 60.4% |
| Start (12 seeds) | 50.9% | — | 12 of 12 | 22.7% |
| Start + critic and orthogonal init (12 seeds) | 26.5% | — | 7 of 12 | 22.1% |

- A search scores the mean of its pick's 8 seeds (1001–1008). The two comparators use seeds 1001–1012.
- The spread between searches fell to about a third of Stage 1's: 0.045 against 0.123 for g0, and 0.046 against 0.131 for no feedback.

### Secondary contrasts (Holm over 5; reported only, not claimed)

| Contrast | Difference | Bootstrap 95% interval | Holm-adjusted p |
|---|---|---|---|
| g0 − no feedback | −0.8 points | −4.7 to +2.8 | 0.680 |
| Menu − start | +21.2 points | +6.8 to +34.8 | 0.028 |
| critic_ortho − start | −24.5 points | −35.7 to −13.4 | 0.002 |
| g0 − menu | +15.9 points | +6.2 to +27.3 | 0.012 |
| No feedback − menu | +16.8 points | +6.9 to +28.1 | 0.009 |

### Descriptive only

**How much lead survives:**

| Arm | Lead over the start, Stage 1 | Lead over the start, retest |
|---|---|---|
| g0 | +52.5 | +37.1 |
| No feedback | +50.9 | +38.0 |
| Menu | +37.7 | +21.2 |

**Rank agreement: Spearman's ρ = 0.25 across the 25 distinct picks.** A pick's Stage 1 rank says little about its retest rank. For example:
- **Moved up:**
  - No-feedback search 1's pick (conv + cell self-attention) went from 44.7% to 85.7%.
  - g0 search 6's pick (GroupNorm body) went from 45.3% to 84.8%.
  - No-feedback search 5's pick (a layer-shared per-z encoder) went from 58.1% to 94.1%, the third best in the retest (corrected 9 November 2026: g0 search 1's pick scored 94.5%; this said "second best").
- **Moved down:** one menu pick (conv, width 32, depth 3, tanh, no normalisation) fell from 69.3% to 36.0%, and only 4 of its 8 seeds learned the skill.
- **Overall:** 23 of the 25 picks scored higher than in Stage 1. That's partly the doubled budget.

**Q is near its ceiling for good networks under this recipe:**
- The best pick averaged 95.4% (no-feedback search 9).
- 70 of the 200 pick-seeds built at least 90% of designs.
- Stage G's reference designs reached 92–97% at twice this budget.

## Where to be careful

**1. The start's seeds matter more than expected.**
- **The gap:** under the same recipe, budget and file, the start averaged:
  - 50.9% on this retest's seeds (1001–1012);
  - 73.7% on Stage G's seeds (300–309).
  - An exact permutation test, outside the protocol, gives p = 0.004 for that 22.7-point gap.
- **Not the pipeline:** I reran seed 300 and seed 1001 alone on the GPU. Each reproduced its original run bit for bit: the same weights, and the same result on every design (`seedcheck/result.json`). So the gap is in the seeds themselves, or it's an unlucky draw. I haven't found a cause.
- **Do the seeds share difficulty?** Each seed sets the order of training designs, the same for every network. Across seeds 1001–1008, the start's score and the picks' mean score correlate at r = 0.54. That's weak evidence, from 8 points, that some seeds are harder for every network.
- **What it means:**
  - The protocol's comparison uses the same seeds for picks and start, as Stage 1 did, so the reading above stands.
  - The size of the lead is less certain than its interval suggests. Against Stage G's start mean, g0's lead would be about 14 points. But the picks weren't run on those seeds, so that isn't a like-for-like comparison.
  - Every g0 and no-feedback search mean (lowest 79.0% and 81.2%) is above even the higher start estimate.
  - Over all 22 seeds, the start averages 61.3% (SD 0.19).

**2. The "two known PPO practices" hurt under the new recipe.** The start plus a critic and orthogonal init scored 24.5 points below the start (Holm-adjusted p = 0.002, secondary). Under the old recipe they were level. I haven't looked into why.

**3. This isn't a search under the new recipe.** These picks were chosen under the old recipe. The retest can't say what a search run under the new recipe would find, or whether feedback would help there.

## What this means for Stage 2

1. **Architecture still matters under a better recipe.** Claude's picks keep most of their lead.
2. **Part of Stage 1's lead came from the old setting,** 13 to 15 points on these seeds. The start gains more from the new recipe and budget than the picks do. The retest changed both, so it can't say how much is due to each.
3. **Feedback still doesn't show:** g0 − no feedback is −0.8 points, with an interval of −4.7 to +2.8. That's tighter than Stage 1's, but these picks weren't searched under this recipe, so it doesn't re-test Level 1b.
4. **Rankings don't carry across recipes (ρ = 0.25).** This supports letting the search choose the recipe and the network together (Stage 2b).
5. **Q has little headroom for good networks.** A harder world is needed (Stage 2a).
6. **Calibrate the start on more than one seed range.** Its mean under the new recipe is somewhere between about 51% and 74%, depending on the seeds.

## Departures and disclosures

1. **The seed-range comparison and the rerun check** (`seedcheck/check.py`) were not in the protocol. I ran them after seeing the start's retest mean, because it disagreed with Stage G's. Neither is part of the reading.
2. **Exact p-values print as 0.0** in `results.json`, because they are rounded to 4 decimals. With 10 and 12 values there are 646,646 splits, so the smallest possible p is about 0.0000015. The table says "below 0.001".
3. **Rounding:** differences in this report are computed from unrounded arm means. g0 − no feedback is −0.847 points, stored as −0.0085.

## Costs

- **Machine time:** 2 h 0 min from start to finish, 4 runs at once. Summed over runs, 7.8 h of training and scoring.
- **Claude calls:** none, so $0.
- **The seed check:** 2 extra runs, about 2 minutes.

## Added after the report: is ρ = 0.25 just seed noise?

This is descriptive and outside the protocol, prompted by Yang et al. (2020, arXiv 1912.12522), who found architecture rankings unstable even between seeds. Source: `noise_ceiling.py` → `noise_ceiling.json`. No new runs.

| Rank agreement across the 25 distinct picks | Spearman ρ |
|---|---|
| Old recipe against new recipe, 8-seed means | 0.25 |
| Old recipe, one half of the 8 seeds against the other (mean of 35 splits) | 0.67 (range 0.49–0.83) |
| New recipe, the same split-half test | 0.85 (range 0.76–0.93) |
| Rough ceiling across recipes if only seed noise differed (Spearman–Brown, then the geometric mean) | about 0.86 |

**What it means:** within each recipe, the picks' rankings are fairly stable from one half of the seeds to the other. Across recipes they nearly vanish. So the low agreement isn't mostly seed noise. Which network looks best depends on how it's trained.

**Caveat:** Spearman–Brown applied to a rank correlation is an approximation. The ceiling is a guide, not a test.
