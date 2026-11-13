# Stage 1: does the search use its experiments? Result

The run went from 07:24:53 to 13:31:53 UTC on 9 November 2026 (6 h 7 min), under `protocol_stage1.md`. The outcome is read with the rules in `addendum_outcomes.md`, which was written while the run was going. By then, 6 of the 20 g0 and no-feedback confirmation results were already in `run.log`, which the user had opened (departure 1).

Sources:
- the rule review: `../campaigns/stage1/review.json`, from `review_stage1.py`;
- every number below: `../campaigns/stage1/results.json`, from `analyze_stage1.py`, with `primary_family.status` "final".

Every score is exact-build success on Q's 512 dev designs. The final split was never read, and `FINAL_OPENED` does not exist.

## Outcome: B, so the gate failed

| Contrast | Difference | Bootstrap 95% interval | Holm-adjusted p | Passes? |
|---|---|---|---|---|
| **P1:** g0 − no feedback (primary) | +1.5 points | −9.1 to +12.0 | 0.791 | no |
| **P2:** g0 − start (primary) | +52.5 points | +37.1 to +64.9 | below 0.001 | yes |
| **S3:** no feedback − start (secondary, Holm over 6) | +50.9 points | +35.3 to +63.7 | below 0.001 | yes |

P1 does not pass, while P2 and S3 do. By the addendum, that is outcome B: a gate failure. The only verdict sentence it allows is:

> Claude's first ideas plus selection beat the start; experiments added nothing measurable.

This was not a test of equality. Before the run, the chance of detecting a true +15-point benefit with 10 searches per arm was estimated at 40–62%. P1's interval runs from −9.1 to +12.0 points, so a benefit of up to about 12 points is not ruled out.

The rule review didn't change the outcome. Its 20 verdicts were all "compliant", so the as-computed analysis is identical, and its outcome is also B.

## Per arm

| Arm | Unit | Mean | SD between searches | Seeds that learned the hidden-cell skill | Selection optimism | Picked the start |
|---|---|---|---|---|---|---|
| g0 (sees its results) | 10 searches × 8 seeds | 75.2% | 0.123 | 78 of 80 | 5.7 points | 0 |
| No feedback (scores hidden) | 10 searches × 8 seeds | 73.6% | 0.131 | 79 of 80 | 5.8 points | 0 |
| Random menu | 10 searches × 8 seeds | 60.4% | 0.183 | 73 of 80 | 12.7 points | 1 (search 2) |
| Start | 12 seeds | 22.7% | — | 2 of 12 | — | — |
| Start + critic and orthogonal init | 12 seeds | 22.1% | — | 4 of 12 | — | — |

- **Learning the skill:** a seed counts as having learned it if it builds at least 10% of the 430 dev designs that have hidden cells.
- **Selection optimism** is how far a pick's 2-seed selection score sat above its 8-seed confirmation score, averaged over searches.

### Secondary contrasts (reported only, not claimed)

| Contrast | Difference | Bootstrap 95% interval | Holm-adjusted p |
|---|---|---|---|
| g0 − menu | +14.8 points | +1.8 to +28.0 | 0.127 |
| No feedback − menu | +13.2 points | −0.1 to +26.6 | 0.151 |
| No feedback − start (S3) | +50.9 points | +35.3 to +63.7 | below 0.001 |
| Menu − start | +37.7 points | +20.3 to +52.7 | 0.004 |
| g0 − critic_ortho | +53.1 points | +42.5 to +62.3 | below 0.001 |
| critic_ortho − start | −0.7 points | −15.3 to +11.6 | 0.942 |

### How each search's best score moved per round

These are 2-seed selection scores averaged over searches, so they carry the 5.7–12.7-point selection optimism above.

| Arm | Round 1 | Round 2 | Round 3 | Round 4 |
|---|---|---|---|---|
| g0 | 62.9% | 75.8% | 80.5% | 81.1% |
| No feedback | 77.7% | 80.8% | 84.0% | 84.7% |
| Menu | 31.4% | 52.2% | 71.1% | 71.1% |

Descriptive only, not a test: on this measure the no-feedback arm started higher and stayed higher. It doesn't change the outcome, which uses the confirmation scores.

## Rule review

- **What ran:** one fresh call per distinct pick written by the scientist, 20 picks in all. Model `claude-opus-5-5` at high effort.
- **What it saw:** only the frozen rule text and the code, in a fixed shuffled order (`Random(2026)`).
- **Result:** 20 of 20 compliant, so no pick was scored as the start.
- **Limit:** the review is blind, not independent. It is the same model that wrote the networks.

## Integrity and runs

- **Clean run:** no `INTEGRITY` stop and no traceback in `run.log`. Hash checks passed before every run.
- **Run counts:** 1,124 training runs: 1,120 finished and 4 hit the 600 s limit. There were no invalid files.
- **The 4 timeouts** were both seeds of two candidates, g0 search 4's r4c3 (09:52 UTC) and no-feedback search 8's r2c3 (11:55 UTC).
  - The pre-launch checks timed each forward-backward pass at 162 ms and 261 ms, against the start's 19 ms. Their FLOP ratios were only 1.11 and 1.42, because the counter doesn't see pooling and similar operations.
  - Both were selection trials that scored as failed, not confirmation runs.
- **My GPU tests overlapped the end of the run** (see `../gpu/GPU_PLAN.md`):
  - They ran from about 12:09 to 13:14 UTC, at low priority.
  - The budgets are compute-matched, so machine load changes only wall time, not what a run computes.
  - No run timed out or failed in that window, and both timeouts came before the first test.

## Costs

Search training only, as `analyze_stage1.py` counts it.

| Arm | Training time, summed over runs | Claude calls | Notional cost |
|---|---|---|---|
| g0 | 20.5 h | 42 | $8.73 |
| No feedback | 20.5 h | 43 | $9.53 |
| Menu | 5.2 h | 0 | $0 |
| **Total** | **46.2 h** | **85** | **$18.26** |

- **Training time** sums each run's own wall time. Up to 10 ran at once, so the whole stage took 6 h 7 min from start to finish.
- **Notional cost** is the CLI's own total per call.
- **Not counted:** confirmation and comparator training, and the 20 review calls.

## Departures and disclosures

1. **The outcome addendum** (`addendum_outcomes.md`, SHA-256 `608edba7…21d`) fixed what "g0 ≈ nofb" means while the run was going.
   - At that point, interim comparator means and some menu and selection numbers had been seen in the logs.
   - It didn't change `protocol_stage1.md`.
   - **Correction found by the fact-check:** the addendum's line 17 says it fixed the phrase "before any result that the phrase concerns exists". That is wrong.
     - When it was saved (09:44:46 UTC), `run.log` already held 6 of the 20 g0 and no-feedback confirmations: g0 searches 1–3 and no-feedback searches 1–3, logged 08:47–09:39.
     - None of them appears in my own session's record before 09:44:46. But the user confirmed on 9 November that they had opened `run.log` before 09:45 UTC, so those results may have been seen.
     - The addendum stays as written, since its hash is recorded, and this note corrects it.
2. **Order of review and analysis.** `run.log`'s last line says to run the analysis, then the review. `review_stage1.py`'s header says the opposite, and I followed the header. Without `review.json`, the analysis marks the primary family "pending", so the order changes nothing.
3. **Input tokens are not logged.** The frozen logger drops the CLI's cache-creation count, so `costs` reports output tokens only. The cost figure is the CLI's own total, so it isn't affected.
4. **Rounding.** `results.json` rounds each contrast's difference to 4 decimals. P1 is stored as 0.0155, but from the unrounded arm means it is 0.01548, so +1.5 points. Figure 4's script now computes differences from the unrounded means.

## What this does and doesn't show

- **Shown:**
  - On fresh seeds, Claude's searches, with or without seeing their scores, picked networks that beat the starting network by about 50 points on the designs they were selected on.
  - Seeing the scores added nothing measurable.
- **Not shown:**
  - That experiments don't help: the test had limited power.
  - That the picks generalise: the held-out designs are still locked.
  - Anything beyond one model, one task family and toy scale.
- **The start was weak by design.** It averaged 22% and learned the skill on 2 of 12 seeds, so beating it was expected (Stage 0 report, decision 1).
- **Next:** by the protocol, a failed gate goes back to the user before Stage 2. Nothing has been decided.
