# Stage 2b: did experiments guide the search on W? Result

**When and how:** 20:27 UTC on 9 November to 15:05 UTC on 10 November 2026, under `PROTOCOL_2B.md`. The run was interrupted once for about 25 minutes (see "Departures and disclosures"). The rule review and the analysis ran straight after, with the pre-registered `review_2b.py` and `analyze_2b.py`.

**Sources:** every number below comes from `campaigns/stage2b/results.json` (status "final"), except the run times, which come from `progress.log`. Scores are exact-build success on W's 512 dev designs. W's final designs were never read, and `FINAL_OPENED_W` doesn't exist.

**What ran:**
- 5 arms × 16 searches. Each search ran 4 rounds of 3 candidates on 2 training seeds, then a retest, and its pick was confirmed on 8 fresh seeds.
- 192 candidates per arm. None was invalid, none failed in training and none timed out.
- Three comparators on 12 seeds each.

## Outcome: the gate failed

| Primary contrast (Holm over 2) | Difference, points | 95% interval | p | Holm-adjusted p | Passes |
|---|---|---|---|---|---|
| **P1, the gate:** sees its scores − scores hidden | +1.5 | −10.1 to +12.2 | 0.81 | 1.00 | no |
| **P2:** sees its scores − reference A | +3.6 | −9.6 to +17.2 | 0.62 | 1.00 | no |

- **The gate** was P1 positive with Holm-adjusted p < 0.05. It failed: searches that saw their scores averaged 57.8%, and the same searches with scores hidden averaged 56.3%.
- **What the protocol allows me to say:** seeing its results didn't measurably help. It doesn't show that experiments never help (see "Power").
- **Consequence:** Stage 2c doesn't run, as the protocol fixed before the run. The result goes back to the user.
- **The as-computed family is identical,** because the rule review found no violation.

## The arms

| Search arm (16 searches each) | Mean confirmed score | SD between searches | Lowest to highest search | Confirmation seeds that learned the hidden-cell skill | Picks with their own recipe | Picks that fell back to the start |
|---|---|---|---|---|---|---|
| Sees its scores (g0) | 57.8% | 0.20 | 15.2% to 86.1% | 84% | 15 | 0 |
| Scores hidden | 56.3% | 0.13 | 30.2% to 81.3% | 84% | 15 | 0 |
| Network only | 55.6% | 0.15 | 31.6% to 81.1% | 95% | 0 | 0 |
| Recipe only | 48.3% | 0.12 | 22.4% to 68.7% | 90% | 16 | 0 |
| Random draws | 26.1% | 0.18 | 12.2% to 71.8% | 41% | 8 | 8 |

| Comparator (12 seeds each) | Mean | SD between seeds | Seeds that learned the hidden-cell skill |
|---|---|---|---|
| Reference A (the best network from the retest, Stage G's recipe) | 54.2% | 0.18 | 11 of 12 |
| The start, Stage G's recipe | 14.2% | 0.05 | 1 of 12 |
| The start, frozen recipe | 9.5% | 0.04 | 0 of 12 |

- **Every search arm beat the start by a wide margin.** The three arms where the AI scientist wrote the network landed within about 2 points of each other, and within 4 points of reference A.
- **The hidden-cell skill** is counted on the 446 dev designs with hidden cells, with the protocol's threshold.

## Secondary family (Holm over 5, reported only)

| Contrast | Difference, points | 95% interval | p | Holm-adjusted p | Passes |
|---|---|---|---|---|---|
| g0 − network only | +2.1 | −10.3 to +13.5 | 0.73 | 1.00 | no |
| g0 − recipe only | +9.4 | −1.8 to +20.1 | 0.11 | 0.33 | no |
| g0 − random | +31.7 | +17.9 to +43.4 | below 0.001 | 0.0005 | yes |
| g0 − the start (Stage G's recipe) | +43.6 | +33.6 to +52.8 | below 0.001 | below 0.001 | yes |
| scores hidden − reference A | +2.1 | −8.7 to +13.9 | 0.72 | 1.00 | no |

- **g0 − random passes,** so the protocol's phrase applies, marked secondary: the AI scientist beat random draws from the same space.
- **g0 − the start passes.** The protocol gives it no phrase.
- **The other three are numbers only.** In particular, searching the recipe as well as the network wasn't measurably better than searching the network alone.

## Descriptive

**Running best during the search, by round** (the runner's selection scores on 2 training seeds, averaged over the 16 searches; they aren't confirmed scores, and in the scores-hidden arm the AI scientist never saw them):

| Arm | After round 1 | Round 2 | Round 3 | Round 4 | Selection optimism, points |
|---|---|---|---|---|---|
| Sees its scores (g0) | 31.4% | 52.2% | 58.1% | 67.9% | +9.1 |
| Scores hidden | 41.6% | 53.2% | 64.5% | 68.1% | +4.5 |
| Network only | 44.3% | 55.2% | 61.3% | 63.9% | +6.2 |
| Recipe only | 7.7% | 23.0% | 35.8% | 47.3% | +2.3 |
| Random draws | 11.2% | 14.5% | 18.4% | 20.2% | −2.8 |

- **Selection optimism** is the pick's selection score minus its confirmed score, averaged over searches. It was largest for the arm that saw its scores.
- **Both score-seeing and score-hidden searches ended round 4 at about the same running best** (67.9% and 68.1%).
- **Rule review:** one blind call per distinct pick from the g0, scores-hidden and network-only arms: 48 picks, all judged compliant. Random and recipe-only picks are compliant by construction. The review is blind, not independent: the reviewer is the same model family as the AI scientist.

## Power

- **The plan** (`../PLAN_STAGE2.md`): with 16 searches per arm and the spread between searches that Stage 1 showed (SD 0.13), about a 56% chance of detecting a true 10-point gain and 88% for 15 points.
- **What happened:** the spread was 0.20 for the arm that saw its scores and 0.13 with scores hidden. The first is larger than planned, so the real chances were lower than those.
- **Holm:** both primary tests failed, so the gate was tested at 0.025, which lowers the chances further.
- **The interval** for the gate runs from −10.1 to +12.2 points, so a benefit of up to about 12 points isn't ruled out.

## Departures and disclosures

- **The interruption.** The run stopped at 11:46 UTC on 10 November, when the session that had launched it ended. At that point 65 of 80 searches were confirmed. It was found at 12:11 UTC and resumed at 12:11:39 UTC with the same command, in a terminal that outlives sessions.
  - **The rule applied before resuming:** every round that hadn't finished was moved aside whole, whatever its partial results, and redrawn with a fresh call. Unfinished runs of fixed candidates (retest and confirmation seeds) were moved aside and rerun.
  - **Why:** on resume the runner asks again for an unfinished round but reuses saved run results, so new code could have been paired with old results.
  - **What was moved:** 77 paths, now in `campaigns/stage2b_interrupted_2026-11-10/`, with the rule and the list in `INTERRUPTION.json`. No finished round, pick or confirmation was touched, and no frozen file was edited.
  - **A side effect:** `progress.log` has duplicate "confirmed" lines from the cached replay after the resume. The analysis reads the result files, not the log.
- **What I'd seen before the analysis.** While monitoring the run I saw lines of the live progress log, including per-round scores and some confirmation scores. No decision used them: the protocol, the analysis code and the endings were fixed before the run.
- **A rounding quirk in the frozen runner,** found while rendering the video. Its running "incumbent" compares a score stored to 4 decimals with the unrounded bar, so a candidate whose 2-seed mean exactly tied the start's was recorded as the incumbent.
  - **How often:** 38 of 276 incumbent updates, in 38 of the 80 searches (sees its scores 14, scores hidden 7, recipe only 15, random 2). In 33 of them it was the first candidate.
  - **What it touches:** in the arms that see their scores, the next round's prompt names the incumbent, so it named that candidate where it would have named the start. With scores hidden, the incumbent is never shown. Picks come from the retest rule, which doesn't use it.
  - **At the end of a search:** the tie was still the incumbent in 1 search (random, s16), and that search's pick was the start.
  - **In the video:** the single-search scene doesn't draw such a tie as a new best. The all-searches chart follows the runner's rule, so there it's a blue dot at the bar's height; it doesn't move a running-best line.
- **Frozen files.** All 39 files in the campaign manifest had their recorded hashes when the run ended and when the analysis finished.
- **Two log files were added** to the campaign folder by the review and analysis commands: `review_2b.log` and `analyze_2b.log`.
- **Stage 2c's W_s calibration ran before this result was known,** with the user's go-ahead, while the analysis was running. It reads nothing from 2b. Its result is in `../2c/PROTOCOL_2C.md` and is unused, because 2c doesn't run.

## Costs

| Arm | AI-scientist calls | Notional cost | Training time, hours (summed over parallel runs) |
|---|---|---|---|
| Sees its scores (g0) | 67 | $15.84 | 14.9 |
| Scores hidden | 68 | $16.40 | 11.7 |
| Network only | 66 | $17.00 | 19.7 |
| Recipe only | 64 | $9.11 | 5.8 |
| Random draws | 0 | $0.00 | 4.1 |

- **Calls:** 265 in the four AI-scientist arms, about $58.35 notional. The 48 rule-review calls are extra and aren't in `results.json`.
- **Elapsed time:** 18 hours 37 minutes from start to `STAGE2B_DONE`, including the 25 minutes stopped. The plan projected about 18 hours.

## What this means

- **Two controlled tests now agree.** In Stage 1 and in Stage 2b, seeing its results didn't measurably help the AI scientist's searches. Here, with scores hidden, they still averaged 56.3%, against 14.2% for the start.
- **Stage 2c stays unrun.** Its runner, analysis and video code are built and tested, and its protocol stays as written.
- **A possible next test,** which would need its own protocol and approval: give each search more tries, and score each try on more seeds, so the feedback carries more signal than noise.
