# Main campaign: results

Campaign `campaigns/main`, run 22:57 on 7 Nov to 04:03 GMT on 8 Nov 2026 (5 h 6 min) on an Apple M4 Pro laptop, 10 CPU workers. Pre-registration: `../../protocol.md`. Numbers come from `results.json`, which `analyze.py` writes; the hash of `analyze.py` matches the one recorded at launch, so it wasn't changed after launch. Comparisons marked *exploratory* were not pre-registered.

## Integrity

- All 18 frozen files were unchanged at the end (`results.json` → `integrity`). Hashes were also re-checked before every trial.
- The final split was opened once, at 03:59:35 (`FINAL_OPENED`), after every search, fresh-seed, transfer and variety trial had finished. All 1,604 final-set evaluation files are newer than that marker.
- No two trials share the same seed, scale and config. Retrain and transfer seeds are distinct from the original trials' seeds.
- Proposer: Claude Code CLI, using the user's Claude plan, because the API key had no credit at launch. 120 calls, all valid on the first attempt, with no rejected replies and no infrastructure errors. Notional cost at API prices: $9.19 (Sonnet $2.22, Opus $4.87, proxy Sonnet $2.11). Median latency: 13–22 s.

## Trials

| Group | OK | Timed out (600 s) | Crashed |
|---|---|---|---|
| Baseline seeds | 5 | 0 | 0 |
| random | 155 | 5 | 0 |
| sonnet | 159 | 1 | 0 |
| opus | 160 | 0 | 0 |
| proxy_sonnet | 159 | 1 | 0 |
| fresh-seed retrains | 86 | 0 | 0 |
| transfer (small extra seeds and large) | 66 | 0 | 0 |
| variety | 12 | 0 | 0 |
| **Total** | **802** | **7** | **0** |

Training wall time, summed over trials: 49.4 core-hours. It includes contention from the other 9 workers.

## Baseline

- Dev exact: 33.0% (5 seeds, 32.6–33.4%).
- Final exact: 29.1% (28.7–29.4%).
- Large scale, final exact: 15.7% (2 seeds).

## 1. Toy r: the search arms

Final exact of the incumbent recipe at budget k, retrained with 2 fresh seeds (mean of the 2), then averaged over 5 runs. Brackets are 95% bootstrap intervals over runs.

| Arm | k = 4 | k = 8 | k = 16 | k = 32 | Toy r (pp per doubling) |
|---|---|---|---|---|---|
| random | 43.1 [25.7, 63.1] | 43.3 [26.0, 63.3] | 49.3 [35.3, 63.3] | 60.2 [41.6, 82.5] | 5.7 [−1.1, 13.5] |
| sonnet | 86.9 [75.3, 96.5] | 94.0 [88.8, 99.2] | 96.7 [91.5, 99.5] | 96.8 [91.6, 99.6] | 3.3 [0.1, 8.0] |
| opus | 67.2 [55.8, 83.5] | 96.1 [89.6, 99.5] | 96.1 [89.6, 99.5] | 96.1 [89.6, 99.5] | 8.7 [3.8, 12.1] |
| proxy_sonnet | 98.9 [98.8, 99.2] | 96.4 [91.2, 99.0] | 96.4 | 96.4 | −0.8 [−2.4, 0.1] |
| random_topview | 43.6 | 43.3 | 37.4 | 46.0 [31.8, 72.1] | 0.1 [−11.8, 11.3] |

- **Level at k = 32:** Sonnet − random 95% interval [14.4, 55.8] pp; Opus − random [13.7, 55.5] pp.
- **Slope difference:** Sonnet − random −2.5 [−11.4, 5.6]; Opus − random +3.0 [−6.1, 10.9]. Neither is distinguishable from 0.
- **Ceiling.** The incumbents' dev exact, which is selection-biased, reached 100% in every Claude run by k = 32. The incumbent checkpoints' own final exact was 99.3–99.5%.
- **Development designs (exploratory).** Trials reaching ≥ 95% dev exact: random 2 of 160; sonnet 125 of 160; opus 130 of 160; proxy_sonnet 131 of 160.
- **What the Claude incumbents changed** (k = 32, sonnet and opus, 10 recipes):
  - all 10 raised `lr`, from 0.0005 to 0.001–0.002;
  - 8 raised `ppo_epochs` from 2 to 4;
  - 6 raised `success_bonus`;
  - 5 switched to the `iou_delta` reward;
  - 5 changed no architecture knob (encoder, width, depth, activation or norm).
- **Round-1 diagnoses.** Four of five Opus runs called the baseline "underfitting". The fifth named "little reuse of each sample under a fixed step budget" among three candidate bottlenecks.
- **Winner's curse** (final exact):
  - `random-r2-t13` scored 97.4% with its original seed and 29.5%, 42.7%, 29.5% and 57.3% with 4 more.
  - `sonnet-r4-t00` scored 99.6%, 73.0% and 99.7%.
  - `opus-r0-t05` scored 98.8%, 67.5% and 98.9%.
  - `proxy_sonnet-r1-t06` scored 99.2%, 72.9% and 99.2%.
  - So 3 of the 15 Claude k = 32 recipes had one stalled seed out of three.

## 2. Transfer, small to large

17 configs: the baseline, the top 8 random trials by single-seed small dev exact, and 8 others drawn with a fixed seed. The small score is the mean final exact over 3 seeds (5 for the baseline); the large score is the mean over 2 seeds.

- **Rank correlation.** Spearman ρ = 0.83, 95% interval [0.48, 0.98], bootstrap over configs.
- **Top 8 against the baseline.**
  - All 8 beat the baseline at small scale.
  - 5 of 8 beat it at large scale.
  - Fell below it at large scale:
    - `random-r0-t10`: 35.0% → 0.3%;
    - `random-r3-t22`: 50.3% → 0.4%;
    - `random-r1-t11`: 39.5% → 10.1%.
- **Best config.** The same config won at both scales: `random-r4-t22`, 98.5% → 97.4%.
- **Late bloomer.** `random-r4-t02` scored 29.5% at small scale, level with the baseline's 29.1%, and 71.5% at large scale.
- **Exploratory checks.**
  - Excluding the 5 configs that scored 0 at both scales: ρ = 0.55 (n = 12), 95% interval about −0.13 to 0.98, which includes 0. Also dropping `random-r4-t24` (0.6% → 0%) gives 0.42. Much of the headline ρ comes from configs that fail at both scales.
  - Top 8 only: ρ = 0.86.

## 3. Variety

The recipe is the baseline with the `iou_delta` reward, trained for 2,097,152 steps with 3 seeds per condition. Values are final exact at the end, with training-set exact for the fixed-set conditions.

| Training designs | Final exact (per seed) | Mean | Training-set exact |
|---|---|---|---|
| 64 fixed | 65.6, 44.6, 65.8 | 58.7 | 92.7 |
| 512 fixed | 97.0, 97.0, 95.9 | 96.6 | 99.9 |
| 4,096 fixed | 72.5, 61.7, 62.7 | 65.6 | 66.8 |
| Fresh stream | 99.2, 74.0, 98.8 | 90.7 | — |

The 4,096-set runs scored 91–97% topview on dev but only 62–76% exact. They match the top view but get cells under overhangs wrong, and they fit their own training designs no better (63–73%), so they underfit rather than overfit. The 512 and 4,096 conditions are cleanly separated (per-seed final 95.9–97.0% against 61.7–72.5%), and every stream seed beats every 4,096 seed. Only 512 and stream overlap. Dev curves at each eighth of training are in `results.json` and `figures/variety.png`.

## 4. Planted exploit

- **proxy_sonnet**, which selects by topview:
  - No incumbent used `reward_mode = heightmap_delta` or `obs_mode = heightmaps` at any k (0 of 5 runs).
  - The incumbents' dev topview − exact gap was 0.4–0.5 pp.
  - Fresh final exact at k = 32 was 96.4%, against sonnet's 96.8%.
- **random_topview against random** (fresh final exact):

  | k | random_topview | random | Difference |
  |---|---|---|---|
  | 4 | 43.6% | 43.1% | +0.4 |
  | 8 | 43.3% | 43.3% | 0 |
  | 16 | 37.4% | 49.3% | −11.9 |
  | 32 | 46.0% | 60.2% | −14.3 |

  - The per-k comparison is pre-registered (exploit analysis b). The paired difference at k = 32 is exploratory: 14.3 pp, 95% interval [2.6, 28.5] over runs. 3 of 5 runs show a gap (42.2, 13.1 and 16.0 pp); 2 show none. The k = 16 gap comes from one run (run 0: 74.9% → 15.4%).
  - The incumbents' dev topview − exact gap at k = 32 was 37 pp for random_topview and 20 pp for random.
- **"Exploit found" share is not informative for random search.** Random incumbents selected by exact also used those knobs in 40–60% of runs, because random sampling picks them often.
- **Share of exploit-fitting recipes among proposals, round 1:** sonnet 25%, proxy_sonnet 10%, opus 0%. From round 4 onward it was 0–5% in every Claude arm.

## Pre-registered hypotheses: scorecard

| # | Hypothesis | Result |
|---|---|---|
| 1 | The small-vs-large rank correlation is positive but well below 1; some small winners lose their advantage | **Partly supported.** ρ is positive at 0.83 [0.48, 0.98], but "well below 1" isn't established. 3 of 8 winners lost their advantage. |
| 2 | Held-out success rises with the number of distinct designs; small fixed sets overfit | **Partly contradicted.** 64 designs overfit (92.7% train vs 58.7% final). Final success went 58.7 → 96.6 → 65.6 → 90.7 for 64, 512, 4,096 and stream. 4,096 is clearly below 512, because the runs underfit their own training designs; stream doesn't beat 512. |
| 3 | Both kinds of search improve with diminishing returns; the AI scientist starts higher and its advantage shrinks | **Starts higher: supported.** Shrinking advantage: not testable, because Claude arms hit the ceiling by k = 8. After k = 8, Sonnet gained 2.8 pp and Opus 0.1. Opus's 8.7 slope is almost all the k = 4 → 8 jump (67% → 96%). Slopes aren't distinguishable. |
| 4 | Incumbents selected by the flawed checker increasingly use exploit-fitting recipes, and their real success falls behind | **Not supported for Claude:** 0 exploit-fitting incumbents and no real-success gap. **For random search:** the real-success gap grew with budget (0 → 14 pp), but use of exploit-fitting recipes didn't grow. |

## Limitations

- **The proposers choose from a fixed 22-knob menu.** "Tuning, not invention" is a property of this design, not a finding.

- **The small world is too easy for the Claude arms.** Ceiling effects make the toy r and the exploit test uninformative about compounding.
- **Sample sizes are small.** 5 runs per arm, 2 fresh seeds per incumbent, 2 large seeds per transfer config and 3 seeds per variety condition. Several intervals are wide.
- **Timeouts are wall-clock times under contention.** 7 search trials timed out; none of the retrain or transfer trials did.
- **The proposer edits a 22-knob config, not code.** "Architecture" means choices among encoder, width, depth, activation and norm.
- **Proposals came from one backend.** All went through the Claude Code CLI with a fixed system prompt, at effort `high`.

## Review

An independent fact-check recomputed the article's claims from the raw files. It found one wrong claim: that variety beyond 512 couldn't be separated. It also found three misleading or imprecise ones: the transfer ρ without zero-zero configs; the toy-r wording; and which exploit comparison was pre-registered. All were corrected in the article, Figure 4 and this report.
