# Stage 1 protocol: does the architecture search use its experiments?

Written on 9 November 2026, after Stage 0 (`../pilot/stage0/REPORT.md`) and before any Stage 1 result. It amends `../protocol.md` for Stage 1 only, as planned in `../PLAN.md` section 5. The code is in this folder, and the file hashes of this protocol, `run_stage1.py`, `analyze_stage1.py`, `review_stage1.py` and `candidates/start_critic_ortho.py` are added to the campaign's frozen set at launch. Every frozen file of the original design stays unchanged and is hash-checked before every run.

## Question

Claim 1b in PLAN.md: **experiments guided the search.** If Claude's search beats the start only because its first ideas are good and the frozen code keeps the best of 12, hiding the results from it should cost nothing. If it uses the results, hiding them should hurt.

## What is the same as `protocol.md`

The following are all unchanged:
- the training code, recipe, compute rule, observation and reward;
- the checks, caps and interface rules;
- the dev scoring, the scientist's model (Claude Opus 5.5, effort high, `claude -p` with no tools), its program g0, the menu, the repair calls, and the 4 rounds of 3 candidates with 2 seeds (0 and 1) per trial.

## What changes

- **World and budget.** Q (8×8×5) only, at 131,072 steps per trial, with a 600 s time limit (Stage 0, decision 1). P is not used in Stage 1.
- **Phase 0.** The start trains seeds 0–3 on Q. The scientist's noise note shows these four scores, as in `protocol.md`.
- **Arms, 10 searches each:**
  - **g0:** the normal search under program g0.
  - **nofb (no feedback):** the same program and prompts, but every success score is removed: the trials' dev and training success, the start's dev score, the current best and the training times. The noise note gives no numbers. The scientist still sees each trial's status, check failures, errors, parameters, compute ratio and step count, and is told that scores are hidden and that selection still uses them. The code selects with the real scores, so the only thing removed is adaptation to results.
  - **menu:** 12 random draws from the 144-option menu, with search seed 500 + i.
- **Retest before selection, in every arm.**
  1. After the 12 trials, the 3 trials with the highest 2-seed dev mean, among those whose seeds all trained, each train seeds 2 and 3. Ties go to the earlier trial.
  2. The pick is the retested trial with the highest 4-seed mean whose seeds all trained. It must strictly beat the start's own 4-seed mean (seeds 0–3); otherwise the pick is the start.
  3. All arms' task text says this instead of "the trial with the highest dev success is selected".
- **Confirmation:** each search's pick trains 8 fresh seeds (1001–1008), scored on Q's 512 dev designs. A seed whose training fails scores 0. Picks with identical code share their runs.
- **Comparators, no search, 12 seeds each (1001–1012):**
  - the start;
  - **critic_ortho:** the start with a separate critic (its own conv body and pooled layer) and orthogonal initialisation (gain √2 for hidden layers, 0.01 for the policy outputs, 1 for the value output, zero biases). It was written before any Stage 1 result to test whether wins are just known PPO practices. It passes the checks. Its training compute is 1.99× the start's, so the compute rule gives it about half the steps, as it would any candidate.
- **Scheduling:** the 30 searches run 6 at a time in interleaved order (g0 1, nofb 1, menu 1, g0 2, …), so every arm meets the same machine load. There are 10 training slots on a 14-core machine.
- **The final split stays locked.** `FINAL_OPENED` is not written in Stage 1. It opens once, after all stages.

## Analysis (`analyze_stage1.py`, frozen at launch)

- **Unit:** one search. Its score is the mean, over the 8 confirmation seeds, of its pick's exact-build success on dev.
- **Primary score: compliant picks only.**
  - After the run, `review_stage1.py` gives a fresh Claude call each distinct pick written by the scientist. The call sees only the rule text and the code, with no scores, no arm and no search, in shuffled order.
  - A search whose pick is judged to break a rule is scored as if it had picked the start, using the start's scores on the same seeds.
  - Start and menu files are compliant by construction.
  - The as-computed score is reported beside the primary score.
- **Primary family (Holm-corrected):**
  - P1 = g0 − nofb, 10 searches against 10;
  - P2 = g0 − start, 10 searches against 12 seeds.
  - Each gets an exact two-sided permutation test over all splits.
- **Gate:** both differences positive with Holm-adjusted p < 0.05.
  - **If g0 beats both:** we may write "experiments guided the search".
  - **If g0 ≈ nofb and both beat the start:** we may write "Claude's first ideas plus selection beat the start; experiments added nothing measurable".
  - **If the gate fails:** that is the result, and it goes back to the user before Stage 2.
- **Secondary family (Holm-corrected, reported only):**
  - g0 − menu, nofb − menu, nofb − start, menu − start;
  - g0 − critic_ortho, critic_ortho − start.
- **Descriptive only:**
  - bootstrap 95% intervals;
  - success on the dev designs with hidden cells;
  - the share of confirmation seeds that learned the hidden-cell skill (at least 10% of those designs built);
  - selection optimism (selection mean minus confirmation mean);
  - per-round best and best-so-far means per arm (best-so-far rises even under pure noise, so compare arms, not slopes);
  - invalid and failed trials;
  - measured training hours, and Claude calls, tokens and notional cost per arm.
- **Expected power:** with Stage 0's measured noise, about 64–87% for a true +20 points and 40–62% for +15. Smaller effects will be reported as not established.

## Rules if something goes wrong

These are `protocol.md`'s rules:
- Infrastructure errors are retried with growing waits and never count as model attempts.
- A stopped run resumes by rerunning the same command. Finished runs are never rerun. A run interrupted without a result is rerun with the same seed.
- Any change to a frozen file stops the campaign.
- Numbers here are not changed after launch, and any departure is listed in the report.

## Dry run (disclosed)

`--preset stage1_dry` exercises every path at 16,384 steps:
- 1 search per arm, 2 rounds of 2 candidates;
- 1 retest seed and 2 confirmation and comparator seeds.

Its outputs are never analysed as results.

## Budget

| Item | Estimate |
|---|---|
| Training runs | About 1,170: 900 search, up to 240 confirmation, 24 comparator, 4 phase-0 |
| Claude calls | About 80 scientist calls plus repairs, about $18–25 notional |
| Review calls | Up to 20 after the run |
| Wall time | About 6–8 hours on this Mac, plugged in and awake |

## Known limitations, stated in advance

- At 131,072 steps the start averages about 22%, not mid-range. So beating the start is easy (Stage 0, decision 1).
- Confirmation uses fresh seeds on the same dev designs that selection used. The final designs stay unopened until after all stages.
- One model (Opus) and one task family. Sonnet replication is Stage 3.
- The no-feedback arm still benefits from the frozen selection. It isolates adaptation, not selection.
