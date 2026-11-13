# Stage 2c protocol: does the procedure improve itself? (Levels 2 and 3)

**Outcome, 10 November 2026, 15:37 UTC: Stage 2c doesn't run.** Stage 2b's gate failed: seeing its scores beat hiding them by 1.5 points, with Holm-adjusted p 1.0 (`../2b/REPORT.md`). This protocol fixed that 2c runs only if that gate passes. Everything below stays as written, as the record of what was planned and built. `LAUNCH_APPROVED.md` was never written.

**Status:** the user approved all 7 proposed defaults on 10 November 2026 ("Approve defaults", about 12:30 GMT). Drafted at about 12:25 GMT, while Stage 2b was still running. No 2b result had been read or analysed. The user asked for it ("Yes, write it now"). When this was drafted, nothing here had run on the GPU and no model had been called. **Update, 10 November 15:20 UTC:** Stage 2b has finished and the W_s calibration has run (its result is under "Build status"). Nothing else of 2c has run, and no model has been called for it. What's built and tested is listed under "Build status", below the build list. Two amendments, both made before 2b finished or was analysed: "Amendment 1" under Lineages (rewrite inputs), and "Amendment 2" under Build status (the W_s calibration ladder continues downward; approved by the user).

**It runs only if both hold:**
- Stage 2b's gate passes. That's the pre-registered rule in `../2b/PROTOCOL_2B.md`.
- You approve this protocol, including the decisions at the end.

If the gate fails, this file stays a draft, and the result goes back to you.

**Sources:**
- `../../PLAN.md` (Stage 2, sections 3 and 8);
- `../PLAN_STAGE2.md` (the 2c section and decision 3);
- the 2b protocol;
- the existing rewrite code: `orchestrate_arch.py: revise()`, and `scientist.py: SYSTEM_L2, l2_prompt, L2_SCHEMA, parse_program`.

## What Stage 2c adds

In Stages 1 and 2b, the AI scientist (Claude Opus 5.5, as in 2b, no tools) follows a fixed research program, g0r. Stage 2c lets it rewrite that program from its own logs. The rewritten program then controls part of what runs. A rewrite is kept only if it tests better than its parent.

The model's weights never change. So the most this stage can show is recursive improvement of a research procedure, at toy scale. It can't show a model improving itself.

## What each comparison can and can't show

**Programs compared on W** (the 2b world), fresh-seed dev scores:
- **g0:** the program 2b used (g0r).
- **g1:** the first rewrite, whether or not it was kept.
- **final:** the last kept program.
- **facts-only:** a rewrite from the same logs as g1, with the procedure locked.
- **shuffled:** a rewrite from the same logs as g1, with every score permuted.

| Comparison | If it holds, we may write | What it can't show |
|---|---|---|
| **P1:** g1 − g0 | "One self-rewrite, written on another world, helped on W" (Level 2a: transfer) | That the procedure, not added knowledge, did it |
| **P2:** g1 − facts-only | "The procedure changes helped, beyond carried-over facts" (Level 2b) | That the scores were used (P3) |
| **P3:** g1 − shuffled | "The rewrite used the evidence, not just the act of rewriting" (Level 2c) | Anything about later rewrites |
| **P4:** final − g1 | "Gains compounded across self-revisions" (Level 2d) | That this generalises beyond two lineages |
| Level 3 (descriptive) | "The program changed what runs, and changes were kept only after testing" | That the changed settings caused a gain, since several things change at once |
| No differences | "No evidence of self-improvement at this scale" | That it never helps: power is limited (see below) |

**What we may write, by tier:**
- **Level 2, "rewriting its own procedure helped, beyond carried-over knowledge":** P1, P2 and P3 all positive, each with Holm-adjusted p < 0.05.
- **"Recursive", for the procedure:** the Level 2 tier plus P4 positive with Holm-adjusted p < 0.05, and at least one kept rewrite that changed the header.
- **Anything less:** reported as measured, without those words.

**Why the controls are matched to g1, not to the final program.** g1, facts-only and shuffled are all written in one rewrite, from the same logs, so each comparison changes one thing:
- **P2** changes whether the procedure may change;
- **P3** changes whether the scores are real.

The plan's literal wording compared the final program with one-shot controls. Then iteration, test-before-keep and evidence would all differ at once. P4 tests iteration separately. This is decision 4 below.

## Setting

### Meta-training world: W_s (new, decision 1)

- **The world:** 8×8×6, with 6–16 bricks, and its own dev and check designs. It's smaller than W, so it's cheaper, and the program meets a grid size it will later be tested off. W's dev designs stay unseen until the meta-test.
- **Its budget:** calibrated before anything else, with Stage 2a's ladder rules unchanged, applied to the start with Stage G's recipe and the two reference networks. No AI-scientist calls.
- **No final split.** W_s is used for meta-training only.
- **Why not Q?** On Q the best picks already reach about 95%, so there's no headroom.

### Meta-test world: W, exactly as in 2b

- 10×10×6, 8–20 bricks, 512 dev designs;
- the budget is fixed training FLOPs equal to 131,072 start steps with Stage G's recipe;
- 900 s per run, the same trainer, `train_2b.py`;
- the same checks, `check_2b.py`;
- the same RECIPE bounds.

### The program and its header (Level 3)

A program is Markdown, at most 1,000 words, as in Stage 1. It starts with a machine-read header that the runner obeys:

```
<!-- header
rounds: 4                 # R, 2-6
candidates_per_round: 3   # K, 2-6
seeds_per_candidate: 2    # 1-3
retest_top: 3             # 1-4, each retested on 2 more seeds
split: {network: 0, recipe: 0, both: 3}   # per round; sums to K
-->
```

**The rules:**
- **Fixed runs per search:** R × K × seeds + retest_top × 2 ≤ 30, before the confirmation runs. The g0 values above use exactly 30, as 2b does.
- **The split:** sets how many candidates each round may change the network only, the recipe only, or both. The checks enforce it per candidate.
- **A bad header:** a header that's missing, unparsable or out of bounds sends the rewrite back for a repair, up to 2 times. After that the parent's header is used, and this is disclosed.
- **Bodies:** g0's body is `../programs/g0r.md` unchanged. Only the header is added, with values identical to what 2b ran.

### Lineages, generations and test-before-keep

**Two independent lineages.** Each starts from g0 and has 3 rewrites (g1, g2, g3), so 4 generations of searches.

**Each generation:**
- 6 searches per lineage on W_s, under the current program;
- each search's pick is confirmed on 3 fresh W_s seeds before any rewrite sees it.

**The rewrite:**
- It reads the current program and the generation's logs. These are each search's trials, scores, selected trial and the code of its top 3, as in `revise()` today but without the per-round analyses (Amendment 1).
- It also reads a table of earlier generations, with what each change was and whether it was kept.

**Test before keeping:**
- A rewrite is kept if the mean confirmed score of its 6 searches is strictly above its parent's.
- Otherwise the parent stays. The next rewrite starts from the parent and is told its change failed.
- Every program is logged, kept or not.

**Definitions:**
- **g1** is the first rewrite, kept or not.
- **final** is the last kept program. If nothing was kept, final is g0, and P4 isn't tested for that lineage.

**Facts-only and shuffled.** Both are written once per lineage, from g0 and the generation-0 logs: the same input g1 got.
- **Facts-only:** the header and every section except a new "## Facts" section must be byte-identical to g0. A checker enforces this, with up to 2 repairs, then the run stops.
- **Shuffled:** within each search, every ok trial's scores (dev success, per-seed dev success and train success) move together to another ok trial, by a permutation with a fixed seed. Failures stay failures. Score strings quoted in trial names and hypotheses are relabelled to match. The code and the rest of each hypothesis stay as written, and the procedure is free to change.

**Amendment 1 (10 November, about 12:55 GMT; before 2b finished or was analysed, and before any 2c run).** Rewrite inputs leave out the per-round analyses in every arm: g1 and later rewrites, facts-only and shuffled. The analyses quote real scores in free text, which a shuffle can't relabel reliably, so keeping them would leak real scores into the shuffled arm. Leaving them out of every arm keeps the three rewrites matched. Two leaks remain and the report will disclose them: a hypothesis can still describe a result in words ("this learned faster"), and a short score form shared by two different scores is left as written. Code: `prog2c.format_logs` and `prog2c.shuffle_search`.

**Rewrites are not tested on W before the meta-test.** Nothing in meta-training reads W's results.

### Meta-test arms on W

Each program arm has 16 searches, 8 per lineage (decision 2). Selection and confirmation are as in 2b: retest the top candidates, then confirm the pick on seeds 1001–1008.

| Arm | Searches | Source |
|---|---|---|
| g0 | 16 | **Reused from 2b's g0 arm** if conditions are identical (decision 5): the task text, trainer, checks and evaluator hashes, the model id and the header-equivalent settings. Otherwise rerun |
| g1 | 16 | 8 under each lineage's g1 |
| final | 16 | 8 under each lineage's final. If final equals g1 in a lineage, those searches are shared, and this is disclosed |
| facts-only | 16 | 8 under each lineage's facts-only |
| shuffled | 16 | 8 under each lineage's shuffled |

**The rule review:** as in 2b, before the primary analysis. A blind AI-scientist call reviews each distinct pick against the rules, and a non-compliant pick scores as the start.

**Program review (descriptive):** each program is also reviewed for advice that breaks the rules, such as hand-written features. That's reported, not scored.

## Analysis (fixed now; `analyze_2c.py` and `review_2c.py`, built 10 November, before any 2c result)

**The method:** exact two-sided permutation tests, Holm's correction within the primary family, and bootstrap 95% intervals (descriptive). The search is the unit, using Stage 1's functions.

**Primary family (Holm over 4):** P1, P2, P3, P4, as defined above.

**Secondary (reported only):**
- final − g0;
- each comparison within each lineage;
- final − 2b's best arm.

**Descriptive:**
- the full text of every program, and diffs from g0;
- the header values per generation;
- kept and rejected rewrites, with the scores behind each decision;
- runs per search actually spent;
- AI-scientist calls and cost;
- time-outs.

**Planned sample size and power.** Planning SD between searches is 0.13 (Stage 1's). It's replaced by 2b's measured g0 SD before launch. If that's larger, the change in searches per arm comes back to you. These are normal approximations, slightly more optimistic than 2b's own figures.

| Searches per arm | Test level | +5 points | +10 | +15 | +20 |
|---|---|---|---|---|---|
| 16 | 0.05 | 19% | 59% | 90% | 99% |
| 16 | 0.05/4 (Holm, first step) | 8% | 37% | 78% | 97% |
| 24 | 0.05 | 27% | 76% | 98% | over 99% |
| 24 | 0.05/4 (Holm, first step) | 12% | 57% | 93% | over 99% |

**So at 16 per arm:**
- a true gain of 15 points or more is likely to be detected;
- a 5-point gain almost certainly won't be;
- and "not established" is the expected outcome for small effects.

**With two lineages,** claims are about these two lineages' programs. They can't be about rewriting in general.

## Held-out designs

W's final split opens **once**, after every 2c confirmation run, for every 2b and 2c arm (decision 3 of the Stage 2 plan). It's irreversible, so it needs your explicit go-ahead at that time. It's never done automatically, and `FINAL_OPENED_W` is written only then.

## Integrity

- **Frozen files:** the same frozen-file hashing as 2b, plus the 2c runner, the header parser, the facts-only checker and the shuffling code. A change stops the run.
- **Long runs:** they launch in a Terminal tab, not as a session background task. On 10 November, 2b was interrupted that way.
- **The resume rule** (see `../../campaigns/stage2b_interrupted_2026-11-10/INTERRUPTION.json`): an unfinished round is moved aside whole and redrawn. The 2c runner will save each round's proposals before training, so a resume reuses them instead of redrawing.
- **Disclosure:** every departure is disclosed in the report.

## Cost and time (estimates; re-estimated after W_s is calibrated)

| Part | Training runs | AI-scientist calls | Machine time |
|---|---|---|---|
| Calibrate W_s (2a's ladder) | 50 (done) | 0 | 10 minutes (done) |
| Meta-training on W_s: 2 lineages × 4 generations × 6 searches × 33 runs | 1,584 | about 206 (searches) + 10 (rewrites) | about 6–10 h: 4 sequential generations, each about 1.5–2.5 h |
| Meta-test on W: 4 new arms × 16 searches × 38 runs | 2,432 | about 275 | about 14.5 h at 2b's measured 168 runs/h |
| Rule and program reviews | 0 | up to about 125: one per distinct pick, plus one per program | under 1 h |
| **Total** | **about 4,050** | **about 615** | **about 22–27 h**, plus about 3.5 h more if g0 must be rerun |

**Notes on these figures:**
- At about $0.21 per call (2b's figure), the AI-scientist calls come to roughly $130 notional.
- Over 600 calls may hit subscription limits. The runner waits and retries, which can stretch the run.
- The training figures assumed W_s runs cost no more than W's. The calibration then measured W_s at 14.4 s per run against W's 21.1 s, and projects 20.6 h of machine time for training (6.3 h on W_s, 14.3 h on W).

## What gets built before launch (after your approval)

1. **`w_s` world:** the spec, its targets (dev and check), and the calibration script reusing 2a's rules.
2. **`run_2c.py`:** a subclass of the 2b runner that adds:
   - the header (parse, bounds, obeying R, K, seeds, retest and split);
   - lineages and generations with test-before-keep;
   - saved proposals per round (for clean resumes);
   - the facts-only checker;
   - the shuffled-log builder;
   - the g0-reuse identity check;
   - the meta-test.
3. **`analyze_2c.py`** and the rule and program reviews, reusing 2b's functions.
4. **Tests,** plus a dry run on W_s at a tiny budget (1 lineage, 2 generations, 1 search each), before any full run.
5. **The video package** (below), tested on the dry run.

## Build status and implementation notes (10 November, about 13:15 GMT; before 2b finished or was analysed)

**Built and tested (CPU only, no GPU training, no model calls):**
- **W_s:** `ws_world.py` (train, eval, and the CPU trainer for the checks), `make_targets_ws.py` and `targets/` (512 dev, 256 check designs).
- **`check_2c.py`:** 2b's check, imported unchanged, pointed at W_s's world, check designs and CPU trainer. It passes the start (12 s on CPU).
- **`prog2c.py`:** the header, the facts-only checker and the rewrite logs (real or shuffled). `test_prog2c.py` passes.
- **`run_2c.py`:** a subclass of the 2b runner, with every item in point 2 of the build list. `test_run_2c.py` runs the whole flow with fake training, checks and model calls: 2 lineages × 3 generations, the controls, and the meta-test. It then reruns it to check that a resume makes no new calls and gives the same result. It also checks, on 2b's real files, that the round-1 prompt 2c would send for g0 on W is byte-identical to 2b's. Every check passes; the only thing blocking reuse is that 2b hasn't finished.

**Implementation choices.** These fill gaps the decisions left open and change none of them:
1. **Seeds:** a header with n seeds per candidate trains seeds 0 to n−1 and retests on n and n+1. In selection, the start's mean is over the same seed ids, so the start is trained on seeds 0–4 on both worlds. W's seeds 0–3 are 2b's own runs, read from 2b's campaign. The scientist's noise note always shows the start's 4 base seeds, as in 2b.
2. **Roles:** each round's candidates come in the order network, recipe, both.
   - "network" uses check mode `fixed` (no RECIPE);
   - "recipe" uses `recipe_only` (the start's network plus a RECIPE);
   - "both" uses `both`.
   A uniform split uses 2b's own task text; a mixed split names each proposal's role in order.
3. **What the scientist sees:** each round shows the program without its header. The header's settings reach the scientist through the task text and the budget lines, as in 2b. For g0 this makes the prompt identical to 2b's (tested).
4. **What a rewrite reads:** each rewrite starts from the current kept program and reads the logs of the generation just run. Those may be from a rejected program; the table says whether it was kept.
5. **A bad header:** the third bad header in one rewrite call falls back to the parent's header with the new body. This is recorded as `header_fallback` in the revision file and disclosed.
6. **W_s confirmation:** seeds 2001–2003, the same for every generation. Confirmations are stored per world.
7. **Resumes:** a saved model reply is reused only if its prompt is byte-identical; otherwise the run stops instead of redrawing.
8. **The rewrite prompt** (`L2C_PROMPT` in `run_2c.py`) is new. It explains the header and its bounds, and otherwise follows Stage 1's rewrite prompt. The facts-only call adds one restriction paragraph and nothing else; the shuffled call differs from g1's only in its logs (both tested).
9. **Launch guards:** `run_2c.py` refuses to start before `STAGE2B_DONE`. The real preset also needs two files:
   - `ws_budget.json`, from the W_s calibration;
   - `LAUNCH_APPROVED.md`, written only on your go-ahead after 2b's gate passes.

**W_s calibration** (`calibrate_ws.py`, built 10 November about 13:20 GMT; run on 10 November, result below):
- **Go-ahead:** the user approved running it as soon as Stage 2b finishes, on the GPU, while 2b's analysis runs on the CPU ("Yes, calibrate then", 10 November about 13:46 GMT, before 2b finished).
  - It covers the calibration only. Stage 2c's launch still needs 2b's gate to pass and a separate approval.
  - The calibration reads nothing from 2b's results and can't change them.
  - This file is in the calibration's manifest, so it isn't edited while the calibration runs.
- **Rules:** it imports Stage 2a's runner and uses its rules unchanged:
  - budgets 262,144, 524,288 and 1,048,576, with the one 131,072 fallback;
  - the same 12 ladder runs per budget;
  - the same reading: too easy if the start averages over 50% or a reference reaches 90%; too hard if both are under 20%;
  - the same noise step, and the same 3,600 s limit per calibration run.
  A diff confirms the ladder logic matches 2a's except for names.
- **What changes:**
  - the world (W_s and its hidden-cell designs);
  - the trainer: 2c's own (`train_2b.py` through `ws_world.py`), so the budget is set on the code 2c runs;
  - the machine-time projection, which counts 2c's runs.
- **Its output:** if a budget is chosen, it writes `ws_budget.json` with that budget and 2c's 900 s limit. If none passes, it writes nothing, and the result goes back to you, as in 2a.
- **Guard:** it refuses to start before `STAGE2B_DONE`.
- **Why it may come out too easy:** W_s is close to Q in size: Q is 8×8×5 and W_s is 8×8×6, both with 6–16 bricks. On Q, Stage G's recipe let the start build about 51–74% of designs, and the best networks about 95% (`PROTOCOL_2A.md`). Under 2a's rules alone, that outcome would stop calibration with no budget.
- **Amendment 2** (approved by the user on 10 November, about 13:18 GMT: "Yes, extend downward"; before any W_s training run):
  - **When:** if the 131,072 fallback is also too easy, the ladder keeps halving: 65,536, then 32,768, then 16,384 steps. Each step uses the same 12 runs and the same reading.
  - **Where it stops:** at the first budget that passes, which is then chosen (the largest passing budget), or at the first that is too hard, with no budget.
  - **If all are too easy:** it stops with no budget, and the result goes back to you.
  - **Never triggered** when the fallback passes or is too hard.
  - **Tested** with fake runs: one budget chosen on the way down, too easy throughout, too hard on the way down, and not triggered.
  - **A consequence:** the budget check needs at least 4 updates.
    - **At 16,384 steps:** the start with Stage G's recipe gets exactly 4 updates of 4,096 steps. A network-only candidate (default recipe) therefore can't cost more per sample than the start; a candidate that may change the recipe can use a smaller batch.
    - **At 32,768 steps:** a network-only candidate can cost up to twice the start per sample.
    - **The same limit applies to the reference networks:** one costing more per sample than the start can't train at 16,384 steps and would score 0 there.

**W_s calibration result** (run 10 November, 15:07 to 15:17 UTC, after Stage 2b finished; files in `calib/`):
- **Chosen budget: 131,072 steps**, the same as W's. `ws_budget.json` holds it, with 2c's 900 s limit per run.
- **The ladder** (12 runs per budget; the start is scored on 6 seeds, each reference on 2):

  | Budget | Start, Stage G's recipe | Better reference | Reading |
  |---|---|---|---|
  | 262,144 | 65.3% | 92.2% | too easy |
  | 131,072 (the fallback) | 21.0% | 60.4% | passes |

- **Amendment 2 wasn't used:** the fallback passed, so the ladder never went below 131,072.
- **Noise at 131,072 steps:**

  | Network and recipe | Seeds | Mean | SD between seeds | Seeds that learned the hidden-cell skill |
  |---|---|---|---|---|
  | Start, Stage G's recipe | 20 | 23.5% | 0.114 | 9 |
  | Start, frozen recipe | 6 | 21.5% | 0.171 | 2 |
  | Reference A | 6 | 69.0% | 0.096 | 6 |
  | Reference B | 6 | 20.4% | 0.116 | 4 |

  - The start's two seed ranges averaged 21.7% and 25.3% (permutation p 0.50), so there's no sign that the seed range matters.
  - Headroom above the start is 76.5 points.
- **Runs:** 50, none failed or timed out. The slowest took 65.5 s, well under the 900 s limit.
- **Time projection for Stage 2c:** 20.6 h of machine time. That's 6.3 h on W_s (1,589 runs at an effective 14.4 s per run, with four running at once) plus 14.3 h on W (2,432 runs at 2a's effective 21.1 s per run).
- **A caveat on the timing:** the calibration ran while 2b's analysis used the CPU, as the user approved. That can only have slowed these runs, so the W_s time is, if anything, an overestimate.

**Analysis and reviews** (`analyze_2c.py` and `review_2c.py`, built 10 November about 13:30 GMT; tested on the fake campaign with fake reviews, not yet run):
- **Tests:** Stage 1's exact permutation test, bootstrap intervals and Holm correction, all imported unchanged. At 16 searches per arm, each exact test enumerates about 601 million splits, about 5–7 minutes each. A full analysis takes roughly half an hour.
- **Rule review:** 2b's reviewer, prompt and schema, imported unchanged, one blind call per distinct pick.
  - **Compliant by construction:** the start, and any pick with the start's network unchanged.
  - **Reused verdicts:** a pick 2b already reviewed (the same file) keeps 2b's verdict. That covers g0's picks when g0 is reused.
  - **Completeness:** the analysis stops if any pick has no verdict.
- **Scoring a violation:** a pick judged a violation scores as the start with Stage G's recipe on the same seeds, using 2b's `start_g` comparator runs. 2c doesn't rerun those, and their conditions are the ones the g0 identity check confirms.
- **P4 when it can't be tested:** if no lineage kept a rewrite, P4 enters Holm with p = 1, so the family stays at 4 and the correction isn't loosened.
- **Program review (descriptive):** one blind call per distinct program, asking whether it advises anything the rules forbid. It's reported, not scored.
- **Post hoc, labelled as such:** "2b's best arm" in the secondary family is the 2b search arm with the highest primary mean, chosen after 2b's results.

**Still to do before launch:**
- the dry run (`--preset 2c_dry`), which needs the user's go-ahead;
- replace the planning SD with 2b's measured g0 SD, once 2b's analysis is final.

## Video build (fixed before any 2c result; the user asked for it on 10 November)

Same format as the other stage videos:
- 1920×1080, 12 fps, captions on screen, no voice-over;
- generic wording: "the AI scientist", no model or person names;
- every number read from 2c's own files and checked against `results.json`;
- the renderer refuses to run unless results are final.

| Scene | About | What it shows | Fixed rule |
|---|---|---|---|
| 1. Where this sits | 8 s | The three-loop system diagram (`../../viz/stages/system/`), with Loop 3 highlighted and its status chip updated | — |
| 2. The program evolves | 25 s | **Left:** a lineage's program, g0 → g1 → … → final, with each change and its evidence, stamped kept or rejected. **Right:** an autoresearch-style chart, one dot per generation (mean confirmed W_s score), kept rewrites labelled, the running best as a step line | Lineage 1, always, whatever its result. The diff text comes from the revision files, unedited |
| 3. One search under the final program | 25 s | The single-frame view (`../../viz/stage2b/search_build.py`): the build, the evaluation grid and the progress chart, on W | The final-arm search at position ⌊m/2⌋ by confirmed score, as in 2b's rule. If the header changed R or K, the chart shows the new number of experiments |
| 4. The answer | 15 s | Every arm on W with intervals: g0, g1, final, facts-only, shuffled; P1–P4 with Holm-adjusted p; which tier was reached, in the pre-written words | Wording from a 2c endings file written before the result, as for 2b |
| 5. The limits | 6 s | Toy scale, two lineages, one model, the model's weights unchanged, and whether the held-out designs were opened | — |

**Checks (the renderer stops on any mismatch):**
- the program diffs must match the saved programs byte for byte;
- the per-generation means must match the meta-training logs;
- the arm means must match `results.json`;
- the scene 3 build must match its run's per-design result, as in `search_build.py`.

**Built** (`../../viz/stage2c/make_2c_video.py`, 10 November about 13:50 GMT; before 2b finished and before any 2c run):
- **Tested on:** the fake campaign from `test_run_2c.py`, with a fake review and a `results.json` made by `analyze_2c.py`. The test video is 79 s, 1920×1080, 12 fps. None of its numbers is a result.
- **Scene 3 on real checkpoints:** tested once, on a search from 2b's dry run on W (tiny budgets, not a result). The replay matched the evaluator's per-design result. No 2c search exists yet, so scene 3 hasn't run on one.
- **Choices fixed in the build:**
  - **Scene 2's lineage:** A, always (the table's "lineage 1").
  - **Quoted changes:** copied from the revision files. A quote longer than 220 characters is cut to 220, ending with "…". It's never reworded.
  - **Scene 3's search:** among final-arm searches whose pick isn't the start and was judged compliant, sorted by primary score, the one at position ⌊m/2⌋. Ties keep the order in `results.json`. If the final arm wasn't run, or no pick is eligible, the scene is a card that says so.
  - **Scene 3's design:** the first dev design with hidden cells, as in `search_build.py`.
  - **Scene 4's words:** one ending from `ENDINGS_2C.md`, written before any result and chosen by `results.json`'s tier.
  - **16 searches per arm:** the endings state the design's power at 16 per arm, so the renderer stops unless g0, g1, facts-only and shuffled each have 16. `--allow-small` is for tests only.
- **Checks in `load()`**, each stopping the render:
  - `results.json` is final;
  - each program file's hash matches the one saved with its generation and its revision;
  - each generation's mean is recomputed from the `confirm/ws` run files;
  - each search's score and each arm's mean match `results.json`.
- **Changes to shared drawing code** (not frozen, not in 2b's manifest):
  - `viz/stages/system/make_system.py` gained `render()`, with status chips and a highlighted loop as arguments. Its default output is byte-identical to before.
  - `viz/stage2b/search_build.py` now takes the world, the training seeds and the round settings from the search's own files. Its Stage 1 stills are byte-identical to before.
  - **Later, on 10 November after Stage 2b's result:** both files changed again for the Stage 2b video (status chips, label placement, ties with the bar), so those outputs no longer match byte for byte. The changes are listed in `../../viz/stage2b/README.md`. The 2c video tests still pass.

**Stage 2b's video build is ready to render after 2b ends:**
- the system diagram;
- the single-frame search view (`search_build.py`, with W's spec and targets now wired in);
- the answer chart (`search_chart_2b.py`);
- the limits card.

## Decisions (approved by the user on 10 November 2026: the proposed default for each)

1. **The meta-training world:** W_s, 8×8×6 with 6–16 bricks, calibrated by 2a's rules (proposed). The alternative is W with a separate set of meta-training designs. That's cheaper to set up, but it can't show transfer across worlds.
2. **Searches per arm on W:** 16 (proposed, matching 2b), or 24. 24 adds about 7 h and raises power for +10 points from 37% to 57% at the first Holm step.
3. **Lineages and generations:** 2 lineages × 3 rewrites, with 6 searches per generation (proposed). More generations test compounding better, at about 2 h each.
4. **What the controls are matched to:** g1 (proposed; each comparison changes one thing), or the final program (the plan's literal wording, which mixes iteration with evidence).
5. **The g0 arm:** reuse 2b's 16 g0 searches if conditions are identical (proposed; saves about 3.5 h), or rerun them.
6. **Header fields and bounds** as listed (proposed), or fewer fields, for example dropping the split.
7. **Held-out designs:** open once after 2c, on your go-ahead at that time (as already decided).
