# Stage 2a protocol: calibrate world W

Written 9 November 2026, before any training on W. The user approved the Stage 2 plan (`../PLAN_STAGE2.md`, decisions 1–7) that day. `run_2a.py` implements exactly these rules, and its `manifest.json` records this file's SHA-256 before the first run. No Claude calls are made.

## Why

Stage 2b asks again whether experiments guide the search (Level 1b). On Q under Stage G's recipe, that question has no room left: the start builds about 51–74% of designs, and the best networks 95%. W has to leave room above the start, without being so hard that nothing learns.

| This stage can show | It can't show |
|---|---|
| A budget on W where the start under Stage G's recipe builds at most half the designs and good Q networks stay under 90% | That a search on W will find better networks; nothing is searched here |
| How much the start varies across seeds and across two seed ranges on W | Why the start varied across seed ranges on Q |
| Roughly how much machine time Stage 2b would take | The exact cost, which depends on what Claude writes |

## The world and its designs

- **W:** 10×10×6, built from 8–20 bricks (`w_world.py`), using the frozen environment code unchanged. On Q it's 8×8×5 with 6–16 bricks.
- **Designs:** `make_targets_w.py` made dev (512, seed 9009), final (1,024, seed 9119) and check (256, seed 9229) designs, once, before this protocol. The sets are disjoint.
  - Dev designs average 31.9 cells, and 87.1% of them have hidden cells.
  - The final set (`holdout/w_final.pt`) is read-only and stays unopened until the end of Stage 2. The evaluator refuses it without a `FINAL_OPENED_W` marker.
- **Training and scoring:** the frozen GPU trainer and the frozen evaluator, unchanged, through `w_world.py`. That file only adds W to their world lookups.

## Networks and recipes

- **The start:** `candidates/baseline.py`.
- **Two reference networks:** the two best retest picks whose median retest run time was at most 3 times the start's (48.6 s):
  - no-feedback search 9's pick, `1c9c0e6b…`: 95.4%, 116.5 s;
  - no-feedback search 5's pick, `722ae3f7…`: 94.1%, 101.3 s.
  - **Why the time rule:** the second-best pick by score, g0 search 1's `ef6b9b74…` (94.5%), took 701 s, 14 times the start's, though its FLOP ratio is 1.03. On W it could take hours per run. This replaces the plan's "g0 search 1's pick". The choice is made from the retest's results, and it touches no claim.
- **Recipes:**
  - **Stage G's:** 256 worlds per update, learning rate 0.001, 32 minibatches. It's used for every network unless marked otherwise.
  - **The frozen recipe** (64 worlds, 8 minibatches): used for the start only, for reference.
- **Setting:** 4 runs at once on the GPU, with a 3,600 s limit per run. A run that fails or passes the limit scores 0.

## Step 1: budget ladder

**Budgets:** 262,144, then 524,288, then 1,048,576 steps.

**At each budget, 12 runs:**
- the start with Stage G's recipe, seeds 2000–2005;
- the start with the frozen recipe, seeds 2000–2001;
- each reference network, seeds 2000–2001.

**Reading each budget,** with S the start's mean under Stage G's recipe and R the better reference network's mean:
- **Too easy:** S > 50% or R ≥ 90%.
- **Too hard:** both S and R below 20%.
- **Passes:** neither.

**Order:**
- Budgets run in ascending order. The ladder stops after the first "too easy" budget, since a larger one would only be easier.
- The chosen budget is the largest one that passes.

**If none passes:**
- **The smallest was too easy:** run 131,072 steps once, with the same 12 runs. If it passes, it's chosen. Otherwise stop: W is too easy.
- **Every budget was too hard:** stop: W is too hard.
- **Either way,** the result goes back to the user. A different world needs a new version of this protocol, disclosed.

## Step 2: noise at the chosen budget

Runs are added so that, at the chosen budget, there are:
- the start with Stage G's recipe on 20 seeds, from two ranges: 2000–2009 and 3000–3009;
- each reference network on 6 seeds, 2000–2005;
- the start with the frozen recipe on 6 seeds, 2000–2005.

Ladder runs are reused, not repeated.

## What's reported (`runs/decisions.json`, `runs/summary.json`)

- **Per budget:** each cell's per-seed scores, mean, and seeds that learned the hidden-cell skill. A seed learned it if it builds at least 10% of W's dev designs with hidden cells, as in Stage 1. Also the reading.
- **At the chosen budget:**
  - the start's mean and SD over all 20 seeds and per range;
  - an exact two-sided permutation test between the ranges (descriptive only);
  - the references' means;
  - the headroom, 1 minus the start's mean.
- **Machine time for Stage 2b:**
  - **The run count:** 80 searches (5 arms × 16) × 38 runs each (12 candidates × 2 seeds, 3 × 2 retest seeds, 8 confirmation seeds), plus 24 comparator runs.
  - **The time per run:** the measured wall time at the chosen budget divided by 4 runs at once. The references' figure stands in for Claude's networks, and the start's for the comparators.
  - **The flag:** if the projection is over 48 hours, it goes to the user with options: fewer searches, fewer arms or a lower budget. It doesn't change this stage's result.
- **The spread Stage 2b's power table needs** (`PLAN_STAGE2.md`): only the start's per-seed SD is measured here. The spread between searches isn't known until Stage 2b's first searches.

## Integrity

- **Before every run,** the runner re-hashes:
  - the frozen files;
  - `gpu/train_gpu.py`, `w_world.py` and `make_targets_w.py`;
  - W's four design files (the final set is hashed, never read);
  - this protocol, the runner, and the three network files.
  Any change stops the run.
- **No Stage 1 or Q files are written.**
