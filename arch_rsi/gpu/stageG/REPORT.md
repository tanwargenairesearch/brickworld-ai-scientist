# Stage G: calibrating the GPU recipe (pilot, disclosed). Result

**When:** 9 November 2026, from 15:07:39 to 15:42:26 UTC (35 minutes).

**Under what rules:** `PROTOCOL_G.md`, SHA-256 `d69804ee…`, recorded in `runs/manifest.json` before the first run.
- No Claude calls; dev designs only.
- The final split was never read, and `FINAL_OPENED` doesn't exist.
- No integrity stop. All 130 runs finished: 120 on the GPU, 10 on the CPU.
- Stage 1's picks were never read. The runner read only Stage 1's 12 comparator scores for the starting network, for check G6a.

**Source of every number:** `runs/decisions.json`, plus one extra check marked as such.

## The headline: a recipe change teaches the starting network the skill

Every row below uses the same world (Q), the same budget (262,144 steps) and the same seeds.

| Recipe | Device | Start network, exact success per seed | Mean | Seeds that learned the hidden-cell skill |
|---|---|---|---|---|
| Frozen: 64 worlds per update, 8 minibatches (Stage 0's sweep) | CPU | 16, 16, 78, 42, 15, 16% | 30.4% | 2 of 6 |
| Stage G's pick: 256 worlds per update, 32 minibatches (G3) | GPU | 54, 54, 61, 83, 79, 82% | 68.8% | 6 of 6 |
| Stage G's pick (extra check, seeds 300–302) | CPU | 86, 54, 76% | 72.1% | 3 of 3 |

**The two recipes do the same amount of optimisation:**
- Both take 8,192 gradient steps of 128 samples.
- Each sample is used 4 times, at the same learning rate and decay.

**What differs is how the samples are collected.** With 4 times as many worlds per update, each update collects 4,096 samples instead of 1,024, and the policy is refreshed 64 times instead of 256.

**It's the recipe, not the device:**
- The extra check gives the same jump on the CPU.
- G6a found no shift from the device alone (below).

**Stage 0's reference designs no longer beat the start.** These two networks were written by Claude in Stage 0's pilot search. Q results at 262,144 steps:

| Network | Frozen recipe (CPU) | Stage G recipe (GPU) |
|---|---|---|
| Start | 30.4% (6 seeds) | 73.7% (10 seeds) |
| r1c1 | 74.2% (4 seeds) | 71.3% (2 seeds) |
| r4c3 | 91.0% (4 seeds) | 71.2% (2 seeds) |

**What this does and doesn't say about Stage 1:**
- **It stands as stated.** Under the frozen CPU recipe at 131,072 steps, Claude's picks beat the start by about 50 points on dev designs.
- **It may be partly a recipe effect.** Some of that advantage may come from architectures that cope with a recipe that starves the start.
  - The two reference designs lost their whole lead under the new recipe, but they are Stage 0 pilot designs, not Stage 1's picks.
  - By design, Stage G never tested Stage 1's picks, so it can't say how much of their lead survives.

## Decisions, step by step

**G1. Throughput: 256 worlds per update, 4 GPU runs at once.** The score is the geometric mean, over three networks, of total training steps per second.

| Worlds per update | 1 run | 2 runs | 4 runs |
|---|---|---|---|
| 256 | 8,019 | 13,896 | 16,621 |
| 1,024 | 9,530 | 15,365 | 17,696 (best) |

- **Why 256:** it scores within 10% of the best, and the rule picks the smaller setting.
- **Against the CPU:** with 4 GPU runs at once, the start network trained at 29,915 steps per second in total.
  - Ten CPU runs at once with the frozen recipe managed 10,359, timed from process start, so not like-for-like. That's about 2.9 times.
  - The two reference networks ran at 11,900–12,900 steps per second on the GPU. They weren't timed on the CPU.

**G2. Recipe: learning rate 0.001, 32 minibatches.** The start network on Q, 524,288 steps, seeds 400–403:

| Learning rate | Minibatches | Per seed | Mean | Learned |
|---|---|---|---|---|
| 0.001 | 8 | 86, 87, 16, 87% | 68.9% | 3 of 4 |
| 0.001 | **32** | 95, 96, 96, 96% | **95.7%** | 4 of 4 |
| 0.002 | 8 | 95, 92, 55, 90% | 83.2% | 4 of 4 |
| 0.002 | 32 | 96, 96, 96, 92% | 95.0% | 4 of 4 |

- **How it was picked:** the two 32-minibatch cells tied (within 5 points), and the tie went to the lower learning rate.
- **Clearer than expected:** the protocol warned this choice might be a coin toss, but the gap between 8 and 32 minibatches was large.

**G3. Q budget: 262,144 steps (64 updates).**

| Budget | Start (6 seeds) | r1c1 (2 seeds) | r4c3 (2 seeds) | Passes (better reference < 90%)? |
|---|---|---|---|---|
| 262,144 | 68.8%, 6 learned | 71.3% | 71.2% | yes |
| 524,288 | 84.4%, 6 learned | 91.8% | 96.7% | no; larger budgets not run |

**G4. P budget: 262,144 steps (64 updates).**

| Budget | Start (6 seeds) | r1c1 | r4c3 | Passes? |
|---|---|---|---|---|
| 262,144 | 87.0%, 6 learned | 84.1% | 84.3% | yes |
| 524,288 | 92.5%, 6 learned | 96.0% | 96.2% | no |

**G5. Noise at the chosen budgets, the start over 10 seeds (300–309):**

| World | Scores | Mean | SD | Learned |
|---|---|---|---|---|
| Q | 54, 54, 61, 83, 79, 82, 76, 85, 76, 86% | 73.7% | 0.127 | 10 of 10 |
| P | 85, 90, 87, 84, 91, 85, 55, 88, 75, 55% | 79.5% | 0.136 | 10 of 10 |

- **The noise is no longer two-humped.** Every seed learned the skill. On Q the spread is 54–86%, and the SD fell from about 0.30 under the frozen recipe to 0.13.
- **The power table doesn't apply.** The protocol's model needs seeds in both humps. With no seed failing to learn, it fell back to Stage 0's levels, so the table in `decisions.json` doesn't describe this recipe. I haven't used it.

**G6. Device checks: passed.**
- **(a) The device alone doesn't shift scores.** The start with the frozen recipe on the GPU (seeds 1001–1012) averaged 29.2%, with 5 of 12 seeds learning the skill. Stage 1's CPU runs on the same seeds averaged 22.7%, with 2 of 12. Exact two-sided permutation test over 2,704,156 splits: p = 0.456, so no stop.
- **(b) The GPU is deterministic.** Repeating a seed gave bit-identical trained weights and the same dev score, for both the frozen recipe (seed 1001) and the new one (seed 400).

## Where the protocol's rules fall short

1. **The ceiling rule doesn't measure headroom any more.**
   - It assumed the reference designs mark how high a good design can go. Under this recipe, they score no higher than the start.
   - At Q's chosen budget, the start is at 74%. That leaves at most 26 points for any search to find, and how much of that is reachable is unknown.
2. **P is near its ceiling for the start:** 87% over 6 seeds, 79.5% over 10. The budget ladder started at 64 updates, so nothing lower was tried. Stage 2 meta-trains on P, where there is little left to gain.
3. **Stage 2's sample sizes need a new noise model.** The decision goes back to the user anyway, because Stage 1's gate failed.

## What this means for Stage 2 (the user's decision)

Stage 2 was going to compare search programs by how well their searches beat the start. Under this recipe, the start already does most of what the searches' designs did. Three ways forward:

1. **Retest Stage 1's picks under the new recipe first.** Retrain the 20 Claude picks and the start on Q at 262,144 steps, before any Stage 2 design work. This settles whether Claude's architecture gains survive a better recipe.
2. **Make the task harder,** so that a good recipe doesn't saturate it: a larger world, or a lower budget.
3. **Let the search change the training recipe as well as the architecture.** Stage G shows the recipe is where the large gains were, and a loop that can rewrite its own training is closer to the goal.

## Departures and disclosures

1. **Extra check outside the protocol:** the start with Stage G's recipe on the CPU, seeds 300–302, run after Stage G finished (`extra_recipe_on_cpu.py`, `extra_cpu_recipe/`). It isn't part of any decision.
2. **Loop timing in the trainer.** Before the run, `train_gpu.py` gained timing of the training loop itself, which G1 uses. The maths is unchanged, and `manifest.json` hashes the edited file.
3. **Machine load:**
   - A 33-second, low-priority re-render of the Stage 1 learning video overlapped G2's first batch. G2 is scored on dev results, not time, but those runs' wall times may read slightly high.
   - Nothing else ran during G1's timing.
4. **Budgets as updates.** The protocol defines budgets as updates at G1's worlds per update. With 256 worlds, G2 ran at 524,288 steps, not the 2,097,152 that the plan assumed for 1,024 worlds.

## Costs

- **Machine:** 35 minutes for Stage G, plus about 3 minutes for the extra check.
- **Claude calls:** $0.
