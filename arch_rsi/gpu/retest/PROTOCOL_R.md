# Retest protocol: do Stage 1's picks still beat the start under the new recipe?

Written 9 November 2026, after Stage G and before any retest run. `run_retest.py` implements exactly these rules, and its `manifest.json` records this file's SHA-256 before the first run. The user approved the retest on 9 November ("Yes to all").

## Why

Stage G found that a recipe change, with no architecture change, lifts the starting network on Q from 30% to about 70% at 262,144 steps. Under that recipe, Stage 0's reference designs no longer beat it.

Stage 1 measured Claude's picks under the old recipe at 131,072 steps, where they beat the start by about 50 points. This retest asks whether that lead survives the better recipe.
- **What it can show:** whether those architectures still help once training stops starving the start.
- **What it can't show:** whether a search run under the new recipe would find better designs. Nothing is selected from these results.

## What runs

- **Networks:** every distinct pick from Stage 1, taken from `campaigns/stage1/confirm/<sha>/candidate.py`:
  - g0: 10 picks;
  - no feedback: 10 picks;
  - menu: 5 distinct picks. Menu searches with the same pick share its runs, and menu search 2 picked the start, so it uses the start's runs.
  - Plus the start and the start with a critic and orthogonal init (`critic_ortho`).
- **Seeds:** as in Stage 1, picks on 1001–1008, and the start and critic_ortho on 1001–1012.
- **Setting:** Q (`xl`) at 262,144 steps, with Stage G's recipe: 256 worlds per update, learning rate 0.001, 32 minibatches. Everything else is frozen. Training is compute-matched as before.
- **Hardware:** on the GPU with `train_gpu.py`, 4 runs at once, scored by the frozen `evaluate_arch.py` on the 512 dev designs. A run that fails or passes the 1,800-second limit scores 0.
- **Integrity:** before every run, the runner re-hashes the frozen files, `train_gpu.py`, this protocol, the runner and every candidate file. Any change stops it. The final split stays locked.

## Analysis (fixed now)

**Scores:**
- A search scores the mean of its pick's 8 seeds.
- The start and critic_ortho score per seed.
- The tests are Stage 1's own functions (`stage1/analyze_stage1.py`): an exact two-sided permutation test over all splits, Holm's correction within each family, and bootstrap 95% intervals (10,000 draws, seed 0), which are descriptive only.

**Primary family (Holm over 2):**
- **R1:** g0 − start.
- **R2:** no feedback − start.

**Reading, fixed before any result:**
- **The lead survives:** both differences positive with Holm-adjusted p < 0.05. Report each beside Stage 1's: +52.5 for g0 and +50.9 for no feedback.
- **The lead is gone:** neither passes. Report each interval's upper end, the largest lead the data still allows.
- **Mixed:** exactly one passes; report it as mixed.

**Secondary family (Holm over 5, reported only):** g0 − no feedback, menu − start, critic_ortho − start, g0 − menu, no feedback − menu.

**Descriptive only:**
- **How much lead survives:** each arm's mean lead over the start, under each recipe.
- **Rank agreement:** the Spearman correlation, across the 25 distinct picks, between each pick's Stage 1 score (8 seeds, old recipe) and its retest score. Does a design that ranked well under the old recipe still rank well?
- **Seeds that learned the hidden-cell skill**, per arm.
