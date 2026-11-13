# Stage 2b protocol: do experiments guide the search when it controls the recipe? (Level 1b again)

Written 9 November 2026, after Stage 2a and before any Stage 2b run. The user approved the Stage 2 plan (decisions 1–7), and then the four 2b changes in `../PLAN_STAGE2.md` ("Yes go for it"). `run_2b.py` implements these rules, and the campaign's `manifest.json` records this file's SHA-256 before the first run.

## What it can and can't show

| Comparison | If it holds, we may write | What it can't show |
|---|---|---|
| **P1 (gate):** g0 > no feedback | "Experiments guided the search" (Level 1b), in a world with headroom where the search controls the recipe and the network | That the procedure improved itself (that's Stage 2c). Anything beyond W, toy scale and one model |
| **P2:** g0 > reference A | Searching on W beats the best network carried over from Stage 1, trained the same way | That it beats a well-tuned human search |
| g0 > network only | Controlling the recipe adds to architecture search (secondary) | Which recipe setting mattered |
| g0 > recipe only | Architecture adds to recipe search (secondary) | |
| g0 > random | Claude beats random draws from the same space (secondary) | |
| No difference anywhere | "First ideas plus selection" again | That experiments never help: power is limited |

## Setting (fixed by Stage 2a)

- **World:** W, 10×10×6 with 8–20 bricks (`../w_world.py`), scored on W's 512 dev designs. The final split stays locked, and it opens once at the end of Stage 2, as decision 3 says.
- **Budget:** fixed training FLOPs, equal to what the start gets in 131,072 steps with Stage G's recipe. The run's cost counts:
  - one forward pass per collected sample, taken as one third of a forward-and-backward pass;
  - one forward and backward pass per sample per PPO epoch.

  So a network with FLOP ratio r (counted by the frozen counter; r below 1 counts as 1) and E epochs trains

  steps = 131,072 × (4 + 1/3) / ((E + 1/3) × r),

  in whole updates of n_envs × rollout_len steps. With E = 4, this is Stage 1's compute matching.
- **Time limit:** 900 s per training run, beside the FLOP budget, because the FLOP counter misses some costs (`../../gpu/profile/REPORT.md`). A run over the limit scores 0, which is disclosed.
- **Trainer:** `train_2b.py`, the frozen GPU trainer's PPO update maths unchanged. It adds only the RECIPE block, its bounds and the FLOP budget. It runs on the GPU, with 4 runs at once.

## The RECIPE block

A candidate file may define one top-level constant, `RECIPE = {...}`. Any key it leaves out takes Stage G's value.

| Key | Stage G default | Bounds |
|---|---|---|
| `lr` | 0.001 | 1e-4 to 3e-3 |
| `n_envs` | 256 | 64–1,024, integer |
| `rollout_len` | 16 | 8–64, integer |
| `n_minibatches` | 32 | 4–64, integer, at least 64 samples each, and must divide n_envs × rollout_len |
| `ppo_epochs` | 4 | 1–8, integer |
| `ent_coef` | 0.01 | 0–0.05 |
| `gamma` | 0.95 | 0.9–0.995 |
| `gae_lambda` | 0.9 | 0.8–0.99 |
| `clip_eps` | 0.2 | 0.1–0.3 |

**Two rules added to the plan's bounds,** so a run can't overspend or crash:
- n_minibatches must divide the batch;
- the budget must allow at least 4 updates.

**Everything else is fixed:** Adam with linear decay to 0, value coefficient 0.5, gradient-norm clip 1.0, normalised advantages, the reward, the observation and the design stream.

## Pre-training checks (`check_2b.py`)

- **As in Stage 1:** the sandbox rules, the parameter cap (1M), the FLOP cap (12×), and the learned-policy rule on W's 256 check designs (the short run is 8,192 steps with the frozen recipe on the CPU).
- **The RECIPE block:** it must be within bounds and fit the budget.
- **Per-arm rules:**
  - network only: no RECIPE block;
  - recipe only: the network must be the start's, so the runner builds the file from the start's code plus the proposed RECIPE.
- **Repairs:** a failing file goes back to Claude for up to 2 repairs, as in Stage 1.

## Arms

There are 16 searches per arm, run interleaved. Each search has 4 rounds of 3 candidates, each trained on seeds 0 and 1.

| Arm | Who proposes | What it may change | What it sees |
|---|---|---|---|
| g0 | Claude Opus 5.5 (high effort, no tools) | network + RECIPE | everything, as in Stage 1 |
| No feedback | Claude | network + RECIPE | no success scores (Stage 1's hidden prompt) |
| Network only | Claude | network; recipe fixed at Stage G's | everything |
| Recipe only | Claude | RECIPE; network fixed (the start) | everything |
| Random | seeded draws | a random menu network (143 options) × a random recipe within bounds | — |

**The program:** every Claude arm reads `../programs/g0r.md`. That's g0 with its goal sentence generalised to "network and training recipe, as the task allows", and nothing else changed. The task text states what each arm may change.

**Random draws:**
- lr: log-uniform;
- n_envs: from {64, 128, 256, 512, 1,024};
- rollout_len: from {8, 16, 32, 64};
- n_minibatches: from {4, 8, 16, 32, 64};
- epochs: 1–8;
- the others: uniform within their bounds.

A draw that breaks a bound or the budget is redrawn. The draw seed is `7919 × search_seed + 17`.

**Selection, as in Stage 1:**
- The 3 trials with the best 2-seed mean each train seeds 2 and 3.
- The pick is the best 4-seed mean among them, if it beats the start's own 4-seed mean (the start with Stage G's recipe, seeds 0–3). Otherwise the pick is the start.
- The pick is then confirmed on 8 fresh seeds (1001–1008). A search scores the mean of those 8.

**Comparators, no search, seeds 1001–1012:**
- reference A (no-feedback search 9's Stage 1 pick) with Stage G's recipe;
- the start with Stage G's recipe;
- the start with the frozen recipe (as a RECIPE block).

## Analysis (fixed now; `analyze_2b.py`)

**Methods:** exact two-sided permutation tests, Holm's correction within each family, and bootstrap 95% intervals (descriptive), all with Stage 1's own functions.

**Primary family (Holm over 2):**
- **P1 (the gate):** g0 − no feedback.
- **P2:** g0 − reference A.

**Gate:** P1 positive with Holm-adjusted p < 0.05.
- If it fails, Stage 2c doesn't run, and the result goes back to the user.
- P2 is reported either way, and it doesn't gate.

**Rule review before the primary analysis:** a fresh Claude call reviews each distinct Claude-written pick against the frozen rules, as in Stage 1. A pick judged non-compliant is scored as the start with Stage G's recipe.

**Secondary family (Holm over 5, reported only):**
- g0 − network only;
- g0 − recipe only;
- g0 − random;
- g0 − (start with Stage G's recipe);
- no feedback − reference A.

**Descriptive:**
- per-arm means, SDs between searches, and seeds that learned the hidden-cell skill;
- selection optimism;
- each round's running best;
- the recipes chosen, by arm;
- time-outs;
- Claude calls and cost.

## Costs and stopping

- **Claude calls:** about 4.3 per search, so about 275 for the four Claude arms, roughly $60 at about $0.21 each. Repairs and the rule review add to that.
- **Machine time:** about 18 hours by Stage 2a's projection.
- **The run stops** on any integrity change (`INTEGRITY_STOP`) or a repeated infrastructure failure.
- **Nothing here opens W's final split.**
