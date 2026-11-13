# GPU profile: where a training run's time goes

Run 9 November 2026, after the retest, at the user's request. It's not part of any experiment.

**What ran:**
- `profile_run.py` is a copy of `train_gpu.py`'s training loop (same maths, same order of random draws), with a timer around each section.
- Each network ran once, alone on the GPU: Q, Stage G's recipe, 262,144 steps, seed 1001.
- Each timer waits for the GPU to finish before reading the clock. Sections are measured exactly, but the GPU can't overlap work across sections, so the total is a little different from an untimed run.

**Fidelity:** each profiled run produced the same trained weights and the same result on every dev design as its retest run, bit for bit. So the copy trains exactly what the trainer trains.

## Results

| Section | Start (`764c7fb5`) | No-feedback search 9's pick (`1c9c0e6b`) | g0 search 1's pick (`ef6b9b74`) |
|---|---|---|---|
| PPO update: backward pass | 15.3 s (42%) | 46.3 s (57%) | 249.6 s (69%) |
| PPO update: gradient clipping + Adam step | 9.9 s (27%) | 16.5 s (20%) | 19.5 s (5%) |
| PPO update: forward pass + loss | 5.7 s (16%) | 11.7 s (14%) | 68.3 s (19%) |
| Environment: step | 2.3 s (6%) | 2.3 s (3%) | 2.7 s (1%) |
| Rollout: policy forward + sampling | 1.6 s (4%) | 2.6 s (3%) | 17.9 s (5%) |
| Environment: valid actions, observation, reward | 1.0 s (3%) | 1.0 s (1%) | 1.1 s (0%) |
| Rollout: storing, GAE | 0.2 s (1%) | 0.3 s (0%) | 1.2 s (0%) |
| **Total (timed)** | **36.3 s** | **80.8 s** | **360.6 s** |

## What it means

1. **The environment isn't the bottleneck.** Stepping the brick world, building observations and computing rewards take about 9% of the start's time, and 4% of a larger network's. A native Metal environment could save at most that much.
2. **The PPO update is 85–92% of the time.**
   - Stage G's recipe makes 8,192 optimiser steps per run (64 updates × 4 epochs × 32 minibatches), each on only 128 samples.
   - At that size, the GPU spends much of its time launching many small operations. Gradient clipping and the Adam step alone take 27% of the start's time, for 58,886 parameters.
3. **The cheapest speed-ups are in the update, not the environment.** Both of these exist in this PyTorch (2.12.1):
   - a fused Adam step (`fused=True`);
   - a single-pass gradient clip (`foreach=True`).
   They change floating-point rounding, so results wouldn't match earlier stages bit for bit. That's fine for a new trainer used by every Stage 2b arm, but not for the frozen trainer. Larger minibatches would also help, but they change the recipe, so in Stage 2b that's the search's choice.
4. **Hand-written kernels for each candidate network** would target the 85% share. That's the heavy path from earlier: forward and backward kernels per network, checked against PyTorch.
5. **Some operations cost far more on this GPU than their arithmetic suggests.** g0 search 1's pick (`ef6b9b74`) is built from 3D convolutions.
   - **The run:** it took 10 times as long as the start (360.6 s against 36.3 s), though the FLOP counter rates it at 1.03 times the start's compute. Its backward pass alone was 69% of the run.
   - **A micro-benchmark** (batch 128) on the GPU, with layers doing the same work (about 2.2 million multiply-adds per sample):

     | Layer | Forward | Forward + backward |
     |---|---|---|
     | 3D convolution, 16 → 16 channels, 5×8×8 | 0.84 ms | 4.17 ms |
     | 2D convolution, 62 → 62 channels, 8×8 | 0.08 ms | 0.73 ms |

     So 3D convolutions run 6–10 times slower here than 2D ones of equal compute.
   - **For Stage 2b:**
     - A fixed-FLOP budget doesn't see this cost. Each run also needs a wall-time limit, and it should be disclosed that the limit penalises networks like this one.
     - If you ever want native kernels for the networks, 3D convolution is the clearest first target.
