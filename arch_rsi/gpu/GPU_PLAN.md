# Moving training to the Apple GPU (Stage 2 onwards)

Written 9 November 2026, while Stage 1 was running. Nothing here changes Stage 1, which finishes on the CPU with its frozen files untouched.

## Decision

From Stage 2 on, training runs on the M4 Pro's GPU (20 cores, Metal 4) through PyTorch's MPS backend.

**Route B (a GPU-native recipe) was chosen by the user on 9 November 2026, about 13:10 UTC.** The calibration it needs is Stage G, below. Stage G runs only after Stage 1's rule review and analysis are done, and only with the user's go-ahead.

Stage 1 stays on the CPU. Its trainer, `train_arch.py`, is hash-checked, so changing it mid-run would stop the campaign. Mixing devices inside one campaign would also mix two sources of float noise.

## Measured so far (rough)

These are single short runs, measured while Stage 1 kept about 10 of the 14 CPU cores busy, so the CPU figures are probably pessimistic.

- **Setup:** world Q (8×8×5) and the frozen PPO recipe.
- **Code:** `bench_train.py` runs the same maths as `train_arch.main`, with every buffer on the chosen device.
- **Networks:**
  - "start" is `candidates/baseline.py`;
  - "footprint" is the pick from Stage 1's g0 search 4, which adds footprint pooling heads.

| Network | Device | Worlds per update | Seconds per update | Steps per second |
|---|---|---|---|---|
| start | CPU, 1 thread | 64 | 0.94 | 1,095 |
| start | GPU | 64 | 0.28 | 3,708 |
| footprint | CPU, 1 thread | 64 | 3.47 | 295 |
| footprint | GPU | 64 | 0.43 | 2,402 |
| start | CPU, 1 thread | 1,024 | 15.1 | 1,085 |
| start | GPU | 1,024 | 1.02 | 16,112 |
| footprint | GPU | 1,024 | 4.87 | 3,368 |
| start, 4 GPU runs at once | GPU | 64 each | 0.54–0.71 each | 6,589 in total |

- **It works.** On the same weights and input, the GPU's outputs match the CPU's to within 6×10⁻⁷. Every loss was finite, and neither network hit an operation the GPU lacks.
- **Per run, the GPU is faster at today's recipe:** 3.4× for the start and 8.1× for the footprint network. This contradicts my earlier guess, which came from timing one isolated tiny layer.
- **Sharing the GPU scales poorly at 64 worlds.** Four runs at once got 1.8× one run's throughput. Ten CPU runs side by side would give roughly 10 × 1,100 steps per second for the start network, but that total wasn't measured.
- **The large gain needs more worlds per update.** At 1,024 worlds, one GPU run of the start network took 16,100 steps per second, about 15× one CPU thread.
  - The gain depends on the network: the footprint network managed only 3,400.
  - The full sweep may show where the footprint network's time goes; the 64-world rows don't settle it.

## Two routes for Stage 2

**A. Same recipe on the GPU (64 worlds per update).**
- Stage 1's comparisons still hold, apart from float differences.
- The gain is mostly per-run speed, 3–8×. The gain for the whole machine is uncertain.

**B. A GPU-native recipe (for example 1,024 worlds per update).**
- Roughly 10× the machine's throughput for networks like the start.
- It changes the RL recipe, so:
  - the budget, noise and ceiling must be re-calibrated, like Stage 0;
  - Stage 2's g0 comparison must be re-run on the new recipe, instead of reusing Stage 1's numbers.
- The payoff is statistical power. At Stage 1's settings, the chance of detecting a true +15-point difference was only 40–62%. More seeds per candidate and more searches raise it.

The user picks between A and B. Either way, these rules hold:

- **The sandbox stays as it is.** Candidates still can't touch `torch.mps`; the frozen trainer moves their network to the GPU. The interface text tells Claude it trains on an Apple GPU.
- **Missing GPU operations.** A candidate that uses an operation the GPU lacks crashes, with no fallback to the CPU, and is logged as a crash. This is fixed before Stage 2 starts.
- **Timeouts** are reset from GPU timings, with headroom.
- **The final set stays locked.**

## Checks before Stage 2 (after Stage 1 finishes)

1. **Clean benchmark** on a quiet machine:
   - ten CPU runs at once against 1, 2, 4, 6, 8 and 10 GPU runs at once;
   - at 64, 256 and 1,024 worlds;
   - for the start network and three Stage 1 picks: a footprint head, a 3D voxel network and an attention network.
2. **Equivalence:** the start network on the GPU with seeds 1001–1012, compared with Stage 1's CPU comparator on the same seeds. Scores won't match exactly; this checks the distribution is the same.
3. **Determinism:** the same seed twice on the GPU. If results differ, say so, and re-check the retest logic.
4. **For route B only:** a calibration like Stage 0 for the new recipe (budget, ceiling and noise), before any Claude calls.

## The GPU trainer (`train_gpu.py`), tested 9 November

- **What it is.** It takes `train_arch.py`'s run file, plus `"device"` and a `"recipe"` field. The recipe field may change only these five settings: worlds per update, rollout length, minibatches, PPO epochs and learning rate.
- **How it differs from the frozen trainer.** It imports the frozen modules rather than editing them. Episode statistics are summed on the GPU and read once per update. The weights are saved on the CPU.
- **Same maths.** On the CPU with the frozen recipe, it reproduced `train_arch.py`'s trained weights exactly (seed 3, 3 updates). Only one logged average differed, in the 7th digit, because it sums in 32-bit floats.
- **End to end on the GPU.** At 1,024 worlds per update, a 65,536-step run of the starting network finished in 8 s. The frozen `evaluate_arch.py` then scored its checkpoint unchanged.
  - The score was 0% exact, since 4 updates is far too few to learn.
  - That 8 s includes start-up.
- **Bad settings are refused.** A recipe field outside the allowed five stops the run.
- **Rule checks stay as they are.** `check.py` keeps its short CPU training run with the frozen recipe. It detects hand-written policies, which doesn't depend on the device.

## Stage G: calibrating the GPU recipe (pilot, disclosed; no Claude calls)

**Conditions**
- It runs after Stage 1's analysis, on a quiet machine, with the user's go-ahead.
- It never reads Stage 1's g0, no-feedback or menu picks. Its reference designs are Stage 0's two sweep designs (probe 1's top two by 2-seed mean, r1c1 and r4c3), so Stage 1's results don't steer the calibration.
- Like Stage 0, it is never used in any campaign analysis. Every rule below is fixed before it runs.

| Step | What runs | What it decides | Rough time |
|---|---|---|---|
| G1. Throughput | Start, r1c1 and r4c3 at 256 and 1,024 worlds; 1, 2 and 4 GPU runs at once; 10 CPU runs at the frozen recipe for comparison | Worlds per update and GPU runs at once: the highest total steps per second, rounded down to the smaller setting if within 10% | 15 min |
| G2. Recipe | Start network on Q, 4 seeds per cell, a fixed 2×2 grid: learning rate 0.001 or 0.002 × 8 or 32 minibatches per epoch, at a fixed 2M steps | Learning rate and minibatches: the best 4-seed mean; a tie (within 5 points) goes to fewer minibatches and then the lower rate | 45 min |
| G3. Budget and ceiling on Q | Start (6 seeds), r1c1 and r4c3 (2 seeds each), at 3 budgets set by G1's speed | Q budget: the largest budget where the better reference design stays under 85–90% (Stage 0's gate); among ties, the one that puts the start nearest mid-range | 45 min |
| G4. The same on P | As G3, on world P (6×6×4), where Stage 2 meta-trains | P budget, by the same rule | 30 min |
| G5. Noise | Start at the chosen budgets, 10 seeds on each world | Seed spread and shape; Stage 2's seeds per candidate and searches per arm, from a new power table | 30 min |
| G6. Device checks | (a) Start with the frozen 64-world recipe on the GPU, seeds 1001–1012, against Stage 1's CPU comparator on the same seeds. (b) The same seed twice on the GPU | (a) If the device alone shifts scores (exact permutation test, p < 0.05), stop and report before Stage 2. (b) Whether a GPU seed gives the same result twice | 15 min |

Total: roughly 3 hours of machine time and $0 in Claude calls. The output is a Stage G report, written like Stage 0's, with each decision and its evidence.

**Fixed details (9 November 2026, before any Stage G run):** `stageG/PROTOCOL_G.md` fixes what this table leaves open, and `stageG/run_stageG.py` implements it. A dry run on tiny budgets (`stageG/dry/`) exercised every step.
- **What it adds:** how throughput is timed, the exact tie rules, budgets defined in updates (64, 128 or 256 at G1's worlds per update), and a stop for larger budgets once a reference design reaches 90%.
- **G5 decides less:** it reports the power table, and Stage 2's sample sizes go to the user, since Stage 1's gate failed.
- **Trainer change:** `train_gpu.py` now also logs the training loop's own start and end times, which G1 uses. The maths is unchanged.

**What Stage G changes in Stage 2, once agreed:**
- **g0 is re-run.** Under the new recipe, g0 is re-run as one of Stage 2's arms instead of reusing Stage 1's numbers. That adds about 6 searches' worth of Claude calls.
- **The interface text** tells Claude it trains on an Apple GPU with the chosen worlds per update.
- **Timeouts** are reset from G1 and G3 timings, with headroom.
- **The cost estimate** for Stage 2 is redone.
- **The article** describes Stage 1 as trained on the CPU and Stage 2 as trained on the GPU.

## Files

- `bench_train.py`: the training loop with a device argument, plus a CPU-against-GPU output check (`parity`).
- `gpu_check.py`: a minimal Metal kernel through `torch.mps.compile_shader`, and a matrix-multiply speed test.
- `train_gpu.py`: the GPU trainer for Stage G and Stage 2.
