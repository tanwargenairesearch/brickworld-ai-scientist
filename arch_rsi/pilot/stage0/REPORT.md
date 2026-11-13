# Stage 0: calibration (pilot, disclosed)

Run 8–9 November 2026, as set out in `../../PLAN.md`. Stage 0 is a pilot: it sets budgets and sample sizes and is never used in any campaign analysis. Every score below is on dev designs. The final split was never read, and `FINAL_OPENED` was never written. The frozen files were hash-checked before every run.

- **Machine:** 154 training runs.
- **Claude:** 8 Opus calls, about $1.85 notional.
- **Interruption:** the Mac slept for about 2 hours during phase 0 (lid closed, on battery). The runs paused rather than failing. Seeds 0 and 1 reproduced the earlier pilots' scores exactly on both worlds.

## 1. Noise: the starting network, 10 seeds per world

| World and budget | Exact success, 10 seeds | Mean | SD | Seeds stuck at the floor |
|---|---|---|---|---|
| P, 262,144 steps | 22, 54, 51, 22, 46, 86, 28, 96, 88, 22% | 51% | 0.29 | 3 |
| Q, 524,288 steps | 55, 30, 16, 16, 49, 90, 16, 26, 89, 88% | 47% | 0.31 | 3 |

- **The floor** is the share of designs with no hidden cells: 22% on P and 16% on Q. Every run builds all of those.
- **The noise is two-humped, not bell-shaped.** On Q, 3 of 10 seeds built no design with hidden cells and scored exactly the floor; 3 built 86–88% of them; 4 fell in between (`summary.json`, `noise.xl.hidden`). *(Corrected 9 November 2026: this line first said every seed either learns or scores exactly the floor.)*
- **Spread:** a seed SD of about 0.30 is 50% more than the 0.20 the plan assumed.

## 2. Ceiling

**Probe 1: one g0 search on Q at the full budget (524,288 steps).**
- The best design scored 96% (two-seed mean, 98% on its better seed), and 9 of the 12 candidates averaged 90% or more.
- Round 1's first candidate, a footprint-shaped placement head, already scored 95% on both seeds.
- This fails the gate: there's no room left to tell programs apart.

**Budget sweep on Q, no Claude calls.** The starting network and probe 1's two best designs (fixed rule: top two by mean):

| Q budget | Starting network | Best design | 2nd design |
|---|---|---|---|
| 131,072 | 14% (6 seeds; 5 at the floor) | 55% (29–82%) | 52% |
| 262,144 | 30% | 91% | 74% |
| 393,216 | 38% | 93% | 67% |
| 524,288 | 47% (10 seeds) | 96% (2 seeds) | 95% |

**Probe 2: one g0 search on Q at 131,072 steps.** Designs tuned for the full budget might not be the best at a quarter of it, so the sweep alone couldn't settle the ceiling.
- The best result per round rose 52% → 60% → 64% → 89%.
- The winner, a 3D voxel network with whole-column context and a separate critic, scored 88% and 90% on its two seeds.
- At this budget, the starting network's seeds 0–3 scored 54, 16, 16 and 52%. Combined with the sweep's 6 seeds, that's a mean of 22% over 10 seeds, and 2 of the 10 learned the skill.
- In its round analyses, the scientist described the two-humped noise, prompted by the noise note's four start scores and its program's instruction to compare against them, and treated small gaps as unresolved. Round 3 went all-voxel after a single voxel result (r2c3: 60% mean, 90% best seed); by round 4 the voxel networks' two-seed means (60–64%) beat the 2D ones (46–52%). *(Corrected 9 November 2026: this line first said it spotted the noise itself and moved to voxels once several had won.)*

**P sweep, no Claude calls.** The same two designs (written for Q) on P:

| P budget | Starting network | Design r1c1 | Design r4c3 |
|---|---|---|---|
| 65,536 | 28% (6 seeds) | 39% | 22% |
| 131,072 | 62% (6 seeds); 28% for seeds 0–3 in probe 2 | 62% | 62% |
| 262,144 | 51% (10 seeds) | 91% | 92% |

## 3. Decisions

1. **Q budget: 131,072 steps,** a quarter of the protocol's. g0's best was 89%, which passes the "under about 85–90%" gate narrowly.
   - **Departure from the plan's wording:** the starting network averages 22% here, not mid-range. No Q budget met both conditions. I chose headroom above the best design over a mid-range start, because the main comparisons are between search arms. Beating the start becomes easy, which only matters for H3.
2. **P budget: 131,072 steps,** half the protocol's. The start averages 49% over 10 seeds, and Q's designs reach 62%, against 91–92% at 262,144. A P search may still reach the ceiling, so one P probe runs before Stage 2.
3. **Confirmation seeds: 8 per selected design,** up from 3. With seed SD around 0.30, 3 seeds leave each search's score too noisy.
4. **Searches per arm: 10.** A model of the two-humped noise, with seeds that either learn or don't and a learning chance that varies between searches, gives these chances of detecting a difference (exact permutation test, p < 0.05):

| Confirmation seeds | Searches per arm | +10 points | +15 points | +20 points |
|---|---|---|---|---|
| 3 | 10 | 15–21% | 31–37% | 48–60% |
| 8 | 10 | 24–37% | 40–62% | 64–87% |
| 8 | 12 | 27–41% | 51–71% | 73–92% |

   The ranges cover a learning chance that varies by 0.15 to 0.25 between searches. False positives stayed at 4–6%. These are model estimates, not measurements.

5. **Cost per run at 131,072 steps:** about 120–240 s each with 10 in parallel, against 440–630 s at the full budget. A Q search with retesting takes about 10 minutes of the machine, against about 22 at the full budget.

## Files

- `stage0.py`: phase 0, noise seeds and probe 1.
- `sweep.py`: the Q budget sweep.
- `probe2.py`: probe 2.
- `psweep.py`: the P budget sweep.
- `analyze_stage0.py`, `summary.json`, and the logs `run.log`, `sweep.log`, `probe2.log`, `psweep.log`.
- Run directories: `campaign/`, `sweep/`, `probe2/`, `psweep/`.
