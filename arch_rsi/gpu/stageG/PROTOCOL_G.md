# Stage G protocol: calibrating the GPU recipe

Written 9 November 2026, after Stage 1's analysis and before any Stage G run. It fixes the details that `../GPU_PLAN.md`'s Stage G table leaves open. `run_stageG.py` implements exactly these rules, and its `manifest.json` records this file's SHA-256 before the first run.

## Conditions

- **It's a pilot.** It sets the recipe, budgets and sample sizes for Stage 2, and is never used in any campaign analysis.
- **No Claude calls.** It runs only frozen code and three fixed networks.
- **Dev designs only.** The final split is never read, and `FINAL_OPENED` is never written.
- **Stage 1's picks are never read,** so they can't steer the calibration. The runner reads Stage 1's results file in one place only: the starting network's 12 comparator scores, for check G6a.
- **Integrity:** before every run, the runner re-hashes the frozen files (the list in `orchestrate_arch.py`) and `train_gpu.py`. Any change stops Stage G.

## Networks

- **start:** `candidates/baseline.py`.
- **r1c1 and r4c3:** Stage 0's two reference designs, probe 1's top two by 2-seed mean (`pilot/stage0/campaign/Q/g0/s1/trials/`). They are the designs Stage 0's sweeps used.

**Worlds:** Q is `xl` (8×8×5) and P is `large` (6×6×4). Scores are exact-build success on each world's 512 dev designs, from the frozen `evaluate_arch.py`. A seed **learned the skill** if it builds at least 10% of the dev designs that have hidden cells, as in Stage 1.

**Recipe:** the frozen `worlds.RECIPE`, except for the settings each step below changes. Only worlds per update, minibatches and learning rate change; rollout length (16) and PPO epochs (4) stay frozen. A run that doesn't finish counts as 0 in every mean, and is flagged.

## G1. Throughput (decides worlds per update and GPU runs at once)

- **What runs:** on Q, each network at 256 and 1,024 worlds per update, with 1, 2 and 4 identical runs at once on the GPU. Each run is 262,144 steps, scaled down by the network's FLOP ratio, as compute matching does. Seeds start at 900.
- **Measure:** total steps per second = the runs' summed steps ÷ the span from the first training loop's start to the last one's end. Imports, model building and FLOP counting are excluded.
- **Score of a setting:** the geometric mean of that measure over the three networks.
- **Rule:** take the best score. Among the settings within 10% of it, choose the fewest worlds per update, then the fewest runs at once.
- **For comparison only:** 10 CPU runs at once of the start network with the frozen recipe and the frozen `train_arch.py` (65,536 steps each; timed from process start, so not like-for-like).

## G2. Recipe (decides learning rate and minibatches)

- **What runs:** the start network on Q, at G1's worlds per update, for 128 updates (2,097,152 steps at 1,024 worlds). A 2×2 grid: learning rate 0.001 or 0.002 × 8 or 32 minibatches per epoch. Seeds 400–403 in every cell.
- **Rule:** the best 4-seed mean wins. Cells within 5 points of the best count as tied; a tie goes to fewer minibatches, then the lower learning rate.
- **Known weakness:** with a seed SD near 0.30, a 4-seed mean has a standard error near 15 points, so this choice is close to a coin toss between similar cells. Every arm in Stage 2 shares the recipe, so a weak choice costs headroom, not fairness.

## G3 and G4. Budget and ceiling (decide Q's and P's budgets)

- **What runs:** on each world, at G1's worlds per update and G2's recipe, three budgets of 64, 128 and 256 updates (at 1,024 worlds: 1,048,576, 2,097,152 and 4,194,304 steps).
  - At each budget: start with seeds 300–305, and r1c1 and r4c3 with seeds 300–301.
  - Budgets run smallest first. If the better reference design's 2-seed mean reaches 90% at a budget, larger budgets aren't run on that world. This saves time and assumes scores rise with budget, which Stage 0's sweeps showed.
- **Rule (Stage 0's gate):** a budget passes if the better reference design's 2-seed mean is below 90%. Choose the largest passing budget. A passing budget whose better-design mean is within 5 points of that budget's counts as tied, and among tied budgets the one with the start's mean nearest 50% wins, then the larger budget.
- **If no budget passes:** choose the smallest, and flag that the ceiling gate failed.

## G5. Noise (reports the seed spread and a power table; the user decides Stage 2's sample sizes)

- **What runs:** the start at each world's chosen budget, with seeds 306–309. Together with seeds 300–305 from G3 or G4, that's 10 seeds per world.
- **Reported:** the mean, SD, scores, and the number of seeds that learned the skill.
- **Power table:** Stage 0's two-humped model. Each seed either learns (score near the learned level) or doesn't (near the not-learned level), and the learning chance varies between searches (SD 0.15 or 0.25).
  - The levels come from Stage G's own runs at the chosen Q budget: the learned level from seeds that learned (all networks), the not-learned level from seeds that didn't.
  - Rows: 8, 12 or 16 confirmation seeds × 10, 12, 16 or 20 searches per arm.
  - Columns: true differences of +10, +15 and +20 points, plus the false-positive rate at 0, with an exact or Monte Carlo permutation test at p < 0.05.
- **Not decided here.** Stage 1's gate failed, so Stage 2's design goes back to the user. G5 gives the table and the measured time per run; the user picks the sample sizes.

## G6. Device checks

- **(a) Does the device alone shift scores?** The start, with the frozen recipe (64 worlds per update) on the GPU, on Q at 131,072 steps, seeds 1001–1012. It is compared with Stage 1's 12 CPU comparator scores for the same network and seeds, by a two-sided exact permutation test on the difference in means.
  - The two devices draw different random numbers, so the seeds aren't paired.
  - **Rule:** if p < 0.05, stop and report before Stage 2.
- **(b) Determinism:** two runs are repeated with the same seed: G6a's seed 1001 and G2's chosen cell at seed 400. It reports whether the dev score and the trained weights match exactly.

## Outputs

- `run.log`: one line per run.
- `decisions.json`: each step's decision and the numbers it used, written as each step finishes.
- `summary.json`: everything above.
- `REPORT.md`: written afterwards, like Stage 0's, with each decision and its evidence.
