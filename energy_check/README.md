# Energies that build the brick-assembly world's designs

Run on 13 November 2026 on one Apple M4 Pro laptop.

| | Fit | Search runs on | Dev designs built exactly |
|---|---|---|---|
| **Part 1. Hand-set energy** | none: the rules are written in by hand | CPU | 2,048 of 2,048 |
| **Part 2. Learned energy** | counted from 4,000 solved designs of the smallest world | Apple GPU, custom Metal kernel | 2,046 of 2,048 |

Every build was made in the frozen `BrickWorld` and scored by the frozen checker in `evaluate.py`.

**The two misses of Part 2.** One tiling needs one brick more than the world's step limit. One design cannot be solved under the 4,000-design fit at all: the count wrongly rules out a pair of labels that the design needs. With 16,000 training designs, no such design was found among 16,384 fresh ones, for three training samples.

This folder was reviewed critically on the same day, and corrected. "What the review changed", near the end, lists the findings.

**What it means.** Once the rules are known, deciding which bricks go where is an easy constraint puzzle, whether the rules are written down or read off solved examples. A high exact-build rate in this world is therefore not evidence of reasoning.

**What it doesn't mean.** It doesn't rank energies against reinforcement learning. The networks of Part 3 were trained at a budget that was kept small on purpose. With four times that budget and a better recipe, the starting network alone builds 95.7% of the 8×8×5 dev designs.

## Provenance: the AI scientist's second suggestion

The energy approach was suggested and written by an LLM, like the networks of the search. The two suggestions were not made under the same rules.

- **How it came about.** A person showed an LLM coding agent the paper below and asked whether it applied to this world. The agent read the paper, the paper's public code and this world's code. It proposed the hand-set energy and the learned one, including the label set, the anchor device, the bottom-up build order and the counting fit. The person approved each step.
- **It is not a product of the automated search.** The search's rules would not allow it.

| | Suggestion 1: inside the search | Suggestion 2: outside the search |
|---|---|---|
| Who suggests | an LLM proposer with no tools | an LLM coding agent with tools (a different LLM), with a person approving each step |
| May change | the network and its training recipe | the whole method |
| Is shown | the interface text, its own earlier experiments and, in one arm, their scores | a paper, the world's code and its own test results |
| Fixed rules | training by reward (PPO), the compute budget and the checker are frozen; the policy and its features must be learned | none of these |

- **Why the search has a learned-policy rule.** In an early dry run the proposer hand-wrote the world's rules into a network as a fixed placement score. Untrained, it built 97% and 95% of check designs (`arch_rsi/protocol.md`, pilots table). The rule was added to stop that. Both energies sit outside it: their labels are designed from the rules.

## Why this check

"Classical Energies Learn to Reason 1,000x Faster than Looped Models" (arXiv 2611.17309) shows that Sudoku and maze benchmarks are solved by small pairwise energies, some with weights set by hand, and warns that a high score on such a benchmark shows little about reasoning. This folder applies both checks to our world. It borrows the method. No code is copied from that paper's repository.

---

## Part 1. The hand-set energy

Files: `brick_energy.py`, `run_check.py`, `test_brick_energy.py`.

One on/off variable per brick: a brick type (1×1, 1×2, 2×1 or 2×2), the layer it rests on, and its anchor cell.

```
E(bricks) = sum over cells of (number of 'on' bricks covering the cell − target cell)²  +  mu × (number of bricks on)
```

- The first term is 0 exactly when the bricks tile the target with no overlap and no gap.
- Expanded, the energy is pairwise: two bricks cost 2 for each cell they share, and each brick has an input cost set by the target cells under it. `pair_form` returns both tables, and a test checks the identity.
- The frozen setting has `mu = 0`, so no number in the energy was tuned.

**Clamps.** A brick is switched off for good unless it fits in the grid, lies inside the target and, above the ground layer, has at least one target cell directly below it. The last rule is the world's drop rule. It is a mask computed from the target, not a pairwise term.

**Search.** Four chains per design start from the empty board. Each move picks a random brick that touches a cell where cover and target differ, then switches it on with probability `sigmoid(−dE / temp)`. A chain stops at the first board whose cover term is 0 and whose brick count is within the world's step limit.

**Build.** The 'on' bricks are placed lowest layer first. With that order, each brick comes to rest on its intended layer.

### Protocol

1. **Tuning designs.** 256 fresh designs per world from the frozen generator (seed 20261113). Every dev and final design is rejected by hash, as the training stream does.
2. **Grid.** `mu` in {0, 0.5} × `temp` in {0.25, 0.5, 1.0}, with 4 chains and a budget of 20,000 moves per chain.
3. **Rule, fixed before the run.** The setting that builds the most tuning designs exactly, summed over the worlds. Ties go to the fewest search moves.
4. **Dev.** Each dev set is scored once with the frozen setting. `run_check.py dev` refuses to run a second time (see "Limits" for how far that guard goes).
5. **Final splits.** Never read.

### Results

**Tuning.** All six settings built all 1,024 tuning designs, so the tie-break chose `mu = 0`, `temp = 0.25` (`settings.json`).

**Dev, scored once (`results/dev.json`).**

| World | Grid | Built exactly | Stopped on a wrong board | Hit the move budget | Search moves, median (max) | Bricks used, mean (max) | Step limit | CPU search time, 512 designs |
|---|---|---|---|---|---|---|---|---|
| small | 5×5×3 | 512 / 512 | 0 | 0 | 12 (55) | 7.3 (17) | 17 | 0.9 s |
| large | 6×6×4 | 512 / 512 | 0 | 0 | 21 (823) | 10.7 (20) | 25 | 1.7 s |
| xl | 8×8×5 | 512 / 512 | 0 | 0 | 31 (105) | 14.7 (31) | 33 | 2.8 s |
| W | 10×10×6 | 512 / 512 | 0 | 0 | 41 (6,641) | 19.6 (33) | 41 | 8.3 s |

- No placement was illegal.
- Between 66% (small) and 87% (W) of the dev designs have an empty cell under a filled one.
- 67 to 88 designs per world have more cells than the step limit, so a build of 1×1 bricks alone would not fit. The search stayed within the limit on every design.
- Times are wall-clock on 4 CPU threads, for the search only.

---

## Part 2. The learned energy

Files: `brick_cram.py` (labels, energy, fits, NumPy reference search, impossibility proof), `metal_search.py` (the GPU kernel), `run_learned.py`, `run_fresh.py`, `run_fitsize.py`, `verify_replay.py`, `test_brick_cram.py`.

**What is learned and what is not.** The costs are learned. Four things are designed from the world's rules:

- the label set, built from the world's brick shapes;
- the input token, which says whether a filled cell rests on something;
- the training labels, made by replaying the generator's bricks through the world's drop rule;
- the bottom-up build order.

`true_rules` in `brick_cram.py` writes the rules down as allowed terms, for tests and diagnostics only. No fit and no search reads it.

### Labels and input

Every filled cell carries one of 25 labels. A label says which piece of which brick the cell is, and which cell of that brick is the brick's **anchor**: a cell that rests on something. The anchor turns the world's support rule into a rule about one cell (the anchor rests on something) and about pairs of cells (the cells of one brick name the same anchor).

The input is one token per cell: empty, filled and resting on the ground or a filled cell, or filled over an empty cell.

### The energy

```
E(labels) = sum over neighbouring cell pairs of K[offset, label, label]  +  sum over filled cells of B[token, label]
```

`K` covers the 3×3 window of one layer. `K` and `B` hold 2,750 numbers, shared by every cell, layer and world size.

### The fit: counting

A pair term or an input term costs 0 if it occurs in a training solution and 1 if it never does. This is one pass of counting, the Hebbian rule of an associative memory.

- **Training data.** 4,000 solved designs of the smallest world only (5×5×3, 49,278 filled cells; 3,998 of the designs are distinct), from the frozen generator with its own bricks. Every dev, final and tuning design is rejected by hash.
- **Fit time.** The counting pass took 5.5 ms on the CPU in the scored run (5.5 to 7.8 ms over three runs). That excludes making and labelling the 4,000 designs (about 55 ms), building the search tables (about 5 ms) and the tuning.
- **Same weights everywhere.** The energy has no grid size, so the weights fitted on 5×5×3 are used unchanged in all four worlds. Two settings (the number of training designs and the search temperature) were chosen on tuning designs from all four worlds.
- **What the count gets right.** A count flags every term it has not seen. So it flags all 1,196 forbidden pair terms and all 9 forbidden input terms by construction, for any number of training designs, and a labelling of energy 0 is always a valid tiling.
- **What it gets wrong.** It also flags 146 allowed pair terms that it never saw. 144 of them are still unseen in 16,000 designs. The other 2 occur in valid solutions: two 2×2 bricks that meet corner to corner, with particular anchors. A design whose every tiling needs one of them has no labelling of energy 0, and no search can stop on it. `brick_cram.unsolvable` proves that for a given design, by arc consistency.

**Stopping test.** Every training solution has energy 0. A chain stops at the first labelling with energy 0. The world's step limit is not part of the test, so energy 0 means a valid tiling, not always a build that fits the step limit.

**Search.** Four chains per design start from random labels. Each move picks a random filled cell that has a violated term and redraws its label from `softmax(−local energy / temp)`.

### The GPU kernel

`metal_search.py` holds a custom Metal compute kernel, compiled with `torch.mps.compile_shader`. One GPU thread runs one chain and owns one row of each buffer, so no two threads write the same memory. Random numbers come from a counter hash of (seed, chain, move), written identically in the kernel and in NumPy.

**Checked against a CPU reference (`test_brick_cram.py`).**
- **Labels.** At temperature 0 the kernel and the NumPy reference give identical labels, move for move, on 96 chains after 0, 1, 37 and 300 moves. No chain stops in that test.
- **Stops.** A second test starts 96 chains at a solution or 1 to 3 wrong labels away from one. The kernel and the reference report the same stop move for every chain, with budgets of 0, 1, 5, 40 and 300 moves and launches of 2, 7 and 64 moves: 32 chains stop at move 0, 55 later and 9 never.
- At temperature above 0, every chain the kernel stopped has no violated term when recomputed from scratch, and no running chain is free of them.
- On 40,000 one-cell chains, the kernel's label frequencies match `softmax(−L / temp)` within 0.01.

**Speed (`results/kernel_speed.json`).** Boards of 600 cells, the size of the largest world. Median of 5 timings.

- **An upper figure.** Every term is flagged in this benchmark, so no chain stops and the flag check exits at its first test. A review probe with the real rule tables measured about 14% fewer moves per second at 2,048 chains (17.9 against 20.8 million; not in a results file).
- **What is timed.** The kernel launches and the read-back of the stop flags. The uploads and the read-back of the labels are not.
- **The NumPy column** times a whole call of 64 moves, set-up included, and NumPy's thread count was not set. The file's key `numpy_reference_4_threads` names PyTorch's setting, which does not govern NumPy.

| Chains | Metal kernel, 256 moves per chain | NumPy reference, 64 moves per chain |
|---|---|---|
| 2,048 | 21 million moves/s | 0.32 million moves/s |
| 16,384 | 125 million moves/s | not run |
| 131,072 | 107 million moves/s | not run |

### Protocol

1. **Tuning designs.** The same 256 per world as Part 1.
2. **Grid.** Training designs in {1,000, 4,000, 16,000} × `temp` in {0.4, 0.6, 0.8}, with 4 chains and 20,000 moves per chain.
3. **Rule, fixed before the run.** Most tuning designs built exactly, summed over the worlds. Ties go to the fewest training designs, then to the fewest search moves.
4. **Dev.** Scored once. `run_learned.py dev` refuses to run a second time (see "Limits" for how far that guard goes).
5. **Final splits.** Never read.

### Results

**Tuning (`results/learned_tune.json`).** With 4,000 or 16,000 training designs, every temperature built all 1,024 tuning designs. With 1,000 it built 1,018 or 1,019. No tuning run stopped on a wrong board. The rule chose 4,000 designs and `temp = 0.6` (`learned_settings.json`).

- **The misses with 1,000 designs are fit errors.** 7 tuning designs have no labelling of energy 0 under that fit (proved by arc consistency), and they are the 7 the search did not stop on.
- **The tie-break chose the weaker fit.** 4,000 and 16,000 tied on the tuning designs, and the rule preferred fewer training designs. 1,024 tuning designs cannot see an error that touches about 3 designs in 10,000.
- **"Built exactly" does not require a stop.** A design the search did not stop on is still decoded and built. With 1,000 training designs, one or two of the 7 impossible designs were built exactly that way. On the dev sets this happened to none.

**Dev, scored once (`results/learned_dev.json`).**

| World | Grid | Built exactly | Stopped on an invalid tiling | Over the step limit | Impossible under the fit | Search moves, median (max) | GPU search time, 512 designs |
|---|---|---|---|---|---|---|---|
| small | 5×5×3 | 511 / 512 | 0 | 1 | 0 | 29 (1,843) | 0.06 s |
| large | 6×6×4 | 512 / 512 | 0 | 0 | 0 | 57 (2,141) | 0.06 s |
| xl | 8×8×5 | 511 / 512 | 0 | 0 | 1 | 90 (3,508) | 0.50 s |
| W | 10×10×6 | 512 / 512 | 0 | 0 | 0 | 111.5 (4,534) | 0.18 s |

- **The two misses.**
  - One small design was tiled validly with 18 bricks where the world allows 17 placements.
  - One xl design (dev index 114) has no labelling of energy 0 under this fit. It was first reported here as "did not reach energy 0 in 20,000 moves", which was wrong: no number of moves would do. The proof was run after the score was seen.
- **The search never failed on a design it could solve.** In every recorded run (tuning, dev, fresh designs), the designs the search did not stop on are exactly the ones proved impossible under that run's fit.
- **Search moves** count the chain that stopped first. The design's other three chains run alongside it until they are cancelled, so the kernel makes several times as many moves.
- The median is taken from the per-design records. The file's summary field holds 111 for W, the lower of the two middle values; the code now writes the mean of the two.
- No placement was illegal.
- GPU times are single runs of the search only. Most of the xl time went on the impossible design, whose four chains ran all 20,000 moves.
- **Reproduced.** `verify_replay.py` replays the scored run with the code now on disk. For all 2,048 designs the moves, the brick count and the outcome equal the record.

### Fresh designs, other seeds (post hoc)

The dev score above is one fit and one search seed. `run_fresh.py` was added after that score was seen, to test whether it depended on either. It selects nothing and changes no setting.

- **Designs.** 4,096 fresh designs per world (16,384 per run), none of them a dev, final, tuning or training design.
- **Runs.** The frozen fit with search seeds 0, 1 and 2, and two refits on other samples of 4,000 training designs.

| Run | Built exactly, of 16,384 | Impossible under the fit (the search did not stop) | Tiled validly, over the step limit | Stopped on an invalid tiling |
|---|---|---|---|---|
| frozen fit, search seed 0 | 16,376 (99.95%) | 1 | 7 | 0 |
| frozen fit, search seed 1 | 16,370 (99.91%) | 1 | 13 | 0 |
| frozen fit, search seed 2 | 16,376 (99.95%) | 1 | 7 | 0 |
| refit 1, search seed 0 | 16,371 (99.92%) | 7 | 7 | 0 |
| refit 2, search seed 0 | 16,368 (99.90%) | 10 | 7 | 0 |

- **Reading.** The score does not depend much on the seed or on the training sample: every run built between 99.90% and 99.95%.
- **Two kinds of miss, and no third.**
  - **The step limit.** The energy doesn't count bricks, so a valid tiling can need more placements than the world allows. Most of these are in the smallest world, where the limit is tightest.
  - **The fit.** Every design the search did not stop on has no labelling of energy 0 under that run's fit: 1 for the frozen fit (the same design with all three search seeds), 7 and 10 for the refits. The proofs were run after the check, with `brick_cram.unsolvable`.
- A design that was not stopped can still be built exactly, so the columns need not add up to the misses.

### More training designs (post hoc)

`run_fitsize.py` was added after the review, to test whether a larger training sample removes the impossible designs. It selects nothing. The scored setting stays 4,000 designs, and no dev set was rescored.

- **Designs.** 4,096 new fresh designs per world (16,384 per run; another seed than above), none of them a dev, final, tuning or training design.
- **Runs.** Three training samples, each at 4,000 and at 16,000 designs. Search seed 0, the frozen temperature.

| Training sample | Training designs | Allowed pair terms flagged | Impossible designs, of 16,384 | Tiled validly, over the step limit | Built exactly |
|---|---|---|---|---|---|
| 1 (the scored one) | 4,000 | 146 | 2 | 3 | 16,379 |
| 1 | 16,000 | 144 | 0 | 3 | 16,381 |
| 2 | 4,000 | 149 | 0 | 3 | 16,381 |
| 2 | 16,000 | 144 | 0 | 3 | 16,381 |
| 3 | 4,000 | 155 | 13 | 3 | 16,370 |
| 3 | 16,000 | 144 | 0 | 3 | 16,381 |

- **With 16,000 training designs, no design is impossible**, for all three samples. All three flag the same number of allowed pair terms, 144.
- **With 4,000 it depends on the sample:** 2, 0 and 13 impossible designs here, and 1, 7 and 10 on the first fresh set. That is 33 of 98,304, or 0.03%.
- **What is left is the step limit:** 3 of 16,384, two in the smallest world and one in the next.
- In every run the search stopped on every design that was not proved impossible, and never on an invalid tiling.
- In sample 3 at 4,000, two of the 13 impossible designs were built exactly all the same, which is why its misses are 14 and not 16.

### The pseudolikelihood runs made here gave no stopping test

The paper fits its energies by pseudolikelihood. `run_learned.py pl` runs that objective by gradient descent on 1,000 of the same training designs: 300 updates, 2.9 s on the GPU. The stopping rule is the one used throughout this folder: a term is violated when it costs more than the highest cost in a training solution. With that rule, the fitted costs do not separate the forbidden terms (`results/learned_pl_diagnostic.json`):

| Fit, three learning rates | Loss | Threshold on pair costs | Median cost of a forbidden pair term | Forbidden pair terms above the threshold | Forbidden input terms above it |
|---|---|---|---|---|---|
| Pseudolikelihood | 0.00001 to 0.00018 | 4.2 to 40.3 | 0.12 to 0.96 | 12 to 18 of 1,196 | 1 of 9 |
| The same with costs kept at 0 or above | 0.00002 to 0.01041 | 1.1 to 29.8 | 0.94 to 6.80 | 159 to 482 of 1,196 | 1 to 6 of 9 |

- **Measured.** In every run the threshold is above the median cost of a forbidden pair term. Some terms that occur in solutions end with high costs, and they set the threshold.
- **This is a finding about these runs, not about the objective.** The counting energy times 10 has a pseudolikelihood loss of 0.00007 on the same designs, inside the range above, and times 20 its loss is 0. Both keep the counting energy's stopping test, with all 1,196 and all 9 forbidden terms flagged. So a perfect stopping test is among the objective's best solutions. Gradient descent from zero costs did not reach it in 300 updates.
- **The threshold rule depends on how the costs are written.** Cost can be moved between pair terms and input terms without changing the loss, and that changes which terms pass a threshold. A review probe moved the count of flagged forbidden pair terms from 12 to 8 to 5 at equal loss (not in a results file).
- **Why the runs leave the support rule unpriced (likely, untested).** Any single-cell change to a valid solution also breaks a brick. Pseudolikelihood compares only single-cell changes, so it never needs a cost on "anchor over an empty cell".

Not tested: the paper's own optimiser and stopping rule, more updates, another starting point, or a fit that samples its own negatives.

---

## Part 3. Compared with the networks trained by reward

`video/make_compare_video.py collect` reads every number of this part from the campaigns' own files and saves it to `video/compare_data.json`.

**What is compared.** The two worlds with confirmed networks: 8×8×5 (Stage 1) and 10×10×6 (Stage 2b). For the networks, every build of every confirmed training run is counted: 12 fresh-seed runs of the starting network, and 8 fresh-seed runs of each pick of the "sees its scores" arm (10 picks in 8×8×5, 16 in 10×10×6). For the energy, the one scored run.

### Every dev design, by size

The 512 dev designs of each world are sorted by number of cells and cut into four groups of 128.

| World | Size group (cells) | Starting network | AI scientist's picks | Learned energy |
|---|---|---|---|---|
| 8×8×5 | 7 to 18 | 47.7% | 92.5% | 128 of 128 |
| 8×8×5 | 18 to 24 | 24.8% | 81.8% | 128 of 128 |
| 8×8×5 | 24 to 30 | 11.2% | 70.2% | 127 of 128 |
| 8×8×5 | 30 to 51 | 7.1% | 56.2% | 128 of 128 |
| 8×8×5 | **all 512** | **22.7%** | **75.2%** | **511 of 512** |
| 10×10×6 | 10 to 25 | 42.2% | 78.4% | 128 of 128 |
| 10×10×6 | 25 to 32 | 11.1% | 63.1% | 128 of 128 |
| 10×10×6 | 32 to 39 | 2.7% | 52.1% | 128 of 128 |
| 10×10×6 | 39 to 55 | 0.7% | 37.6% | 128 of 128 |
| 10×10×6 | **all 512** | **14.2%** | **57.8%** | **512 of 512** |

- **Size is what hurts the networks.** Their rate falls steadily from the smallest designs to the largest. The energy's does not.
- **Designs no run builds.** In 10×10×6, 330 of the 512 designs are built by none of the starting network's 12 runs, and 1 by none of the picks' 128 runs. In 8×8×5 the counts are 87 and 1.
- **Set-ups differ between the two rows of worlds.** 8×8×5 is a network-only search with the original recipe on the CPU. 10×10×6 is a network-and-recipe search on the GPU, and its starting network uses the later recipe (with the original recipe it builds 9.5%).
- **Selection.** The picks were chosen on seeds 0 and 1 of these same dev designs and then confirmed on fresh seeds. The energy's settings were chosen on separate tuning designs. This asymmetry favours the networks.

### Eight designs, chosen by size alone

The rule was fixed before any of these builds was looked at: sort the 512 dev designs of 10×10×6 by number of cells (ties by dev index) and take the designs at ranks 0, 73, 146, 219, 292, 365, 438 and 511. The networks shown are the median training run of each group: the starting network's run that builds 12.9% of dev designs, and a run of the picks that builds 68.9%.

| Cells | Starting network (median run) | Built by, of its 12 runs | AI scientist's network (median run) | Built by, of the picks' 128 runs | Energy | Search moves | Bricks |
|---|---|---|---|---|---|---|---|
| 10 | built | 10 | built | 115 | built | 19 | 10 |
| 21 | not built | 0 | built | 74 | built | 68 | 16 |
| 26 | not built | 1 | not built | 64 | built | 70 | 19 |
| 30 | not built | 0 | built | 81 | built | 232 | 22 |
| 34 | not built | 0 | built | 96 | built | 75 | 25 |
| 38 | not built | 0 | built | 26 | built | 101 | 28 |
| 43 | not built | 0 | not built | 25 | built | 1,657 | 23 |
| 55 | not built | 0 | not built | 41 | built | 307 | 35 |

Of these eight, the starting network's median run builds 1, the picks' median run builds 5 and the energy builds 8.

### The training budget is what the gap mostly measures

The starting network was trained for 131,072 practice placements. Each pick got the same compute, which is 94,208 to 278,528 placements in 10×10×6 depending on the network's cost. Earlier stages of this project ran larger budgets:

| World | Budget | Starting network | AI scientist's picks | Source |
|---|---|---|---|---|
| 8×8×5 | 131,072 | 22.7% (12 runs) | 75.2% (80 runs) | Stage 1, original recipe |
| 8×8×5 | 262,144 | 50.9% (12 runs) | 88.1% (80 runs) | `arch_rsi/gpu/retest`, later recipe |
| 8×8×5 | 524,288 | 95.7% (4 runs) | not run | `arch_rsi/gpu/stageG`, later recipe |
| 10×10×6 | 131,072 | 14.2% (12 runs) | 57.8% (128 runs) | Stage 2b, later recipe |
| 10×10×6 | 262,144 | 33.4% (6 runs) | not run; two reference networks built 91.6% and 86.9% (2 runs each) | `arch_rsi/stage2/2a` |

- **The budget was chosen to leave headroom.** In 10×10×6, twice the budget was rejected as too easy for a search, because a reference network already built 91.6%.
- **With enough practice the networks get close.** At four times the budget, with the later recipe, the starting network alone builds 95.7% of the 8×8×5 dev designs, against the energy's 511 of 512.
- The rise from 131,072 to the larger budgets in 8×8×5 also includes the change of recipe, so it is not from practice alone.

### What each method is given

| | Networks trained by reward | Learned energy |
|---|---|---|
| Signal | a reward per brick for cells placed inside the target, less cells outside it; a bonus for an exact match; a small cost per placement | 4,000 solved designs of the 5×5×3 world, with their bricks |
| Input | the build, the target, maps of missing and extra cells, height maps and the share of the step limit used; a mask of legal moves | one token per cell: empty, resting on something, or over an empty cell |
| Built-in structure | the network's code may use the brick shapes; the proposer is told the drop rule in words | 25 labels built from the brick shapes; the bottom-up build order |
| Cost to learn | one training run took 36 to 362 s with four runs sharing the GPU; one search of 12 experiments took about 56 minutes of training | one counting pass of 5.5 ms; the session that designed it is not timed |
| Cost to build | one pass of the network per brick | a search: 111.5 moves at the median in 10×10×6 and 4,534 at most, counting the chain that stops first |

---

## Limits

- **Not a like-for-like comparison.** The two suggestions were made under different rules (see "Provenance"), given different information, and the networks' budget was kept small on purpose. The gap at that budget is not a ranking of methods.
- **Both energies are designed from the rules.** Part 1 writes them in. Part 2 learns its costs, but its labels, its input token and its build order come from the rules.
- **Part 2's fit is counting, not gradient training.** The pseudolikelihood runs made here gave no usable stopping test. That says little about the objective: the counting energy, scaled, is one of its best solutions.
- **The count can rule out a valid design.** With 4,000 training designs, about 0.03% of fresh designs have no labelling of energy 0. With 16,000, none was found. The scored setting is the 4,000 one, as the rule chose it.
- **Part 2 does not count bricks.** A valid tiling can exceed the world's step limit. With enough training designs this is its only failure seen on fresh designs.
- **"Scored once" is a file check.** `dev` writes a marker before it scores and refuses to run when the marker or the result exists. The marker was added after the review; the two scored runs predate it. Nothing but the protocol stops someone deleting the files.
- **Dev designs were replayed.** `verify_replay.py` and the video script run the frozen search again on dev designs, with the frozen settings, and require the same moves, bricks and outcomes as the record. They select nothing.
- **Not under version control.** `learned_tune.json` is rewritten by every `tune` run, so the number of tuning runs made before the rule was fixed cannot be shown. Exploratory runs on training and tuning designs were made while the method was being written, among them the switch from pseudolikelihood to counting. To my knowledge none touched a dev design; the files cannot show that. Two source files were last edited seconds after the scored run. `verify_replay.py` shows that the code now on disk reproduces that run for all 2,048 designs.
- **Part 1's support rule is a mask, not a pairwise cost.** It needs up to four cells of the layer below at once.
- **The placement order is hand-written in both parts.** The energy decides which bricks; a fixed rule decides when.
- **One scored fit.** The dev score is one fit and one search seed. The two checks on fresh designs, with other seeds, refits and fit sizes, were added afterwards.
- **No physics.** This is the lattice world. The stability check in `physics/` balances forces across the whole structure and is not pairwise.
- **One laptop, small boards.** The GPU numbers are for an M4 Pro and boards of at most 600 cells.
- **Repeatable, not independent per design, in Part 1.** One random stream serves a whole batch there. In Part 2 each chain has its own counter-hash stream.

## What the review changed

Two independent reviews were run on 13 November: one of the code, one of the claims and of the fairness of the comparison. Both were asked to find faults, not to confirm.

| Finding | What was done |
|---|---|
| Misses were reported as "ran out of moves". They are designs the fit makes impossible. | Proved with `unsolvable`. Tables and text corrected. `run_fitsize.py` run on new fresh designs. |
| "The paper's objective gave no stopping test" claimed more than was measured. | Reworded as a finding about the runs made here. The scaled counting energy was measured under the same objective. |
| The comparison used one training run per network and one design chosen by eye. | Every confirmed training run is counted. Designs are chosen by a size rule and the median runs are shown. |
| The gap to the networks was shown without the training budget. | A budget table and a video scene: at four times the budget the starting network builds 95.7% in 8×8×5. |
| The networks were said to learn "from reward alone", with none of the structure. Their reward is paid per brick, and they are given a mask of legal moves and maps of missing cells. | Corrected in "What each method is given" and in the video. |
| The energy's origin and its different rules were not stated. | The "Provenance" section, and the video's first scene. |
| The test of the rounding fix could not fail. | Replaced by a comparison with integer sums. |
| The kernel test never exercised a stop, and a budget of 0 moves reported no stop. | A test of stops added. The kernel now makes a launch of no moves in that case. |
| Test designs could include dev or final designs (one did). | Excluded by hash. |
| "Scored once" checked only for the result file, which is written when a run ends. | A marker is written before scoring. It is still a file check. |
| The recorded median was the lower of the two middle values. | Fixed in the code. The tables use the true median. |
| Timings and the speed table lacked their conditions. | Added. |
| A docstring said the energy was "fitted by pseudolikelihood" and "holds no rule of the world". | Corrected. An unused value was removed. |

**Not fixed.**

- The folder has no version control, so the order of edits and runs cannot be audited.
- The scored setting is still the 4,000-design fit. The rule chose it, and changing it after seeing dev results would break the protocol.
- The energy still does not count bricks.
- The picks were not trained at the larger budgets in 10×10×6, so the budget table has a gap there.
- The pseudolikelihood fit was not pursued further.

**Checked by the code review and found sound.** The hand-set energy's formulas against integer references; the label tables, the 2,750 parameters and the 1,196 + 9 forbidden terms; `violations` against a brute-force evaluation; the kernel against the reference at temperature 0.6 on 64 chains over 3,000 moves; the random numbers (no repeated counter, no measurable correlation); the exclusion of dev, final and tuning designs from training; no path from `true_rules` into a fit or a search; every summary number against its per-design records.

## Video

`video/energy_vs_network.mp4` (1920×1080, 128 s, no sound) puts the AI scientist's two suggestions side by side. It is an illustration: it scores, selects and tunes nothing.

- **Scenes.** The problem and the two suggestions; the median networks building one design; the energy's search and build of the same design; eight designs chosen by size; every dev design of two worlds; what more practice does for the networks; a table of what differs; what the comparison shows and doesn't.
- **Rules.** The designs and the networks shown are fixed by the rules of Part 3, written before the builds were looked at.
- **Energy search.** A replay of the scored run with the same batch and random numbers. It must reproduce the recorded moves and brick counts of all 512 designs, or the script stops.
- **Numbers.** Every measured number on screen is read from a results file and saved to `video/compare_data.json`.
- **Wording.** The script stops if a banned word appears in any string it draws.
- **Stills.** `video/stills/` holds one or two frames of each scene, and `video/cover.png` the eight designs.

## Reproduce

```bash
.venv/bin/python -I -B energy_check/test_brick_energy.py
.venv/bin/python -I -B energy_check/run_check.py tune
.venv/bin/python -I -B energy_check/run_check.py freeze
.venv/bin/python -I -B energy_check/run_check.py dev          # refuses while results/dev.json or dev.started exists

.venv/bin/python -I -B energy_check/test_brick_cram.py         # needs an Apple GPU
.venv/bin/python -I -B energy_check/run_learned.py tune
.venv/bin/python -I -B energy_check/run_learned.py freeze
.venv/bin/python -I -B energy_check/run_learned.py dev         # refuses while results/learned_dev.json or .started exists
.venv/bin/python -I -B energy_check/run_learned.py pl
.venv/bin/python -I -B energy_check/run_learned.py speed
.venv/bin/python -I -B energy_check/run_fresh.py               # refuses while results/learned_fresh.json exists
.venv/bin/python -I -B energy_check/run_fitsize.py             # refuses while results/learned_fitsize.json exists
.venv/bin/python -I -B energy_check/verify_replay.py           # replays the scored dev run; writes nothing

.venv/bin/python -I -B energy_check/video/make_compare_video.py collect   # needs an Apple GPU, for a few seconds
.venv/bin/python -I -B energy_check/video/make_compare_video.py stills
.venv/bin/python -I -B energy_check/video/make_compare_video.py video
```

No frozen file was edited. After the last run, 37 of the 39 hashes in `arch_rsi/campaigns/stage2b/manifest.json` were recomputed and all match. The other 2 are the final splits, which were not opened; their modification dates are 8 and 9 November.

Tests: 9 in `test_brick_energy.py` and 13 in `test_brick_cram.py`, all passing after the fixes.

## A numerical trap worth knowing

PyTorch's batched convolution returned 0.99999994 for a sum that is exactly 1, which made an exact comparison reject valid bricks in Part 1. Every sum there is a whole number, so `brick_sums` and `cover` round their results. A test compares both with sums made from integer slices. The first version of that test could not fail: it compared rounded values with themselves.
