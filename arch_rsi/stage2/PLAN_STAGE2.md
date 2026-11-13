# Stage 2 plan (approved by the user on 9 November 2026: decisions 1–7 as proposed)

Drafted 9 November 2026, after Stage 1, Stage G and the user's direction ("Yes to all"):
- retest Stage 1's picks under the better recipe first;
- make the task harder;
- let the search change the training recipe as well as the architecture.

The user approved it on 9 November ("yes, I approve those decisions"). Each sub-stage gets a protocol file, fixed before any of its results, as Stage 1 and Stage G did. Stage 2b spends money on Claude calls, so it goes back to the user with its protocol and 2a's numbers before it launches.

## Where we are

- **Stage 1:** under the old recipe, Claude's picks beat the start by about 50 points, but hiding the scores didn't hurt: g0 − no feedback was +1.5 points, so the gate failed. Level 1b ("experiments guided the search") isn't shown.
- **Stage G:** a recipe change alone teaches the start the skill on every seed. Q at 262,144 steps is close to saturated: Stage 0's reference designs reach 92–97% with twice the steps.
- **Retest** (`../gpu/retest/REPORT.md`): under Stage G's recipe and budget, Claude's picks still beat the start, but by less.
  - **The lead:** +37.1 points for g0 and +38.0 for no feedback, against +52.5 and +50.9 in Stage 1. The primary reading is "lead survives".
  - **Feedback:** g0 − no feedback is −0.8 points (interval −4.7 to +2.8).
  - **Spread:** the SD between searches fell to 0.045.
  - **Rankings:** Stage 1's ranking of picks barely carries over (Spearman ρ = 0.25).
  - **Ceiling:** the best pick reaches 95.4%, so Q has little headroom left.
- **The start depends on its seeds:**
  - Under the new recipe, it averaged 73.7% on Stage G's seeds (300–309) and 50.9% on the retest's (1001–1012). The exact permutation p is 0.004.
  - Rerunning one seed from each set reproduced both bit for bit, so this isn't the pipeline.

## The order matters

Self-revision (Levels 2 and 3) asks whether rewriting the search's own program makes later searches better. That only means something if the search uses what its experiments tell it. Stage 1 couldn't show that.

So Stage 2 re-asks Level 1b first, in a task with headroom where the search controls both the architecture and the recipe. Self-revision runs only if that gate passes.

## Stage 2a: a harder world, calibrated (no Claude calls, about 1–2 hours)

**The world.** A new world, W, made with the existing environment code, which isn't changed:
- **Size:** the proposal is 10×10×6 with 8–20 bricks, against Q's 8×8×5 with 6–16.
- **Designs:** fresh dev (512) and final (1,024) designs from new seeds, with held-out hashes, generated once like `make_targets.py`.
- **The xl final split stays locked** (see decision 3).

**Calibration, as in Stage G:**
- The start with the frozen recipe and with Stage G's recipe.
- Two reference designs: the two best retest picks, so the reference designs are good under the new recipe. Using them to calibrate W is a choice made from the retest's results. It's disclosed, and it touches no claim.
  - **Changed in `2a/PROTOCOL_2A.md`, before any W run:** they are no-feedback search 9's pick (95.4%) and no-feedback search 5's (94.1%).
  - **Why:** g0 search 1's pick (94.5%) took 701 s per retest run, 14 times the start's, though its FLOP ratio is 1.03.
  - **The rule:** the two best retest picks that run within 3 times the start's time.
- A budget ladder.
- 20 seeds of the start at the chosen budget, from two separate seed ranges, because the retest found that the start's mean depends on which seeds are used.

**Pass if:**
- the start with Stage G's recipe stays at or below about 50%, leaving room for recipe and architecture search;
- the better reference design stays under 85–90%;
- seed noise is measured.

If W is too easy, try a lower budget or a larger world. If it's too hard (nothing learns), try a smaller one. Each try is disclosed.

## Stage 2b: does the search use its experiments when it controls the recipe? (Level 1b again)

**What the search may change.** A candidate file gains a `RECIPE` block next to the network. The frozen trainer reads it and enforces these bounds:

| Setting | Range |
|---|---|
| Learning rate | 1e-4 to 3e-3 |
| Worlds per update | 64–1,024 |
| Rollout length | 8–64 |
| Minibatches | 4–64, with at least 64 samples each |
| PPO epochs | 1–8 |
| Entropy coefficient | 0–0.05 |
| γ | 0.9–0.995 |
| GAE λ | 0.8–0.99 |
| Clip | 0.1–0.3 |

The reward, design stream, evaluator and learned-policy checks stay frozen.

**Budget: fixed training FLOPs per run.**
- Each run gets the same number of floating-point operations, counted by the frozen counter (`check.train_flops`): one forward pass per collected sample, and a forward and backward pass per sample per epoch.
- So a bigger network or more epochs means fewer samples, and nothing can win by spending more compute.
- **Alternative (rejected):** a fixed number of GPU seconds, like a fixed 5-minute budget per run. With 4 runs sharing the GPU, a run's time would depend on its neighbours, and the result would be tied to this machine.

**Arms on W, all with the same number of searches:**

| Arm | Claude? | What it can change | Role |
|---|---|---|---|
| **g0** | yes, sees scores | network + recipe | the search |
| **No feedback** | yes, scores hidden | network + recipe | **primary control:** do experiments guide the search? |
| **Network only** | yes, sees scores | network; recipe fixed at Stage G's | secondary: does recipe control matter? |
| **Recipe only** | yes, sees scores | recipe; network fixed (the start) | secondary: do networks matter, given recipe control? |
| **Random** | no | random menu network × random recipe within bounds | secondary: is Claude better than random draws from the same space? |

**Comparators, no search:** the start with the frozen recipe, and the start with Stage G's recipe, the known good recipe.

**Primary family (Holm over 2):**
- g0 − no feedback;
- g0 − (start with Stage G's recipe).

**Gate:** both positive with Holm-adjusted p < 0.05, on fresh-seed dev scores. If it fails, the result goes back to the user, and Stage 2c doesn't run.

### What each comparison can prove

| Comparison | If it holds, we may write | What it can't prove |
|---|---|---|
| g0 > no feedback | "Experiments guided the search" (Level 1b), in a task where recipe and network both matter | that the procedure improved itself; that any of it generalises beyond this toy |
| g0 > start + Stage G recipe | the search beats a good human-chosen recipe, not just a starved start | that it beats a well-tuned human search |
| g0 > network only | controlling the recipe adds to architecture search | which part of the recipe mattered (descriptive only) |
| g0 > recipe only | architecture adds to recipe search | |
| g0 > random | Claude beats random draws from the same space | |
| no difference anywhere | "first ideas plus selection" again, now with recipe control and headroom | that experiments never help (power is limited) |

**Sample sizes.** The spread between searches on W is unknown until Stage 2a measures it. The two spreads measured so far bracket it:
- **SD 0.045:** the retest, near Q's ceiling. Scores bunch up there, so this is likely too optimistic for a world with headroom.
- **SD 0.13:** Stage 1, with room to improve. This is the more likely case for W.

The table gives the chance of detecting a true g0 − no-feedback gain of +5, +10 and +15 points:

| Searches per arm | SD 0.045 | SD 0.09 | SD 0.13 |
|---|---|---|---|
| 10 | 65 / >99 / >99% | 22 / 65 / 94% | 13 / 37 / 68% |
| 12 | 74 / >99 / >99% | 26 / 74 / 97% | 15 / 44 / 77% |
| 16 | 86 / >99 / >99% | 33 / 86 / >99% | 18 / 56 / 88% |
| 20 | 93 / >99 / >99% | 40 / 93 / >99% | 22 / 66 / 94% |

- **How it's computed:** a normal approximation (a two-sample t-test at α = 0.05), not the exact permutation test, so it's somewhat optimistic.
- **The α assumption:** Holm tests the larger of the two primary p-values at 0.05, which assumes g0 − start passes easily. If it doesn't, the test is at 0.025, and the chances fall by up to about 12 points.
- **Proposal: 16 searches per arm.** At SD 0.13 that gives an 88% chance at +15 points and 56% at +10. Re-check once 2a has measured W's spread, and move to 20 if the SD is above 0.13. Twenty per arm would cost about 344 calls, roughly $72.

**Rough cost:**
- About 4.3 Claude calls per search (Stage 1 measured 85 calls for 20 searches), about $0.21 each.
- Four Claude arms × 16 searches is about 275 calls, roughly $60.
- Machine time depends on W's budget and is estimated in Stage 2a.

## Stage 2c: self-revision (Levels 2 and 3), only if 2b's gate passes

As in `../PLAN.md` Stage 2, moved to the new setting:
- **Meta-training** on a smaller version of W, in 2 lineages.
- **The program controls what runs:** a machine-read header sets candidates per round, seeds per candidate and how many to retest, within a fixed run budget.
- **Test before keeping:** a rewrite is kept only if its searches beat its parent's. A rewrite may also change how much of the budget goes to recipe and to network.
- **Controls on W:**
  - the final program against g0;
  - a facts-only rewrite, with the procedure locked;
  - a rewrite from shuffled scores;
  - the first revision against the final one, to test compounding.
- **Then:** W's final split opens once, for every arm.

**What it may let us write:** "rewriting its own procedure helped, beyond carried-over knowledge", only if the final program beats g0, facts-only and shuffled. Always qualified: at toy scale, with Claude's weights unchanged.

## Decisions for the user

1. **The world:** 10×10×6 with 8–20 bricks (proposed), or another size.
2. **The budget unit:** fixed FLOPs (proposed), or fixed GPU seconds.
3. **The final sets:**
   - Proposed: W's own final split, opened once at the end of Stage 2.
   - The xl final split stays locked. It could be opened once later, to check Stage 1's picks on held-out designs.
4. **The arms:** all five (proposed), or a cheaper minimum of g0, no feedback and the comparators.
5. **Claude model:** Opus 5.5 throughout (proposed), with Sonnet 5.5 replication in Stage 3.
6. **What each 2b search starts from:**
   - **Proposed:** the same starting network as before, with Stage G's recipe filled into its `RECIPE` block. Every search then has to beat a strong, known-good point, not a starved one.
   - **Not proposed:** starting from Stage 1's best pick. Searches would no longer be independent of earlier results, and a win could be inherited rather than found.
7. **Searches per arm:** 16 (proposed), re-checked against W's measured spread after 2a. Or 20 from the start.

## What the approved decisions change in the analysis

| Decision | What it changes | The cost or risk |
|---|---|---|
| 1. World W | The g0 − no-feedback test has room to show a difference. On Q, the best picks already reach 95% | Claims are about W, a new world. Q's results can't be pooled with W's |
| 2. Fixed-FLOP budget | No candidate can win by spending more compute, whether through a bigger network, more epochs or more samples | The FLOP counter misses some operations: one pick ran 14 times slower than the start at a FLOP ratio of 1.03. So 2b also needs a wall-time limit per run, which favours fast networks. It will be disclosed |
| 3. Final sets | W's held-out designs open once, at the end, for every arm. Claims about held-out designs are on W | The xl final stays locked, so Stage 1's picks are still untested on held-out designs |
| 4. Five arms | Separates four questions: does feedback help, does recipe control matter, do networks matter, and is Claude better than random | Roughly twice the cost of the minimum (g0, no feedback, comparators) |
| 5. Opus 5.5 throughout | One model, so arms differ only in what they see and can change | Generality waits for Stage 3's Sonnet 5.5 replication |
| 6. Every search starts from the start plus Stage G's recipe | Searches stay independent, and the bar is a known-good point (51–74% on Q), not a starved one | It's harder to beat, so effects may be smaller than in Stage 1 |
| 7. 16 searches per arm | About an 88% chance of detecting a true 15-point gain, and 56% for 10 points, if searches vary as much as in Stage 1 (SD 0.13) | It's re-checked once 2a measures W's spread |

## Stage 2a result, and proposed changes for 2b (need the user's OK)

**2a result** (`2a/REPORT.md`): W's chosen budget is 131,072 steps.
- **Why that budget:** at 262,144, reference A reached 91.6%, so that budget counted as "too easy"; the fallback, 131,072, passed.
- **The start at 131,072**, with Stage G's recipe, over 20 seeds: 12.3% (SD 0.072). It learned the hidden-cell skill on 2 of 20 seeds. There was no seed-range effect (p = 0.42).
- **The references:** A averages 55.2% (6 seeds) and B 4.1%.
- **Machine time:** the projection for 2b is about 18 hours.

**Proposed for 2b's protocol:**
1. **Budget: 131,072 steps,** the protocol's choice. Using 262,144 instead would be a departure: the start would be about 33%, the references 87–92% (near the ceiling again), and machine time would roughly double.
2. **The second primary contrast moves to reference A.** It becomes g0 − reference A (Stage 1's best network, with Stage G's recipe, at W's budget). The start is weak on W, so g0 − start would pass trivially. Reference A asks whether searching on W beats the best network carried over from Q. g0 − (start with Stage G's recipe) moves to the secondary family. The gate stays g0 − no feedback.
3. **A wall-time limit of 900 s per run,** about 22 times the start's 40.5 s with 4 runs at once. It sits alongside the fixed-FLOP budget, because the FLOP counter misses costs like 3D convolutions. A run over the limit scores 0, as in Stage 1, and this is disclosed.
4. **The 2b trainer keeps the frozen PPO update maths** (no fused optimiser), so 2b's runs match 2a's calibration. Its only additions are the RECIPE block and its bounds. At about 18 hours, speed doesn't justify the change.

## Status, 9 November 2026

- **2a:** done (`2a/REPORT.md`).
- **The four 2b changes:** approved by the user ("Yes go for it").
- **2b:**
  - protocol: `2b/PROTOCOL_2B.md`;
  - code: `2b/run_2b.py`, `train_2b.py`, `check_2b.py`, `w2b.py`, `programs/g0r.md`, `2b/analyze_2b.py`, `2b/review_2b.py`;
  - the dry run passed;
  - the full run launched on 9 November (`campaigns/stage2b`).

- **2c protocol (draft):** `2c/PROTOCOL_2C.md`, written 10 November before any 2b result was read. It runs only if 2b's gate passes and the user approves its decisions.
