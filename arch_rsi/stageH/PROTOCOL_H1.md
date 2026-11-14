# Stage H1: hidden-rule worlds. Is there something to find? (protocol)

**Status:** written 14 November 2026, before any Stage H code or run. It's a draft for the user's approval. Nothing gets built or run until the user says so.

**Place in the plan:** this is Step 1 of three. Step 1 has no AI scientist and makes no model calls. Steps 2 and 3 get their own protocols, written after this step's result.

## Why this step

- **What happened before:** in Stage 2b, searches that saw their scores averaged 57.8% on W, and the same searches with scores hidden averaged 56.3% (+1.5 points, p 0.81). So seeing its results didn't measurably help, and Stage 2c wasn't run (`arch_rsi/stage2/2b/REPORT.md`).
- **Three likely reasons:**
  - There was little to discover: prior knowledge of good networks already solved most of the task.
  - The feedback was thin: one number per seed.
  - There was no positive control: no world where we knew in advance that a findable change helps a lot.
- **This step fixes the third reason.** It builds worlds with a hidden rule and checks, without any AI scientist, that one small, legal change aimed at the rule wins back a large loss. Without such a control, a null result in Step 2 couldn't separate "the AI scientist can't discover things" from "there was nothing to discover".

## The question

For each of four hidden rules: on a world with that rule, does reference A lose a lot? And does one small, legal change aimed at the rule win it back, by far more than noise and far more than the same change helps on the plain world?

## Base world and base network

- **World:** W_s, 8×8×6 with 6–16 bricks, 257 actions and 33 steps per episode. Training uses 131,072 steps with Stage G's recipe and a 900 s limit, which is Stage 2c's calibrated setting (`arch_rsi/stage2/2c/ws_budget.json`).
- **Base network:** reference A (`arch_rsi/campaigns/stage1/confirm/1c9c0e6b521fc0bb/candidate.py`), with Stage G's recipe.
- **Why reference A and not the start network:**
  - On plain W_s at this budget, the start averaged 23.5% (20 seeds) and reference A averaged 69.0% (SD 0.096, 6 seeds) (`arch_rsi/stage2/2c/calib/summary.json`).
  - From the start network, a search on a rule world could gain about 45 points just by rediscovering what reference A already knows. Prior knowledge supplies that, which is the confound that hid any effect of feedback before.
  - Starting from reference A leaves the hidden rule as the main thing left to find. Step 2's searches will start from reference A too.
- **What reference A can see:**
  - It has a 1×1 stem, then two 3×3 convolutions with zero padding, then a head that pools over each brick's footprint.
  - So a placement decision at column x sees columns x−2 to x+3, and the same in rows, and nothing beyond. (This is read from the code.)

## The four hidden rules and their fixes

| Rule | What changes | Why reference A should struggle | The fix: one change to reference A | Kind |
|---|---|---|---|---|
| **H1, shifted booklet** | The booklet channels (target, missing, extra, target height) are shown 4 columns to the right, wrapping around. The world and its designs don't change. | A placement at x needs the booklet at x+4 and x+5, which are outside its view of x−2 to x+3. | Roll those channels 4 columns back before the stem. | Network: re-indexing the input |
| **H2, mirrored hands** | Every placement lands at the left-right mirror of the chosen anchor: anchor x of a brick dx wide lands at 8−dx−x. The valid-action mask is given in the agent's own action terms. Designs don't change, because the design generator is mirror-symmetric. | Except near the middle columns, where a brick lands is outside the view of where it was chosen. | Flip the whole input left to right before the stem. | Network: re-indexing the input |
| **H3, wraparound edges** | The board wraps both ways: a brick over the right edge continues at the left, and one over the bottom edge continues at the top. Every anchor is valid. Designs are generated under the same rule. | Zero padding hides the far side of the board from edge placements. A wrapped brick resting on uneven columns leaves hidden cells, and single cells can't replace it. | Circular padding in the two 3×3 convolutions and in the footprint pooling. | Network: padding |
| **H4, late feedback** | All reward is held back and paid as one sum on the episode's last step. That's the per-step reward (cells right minus cells wrong as a share of the target, minus the 0.01 step penalty) plus the success bonus. Observations, actions, designs and scoring don't change. | Stage G's recipe discounts steeply (γ 0.95, λ 0.9). The advantage estimate weights a reward 20 steps ahead by (γλ)^20 ≈ 0.04. | A recipe block with γ 0.995 and λ 0.99, the top of the allowed bounds. That weight becomes ≈ 0.74. | Recipe |

- **H1 and H2 are complete by construction:**
  - With its fix, H1's input is exactly the plain world's input.
  - With its fix, H2 is the plain world seen in a mirror.
  - H3's and H4's fixes are the natural ones, but they may not recover everything.
- **Each rule is one member of a family:**
  - H1: the offset's size and axis.
  - H2: the mirror's axis.
  - H3: the wrap axis.
  - H4: the delay length.
  - Step 3 needs worlds whose facts differ, so it would draw other members, and each new member needs its own calibration first.
- **The rules stay hidden from the AI scientist:**
  - They're written here so the analysis can be fixed in advance.
  - In Step 2, the arm that isn't told the rule must never see this file, its code or its results, so they will be kept outside its sandbox.
  - Step 1 has no AI scientist, so nothing can leak during this step.

## Design

- **Cells for each rule r:**
  - plain world with reference A (shared by all four rules);
  - plain world with fix r;
  - rule world r with reference A;
  - rule world r with fix r.
- **Seeds:**
  - 6 training seeds per cell, 4000–4005. No earlier stage used 4000–4999 (checked on 14 November).
  - Every cell uses the same seeds.
- **Runs:** 6 + 4 rules × 3 cells × 6 = 78.
- **Designs:** new sets, made before any run:
  - H-plain: 512 dev and 256 check designs from the plain W_s generator (generator seeds 4141 and 4242).
  - H-torus: 512 dev and 256 check designs from the wraparound generator (generator seeds 4343 and 4444).
  - Each set excludes W_s's held-out designs (`ws_heldout_hashes.pt`, the hashes only) and the other sets.
  - Every training stream rejects all of them.
  - These worlds have no final split.
- **Scoring:**
  - Each run is scored once, by exact-build success on its world's 512 dev designs.
  - H1, H2, H4 and the plain cells use H-plain's dev designs. H3's rule cells use H-torus's dev designs.
  - Check designs are used only for the legality checks.
- **H3's comparison spans two design sets:** the plain world can't build wrapped designs, so H3's plain cells use H-plain. The result will say so.

## Pass rules, fixed before any run

A rule passes only if all four of these hold:

1. **Repair:**
   - Rule world with the fix minus rule world with reference A is at least 25 points.
   - Stage 1's exact two-sided permutation test gives a positive difference with Holm-adjusted p < 0.05 across the four rules. With 6 against 6 there are 924 splits, so the smallest possible p is 2/924 ≈ 0.0022.
   - Why 25 points:
     - At reference A's seed SD on W_s (0.096), a 6-against-6 difference has a standard error of about 5.5 points, and 25 points is about 4.5 of those.
     - Step 2 needs a target that noise can't fake.
2. **Specific to the rule:**
   - The fix's gain on the rule world minus its gain on the plain world is at least 20 points: (rule-fix − rule-A) − (plain-fix − plain-A). This uses the point estimate, with a bootstrap 95% interval reported.
   - So the fix can help the plain world a little, as long as most of its gain on the rule world comes from the rule.
3. **Legal:** the fix must be something the AI scientist is allowed to write.
   - The fixed network passes the learned-policy checks on its world's check designs:
     - untrained, it builds at most 2% exactly;
     - untrained, its clean fill is at most 10%;
     - after one 8,192-step run, it builds at most 25% exactly.
   - The short run uses the copied trainer on the CPU, with the frozen recipe and seed 0, in the rule world.
   - H4's recipe must be within bounds.
4. **Complete:** none of its runs failed or timed out.
   - A failed or timed-out run is rerun once on the same seed.
   - A second failure fails the rule.

**One fallback:**
- **When it applies:** a rule clearly hurts (plain-A − rule-A is at least 25 points) but fails rule 1.
- **What happens:**
  - Its three cells and the shared plain-A cell are rerun once at 262,144 steps.
  - All four pass rules are then judged on those runs.
  - Its p-value is doubled before Holm, because there were two looks.
  - Nothing else is rerun or retuned.

**What happens next, fixed now:**
- **3 or 4 rules pass:** I write the Step 2 protocol with the passing rules, and Step 3 stays possible.
- **2 rules pass:** Step 2 can go ahead with those two. Step 3 would need more rules calibrated first.
- **0 or 1 rule passes:** stop and report. No rule is redesigned or retuned under this protocol; a new rule needs a new protocol.
- **Every case:** all four rules are reported, whether they pass or not.

## Also recorded (descriptive, no decisions)

- **Damage and recovery:** damage is plain-A − rule-A. Recovery fraction is (rule-fix − rule-A) / (plain-A − rule-A).
- **Hidden cells:** the exact rate on designs with hidden cells, using the existing measure.
- **Where the wrong cells are:**
  - Wrong-cell counts by column and row.
  - For H3, the share of wrong cells in the wrapped edge rows and columns, compared with those cells' share of the board.
  - This tells Step 2 whether each rule leaves a trace that a scientist could see.
- **Learning curves:** from the training logs.
- **H1 identity check:**
  - H1 with its fix and plain-A get identical inputs on the same seeds, so they should match.
  - Whether they match exactly is reported, not used as a pass rule, because training on the Mac's GPU may not be bit-reproducible.
- **Reference A on plain W_s:** re-measured on H-plain's dev designs and compared with 2c's 69.0%. The start network isn't run.

## What each comparison can and can't show

| Comparison | Can show | Can't show |
|---|---|---|
| Rule world against plain world, both with reference A | The rule hurts reference A, and by how much | Why it hurts, or whether some other change would help |
| Fix against reference A, on the rule world | A known, legal change wins back the loss | That the AI scientist can find it (that's Step 2) |
| Gain on the rule world against gain on the plain world | The gain comes from the rule, not from a generally better network or recipe | For H3, how much of the gap comes from the different design set |
| Legality checks | The fix is inside the space the AI scientist may search | That it's the best possible change |
| All four rules together | How many worlds Steps 2 and 3 can use | Anything about self-improvement, since Step 1 has no AI scientist |

## Integrity rules

- **Frozen files:**
  - No frozen file is edited.
  - The 39 hashes in `arch_rsi/campaigns/stage2b/manifest.json` are checked before the first run and after the last. Any change stops the run.
- **Where new code goes:**
  - New code goes in `arch_rsi/stageH/` only.
  - Rules are applied by wrapping the world, as `ws_world.py` does, never by editing `brickworld/`.
- **Copied trainer and evaluator:**
  - They're copies of the frozen Stage 2b trainer and the frozen evaluator.
  - They add the world constructor and four hooks: observation, action, valid mask and reward. Nothing else differs.
  - Before any run, with all rules off:
    - the copy must train to the same final weights as the frozen trainer on the same spec (CPU, 8,192 steps, SHA-256 of the weights);
    - the copied evaluator must give the same per-design results as the frozen one on H-plain's check designs.
- **Rule tests:**
  - H1: rolling the input back gives exactly the plain input.
  - H2: a placement in the rule world gives the same occupancy as the mirrored placement in the plain world, and the masks match.
  - H3:
    - a wraparound world given only non-wrapping placements behaves exactly like the plain world;
    - a wrapped brick covers both edges;
    - replaying each generated design's own placements rebuilds it exactly.
  - H4: each episode's total reward equals the plain world's, and all of it is paid on the last step.
- **Step 1 manifest:** all Step 1 files are hashed into `arch_rsi/stageH/manifest.json` before the first run. The runner refuses to start if any of them changed.
- **Final splits:** these worlds have no final split. No holdout or final file is read, and no FINAL_OPENED marker is written.
- **No AI scientist:** no model calls in Step 1.
- **Stage 2e:** stays paused, and nothing in 2c or 2e is written.
- **How runs go:**
  - In a Terminal tab, four at once, on the Mac's GPU, niced.
- **Order:** this protocol is committed before any Step 1 code is written. Any departure is reported with the result.

## Cost

- **Main runs:**
  - 78 runs.
  - In 2c's calibration, reference A's runs at this budget took a median of 64.75 s each, so four at once is about 21 minutes.
  - The wraparound world may run slower, so allow an hour.
- **Tests and legality checks:** a few minutes.
- **Fallback, only if triggered:**
  - At most 78 runs at 262,144 steps, if all four rules needed it.
  - Reference A's two runs at that budget took a median of 127.45 s, so that's about 41 minutes.
- **No API cost.**

## Build list, after approval

- `rules.py`: the four rules, as world wrappers, plus a wraparound design generator.
- `make_targets_h.py`: the H-plain and H-torus design sets and their exclusion hashes.
- `train_h.py` and `eval_h.py`: the hooked copies of the frozen trainer and evaluator.
- `fixes/`: four files, each reference A plus one change.
- `run_h1.py` and `analyze_h1.py`: the 78 runs, the legality checks, the pass rules and the report. They use Stage 1's `perm_p`, `holm` and `boot_diff`, imported unchanged.
- Tests: `test_rules_h.py` and `test_identity_h.py`.

## After Step 1 (not part of this protocol)

- **Step 2:**
  - The AI scientist searches from reference A on the passing rule worlds, with richer readouts than one number per seed.
  - The arms are: told the rule, not told the rule, and scores hidden.
  - The question: does it find the fix when it isn't told?
- **Step 3:**
  - It rewrites its own procedure across worlds that use different members of the passing families.
  - Facts learned in one world can't carry over, so only procedure can.

## Amendments before any run (14 November 2026)

These were made while building Step 1, before any of the 78 runs or the four fixes' legality checks ran. The pass rules, cells, seeds, designs and budget are unchanged.

1. **GPU instead of CPU (the user's instruction, 14 November):**
   - Training, evaluation, the legality checks' greedy runs and the legality short run all run on the Mac's GPU.
   - Two things stay on the CPU because frozen code requires it:
     - The frozen evaluator runs only as the reference in the identity test. It is hard-coded to the CPU, and changing that would mean editing a frozen file.
     - check_2b's small static checks run as before: the FLOP count, the parameter and shape check, and the forward/backward timing.
   - Training on the GPU turned out to be bit-reproducible: two runs of the same spec gave identical weights. So the identity tests compare weight hashes on the GPU.
2. **Identity test for the trainer:**
   - It uses 32,768 steps (7 updates), not 8,192. Under Stage G's recipe, a 4,096-step batch gives fewer than 4 updates at 8,192 steps, and the frozen trainer refuses that.
   - It also checks that the shift fix in the shift world trains to the same weights as reference A in the plain world.
3. **Identity test for the evaluator:**
   - It uses Stage 2c's calibration checkpoint of reference A (131,072 steps, seed 2000), read only, on H-plain's check designs.
   - The reason: a 32,768-step checkpoint builds nothing, so it can't show that two evaluators agree.
4. **The legality short run:**
   - It's train_h.py with the frozen recipe. Any RECIPE block in the file is replaced by w2b.FROZEN_AS_BLOCK.
   - At reference A's FLOP ratio (1.0433), the fixed-FLOP budget gives 7 updates of 1,024 steps. That's the same count the frozen CPU trainer gives.
5. **Design sets:** besides W_s's dev designs, the four sets also exclude W_s's 256 check designs.
6. **Limits and reruns:**
   - The 900 s limit is a timeout on each training process, and evaluation has its own 600 s limit.
   - A run cut short because the runner itself stopped is redone from scratch. It doesn't count as a failed attempt.
7. **Order:** the code, the design sets and this protocol are hashed into `manifest.json` and committed before the first run.
