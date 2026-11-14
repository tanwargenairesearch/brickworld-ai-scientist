# Step 3a: more hidden rules for Step 3 (calibration protocol)

**Status:** approved by the user on 14 November 2026 at 18:58 GMT ("Approve defaults"), with the three proposed defaults under "Decisions for the user". It was written at about 18:55 GMT while Step 2's full run was going, and no Step 2 result was read to write it. Per decision 3, the code is built and its CPU tests run while Step 2 runs. The GPU identity tests and the 150 runs wait until Step 2's full run is final, and then run before Step 2S.

**Place in the plan:** Step 3 rewrites the AI scientist's own procedure across worlds whose hidden facts differ, so only procedure can carry over. Step 2's protocol says what it needs first: at least three usable rule families, calibrated under a new protocol. This is that protocol. Like Step 1, it has no AI scientist and makes no model calls.

## Why this step

- **What Step 1 found** (`../REPORT.md`): two rules passed, one from each of two families:
  - H1, the shifted booklet (translation);
  - H2, the mirrored hands (reflection).
- **What failed, and why that shapes the candidates:**
  - The two rules whose fixes weren't complete by construction both failed: wraparound edges (H3) and late feedback (H4).
  - The two whose fixes undo the rule exactly both passed.
  - So every candidate here has a fix that undoes its rule exactly, or exactly in distribution.
- **What Step 3 needs:**
  - At least three families, so one whole family can be held out: facts learned on the others can't fix it.
  - At least two members per family, so worlds within a family also differ in their facts.

## Families and members

- **What a rule is:** a fixed transform, applied either to the booklet or to the hands.
  - The booklet is the agent's view of the target: the target, missing, extra and target-height channels.
  - The hands are where placements land.
- **Family:** the transform.
- **Member:** what it applies to, and the axis.
- **No change to the world:** under every rule below the world itself is unchanged. Bricks, gravity, designs and scoring are as in the plain world.
- **Fixed in advance:** each rule's family is set now, before any run.

| Id | Family | Rule | The fix: one change to reference A | What reference A can still see | Undoes the rule |
|---|---|---|---|---|---|
| H1 (passed) | Translation | Booklet shown 4 columns to the right, wrapping | Roll the booklet back 4 columns | Nothing | Exactly |
| **H5** | Translation | Booklet shown 4 rows down, wrapping | Roll the booklet back 4 rows | Nothing: 4 rows is outside its rows y−2 to y+dy+1 | Exactly: the plain input |
| **H6** | Translation | Hands shifted 4 columns: a brick chosen at anchor x lands at anchor (x+4) mod 8, same row and type. The valid mask is given in the agent's own action terms, as in H2 | Roll the placement logits back 4 columns. The DONE logit is unchanged | Nothing: where a brick lands is 4 columns from where it was chosen | In distribution: the same placements as the plain policy, though not bit for bit, because sampling sees the logits in a different order |
| H2 (passed) | Reflection | Hands mirrored left to right | Flip the input left to right | Middle columns | In distribution |
| **H7** | Reflection | Hands mirrored top to bottom: anchor y of a brick dy tall lands at 8−dy−y. The mask is in the agent's terms | Flip the input top to bottom | Middle rows | In distribution. The design generator is symmetric top to bottom, as it is left to right: each drop picks uniformly among valid anchors |
| **H8** | Reflection | Booklet mirrored left to right: column x shown at 7−x | Flip the booklet back | Middle columns | Exactly |
| **H9** | Transposition | Booklet transposed: cell (y, x) shown at (x, y) | Transpose the booklet back | Cells within about 2 of the main diagonal | Exactly: the board is square |
| **H10** | Transposition | Booklet transposed on the other diagonal: cell (y, x) shown at (7−x, 7−y) | Apply the same transpose again | Cells within about 2 of the other diagonal | Exactly |
| **H11** | Rotation | Booklet turned a quarter turn: cell (y, x) shown at (7−x, y) | Turn the booklet back | A small central region | Exactly |
| **H12** | Rotation | Booklet turned three quarters: cell (y, x) shown at (x, 7−y) | Turn the booklet back | A small central region | Exactly |

- **The view behind the fourth column:**
  - A placement at anchor (y, x) sees rows y−2 to y+dy+1 and columns x−2 to x+dx+1, and nothing beyond. That comes from a 1×1 stem, two 3×3 convolutions and footprint pooling.
  - The fourth column reasons from that view. It is a prediction, not a result.
  - Exact building needs every cell right, so seeing part of the board isn't expected to be enough. H2 also left the middle columns visible, and reference A built 0.0%.
- **Six of the eight are booklet rules:**
  - Their fix gives the network exactly the plain input.
  - So on the same seeds, the fixed run should match plain reference A design by design, as H1 did.
  - For them, the calibration is really asking one thing: does the rule hurt reference A by at least 25 points?
- **The families overlap as transforms:**
  - A quarter turn is a transpose followed by a flip.
  - H1 and H6 both roll.
  - Step 3's protocol must choose its held-out family with these overlaps in mind. This step only labels them.

## Design

- **Base:** as Step 1:
  - world W_s, 8×8×6, with 6–16 bricks;
  - reference A, with Stage G's recipe;
  - 131,072 steps;
  - a 900 s limit per run.
- **Cells per new rule:**
  - plain world with the fix;
  - rule world with reference A;
  - rule world with the fix.
- **Shared cell:** plain world with reference A, rerun.
- **Seeds:** 4000–4005, the same as Step 1, so two identity checks are possible:
  - the rerun plain cell should reproduce Step 1's exactly, since training on the Mac's GPU was bit-reproducible;
  - each booklet rule's fixed cell should match the plain cell design by design.
- **Runs:** 6 + 8 rules × 3 cells × 6 seeds = 150.
- **Designs:** Step 1's H-plain sets, unchanged:
  - 512 dev designs for scoring;
  - 256 check designs for the legality checks.
  - No rule here changes the designs, so no new set is needed.
- **Scoring:** each run is scored once, by exact-build success on the dev designs.
- **No final split:** these worlds have none, and no holdout or final file is read.

## Pass rules (unchanged from Step 1)

A rule passes only if all four hold. These are `../PROTOCOL_H1.md`'s rules and thresholds, with its amendments:

1. **Repair:**
   - Rule world with the fix minus rule world with reference A is at least 25 points.
   - Stage 1's exact two-sided permutation test gives Holm-adjusted p < 0.05.
   - Holm runs across the eight new rules. H1 and H2 aren't retested.
   - The smallest p a 6-against-6 test gives is 2/924 ≈ 0.0022. Even if all eight rules reach it, each one's Holm-adjusted p is about 0.017.
2. **Specific to the rule:** (rule-fix − rule-A) − (plain-fix − plain-A) is at least 20 points. That's the point estimate, with a bootstrap 95% interval reported.
3. **Legal:** the fixed network passes the learned-policy checks on the check designs, in its rule world, on the GPU:
   - untrained, it builds at most 2% exactly;
   - untrained, its clean fill is at most 10%;
   - after the short run (Step 1's amendment 4, with the frozen recipe), it builds at most 25% exactly.
4. **Complete:** none of its runs failed or timed out.
   - A failed run is rerun once on the same seed.
   - A second failure fails the rule.

**One fallback, as in Step 1:**
- **When it applies:** a rule hurts by at least 25 points but fails rule 1.
- **What happens:**
  - Its three cells and the shared plain cell rerun once at 262,144 steps.
  - All four pass rules are judged on those runs.
  - Its p-value is doubled before Holm.
- **Who it can apply to:** the booklet rules' fixes restore the plain input, so for them repair equals damage. In practice, only H6 and H7 can need it.

## What happens next, fixed now

- **A usable family:** one with at least two passing members, counting H1 and H2.
- **3 or 4 usable families:**
  - Step 3's protocol can hold out a whole family.
  - It gets written after Step 2's result, and after Step 2S's if Step 2S runs.
- **2 usable families:**
  - Step 3 can't hold out a family. Holding out a member within a family is weaker, because a family-level fact ("roll the booklet") would carry over.
  - The options go back to the user.
  - No rule is redesigned or retuned under this protocol.
- **0 or 1 new rule passes:** stop and report.
- **Every case:** all eight rules are reported, whether they pass or not.

## Also recorded (descriptive, no decisions)

- **Damage and recovery:**
  - Damage is plain-A − rule-A.
  - The recovery fraction is (rule-fix − rule-A) / (plain-A − rule-A).
- **Hidden cells:** the exact rate on designs with hidden cells.
- **Wrong cells:** counts by row and column, as in Step 1. They show whether each rule leaves a trace that readouts could reveal.
- **Learning curves:** from the training logs.
- **Identity checks:**
  - the plain cell against Step 1's, per design;
  - each booklet rule's fixed cell against the plain cell, per design.

## What each comparison can and can't show

| Comparison | Can show | Can't show |
|---|---|---|
| Rule world against plain world, both with reference A | The rule hurts reference A, and by how much | Why, or whether another change would help more |
| Fix against reference A, in the rule world | A known, legal change wins back the loss | That the AI scientist can find it (that's Step 3's inner search) |
| Gain in the rule world against gain in the plain world | The gain comes from the rule, not from a better network | That it's the best possible fix |
| Families that pass | Which families Step 3 can hold out | That transfer across them tests procedure rather than a shared fact (see below) |

## What this step doesn't fix

- **The same kind:** all eight candidates are re-indexings, as H1 and H2 were.
  - So Step 3's transfer would be across geometric re-indexings.
  - A procedure that says "check whether the input is re-indexed" might carry over as a fact about this set of worlds rather than a better procedure.
  - Step 3's facts-only control can test that, but this step can't.
- **No rule of another kind is proposed:**
  - The two tried, padding (H3) and the recipe (H4), failed.
  - I haven't found another kind whose fix undoes the rule exactly and stays inside what the AI scientist may write.
  - A new kind needs its own protocol.
- **Scale:** toy scale, one world size, one base network.

## Integrity rules

- **Frozen files:**
  - No frozen file is edited.
  - These are checked before the first run and after the last, and any change stops the run:
    - Stage 2b's 39 hashes;
    - Stage H's 26 files (`../manifest.json`);
    - Step 2's 12 files;
    - Step 2S's 12 files.
- **Where new code goes:** in `arch_rsi/stageH/step3a/` only.
- **Copied trainer, evaluator and check:**
  - Stage H's `train_h.py`, `eval_h.py` and `check_h.py` import the rules from the frozen `rules.py` by name.
  - So this step uses copies with only that import changed, the way Step 1 copied the frozen trainer.
  - Before any run, each copy must match its Stage H original under the rules plain, shift and mirror:
    - the same weights (GPU, 32,768 steps);
    - the same per-design results;
    - the same check results.
- **Rule tests:**
  - **Booklet rules (H5, H8–H12):** applying the fix to the rule's input gives exactly the plain input.
  - **Hands rules (H6, H7):** a placement in the rule world gives the same occupancy as the transformed placement in the plain world, and the masks match.
  - **The old rules:** the new rules module's shift and mirror reproduce `rules.py`'s exactly.
- **Manifest:** this protocol, the code and the fixes are hashed into `manifest.json` and committed before the first run.
- **When it runs:**
  - Only after Step 2's full run is final, and not while Step 2S runs. They share the GPU, and Step 2's trials have a 900 s limit.
  - It runs in a Terminal tab, four runs at a time, niced.
- **What stays untouched:**
  - Step 2's and Step 2S's campaigns aren't read or written.
  - Stage 2e stays paused.
  - No FINAL_OPENED marker is written.
  - No model calls.
- **Disclosure:** any departure is reported with the result.

## Cost

| Part | Runs | Time |
|---|---|---|
| Main runs | 150 | About 42 minutes. Step 1 ran 78 in 21.7 minutes, four at once, at a median of 64.78 s each |
| Legality checks and identity tests | 8 checks, 3 identity tests | A few minutes |
| Fallback (H6 and H7 at most) | Up to 42, at 262,144 steps | About 25 minutes. Step 1's 24 took 14.2 |
| **Total** | | **About 1 to 1.5 hours of machine time** |

There's no API cost.

## Build list, after approval

- `rules3a.py`: the rules module for this step. It covers plain, shift and mirror, matching `rules.py`, plus H5–H12.
- `train_3a.py`, `eval_3a.py` and `check_3a.py`: the copies.
- `make_fixes_3a.py` and `fixes/`: eight files, each reference A plus one change.
- The runner and analysis:
  - `run_3a.py`;
  - `analyze_3a.py`, with Stage 1's `perm_p`, `holm` and `boot_diff` imported unchanged;
  - the report tables.
- Tests and manifest: `test_rules_3a.py`, `test_identity_3a.py`, `make_manifest3a.py` and `manifest.json`.

## Decisions for the user (approved 14 November 2026, 18:58 GMT: the proposed default for each)

1. **The candidates:** the eight rules and their families, as listed (proposed). The alternative is to drop the rotations (H11, H12), because a quarter turn is a transpose plus a flip. That saves about 10 minutes but leaves no fourth family as a spare.
2. **Order after Step 2:** this calibration first, then Step 2S (proposed). It's about 1 to 1.5 hours with no model calls, and Step 2S needs your approval of its dry run anyway. The alternative is Step 2S first.
3. **When to build:** write the code and run its CPU tests while Step 2 runs. The GPU identity tests and the runs wait until Step 2 is final (proposed). The alternative is to build nothing until Step 2 is final.

## Changes made while building (14 November 2026, before any run)

The rules, fixes, families, cells, seeds, designs, pass rules and decision are unchanged. These are details the protocol didn't fix:

1. **How the copies are made:** `make_copies_3a.py` writes them from Stage H's files by exact text replacement, as `make_train_h.py` did. Besides the import, three things had to change:
   - the folder paths, because the copies live one level deeper;
   - the table of design sets per rule, which now comes from `rules3a.py` too;
   - `check_3a.py`'s short run calls `train_3a.py`.
2. **The old rules run the old code:** `rules3a.Rule` subclasses the frozen `rules.Rule` and hands it every Stage H rule. A test checks that plain, shift, mirror, wraparound and late feedback behave identically.
3. **The runner runs the GPU tests itself, first:**
   - It runs `test_rules_3a.py` on the GPU, then `test_identity_3a.py`, and stops if either fails.
   - It refuses to start until Step 2's `results.json` exists (only its existence is checked) and no Step 2 or Step 2S runner process is alive. The identity test refuses the same way.
4. **Two identity checks beyond the protocol's list:**
   - each booklet fix, trained in its rule world, gets reference A's plain-world weights (32,768 steps);
   - each new fix except H7, carrying trained weights, scores like the plain world through `eval_3a.py`.
5. **Reading "2 usable families":** when at least two new rules pass but fewer than three families are usable (0, 1 or 2), the options go back to the user. The protocol named only the case of 2.
6. **Added tests:**
   - `test_analyze_3a.py`, on made-up runs;
   - a sanity check in `test_rules_3a.py`: reference A trained in the plain world, run under each new rule. That's not a pass rule.
7. **Wrong cells:** recorded as the full 8×8 counts plus sums by row and by column.
8. **Time:** the identity test trains 12 short runs and runs 4 checks, so allow 10–20 minutes before the main runs, not "a few".

## Build status (14 November 2026, about 19:09 GMT)

- **Built:**
  - `rules3a.py`;
  - `make_copies_3a.py`, with `train_3a.py`, `eval_3a.py` and `check_3a.py`;
  - `make_fixes_3a.py` and `fixes/` (F5–F12);
  - `analyze_3a.py`, `run_3a.py` and `make_report_tables_3a.py`;
  - the three tests and `make_manifest3a.py`.
- **Run on the CPU while Step 2 held the GPU (decision 3):**
  - `test_rules_3a.py --device cpu`: 11 of 11 pass.
  - `test_analyze_3a.py`: 8 of 8 pass.
  - In those tests, with Stage 2c's trained reference A weights on 64 designs:
    - reference A built 62% in the plain world;
    - each fix in its rule world built exactly what reference A built in the plain world (for H7, the mirror image);
    - reference A, trained in the plain world, built 0% under every new rule. That isn't a calibration result: the calibration trains reference A in each rule world.
- **Not yet run:** the GPU rule tests, the identity tests, the eight legality checks and the 150 runs. They wait until Step 2's full run is final.
