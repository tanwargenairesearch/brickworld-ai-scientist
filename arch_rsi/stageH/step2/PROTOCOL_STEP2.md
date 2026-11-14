# Step 2: can the AI scientist find a hidden rule from evidence? (protocol)

**Status:** written 14 November 2026, after Stage H1's result and before any Step 2 code or run. It's a draft for the user's approval. Nothing gets built or run until the user says so.

**Place in the plan:** this is Step 2 of three. Step 1 (`../PROTOCOL_H1.md`, result in `../REPORT.md`) found two hidden rules where a known, legal change wins back a large loss. Step 2 asks whether the AI scientist finds such a change when nobody tells it the rule.

## What Step 1 gives Step 2

- **Two rule worlds,** both W_s (8×8×6, 6–16 bricks):
  - **Shifted booklet:** the agent's view of the target is shifted 4 columns.
  - **Mirrored hands:** every brick lands at the left-right mirror of where it was placed.
- **The size of the prize:**
  - In both worlds, reference A builds 0.0% of designs.
  - It learns to press DONE at once and places nothing.
  - One input re-indexing change brings it back to 76.6% (shifted booklet) and 74.8% (mirrored hands). Reference A builds 76.6% on the plain world.
- **The limit this sets:** both fixes are the same kind of change, so Step 2 tests one kind of discovery, not two.

## The question

Starting from reference A in a world with a hidden rule, does the AI scientist find a large improvement more often when it sees readouts from its own runs than when it sees only scores, or no scores?

## What each comparison can and can't show

| Comparison | If it holds, we may say | What it can't show |
|---|---|---|
| **P1:** readouts − scores hidden | Evidence from its own runs led the AI scientist to fixes that prior knowledge alone didn't | That the procedure improved itself (that's Step 3). Anything beyond input re-indexing rules, toy scale and one model |
| **P2:** readouts − scores only | Readouts beyond the score were what made the difference | Which readout mattered |
| Told the rule − readouts (secondary) | How much knowing the rule is worth beyond finding it | |
| Scores only − scores hidden (secondary) | Whether scores alone lead to fixes | |
| **Check:** told the rule finds the fix | The AI scientist can write the fix once it knows the rule, so a failure to discover isn't a failure to implement | That it would discover the rule |
| No difference anywhere | It didn't find hidden rules from evidence, here | That it never could: power is limited, and the readouts are one choice |

## Setting

- **Worlds:** the two passing Stage H1 rule worlds, with Stage H1's code unchanged (`rules.py`, `train_h.py`, `eval_h.py`, `check_h.py`).
- **Training:**
  - Budget: 131,072 steps of Stage G's recipe, compute-matched as in Stage 2b.
  - Time limit: 900 s per run.
  - Hardware: the Mac's GPU, four runs at a time.
- **Scoring:** exact-build success on H-plain's 512 dev designs, under the world's rule.
- **Start:** reference A with Stage G's recipe. Its code is shown to the AI scientist as the starting architecture.
- **What the AI scientist may change:** the network and a RECIPE block, within 2b's bounds and rules. That's the same as 2b's g0 arm.
- **Program:** `arch_rsi/stage2/programs/g0r.md`, unchanged, for every arm.
- **Checks:**
  - Stage H1's `check_h.py`, in the search's rule world.
  - Up to 2 repairs per round for files that fail, as in 2b.
- **The AI scientist:**
  - It's the same model, effort and call settings as Stage 2b.
  - It has no tools and works in an empty directory, so it sees only its prompt.
  - It can't see this protocol, Step 1's report or any code.

## Arms

There are 12 searches per arm in each of the two worlds, so 24 per arm.

| Arm | Told the rule? | What it sees after each round |
|---|---|---|
| **Told the rule** | Yes: one exact sentence stating the rule | Scores and readouts |
| **Readouts** | No: a note that the world may differ from its description | Scores and readouts |
| **Scores only** | No: the same note | 2b's feedback: success per seed, training success, parameters, compute, steps, time |
| **Scores hidden** | No: the same note | Stage 1's hidden-score prompt: no success scores, and no start score |

**The note, the same for every arm that isn't told:** "This world may not behave exactly as described above. Nothing more about it is stated."

**The sentences for the told arm:**
- **Shifted booklet:** "In this world the agent's observation shows the target shifted: the target, missing, extra and target-height channels are rolled 4 columns toward higher x, wrapping around. The world, its designs and scoring are as described."
- **Mirrored hands:** "In this world each placement lands at the left-right mirror of the chosen anchor: a brick dx cells wide chosen at column x lands at column 8 − dx − x, in the same row. The valid-action mask is in the agent's own action terms. The designs and scoring are as described."

**Everything else in the prompt is 2b's:** the task text (filled in for W_s), the interface, the program, the starting code, the trial history and the reply format.

## Readouts (fixed now; the same for both worlds and every trial)

Harness code computes these from each trained trial's seed-0 run, and from the start's. No dev design is ever shown.

1. **Behaviour on the dev designs** (aggregates only):
   - mean placements per episode;
   - the share of episodes that end with DONE before any placement;
   - mean IoU;
   - mean cells placed inside and outside the target per episode.
2. **Learning curve:** training success rate and mean episode length at 25%, 50%, 75% and 100% of training.
3. **Two greedy traces,** one on each of two diagnostic designs:
   - The diagnostic designs are a new set from the plain generator (seed 4646). They're never scored, and since they're never scored they aren't excluded from training.
   - Each trace shows the design's height map (8 rows of 8 digits).
   - It shows the target-height channel as the agent observed it at the first step, in the same format.
   - For each step it shows the action (DONE, or the brick type and anchor row and column), the cells that became filled (row, column, layer) and the reward.
   - It ends with the final build's height map.
4. **One untrained trace:** the same format, for the trial's network at its seed-0 initialisation, sampling actions as training does (sampling seed 0), on diagnostic design 1.

**Prompt-size rule:** full readouts go to the start, the current best and the latest round's trials. Earlier trials get items 1 and 2 only.

**Why these readouts:**
- They're what a careful experimenter would log: behaviour, learning, and step-by-step traces of what the agent saw and what the world did.
- They name no rule.
- Both rules leave a trace a careful reader could spot:
  - **Shifted booklet:** the observed target-height map is the design's shifted by 4 columns.
  - **Mirrored hands:** in the untrained trace, the filled cells sit at the mirror of the chosen anchors.
- So Step 2 tests whether the AI scientist reads that evidence and acts on it, not whether the evidence exists.

## Each search

- **Rounds:** 4 rounds of 3 candidates. Each candidate trains seeds 0 and 1.
- **The start:** trains seeds 0 and 1 too, before round 1. Its numbers are the "starting architecture" in the prompts.
- **The pick:**
  - The pick is the trial with the best 2-seed mean, if it beats the start's 2-seed mean. Otherwise the pick is the start.
  - There's no retest step. In these worlds the gap between failing and fixed is about 75 points, and fresh seeds confirm the pick anyway.
- **Confirmation:**
  - The pick is confirmed on 4 fresh seeds, 5001–5004, and the search's score is the mean of those four.
  - The start is also run on 5001–5004 in each world, to score picks that fall back to it or break the rules.
- **Rule review:**
  - One blind call per distinct pick, using 2b's reviewer, prompt and schema, unchanged.
  - A pick judged a violation scores as the start.
- **Diagnosis judge (descriptive only):**
  - One blind call per search, given the rule, the search's last analysis and the pick's hypothesis.
  - It answers whether the text states the rule: yes, partly or no.
  - It never affects a score.

## Analysis, fixed now

**Unit:** one search, scored by its confirmed mean.

**Test:**
- The difference in mean score between two arms, pooled over both worlds.
- Two-sided permutation test that shuffles arm labels within each world.
- 1,000,000 seeded random shuffles, because exact enumeration needs about 7.3 × 10¹² splits.
- Bootstrap 95% intervals.

**Primary family, Holm over 2:**
- **P1:** readouts − scores hidden.
- **P2:** readouts − scores only.
- Each passes if positive with Holm-adjusted p < 0.05.

**The check, a precondition for reading P1 and P2 as discovery:**
- At least 18 of the told arm's 24 searches (75%) must score at least 38.3% (half of reference A's 76.6% on the plain world).
- If the check fails, P1 and P2 are still reported, but no claim is made about discovery. The AI scientist couldn't reliably write a stated fix.

**Secondary family, Holm over 2, reported only:**
- told − readouts;
- scores only − scores hidden.

**Descriptive:**
- arm means per world;
- searches scoring at least 38.3%, with an exact Fisher test per world;
- the round of each search's first trial at or above 38.3%;
- the diagnosis judge's verdicts;
- the kinds of fixes picked (read from the code and labelled by hand, as descriptive);
- calls, cost and time.

**Endings, fixed now:**
- **P1 and P2 pass:** "With readouts of its own runs, the AI scientist found fixes for hidden rules it wasn't told, more often than with scores only or with none."
- **Only P1 passes:** "...more often than with no scores. Readouts beyond the score weren't shown to matter."
- **Neither passes:** "It didn't measurably find hidden rules from evidence here."
- **Every ending** states: two rules of one kind, toy scale, one model, whose weights never changed.

## Integrity rules

- **Frozen files:**
  - No frozen file is edited.
  - Stage 2b's 39 hashes and Stage H's 26 hashes are checked before the first run and after the last. Any change stops the run.
- **Where new code goes:**
  - New code goes in `arch_rsi/stageH/step2/` only.
  - It imports Stage H1's and 2b's code rather than copying it.
  - Anything copied must be shown identical by a test.
- **Keeping the rule hidden:**
  - A test fails the build if any harness-written text in a not-told prompt names the rule: "shift", "mirror", "flip", "roll", "offset", "reflect", "wrap", "rotate" or "4 columns".
  - Text the AI scientist wrote itself, such as its own hypotheses and code, is exempt.
- **Final splits:** no final split exists for these worlds, and none is read.
- **Stage 2e:** stays paused.
- **Tests before any real call:**
  - with fake training and fake calls: prompts per arm, readouts, selection, confirmation, resume after a crash, and the review;
  - then a dry run with real calls: 1 search per arm per world (8 searches), excluded from the analysis.
  - If the dry run shows a bug, it's fixed and disclosed. Nothing else in this protocol changes because of it.
- **Launch:** the run starts only after the user approves the dry run's output.
- **How runs go:**
  - In a Terminal tab, four trainings at once.
  - Up to 10 searches in flight, so calls overlap training.
- **Recording:** every prompt and reply is saved.

## Cost

- **Training runs:**
  - 96 searches × (24 trial runs + 4 confirmation runs) + 2 worlds × 6 start runs = 2,700 runs.
  - At Stage H1's measured pace (78 runs in 21.7 min, four at once, about 16.7 s per run) that's about 12.5 h.
  - Slower networks can take longer, up to the 900 s limit each, so allow 13–16 h.
- **AI-scientist calls:**
  - About 4.1 per search in 2b (265 calls for 64 searches, $58.35 notional, about $0.22 each), so about 400 calls here.
  - Readouts lengthen the prompts, so allow up to twice 2b's cost per call for the two readout arms.
  - That gives about $90–130 notional, plus up to 96 rule-review calls and 96 judge calls.
- **Dry run:** 8 searches, roughly a tenth of the above.

## Why this size

- **Searches per arm:** 24, pooled over the two worlds.
- **What it detects** (illustrated with Fisher's exact test on the counts of searches that find a fix):
  - readouts 72% against scores only 25%: p ≈ 0.003;
  - 60% against 25%: p ≈ 0.04, right at the edge.
- **What it doesn't:** a gap of 42% against 25% gives p ≈ 0.36, so it would be missed.
- **Why not more:** doubling the searches would double the cost for a gap this step doesn't need to detect.

## After Step 2 (not part of this protocol)

- **Step 3** rewrites the AI scientist's own procedure across worlds whose hidden facts differ, so only procedure can carry over.
- **What it needs first:** at least three usable rule families. Step 1 found two, so Step 3 needs more rules calibrated first, under a new protocol.
- **What a P1 pass would add:** the evidence-reading part of the loop works, so Step 3 would test whether the loop can improve the procedure.
