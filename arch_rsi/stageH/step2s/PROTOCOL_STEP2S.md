# Step 2S: does experimenting on the world help the AI scientist find a hidden rule? (protocol)

**Status:**
- **Written:** 14 November 2026, before Step 2's full run and before any Step 2S code or run.
- **Approved:** by the user on 14 November 2026, as written, with the told arm kept.
- **Not built or run yet.** Building waits for the user's go-ahead, and the run waits for Step 2's full-run result.

**Place in the plan:**
- This step runs after Step 2 (`../step2/PROTOCOL_STEP2.md`). The user chose a separate step over a fifth arm in Step 2.
- Which worlds it runs in is set below by Step 2's results, using rules fixed now.

**Disclosure:**
- While writing this, I saw four lines of Step 2's dry-run log. The dry run is excluded from Step 2's analysis.
- **What they showed, in the shifted-booklet world, round 3 of one search per arm:**
  - the told arm's best trial built 78.5%;
  - the scores-only and scores-hidden arms built 0.0%;
  - one mirrored-hands scores-hidden search also built 0.0%.
- **Why the design doesn't depend on them:** it was chosen for the reason given under "Why scores only", which comes from Step 2's protocol. Its world rule uses Step 2's full-run results only.

## The question

In a world with a hidden rule, does the AI scientist find a fix more often when it can run its own experiments on the world than when it sees scores only?

This is dig.bench's creative mode one level up:
- **The sandbox:** steps there don't count, and the player chooses what to try.
- **The difference here:** the experimenter isn't playing the game. It's designing the learner that will play it.

## Why scores only, not readouts, as the base

- **Step 2's readouts already show both rules without any experiment.** That's in Step 2's protocol, under "Why these readouts":
  - each trace shows the observed target-height map next to the design's, which exposes the shifted booklet;
  - the untrained trace shows bricks landing at the mirror of the chosen anchors.
- **So on top of readouts,** a sandbox would mostly test whether more of the same evidence helps.
- **On top of scores only,** it tests experimentation itself: the AI scientist gets no traces unless it designs the experiment that produces one.

## The sandbox (fixed now)

**What a probe is:** before proposing each round's candidates, the AI scientist may run up to 3 probes. A probe places bricks it chooses, one after another, on one of four probe designs. Nothing is trained.
- **Probe designs:**
  - 4 new designs from the plain W_s generator, seed 4747.
  - They're never scored, so they aren't excluded from training, as with Step 2's diagnostic designs.
  - No dev or check design is ever used in a probe.
- **Placements:**
  - Up to 16 per probe, each given as brick type, anchor row and anchor column.
  - They run in the search's rule world through Stage H1's `rules.py`, unchanged, on the GPU.
  - The probe ends early if a placement is illegal, as an episode does.
- **What comes back** (the format of Step 2's traces, minus the design's own height map):
  - the target-height channel as the agent observes it at the start (8 rows of 8 digits);
  - for each placement: the action, the cells that became filled (row, column, layer), cells inside and outside the target, and the reward;
  - the board's height map at the end.
- **Why the design's own map is left out:** with it, the shifted booklet would show without any experiment. Without it, the AI scientist has to test the booklet. For example, it can place a brick where the booklet says one belongs and see the world count it as outside the target.
- **How the AI scientist sees it:**
  - **The call:** one extra call per round. The AI scientist replies with its probes, or with none, in a fixed JSON format.
  - **What happens next:** the harness runs them, and the proposal call that follows gets the results.
  - **Later rounds:** they see the last round's probes in full and earlier probes as their requests only.
- **The sandbox text** (the same in every sandbox arm, and naming no rule): "Before you propose this round's candidates, you may run up to 3 probes in this world. A probe places bricks you choose, one after another, on one of four probe designs, and trains nothing. For each probe you see what the agent observes at the start and what the world does after each placement. Probes don't count against your trials. Reply with your probes, or with none."

## Arms

| Arm | Told the rule? | Sees after each round | Sandbox | Searches |
|---|---|---|---|---|
| **Sandbox** | No: Step 2's note | Step 2's scores-only feedback | Yes | 24 |
| **Scores only** | No: Step 2's note | Step 2's scores-only feedback | No | 24 |
| **Told, with sandbox** (descriptive) | Yes: Step 2's exact rule sentence | Step 2's scores-only feedback | Yes | 12 |

- **Scores only is run again** in the same campaign rather than reused from Step 2, so a change in the model service between days can't masquerade as an effect.
- **Everything else is Step 2's, imported, not copied:**
  - the start (reference A with Stage G's recipe);
  - the program (`g0r.md`) and prompts;
  - 4 rounds of 3 candidates;
  - trial seeds 0 and 1, and confirmation seeds 5001–5004;
  - the pick rule, checks, repairs, rule review and diagnosis judge;
  - the AI scientist's model and settings, with no tools and an empty directory.

## Which worlds run (fixed now, read from Step 2's full run)

- **Implementable:** a world runs only if at least 9 of Step 2's 12 told-arm searches in it scored at least 38.3%. Otherwise the AI scientist can't reliably write the fix even when told, and a sandbox can't help with that.
- **Room left:** a world runs only if fewer than 9 of Step 2's 12 scores-only searches in it scored at least 38.3%. Otherwise scores alone already find the fix, and there's nothing for the sandbox to add.
- **Both worlds qualify:** 12 searches per world in the sandbox and scores-only arms, and 6 per world in the told arm.
- **One world qualifies:** all searches run in that world.
- **Neither qualifies:** Step 2S doesn't run. The result says why, and a sandbox test would need other worlds under a new protocol.
- **If Step 2's full run doesn't happen,** neither does this step.

## Analysis, fixed now

- **Unit:** one search, scored by its confirmed mean, as in Step 2.
- **Primary, one test:** P1S is sandbox − scores only.
  - The difference in mean score, pooled over the worlds that run.
  - Two-sided permutation test shuffling arm labels within each world, with 1,000,000 seeded shuffles, using Step 2's `analyze_step2.py` function.
  - It passes if positive with p < 0.05.
- **Precondition for reading P1S as discovery:** at least 9 of the told-with-sandbox arm's 12 searches score at least 38.3%. If not, P1S is reported but no claim about discovery is made.
- **Reported, no decisions:**
  - Searches scoring at least 38.3% per arm and world, with Fisher's exact test.
  - The round of each search's first trial at 38.3% or more.
  - The diagnosis judge's verdict per search (does its last analysis state the rule?), by arm.
  - **Probe use:**
    - probes per search;
    - placements per probe;
    - how many searches used no probes.
  - **dig.bench's prediction, stated now:** a player who knows the rules hardly uses creative mode, so the told arm should request fewer probes than the sandbox arm. Reported as the difference in mean probes per search, with a permutation p.
  - **What probes did:** described by hand from the saved requests, labelled as descriptive. For example: single bricks on a probe design, or placements where the observed booklet says a brick belongs.
  - **Drift check:** this step's scores-only arm against Step 2's scores-only arm in the same worlds.
  - Calls, cost and time.
- **Endings, fixed now:**
  - **P1S passes, precondition holds:** "Given a sandbox to experiment in, the AI scientist found fixes for hidden rules it wasn't told more often than with scores alone."
  - **P1S passes, precondition fails:** "Sandbox searches scored higher than scores alone, but even told the rule it couldn't reliably write the fix, so I make no claim about discovery."
  - **P1S fails:** "Experimenting in a sandbox didn't measurably help it find hidden rules here."
  - **Every ending states:** one or two rules of one kind, toy scale, one model, whose weights never changed.

## What each comparison can and can't show

| Comparison | If it holds, we may say | What it can't show |
|---|---|---|
| **P1S:** sandbox − scores only | Experiments the AI scientist designed itself led it to fixes that scores alone didn't | That a sandbox adds anything on top of good readouts (Step 2's readouts already show both rules) |
| Told with sandbox: share that find the fix | The sandbox doesn't get in the way of writing a known fix | Anything about discovery |
| Probes, told against not told | Whether it experiments when it needs to, as dig.bench predicts | Whether the probes it chose were good ones |
| Diagnosis judge, sandbox against scores only | Whether experiments made it state the rule, not just stumble on a fix | Whether stating the rule caused the fix |
| No difference | It didn't find hidden rules by experimenting, here | That it never could: 24 searches per arm detect only large gaps |

## Power

24 against 24, as in Step 2:
- if scores only finds the fix in 25% of searches, a sandbox at 72% is clear (Fisher p ≈ 0.003);
- 60% is at the edge;
- 42% would be missed.

## Integrity rules

- **Frozen files:** no frozen file is edited. Stage 2b's 39 hashes, Stage H's 26 files and Step 2's manifest are checked before the first run and after the last.
- **Where new code goes:** `arch_rsi/stageH/step2s/` only. It imports Step 2's and Stage H1's code, and anything copied must be shown identical by a test.
- **Keeping the rule hidden:** Step 2's forbidden-word test covers every harness-written text in the not-told prompts, including the sandbox text and probe results.
- **Probe tests:**
  - A probe's filled cells match the rule world's own step, placement by placement.
  - In the plain world they match `BrickWorld`.
  - No probe ever uses a dev or check design.
  - A resume after a crash repeats no call.
- **Before the full run:**
  - Tests with fake training and fake calls.
  - Then a dry run with real calls, 1 search per arm per world that runs, excluded from the analysis.
  - The full run starts only after the user approves the dry run's output.
- **Final splits:** no final split exists for these worlds, and none is read. No FINAL_OPENED marker is written.
- **Stage 2e:** stays paused.
- **How runs go:**
  - In a Terminal tab on the Mac's GPU, four trainings at once.
  - Every prompt, probe and reply is saved.

## Cost

- **Training:** (24 + 24 + 12) searches × 28 runs + 12 start runs = 1,692 runs. At Stage H1's pace (about 16.7 s per run, four at once) that's about 7.8 h, so allow 8–10 h.
- **AI-scientist calls** (rough, scaled from Step 2's cost section):
  - **Scores only:** about 4.1 calls per search at Step 2's $0.22–0.33 per call, so about $22–33 for 24 searches.
  - **Sandbox arms:** about 8.1 calls per search (one probe call per round). Probe results lengthen prompts, so allow up to twice Step 2's cost per call, about $3.6–5.3 per search. That's about $130–190 for 36 searches.
  - **Total:** about $150–225 notional, plus up to 60 rule-review and 60 judge calls.
  - This is higher than the rough $70–110 I gave in chat, which left out the told arm and the long probe-result prompts.
- **Where to cut if needed:** dropping the told-with-sandbox arm saves about $43–64 and 1.6 h. It would also drop the precondition and dig.bench's prediction, so I'd keep it.
- **Dry run:** 1 search per arm per world, roughly a twentieth of the above.

## After Step 2S (not part of this protocol)

- **If P1S passes:** probing is a skill the loop has. Step 3 can let the AI scientist rewrite how it probes across worlds whose rules differ.
- **If it fails:** the next question is whether better-chosen experiments would help. That would come from rewriting the procedure, which is again Step 3.
