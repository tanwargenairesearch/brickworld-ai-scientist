# Step 3: can the AI scientist improve its own research procedure? (protocol)

**Status:** approved by the user on 15 November 2026 at about 12:37 GMT ("Approve defaults"), with the proposed default for each of the six decisions at the end. It was written at about 11:45 GMT, after Step 2's result (`../step2/REPORT_STEP2.md`) and Step 3a's (`../step3a/REPORT_STEP3A.md`). The build follows the build list. The dry run needs the user's go-ahead, and the full run waits for the user's approval of the dry run.

**Place in the plan:** this is Loop 3, the last of the three loops:
- **Loop 1:** an RL agent learns to build.
- **Loop 2:** the AI scientist designs that agent's network and recipe in a search. Step 2 showed it finds hidden rules from readouts of its own runs.
- **Loop 3 (this step):** the AI scientist rewrites the research procedure that runs its own searches, on worlds of some kinds. The rewritten procedure is then tested on worlds of a kind it never saw. Facts about one world can't fix another kind, so only a better procedure can carry over.

Step 2S, the sandbox, runs after this step, by the user's decision of 14 November.

**What this step reuses:**
- Stage 2c's self-rewrite design (`../../stage2/2c/PROTOCOL_2C.md`): a program with a machine-read header, lineages, test-before-keep, and the facts-only control. 2c was built and tested but never ran.
- Step 2's search runner, prompts and analysis.
- Step 3a's ten passing rule worlds and their code.

## What Steps 2 and 3a give Step 3

- **Ten passing rule worlds in four families** (Step 3a):

  | Family | Worlds | Fix |
  |---|---|---|
  | Translation | H1, H5, H6 | Roll the booklet back (H1 columns, H5 rows), or roll the placement logits back (H6) |
  | Reflection | H2, H7, H8 | Flip the input (H2 left to right, H7 top to bottom), or flip the booklet back (H8) |
  | Transposition | H9, H10 | Transpose the booklet back, on either diagonal |
  | Rotation | H11, H12 | Turn the booklet back a quarter or three quarters |

  In each one, reference A builds 0.0% and its one-change fix builds 71–77%.
- **Under readouts, little room is left** (Step 2): the readouts arm found a fix in 23 of 24 searches, at 76.2% on average, against 76.6% for reference A in the plain world. In 19 of 24 searches a trial reached 38.3% in round 1. A better procedure has almost nothing left to win there.
- **Under scores only, there is room, and the gap looks procedural.** Step 2's scores-only arm found nothing (0 of 24). Two things stand out in its record. Both were read from it after its result, so they're observations that shaped this design, not results:
  - The diagnosis judge found it naming the right kind of change in 16 of 24 searches.
  - In the mirror world, 12 of its trials built flipped and transposed copies (up to all 8 orientations of the square) inside one network, and gated or mixed them. One more was a control without them. None tested a single flip as its own candidate. None reached 38.3%; the best reached 11.5% on its 2 seeds.
  - In the shift world it never used a roll.
  - So it suspected the right kind of rule but didn't turn the suspicion into experiments that could tell the hypotheses apart. A better procedure could close that gap; a new fact alone may not.

## The question

The AI scientist rewrites its own research procedure from the logs of its searches in rule worlds of three families. Then, in worlds of the fourth family, with scores only as feedback, does the rewritten procedure find fixes more often than:
- the original procedure;
- a rewrite that may only add facts;
- a rewrite written without any logs?

## What each comparison can and can't show

All comparisons are on the held-out worlds, with scores-only feedback.

| Comparison | If it holds, we may say | What it can't show |
|---|---|---|
| **P1:** g1 − g0 | One rewrite of its own procedure, written in worlds of other kinds, helped in a kind it never saw (transfer) | That the procedure, not facts it added, did it (P2), or that the logs were used (P3) |
| **P2:** g1 − facts-only | The changes to the procedure helped, beyond facts added from the same logs | Which change helped |
| **P3:** g1 − no-logs | The evidence from its own searches mattered, not just being asked to rewrite | That it read the evidence correctly |
| **P4:** final − g1 | Gains grew across its own kept rewrites (compounding) | That this holds beyond two lineages |
| **Check:** g0 with readouts finds the held-out fixes | The held-out worlds' fixes can be found and trained, so a null result is about the procedure, not unreachable fixes | Anything about the rewritten procedure |
| Header changes kept after testing (descriptive) | The program changed how its own searches run, and changes were kept only after testing | That a header change caused a gain, since several things change at once |

**What we may write, by tier** (as in 2c):
- **"Rewriting its own procedure helped, beyond added facts and beyond rewriting without evidence":** P1, P2 and P3 all pass.
- **"Recursive", for the procedure:** that tier, plus P4 passes and at least one kept rewrite changed the header.
- **Anything less:** reported as measured, without those words.

## Setting

- **Worlds:** Step 3a's code unchanged (`rules3a.py`, `train_3a.py`, `eval_3a.py`, `check_3a.py`). W_s, 8×8×6, with 6–16 bricks.
- **Training:** 131,072 steps of Stage G's recipe, a 900 s limit per run, on the Mac's GPU, four runs at a time.
- **Scoring:** exact-build success on H-plain's 512 dev designs, under each world's rule.
- **Start:** reference A with Stage G's recipe, as in Step 2. It scores 0.0% in every rule world here.
- **What a search may change:** the network and a RECIPE block, within 2b's bounds, as in Step 2.
- **Feedback in every search:** scores only, as Step 2's scores-only arm. That is 2b's feedback: success per seed, training success, parameters, compute, steps and time. The one exception is the check arm (below).
- **The note,** in every search prompt, as in Step 2: "This world may not behave exactly as described above. Nothing more about it is stated."
- **The AI scientist:** the same model, effort and call settings as Step 2. No tools, and an empty directory. It can't see this protocol, any report or any code.

### Which family is held out (decision 2)

- **Held out: translation** (H1, H5, H6), for meta-testing only.
- **Meta-training worlds: the other seven** (H2, H7, H8, H9, H10, H11, H12).
- **Why translation:** the seven training rules are all among the 8 symmetries of the square: flips, transposes and turns. A roll isn't one of them.
  - Holding out rotation or transposition would leave a held-out rule that two training rules compose: a quarter turn is a transpose followed by a flip.
  - With translation held out, a fact such as "try the square's symmetries" can't fix a held-out world; only a more general procedure can.
- **H1 was in Step 2.** The AI scientist keeps no memory between calls, and no Step 2 log is ever shown, so this leaks nothing. Under scores only, Step 2's g0 found 0 of 12 there.

### The program and its header

- **The program:** Markdown, at most 1,000 words, as in 2c. g0 is `arch_rsi/stage2/programs/g0r.md` unchanged, with the header below added.
- **The header (Level 3):** read by the runner, which obeys it.

  ```
  <!-- header
  rounds: 4                 # R, 2-6
  candidates_per_round: 3   # K, 2-6
  seeds_per_candidate: 2    # 1-3
  -->
  ```

- **The bound:** R × K × seeds ≤ 24, Step 2's trial runs per search, so every program has the same training budget per search. g0's values are Step 2's exactly.
- **Changes from 2c's header:**
  - The retest step is dropped: Step 2 had none, because the gap between failing and fixed is about 75 points.
  - The network/recipe split is dropped: every candidate may change both, as in Step 2.
- **What the header makes possible:** a program can trade replication for more hypotheses. For example, 6 rounds × 4 candidates × 1 seed tests twice as many ideas as g0, at the risk that seed collapse hides a right fix. (In Step 2, 4 of the told arm's 5 misses looked like seed collapse.)
- **A bad header** (missing, unparsable or out of bounds) goes back for repair up to 2 times. After that the parent's header is used, and this is disclosed (2c's rule).
- **The pick and confirmation,** as in Step 2:
  - The pick is the trial with the best mean over its seeds, if that beats the start's mean on the same seed ids. Otherwise it's the start.
  - The pick is confirmed on seeds 5001–5004, and the search's score is the mean of those four.

### Lineages, generations and test-before-keep (meta-training)

- **Two independent lineages,** each starting from g0, with 3 rewrites (g1, g2, g3): 4 generations.
- **Each generation:** 7 searches per lineage, one in each meta-training world, under the lineage's current program. Every pick is confirmed before any rewrite sees it.
- **The rewrite** reads:
  - the current program;
  - the generation's logs: each search's trials (name, hypothesis, scores per seed, training success, parameters, compute, steps, time), its pick and the code of its top 3 trials;
  - each round's analysis, written by the AI scientist. 2c left these out only because its shuffled control couldn't relabel scores in free text. This step has no shuffled control (decision 5), so they stay in;
  - a table of earlier generations: each change and whether it was kept.
- **World labels:** worlds appear as "world 1" to "world 7", in a fixed random order per lineage. No label, family or rule is ever named.
- **What the rewrite is told:** the searches see scores only. The program will next be used in new worlds, which may each differ from their description in ways these worlds didn't.
- **Test before keeping:**
  - A rewrite is kept if the mean confirmed score of its 7 searches is strictly above its parent's.
  - Otherwise the parent stays, and the next rewrite starts from the parent and is told the change failed.
  - Every program is saved, kept or not.
- **Definitions** (2c's):
  - **g1** is the first rewrite, kept or not.
  - **final** is the last kept program. If nothing was kept in a lineage, final is g0 there, and P4 isn't tested for it.

### The two controls, each written once per lineage from the same input as g1

- **Facts-only:** the rewrite of g0 from the generation-0 logs, but the header and every section except a new "## Facts" section must be byte-identical to g0. 2c's checker enforces this, with up to 2 repairs, then the run stops.
- **No-logs:** the rewrite of g0 with the same prompt as g1, except that the logs and the table are replaced by one line saying there are none.
- **Why no-logs replaces 2c's shuffled control** (decision 5): under scores only, generation 0's scores are likely to be almost all 0. Shuffling zeros changes nothing, so a shuffled rewrite would see the same input as g1. No-logs still tests whether the evidence mattered.

### Meta-test (held-out worlds H1, H5 and H6, scores only)

| Arm | Searches | Program |
|---|---|---|
| g0 | 24 | g0, 8 per world |
| g1 | 24 | Each lineage's g1, 4 per world |
| final | 24 | Each lineage's final, 4 per world. If final is g1 in a lineage, those searches are shared, and this is disclosed |
| facts-only | 24 | Each lineage's facts-only, 4 per world |
| no-logs | 24 | Each lineage's no-logs, 4 per world |
| **Check: g0 with readouts** | 12 | g0, 4 per world, with Step 2's readouts arm's feedback and readouts |

- **No meta-test result reaches any rewrite,** and no held-out world appears in meta-training. Both are tested before any run.
- **Scheduling:** g0 and the check need no rewrite, so they run alongside meta-training. g1, facts-only and no-logs start once their programs exist, and final starts after the last generation.
- **Rule review:** before the analysis, one blind call per distinct pick, with 2b's reviewer, prompt and schema, unchanged. A pick judged a violation scores as the start.
- **Program review (descriptive):** one blind call per distinct program, asking whether it advises anything the rules forbid, such as hand-coding the world into the network. It's reported, not scored.
- **Diagnosis judge (descriptive):** one blind call per meta-test search, with Step 2's judge and the world's rule. It answers whether the search's analyses state the rule: yes, partly or no. It never affects a score.

## Analysis, fixed now

- **Unit:** one search, scored by its confirmed mean.
- **Test:** for each comparison, the difference in mean score between two arms, pooled over the three held-out worlds:
  - a two-sided permutation test that shuffles arm labels within each world, with 1,000,000 seeded shuffles;
  - bootstrap 95% intervals.
- **Primary family, Holm over 4:** P1, P2, P3 and P4. Each passes if positive with Holm-adjusted p < 0.05. A P4 that can't be tested enters Holm with p = 1, so the family stays at 4.
- **The check:** at least 9 of the check arm's 12 searches score at least 38.3%. It doesn't change any test. If it fails and P1 also fails, the ending says the test can't tell (ending D).
- **Secondary, reported only:** final − g0; each comparison within each lineage; each arm per world.
- **Descriptive:**
  - every program's full text, and its diff from g0;
  - header values per generation;
  - kept and rejected rewrites, with the scores behind each decision;
  - searches scoring at least 38.3% per arm, with an exact Fisher test per world;
  - the round of each search's first trial at or above 38.3%;
  - candidates per search actually tried;
  - the diagnosis judge's verdicts;
  - which kinds of change each program names, counted with a fixed word list;
  - calls, cost and time.

## Endings, fixed now

Use exactly one. The post's Loop 3 line comes from the selected ending, in these words.

- **A. P1, P2, P3 and P4 pass, and a kept rewrite changed the header:** "Rewriting its own research procedure, in worlds of other kinds, helped the AI scientist find fixes from scores alone in worlds of a kind it never saw. That beat a rewrite that could only add facts and one written without evidence, and the gain grew across its own rewrites."
- **B. P1, P2 and P3 pass, but not A:** "Rewriting its own research procedure, in worlds of other kinds, helped the AI scientist find fixes from scores alone in worlds of a kind it never saw. That beat a rewrite that could only add facts and one written without evidence. Further rewrites didn't measurably add to it."
- **C. P1 passes, but P2 or P3 doesn't:** "One rewrite of its own procedure helped the AI scientist find fixes from scores alone in worlds of a kind it never saw. The test didn't show that the procedure, rather than {added facts / the act of rewriting / either}, made the difference."
- **D. P1 fails, and the check fails:** "Even with readouts, it couldn't reliably fix the held-out worlds, so this test can't say whether a better procedure would help."
- **E. P1 fails, and the check passes:** "Rewriting its own procedure didn't measurably help it find fixes from scores alone in worlds of a kind it never saw."

**Every ending also states:**
- one model, whose weights never changed;
- two lineages, at toy scale;
- every rule re-indexes where things are, and one family was held out.

## What this step can't show

- **Other kinds of rule:** all ten rules are re-indexings. A procedure such as "test candidate remappings one per candidate" could carry over partly as a fact about this set of worlds.
  - The facts-only control tests that: whatever goes in its Facts section counts as a fact.
  - A family-level hint written into the procedure itself ("the world may remap where things go") sits on the line between fact and procedure. The report will quote any such text.
- **Another held-out family:** only translation is held out. Another choice could give another answer (decision 2).
- **Choosing what to look at:** feedback is scores only, chosen by us. Choosing its own probes is Step 2S's question.
- **A model improving itself:** the model's weights never change. At most, this shows a research procedure improving itself, at toy scale.

## Why this size

The figures below illustrate power using Fisher's exact test on the counts of searches that find a fix (4,000 simulations each). The real tests compare mean scores.

| Searches per arm | Comparison | 30% vs base | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|
| 24 | against about 0% (P1, if g0 stays at 0) | 61% | 91% | 99% | 100% | 100% |
| 24 | against 20% (P2, P3, if the controls also find some) | 3% | 9% | 28% | 54% | 80% |
| 18 | against about 0% | 28% | 64% | 88% | 98% | 100% |
| 18 | against 20% | 2% | 6% | 18% | 37% | 60% |

Each power figure is at α = 0.0125, the first Holm step.

- **At 24 per arm:** P1 is likely to pass if g1 finds fixes in 40% of searches or more.
- **P2 and P3 are harder:** if the facts-only or no-logs rewrites also find fixes in some searches, only large gaps will show. A C ending is a likely outcome even if the procedure does help.

## Integrity rules

- **Frozen files:**
  - No frozen file is edited: Stage 2b's 39, Stage H's 26, Step 2's 12, Step 2S's 12 and Step 3a's 22.
  - All are checked before the first run and after the last, and any change stops the run.
  - Step 3's own manifest is committed before the dry run.
- **Where new code goes:**
  - New code goes in `arch_rsi/stageH/step3/` only.
  - It imports Step 2's runner, Step 3a's rules and 2c's header and facts-only code rather than copying them.
  - Anything copied must be shown identical by a test.
- **Keeping the rules hidden:**
  - A test fails the build if any harness-written text in any search, rewrite or review prompt names a rule: "shift", "mirror", "flip", "roll", "offset", "reflect", "wrap", "rotate", "turn", "transpose", "diagonal", "quarter", "4 columns" or "4 rows".
  - Text the AI scientist wrote itself is exempt: hypotheses, analyses, code and programs. A program may carry what it learned in meta-training.
- **Separation:**
  - Tests check that rewrites read only meta-training logs, and that no held-out world appears in meta-training.
- **Identity tests:**
  - g0's round-1 prompt in H1 must be byte-identical to Step 2's scores-only prompt in the shift world.
  - The check arm's must be byte-identical to Step 2's readouts arm's.
- **Final splits:** none exists for these worlds, and none is read.
- **Stage 2e:** stays paused.
- **Tests before any real call:** fake training and fake calls for the whole flow: header, generations, keep and reject, both controls, the meta-test, and resume after a crash without repeating a call.
- **Dry run:** with real calls, at full training budget but tiny size: 1 lineage, 2 generations of 1 search each, and 1 meta-test search per arm. It's excluded from the analysis.
  - A bug the dry run shows is fixed and disclosed.
  - Nothing else in this protocol changes because of it.
- **Launch:** the full run starts only after the user approves the dry run, and `LAUNCH_APPROVED.md` is written.
- **How runs go:** in a Terminal tab, four trainings at once, up to 10 searches in flight. Every prompt and reply is saved.
- **Disclosure:** every departure is reported with the result.

## Cost (estimates)

| Part | Training runs | AI-scientist calls | Machine time |
|---|---|---|---|
| Starts: 10 worlds × 7 seeds | 70 | 0 | under 1 h |
| Meta-training: 2 lineages × 4 generations × 7 searches × 28 runs | 1,568 | about 240 for searches, plus 10 rewrites | about 7.5–9 h; the generations run one after another |
| Meta-test: 5 arms × 24 searches + 12 check searches, × 28 runs | 3,696 | about 560 | about 17–21 h |
| Reviews and judge | 0 | up to about 330 | under 1 h |
| **Total** | **about 5,300** | **about 1,150** | **about 25–30 h** |

- **The pace** comes from Step 3a (16.9 s per run, four at once) and Step 2 (20.0 s per run, including waits for calls).
- **Wall clock:** g0 and the check arm run alongside meta-training, so it is close to the machine time.
- **Notional cost:** about $250–350. Step 2 averaged $0.26 per call, and scores-only prompts are shorter than readout prompts.
- **Savings:** if final is g1 in both lineages, the final arm is shared, saving 672 runs (about 3.5 h).
- **Call limits:** over 1,000 calls may hit subscription limits. The runner waits and retries, which can stretch the run.
- **At 18 per arm** (decision 3): about 4,500 runs, 21–25 h and $200–280.

## Build list, after approval

1. **`run_step3.py`:** a subclass of Step 2's runner. It adds:
   - the header;
   - lineages and generations with test-before-keep;
   - the rewrite calls (g, facts-only, no-logs);
   - neutral world labels;
   - the meta-test and its schedule;
   - the check arm;
   - saved proposals per round, for resumes.
2. **`prompts3.py`:** the rewrite prompt, adapted from 2c's (`L2C_PROMPT`), with the scores-only description, the neutral labels and the no-logs variant.
3. **`analyze_step3.py`:** P1–P4 with Holm, the check, the endings and the descriptive tables, reusing Step 2's functions.
4. **`make_report_tables3.py`,** the manifest builder and the launcher.
5. **Tests:** all the integrity tests above.
6. **The dry run,** after the user's go-ahead.

**About the build:**
- No GPU use and no model calls before the dry run.
- The CPU tests take minutes.
- The build itself is a few hours of work.

## Decisions for the user (approved 15 November 2026: the proposed default for each, listed first)

1. **Feedback in the searches:** scores only (proposed), because it leaves room to improve and Step 2's record points to a procedural gap. The alternative is readouts with a smaller search budget. But Step 2's readouts arm reached 38.3% in round 1 in 19 of 24 searches, so even one-round searches would leave only about 20 points of room in the share of searches that find a fix.
2. **The held-out family:** translation (proposed). The alternatives:
   - rotation: cheaper to win, but a quarter turn is a transpose followed by a flip, both in training;
   - each family held out in turn: stronger, but it needs four meta-trainings, about 4 times the meta-training cost.
3. **Searches per meta-test arm:** 24 (proposed; about 25–30 h, $250–350) or 18 (about 21–25 h, $200–280; P1 at 40% has 64% power instead of 91%).
4. **The readouts check arm:** keep it (proposed; 12 searches, about 2 h). It's what lets a null result say something. Or drop it.
5. **Controls:** facts-only and no-logs (proposed), or 2c's shuffled control as well (about 3.5 h more).
6. **Lineages and generations:** 2 lineages × 3 rewrites, with 7 searches per generation, one per meta-training world (proposed, 2c's shape), or 1 lineage, which halves meta-training but leaves claims resting on one lineage.
