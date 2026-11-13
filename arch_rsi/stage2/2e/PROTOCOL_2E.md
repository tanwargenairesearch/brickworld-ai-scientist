# Stage 2e protocol: do experiments guide a longer search? (Level 1b, third test)

**Status, 10 November 2026: built, tested, dry-run and approved for launch.** The user approved building option A ("Yes, build A") and then the dry run ("Yes, dry run"). The runner, the analysis, the rule review, the tests and the endings exist. The tests pass with fake training and fake model calls, and the dry run passed with the real trainer, checks and model at tiny budgets. When this file was frozen, no full-budget Stage 2e run had been made. The user then approved the launch and set both ceilings (see "Decisions for the user"). This file doesn't change after that: the approval is in `LAUNCH_APPROVED.md`, with this file's hash, and the launch and everything after it are in the campaign's `progress.log`.

Written 10 November 2026, after Stage 2b's result (`../2b/REPORT.md`), because the user asked for a draft ("Yes, draft it"). Writing it trained nothing and made no LLM call: it only read Stage 2b's saved results. The section "Changes made while building" lists what moved between the draft and the code.

**The name:** 2c is the self-rewrite stage, which didn't run. 2d is "gains compound" in the article's ladder. So this is 2e.

## What it can and can't show

| Result | What we may write | What it can't show |
|---|---|---|
| **The gate passes:** seeing its scores beats hiding them | "With 24 tries per search, experiments guided the search" (Level 1b), on W. Both branches start from the same first ideas and use the same selection rule, so the only difference is whether the AI scientist saw its scores | That the procedure improved itself (that's Stage 2c). Anything beyond W, toy scale and one model. Which part of the feedback mattered |
| **No pass, and the interval's upper end is below +10 points** | "Three tests agree. With twice the tries, any benefit from seeing its scores was under X points" (X is the measured upper end) | That feedback never helps. A smaller benefit, or one at a larger budget, stays possible |
| **No pass, and the upper end is +10 points or more** | "Not shown again. A benefit of up to X points isn't ruled out" | Either direction |
| **Seeing its scores is significantly worse** | "At this budget, seeing its scores made the picks worse" | Why |

## What Stage 2b's data says about the next test

I measured four things in `campaigns/stage2b` before choosing a design. The scripts and their output are in `planning/` (`planning_output.txt`). "g0" is the arm that saw its scores; "hidden" is the same search with scores hidden.

| Measured in Stage 2b | g0 | hidden |
|---|---|---|
| SD between the 2 seeds of one candidate | 0.163 | 0.180 |
| True SD among one search's 12 candidates (seed noise removed) | 0.193 | 0.176 |
| Share of a 2-seed score that is signal (4 seeds would give) | 74% (85%) | 66% (79%) |
| The top 3 on their first 2 seeds, then on 2 new seeds | 59.3% → 54.5% | 54.3% → 43.3% |
| SD between the 8 confirmation seeds of one pick | 0.224 | 0.246 |
| Standard error of a pick's 8-seed score (16 seeds would give) | 0.079 (0.056) | 0.087 (0.061) |
| SD between searches' confirmed scores | 0.196 | 0.129 |
| Running best after round 1, before either arm has seen a candidate's score | 31.4% | 41.6% |
| Correlation of the round-1 best with the final confirmed score | +0.24 | −0.06 |
| Picks by round (1, 2, 3, 4) | 0, 4, 2, 10 | 3, 3, 9, 1 |

**"Signal"** is the share of the spread among one search's 2-seed scores that comes from real differences between candidates, not from seed luck.

Three things follow.

1. **The feedback was already mostly signal.** A 2-seed score was 66–74% signal. Doubling the seeds per try would raise that to 79–85%, at twice the training cost of every try. So noisy feedback is probably not the main reason 2b's test showed no benefit.
2. **The outcome is the noisy part.** The difference between one g0 search and one hidden search had an SD of 0.235. With 16 searches per arm, that gives the interval 2b reported, about ±11 points. Two parts of it can be bought down: the number of searches, and the 8-seed confirmation, whose standard error was 8–9 points per pick.
3. **Round 1 says little about the end.** The arms were 10 points apart after round 1, before either had seen a candidate's score. Chance is one cause. The prompts are another: in 2b the score-seeing prompt already showed the start's own score and its spread over seeds in round 1, and the hidden prompt didn't. The data can't separate the two. Either way, the round-1 best explained almost none of the final score (2% of its variance, both arms pooled). Both are noisy measures, so the real link is somewhat stronger, but it is still weak. Most picks came from later rounds.

**What I changed in my own plan because of this.** I had expected to recommend more seeds per try, and to gain power by pairing searches on a shared first round. The data supports neither strongly. The recommended design below spends the compute on more rounds, more searches and a cleaner outcome. It keeps the shared first round, because it costs nothing and removes that round-1 gap whatever its cause, but the power figures assume it cuts no noise.

## Design (recommended: option A)

### Setting, exactly as in Stage 2b

- **World:** W, 10×10×6 with 8–20 bricks, scored on W's 512 dev designs.
- **Budget per training run:** fixed FLOPs, equal to the start's 131,072 steps with Stage G's recipe, and the 900 s limit.
- **What a candidate may change:** the network and the RECIPE block, with 2b's bounds and 2b's pre-training checks and repairs.
- **The AI scientist:** the same model, effort and call settings as 2b (`scientist.py`, frozen), and the same program (`../programs/g0r.md`).
- **Nothing opens a held-out split.** W's final designs and the xl final stay locked. `FINAL_OPENED_W` is not written.

### Forked pairs

The unit is a **pair** of searches that share their first round.

1. **Round 1, once per pair.** One call with the scores-hidden prompt writes 3 candidates. They pass the checks and train on seeds 0 and 1.
2. **The fork.** The saved state after round 1 is copied into two branches.
3. **Rounds 2 to 8, per branch.**
   - **Seen:** the prompt is 2b's g0 prompt. It shows every earlier trial's code, hypothesis and scores, the start's score and the current best.
   - **Hidden:** the prompt is 2b's hidden prompt. It shows every earlier trial's code and hypothesis, whether it trained without error, and no score.
4. **Why round 1 uses the hidden prompt:** nothing in the hidden branch has then ever been shaped by a score. The seen branch loses one thing in round 1: the start's own score. It sees that from round 2.

Each branch makes 24 tries in all (8 rounds of 3), against 12 in 2b. Seven of its rounds can use feedback, against three.

### Selection and confirmation

- **Selection is the same code in both branches,** and it always uses the real scores, as in 2b. The 3 tries with the best 2-seed mean each train seeds 2 and 3. The pick is the best 4-seed mean among them, if it beats the start's own 4-seed mean. Otherwise the pick is the start.
- **Confirmation:** each pick trains 16 fresh seeds (2001–2016). A branch scores the mean of those 16. A failed or timed-out seed scores 0.
- **Shared runs:** if both branches retest the same round-1 candidate, or pick the same code, the runs happen once and both branches use them. Two branches with the same pick differ by exactly 0.
- **Comparator:** the start with Stage G's recipe, on seeds 0–3 and on the 16 confirmation seeds.

### What changes from Stage 2b

| | Stage 2b | Stage 2e (option A) | Why |
|---|---|---|---|
| Searches | 5 arms, 16 independent searches each | 2 branches per pair, forked after a shared round 1 | The first round is the same for both branches by construction: same prompt, same candidates, same scores |
| Tries per search | 12 (4 rounds of 3) | 24 (8 rounds of 3) | Feedback gets 7 rounds to act, not 3 |
| Seeds per try | 2 | 2 | 2-seed scores are already 66–74% signal |
| Confirmation seeds per pick | 8 | 16 | The standard error per pick falls from 8–9 points to about 6 |
| Primary tests | 2, with Holm's correction | 1 | The gate is tested at 0.05, not 0.025 |
| Searches behind the gate | 16 + 16 | 32 pairs, or 48 if the spread is high | Power |
| Arms dropped | | network only, recipe only, random draws, reference A | Their questions were answered in 2b |

**What dropping those arms loses:** this run can't re-check that the AI scientist beats random draws (2b: +31.7 points) or reference A (2b: +3.6 points, not shown). It tests one thing.

**Two fixes to the runner's rules.**
- **The current best.** In 2b it compared a mean rounded to 4 decimals with the unrounded bar, so an exact tie with the start counted as a new best (`../2b/REPORT.md`, "Departures"). In 2e a try becomes the current best only if its unrounded mean is strictly higher. This changes only the line the seen branch reads.
- **The pick.** 2b's pick rule mixed rounded and unrounded scores in the same way. In 2e it uses unrounded scores throughout. I checked 2b's 80 searches: no pick would have changed.

## Analysis (to be fixed, with its code, before launch)

**Unit:** one pair. d = the seen branch's confirmed score minus the hidden branch's.

**Rule review first,** as in 2b: a fresh call reviews each distinct pick against the frozen rules, without knowing its branch. A pick judged non-compliant is scored as the start on the same 16 seeds. The review is blind, not independent.

**The one primary test (the gate):** an exact two-sided sign-flip test on the mean of d.
- It asks: if the labels "seen" and "hidden" were swapped at random within each pair, how often would the mean be at least this far from 0?
- It is computed over all sign patterns exactly, not sampled. Each score is a whole number of exact builds out of 8,192 (512 designs × 16 seeds), so the sums are integers.
- **The gate passes** if the mean of d is positive and p < 0.05. There is no second primary test, so no correction.

**Interval:** a bootstrap 95% interval for the mean of d, resampling pairs (10,000 draws, seed 0).

**The four readings,** fixed now:
1. **Pass:** the gate passes.
2. **Bounded:** no pass, and the interval's upper end is below +10 points. The bound reported is the measured upper end.
3. **Open:** no pass, and the upper end is +10 points or more.
4. **Worse:** the mean of d is negative with p < 0.05.

**Why +10 points:** it is the smallest gain this design has a good chance of detecting (about 85%, below). A bound tighter than the test can see would be an overclaim.

**Reported only, no tests:**
- each branch's mean, SD between pairs, and share of confirmation seeds that learned the hidden-cell skill;
- the paired difference in running best after each round, which shows where the branches part, if they do;
- pairs with the same pick in both branches, and picks by round;
- each branch against the start on the same 16 seeds;
- selection optimism, invalid candidates, time-outs, recipes chosen, calls and cost.

### The second stage: 16 more pairs if the spread is high

The spread of d under this design is unknown until it runs. So the number of pairs is set by one rule, fixed in advance.

- **Pairs are numbered 1 to 48 before launch.** Pairs 1–32 run first.
- **Then the runner reviews those picks and logs one number:** the root mean square of the 32 differences. It doesn't log the mean or any sign.
- **If that number is above 0.20, pairs 33–48 run.** Otherwise the run ends at 32.
- **The test runs once,** on every pair that ran.

**Why 0.20:** at that spread, 32 pairs give about a 78% chance of detecting a true 10-point gain (`planning/power_extra.py`). Above it, 32 pairs are too few.

**Why the test stays exact.** The test assumes one thing under "no effect": within a pair, it makes no difference which branch saw its scores, and pairs are independent. The size of a difference is the same whichever way the labels go. A rule that reads only the 32 sizes therefore uses nothing the test depends on: given the sizes, every pattern of signs is still equally likely. The same holds for a stop that depends only on time or cost. A simulation agrees: the false-positive rate was 0.044–0.056 with the rule, the same range as without it (`planning/power_2e.py`; Monte Carlo error about ±0.007).

## Options: cost and power

**How the columns are worked out:**
- **Training runs per pair:** the shared round (3 tries × seeds), then per branch 7 rounds of 3 tries, 3 retests and 16 confirmations. Option A: 6 + 2 × (42 + 6 + 16) = 134. The start adds 20 runs once (24 in option B).
- **Machine hours:** 26 s per run. In 2b, runs of these two arms took 25.0 s of machine time each (99.8 s with 4 at once), and their picks' confirmation runs 26.7 s. The GPU was the bottleneck: 2b's elapsed time matched its summed run time divided by 4.
- **LLM calls:** one per round, so 15 per pair in options A and B, plus 5% for repairs (2b: 7 repairs on 128 calls in these arms).
- **Notional dollars:** 2b's calls cost $0.22–0.26 each in rounds 1–4 and didn't rise as the prompt grew from 12,000 to 34,000 characters. Rounds 5–8 are unmeasured. The prompt would reach about 64,000 characters, so I allow $0.30 per call there. The rule review adds about $0.04 per pick.
- **Power:** the chance the gate passes for a true gain of +5, +10 or +15 points, by simulation of the sign-flip test (`planning/power_2e.py`). The central figure assumes the SD of d is 0.22: 2b's 0.235, with 16 confirmation seeds, and the two branches independent. The range is for an SD from 0.26 down to 0.17.

| Option | Pairs | Tries per search | Seeds per try | Training runs | Machine hours | LLM calls | Notional cost | Power at +5 / +10 / +15 | Power at +10, range |
|---|---|---|---|---|---|---|---|---|---|
| **A, two-stage (recommended)** | 32, then 16 more if the spread is high | 24 | 2 | 4,308 to 6,452 | 31 to 47 | about 505 to 755 | $140 to $210 | 32% / 85% / 99% | 73% to 91% |
| A, fixed at 32 | 32 | 24 | 2 | 4,308 | 31 | about 505 | $140 | 25% / 70% / 96% | 56% to 89% |
| A, fixed at 40 | 40 | 24 | 2 | 5,380 | 39 | about 630 | $175 | 27% / 80% / 99% | 66% to 95% |
| B, more seeds per try | 24 | 24 | 4 | 5,688 | 41 | about 380 | $105 | 19% / 56% / 88% | 43% to 79% |
| C, 2b-sized searches | 64 | 12 | 2 | 5,524 | 40 | about 470 | $115 | 44% / 95% / 100% | 86% to 100% |

**These are planning estimates for a design that hasn't run.** The spread could fall outside the range. Machine time could be off by a fifth or more, because candidates written in rounds 5–8 may train slower or faster than 2b's.

**Reading the options:**
- **A, two-stage, is what I recommend.** It doubles the tries, which is what the hypothesis needs, and the rule buys more pairs only if the spread calls for them. At 2b's spread it would extend in about three runs out of four, so plan for the higher cost.
- **B is the test as the article describes it,** with more seeds per try as well. For the same machine time it affords 24 pairs, so its power is the lowest. Its cleaner selection might narrow the spread; 2b's data can't say by how much.
- **C has the most power per hour,** but it repeats 2b's 12-try budget, where the measured gain was +1.5 points twice. It would mostly tighten the bound on a question already asked.
- **No option can detect a 5-point gain.** An 80% chance of that would take about 155 pairs at this spread (`planning/power_extra.py`).

**If the earlier two tests are a guide,** the most likely result is reading 2, a tighter null. When the true gain is 0, the interval's upper end falls below +10 points in about 73% of simulated runs at 32 pairs and 88% at 48.

## What each result would mean for the article and for Stage 2c

| Reading | The ladder's row 1b | Stage 2c |
|---|---|---|
| 1. Pass | "Shown at 24 tries (Stage 2e). Not shown at 12 tries (Stages 1 and 2b)." All three tests are reported, not only the one that passed | Eligible again. It still needs its own approval, and its protocol names 2b's gate, so that sentence needs an amendment first |
| 2. Bounded | "Not shown, three times. At 24 tries, any benefit is under X points" | Doesn't run |
| 3. Open | "Not shown, three times. The third test couldn't rule out a benefit of up to X points" | Doesn't run |
| 4. Worse | "Not shown. At 24 tries, seeing its scores made picks worse" | Doesn't run |

**A sentence in the current article would need to change.** It says the next test would "score each try on more seeds". Under option A it confirms each pick on more seeds and keeps 2 per try. I haven't edited the article.

## Costs, ceilings and stopping

- **Ceilings:** 50 machine hours and $230 of notional LLM use, set by the user on 10 November 2026. Both are in the approval file, and the runner reads them from there.
- **A new pair starts only while both are under their ceilings.** Hours are the time the runner is alive. Dollars are the notional cost of every call so far. Pairs already started always finish, so a run can end a little over. If it ends there, the analysis uses the pairs that finished, and the report says so.
- **No stop depends on a score.** The only rule that reads results is the second-stage rule, and it reads sizes only.
- **Integrity:** as in 2b, the campaign's manifest records the hash of every frozen file, this protocol included, and the run stops on any change.
- **A pair that can't finish** (a repeated infrastructure or model failure) stops the run for the user. Pairs aren't replaced silently.
- **Interruptions:** the run is launched in a terminal that outlives sessions. Every reply is saved before its candidates train. On resume, an unfinished round reuses its saved reply, after checking that the prompt is the same, and reuses a training run only if its folder holds the same code. Nothing is redrawn. In 2b a rule had to be applied by hand after the interruption; here the runner needs none, and a test crashes a round and checks the resumed run ends in the same result. Each interruption is logged and disclosed.

## Limits, stated now

- **One world, one model, toy scale.**
- **The dev designs have been used before.** 2b's 80 searches chose on the same 512 designs. 2e scores on them again, with fresh training seeds. Any fit to those designs affects both branches alike. The held-out designs stay locked.
- **"Scores" means everything the g0 prompt shows:** dev success per seed and on average, training success, and training time. The hidden branch sees none of these.
- **The value of knowing the start's score in round 1 isn't tested,** because round 1 is shared.
- **I will see the progress log while the run is monitored,** as in 2b. No decision uses it.
- **A pass after two nulls is one of three tests.** The report and the article give all three.

## Changes made while building (10 November 2026)

Three things differ from the draft the user approved. None changes the design, the test or the cost.
- **Resume reuses saved replies.** The draft said an unfinished round would be discarded and redrawn. The code reuses the saved reply, so an interruption adds no new draw.
- **The ceiling rule is exact.** The draft said the run stops when it is "on course" to pass a ceiling. The code's rule is the one above: no new pair starts once a ceiling is reached.
- **The pick rule uses unrounded scores.** I found the second rounding issue while writing the retest step.

## What was built, and what is left

**Built on 10 November 2026, after "Yes, build A":**
1. **`run_2e.py`:** a new runner that imports the frozen 2b runner. It reuses, unchanged, the two prompt builders, the checks, the trainer, the pick rule and the confirmation code. It adds the shared round, the fork, the two rounding fixes, saved replies, retests shared inside a pair, 16 confirmation seeds, the second-stage rule and the ceilings. No frozen file was edited.
2. **`analyze_2e.py`:** the exact sign-flip test, the bootstrap interval, the four readings and the descriptive tables.
3. **`review_2e.py`:** 2b's rule review, imported unchanged and pointed at pairs. A pick is never reviewed twice.
4. **`ENDINGS_2E.md`:** the article's wording for each of the four readings.
5. **Tests, with no GPU and no LLM calls** (`test_run_2e.py`, `test_analyze_2e.py`). They all pass:
   - both branches start from an identical round-1 state;
   - no prompt in a hidden branch or a shared round contains a score, and every seen prompt after round 1 does;
   - with 2b's baseline, 2e's round-1 prompts equal 32 of 2b's saved prompts word for word, apart from the number of rounds and tries;
   - an exact tie with the start isn't a new best;
   - a crash in the middle of a round, then a resume, makes no second call for that round and ends in the same result as a clean run;
   - a reply saved for another prompt, a trial folder with other code and a changed fork record each stop the run;
   - a retest of a round-1 candidate runs once for both branches, and two branches with the same pick differ by exactly 0;
   - a pick judged to break a rule is scored as the start;
   - the sign-flip test matches brute-force enumeration on 300 random cases;
   - the second-stage rule gives the same answer when any pair's labels are swapped;
   - the ceilings stop new pairs, and the real preset refuses to start without the approval file.

6. **The dry run** (`--preset 2e_dry`, `campaigns/stage2e_dry`), 10 November 2026, 18:43 to 18:55 UTC, after the user's "Yes, dry run". It was the first time the new runner met the real trainer, checks and model.
   - **What ran:** 2 pairs of 3 rounds of 2 candidates, at a quarter of the training budget, with 2 confirmation seeds. 61 training runs, all finished. 10 search calls and 4 review calls, $1.92 notional. 11.5 minutes.
   - **What it showed:** both forks were clean copies of their shared round; no hidden or shared prompt held a score, and every seen prompt after round 1 held them all; the rule review judged the 4 picks compliant; the second-stage rule read 0.053, below 0.20, and ended the run at 2 pairs; the analysis ran to a final status. No frozen file changed during it.
   - **What it didn't exercise:** the extension to more pairs, a repair call, a rule violation and a ceiling stop. The fake tests cover those.
   - **Its scores aren't evidence:** 2 pairs, tiny budgets.

**Left, each needing the user's word:**
7. **The approval file,** `LAUNCH_APPROVED.md`, with the user's words, the ceilings (`ceiling_hours:`, `ceiling_usd:`) and this file's hash (`protocol_sha256:`). The runner refuses to start without it, and refuses if this file changed after approval.
8. **The launch,** in a terminal that outlives sessions.

## Decisions for the user

1. **Which option:** A. Decided on 10 November 2026 ("Yes, build A").
2. **How many pairs:** built as the two-stage rule, 32 then 48, the recommended default. Say so if you want a fixed number instead.
3. **The bound for reading 2:** built as +10 points, the proposed default.
4. **The dry run:** done on 10 November 2026 ("Yes, dry run"). It passed.
5. **The ceilings:** 50 machine hours and $230 of notional LLM use. Decided on 10 November 2026: the session the user was talking to asked "Set Stage 2e's limits at 50 machine hours and $230 of notional LLM use?" and the user answered "Yes, set both". That session relayed the answer here.
6. **The launch itself:** approved on 10 November 2026, about 18:50 UTC. The user's words, typed in the same other session and relayed here: "lets complete stage 2e and then complete pending stages to compete this entire pipeine". This approval is used for Stage 2e only. It costs 31–47 hours of machine time and about $140–210 of notional LLM use, and the likely result is a tighter null. Nothing after Stage 2e starts on it: Stage 2c, W's held-out split and Stage 3 each need their own approval.
