# Architecture search by a self-revising AI scientist: the plan

Proposal, 8 November 2026. Nothing has launched and no frozen file has changed. If you approve a direction, `protocol.md` gets rewritten before anything runs.

## 1. The short version

The frozen design would run and produce numbers, but it can't support the two claims we care about:

- **"It searched by experiment."** No arm withholds results, so we can't tell whether running experiments helped, or whether Claude just wrote good networks from what it already knows.
- **"It improved its own procedure."** The rewritten program can carry over the winning designs from the first world. A win could be copied knowledge, not a better procedure.

Launching it also spends something we can't get back: the held-out final set can only be opened once.

This plan keeps the machinery and changes what gets compared. It has three stages, and each must pass before the next one runs.

## 2. How the loop works

```
LEVEL 2: the scientist improves its own procedure
  program g0 -> searches on world P -> read the logs -> rewrite program -> g1 -> ... -> g3
                     |
LEVEL 1: one search  v
  read the program and every result so far
  write 3 PyTorch networks, each with a hypothesis
  check each file against the sandbox and learned-policy rules   (frozen code)
  train it with the fixed recipe and equal compute               (frozen code)
  score it on 512 dev designs                                    (frozen code)
  repeat for 4 rounds, keep the best network

TEST: run the programs on world Q, which the rewrite never saw.
      Retrain each pick with fresh seeds. Score once on hidden designs.
```

- **One experiment** is one training run of a network Claude wrote: about 2 minutes on P and 7 on Q.
- **The search space** is any network that fits the interface, up to 1M parameters and 12× the starting network's compute. A bigger network trains for fewer steps, so it can't win just by being bigger.
- **The thing being improved at Level 2** is the program: 300–1,000 words of instructions, including the rules for rewriting itself. That self-reference is the recursive part.
- **What never changes:** Claude's weights, the loop code, what gets logged, and how a revision is accepted.

**A real example (dry run 3, tiny budget, not evidence).** On P, Claude tried a separate critic with orthogonal initialisation, a footprint-pooled placement head, a residual trunk and a 3D voxel encoder. The separate critic scored best (16.8% against 13%). The rewritten program then told later searches to start from that design, stay at or under 1× compute, and treat gaps under 4 points as noise. So the loop does work mechanically. The example also shows the problems: the program mixed knowledge with procedure, the evidence was single seeds, and the 4-point noise figure came from a tiny budget where nothing had learned.

## 3. Each claim needs its own control

| Level | Claim | What tests it | Today |
|---|---|---|---|
| 0 | Tuning a recipe works | Random search | Done (main campaign) |
| 1a | Claude's networks beat the start | Start retrained on the same seeds | Yes |
| 1b | **Experiments guided the search** | Same arm with scores hidden | **Missing** |
| 2a | A rewritten program helps on a new world | g3 against g0 on Q | Yes |
| 2b | **The procedure improved, not just the knowledge** | A rewrite that may only add facts, with the procedure locked | **Missing** |
| 2c | **The rewrite used the evidence** | A rewrite from logs with shuffled scores | **Missing** |
| 2d | **Gains compound** | Intermediate programs on Q | **Missing** |
| 3 | **The change alters what actually runs** | The program sets how compute is spent, and revisions are tested before being kept | **Missing** |
| 4 | The model improves its own weights | Out of scope, and we say so | n/a |

The honest name for what is frozen today is *iterated self-revision of a research prompt*. Level 3 is what starts to earn the word "recursive" for the procedure, because STOP and the Darwin Gödel Machine both score new versions before building on them.

## 4. What is wrong with the frozen design

I checked each of these against the files. Simulations are planning estimates (see section 10).

| # | Problem | Evidence | Fix |
|---|---|---|---|
| 1 | Can't show experiments matter | No arm hides results | No-feedback arm |
| 2 | Knowledge can pose as procedure | The revision sees the top 3 networks' code (`orchestrate_arch.py:309-311`). Dry-run g1 named a default base design | Facts-only arm |
| 3 | Only the endpoints are tested, from one lineage | Q runs g0 and g3 only (`orchestrate_arch.py:359`) | Test g1 too; 2 lineages |
| 4 | The score is two-humped, not bell-shaped | Two pilot runs at different budgets passed exactly the same 113 of 512 designs: every design with no hidden cells, and none with. Q's hidden-design success was 47% and 17% at 524k, and 96% at 1M | Report hidden-design success and how often seeds learn the skill |
| 5 | The loop learns from luck | The revision only sees unconfirmed best scores (`orchestrate_arch.py:302-304`). Estimated overstatement of a pick: about 19 points | Retest before selecting and before revising |
| 6 | Ceiling risk | Doubling Q's steps took the start to 96.7%, so a design that learns 2× faster saturates. No non-start network was piloted at full budget | Ceiling probe in Stage 0 |
| 7 | The decision rule is too loose | Bootstrap "interval excludes 0" at 4 against 4 gives about 15% false positives, not 5% | Exact permutation test with Holm correction |
| 8 | Too few searches | At 4 per arm, a true +20-point effect is detected about 22% of the time | 6 per arm or more |
| 9 | Weak comparison | 72 of the 144 menu options are MLPs (`menu.py`) | Keep it, but call it what it is: random draws from a small menu |
| 10 | Rule-breaking doesn't affect the headline | The primary analysis includes violators (`protocol.md:66`) | Primary analysis on compliant picks only |
| 11 | FLOP matching isn't time matching | 10× FLOPs cost 4.9× wall time | Keep FLOPs; also report measured cost |

The independent reviewer raised items 2–11 or variants of them. I confirmed the code references, the pilot figures and the menu count myself.

## 5. The plan

### Stage 0: calibrate the instrument (about 1 hour, about 4 Claude calls)

Nothing about the science can be planned well until we know the noise and the ceiling.

- Train the start on P and Q with 10 seeds each, split into hidden and no-hidden designs.
- Run one real g0 search on Q at full budget, as a ceiling probe. It's disclosed as a pilot and never used in the analysis.
- **Pass if:** the start sits mid-range and g0's best stays under about 85–90%. Otherwise I lower Q's budget until both are true.
- This stage sets the final number of searches per arm.

**Result (9 November, `pilot/stage0/REPORT.md`):**
- Seed noise is about 0.30 and two-humped.
- At the full budget, Opus's best design hit 96%.
- Both worlds now train for 131,072 steps.
- Stage 1 uses 10 searches per arm and 8 confirmation seeds.

### Stage 1: does it search by experiment? (one night, about 7–8 hours, about 90 Claude calls)

Pre-registered in `stage1/protocol_stage1.md` (9 November 2026, before any Stage 1 result). The code is in `stage1/`, and the original frozen files are unchanged.

Arms on Q at 131,072 steps, 10 searches each, with each pick confirmed on 8 fresh seeds:

- **g0 closed loop:** the normal search.
- **No-feedback g0:** same prompts, but the scores are hidden from Claude. It still sees crashes and invalid files. The frozen code still selects the best network using real scores, so the only thing removed is adaptation.
- **Random menu.**

Comparators, 12 seeds each, no search: the start, and the start with a separate critic and orthogonal initialisation, written before any results. The second one checks whether wins are just known PPO tricks.

Selection changes in every arm: after 12 trials with 2 seeds, the top 3 get 2 more seeds, and the pick is the best 4-seed mean. That is 30 runs per search instead of 24.

**Gate:** using fresh-seed dev scores, g0 must beat no-feedback and beat the start. If it doesn't, we report that, and I bring it back to you before Stage 2. The final set stays locked throughout.

### Stage 2: does the procedure improve itself? (about 13 hours, about 220 Claude calls)

Meta-training on P, in 2 independent lineages:

- Each P search's pick is confirmed with 3 fresh seeds before the rewrite sees it.
- **The program controls what runs (Level 3).** It carries a small machine-read header: candidates per round, seeds per candidate, how many to retest. The orchestrator obeys it, with total runs per search fixed.
- **Test before keeping.** A new program is kept only if its P searches beat its parent's. Otherwise the parent stays, and the next rewrite is told its change failed. This uses searches we'd run anyway.

Arms on Q, 3 searches per lineage per arm (6 total), against Stage 1's g0:

- final accepted program;
- first revision (g1), for the compounding question;
- facts-only rewrite: same logs, but Claude may only add facts and the procedure section is locked;
- shuffled-score rewrite: same logs with scores permuted, costing one Claude call and no P compute.

Then retrain each pick with 3 fresh seeds and open the final set once, across both stages.

### Stage 3: review and replicate (after the numbers)

- A fresh reviewer checks every selected network and program against the rules, without seeing scores.
- Replicate with Sonnet 5.5 to answer the model question directly. Opus stays for Stages 0–2, so a null result can't be blamed on a weak scientist.
- Only then does anything go into the article.

## 6. What stays the same

The sandbox, compute matching, the fixed recipe, the learned-policy checks, worlds P and Q, the locked final set, hash-based integrity checks and the search structure. This is a change to what we compare, not a rebuild.

## 7. Statistics and reporting

- **Primary test:** exact two-sided permutation, Holm-corrected within each stage's family. Bootstrap intervals are descriptive only.
- **Primary score:** exact-build success on final designs, compliant picks only. As-computed results are reported beside it.
- **Also reported:** hidden-design success, share of seeds that learn the hidden-cell skill, per-round means (best-so-far curves rise even under pure noise), measured compute, and every program's full text.

With Stage 0's measured noise, 10 searches per arm and 8 confirmation seeds give about 64–87% power for +20 points and 40–62% for +15. Small effects will stay "not established", and we'll say so.

## 8. What each result lets us say

| Result | What we may write |
|---|---|
| g0 beats no-feedback | "Experiments guided the search" |
| g0 ≈ no-feedback, both beat the start | "Claude's first ideas plus selection beat the start; experiments added nothing measurable" |
| Final program beats g0 only | "A program written from another world's logs helped": transfer, not self-improvement |
| Final beats g0, facts-only and shuffled | "Rewriting its own procedure helped, beyond carried-over knowledge" |
| Also final > g1 > g0 | "Gains compounded across self-revisions" |
| No differences | "No evidence of self-improvement at this scale" |

Even the best case is qualified: recursive improvement of a research procedure, at toy scale, with model weights unchanged.

## 9. What you'll be able to show

- A **search trace**: one search told round by round, as hypothesis, code, result and what changed next. This is the actual demonstration of architecture search.
- A **program diff** from g0 to the final program, with the evidence behind each change.
- The **selected architectures** with a plain description of each.
- The **control ladder chart**: start, menu, no-feedback, g0, facts-only, shuffled, g1, final.

## 10. Time, cost and assumptions

| | Machine time | Claude calls | Notional cost |
|---|---|---|---|
| Frozen design | 6–7 h | about 100 | about $25–35 |
| Stage 0 (done) | about 3 h, 2 of them lost to sleep | 8 | about $1.85 |
| Stage 1 | about 7–8 h | about 90 | about $18 |
| Stage 2 | re-estimated before Stage 2: cheaper per run at the new budgets | about 220 | about $45–60 |

- Times use the measured 111 s (P) and 440 s (Q) per run on 10 workers, plus an allowance for Claude latency.
- Costs use the dry runs' $0.25–0.29 per call, and longer logs will raise them.
- Over 250 Opus calls may hit subscription limits. The runner waits and retries, which can stretch a night.
- The code changes are mine to make before each stage, with tests and a dry run.

**Simulation assumptions.** Seed SD 0.20 (pilot pairs and main-campaign reruns range from about 0.13 to 0.46), spread between searches 0.10, architecture spread 0.10, 3 confirmation seeds. The noise model is bell-shaped, while the real noise is two-humped, so these are rough. Stage 0 replaces them with measurements.

Scripts are in the session scratchpad, and no training was run to produce these figures.

| Searches per arm | False positives, current rule | Detects +10 | Detects +20 | Detects +30 |
|---|---|---|---|---|
| 4 | 15% | 8% | 22% | 47% |
| 6 | 11% | 18% | 53% | 87% |
| 8 | 9% | 21% | 68% | 95% |
| 12 | 7% | 33% | 87% | over 99% |

Detection columns use the permutation test.

## Sources

- [NAS best-practices checklist](https://www.automl.org/nas_checklist.pdf) and [Lindauer and Hutter, JMLR 2020](https://www.jmlr.org/beta/papers/v21/20-056.html)
- [Agarwal et al., few-run RL evaluation, NeurIPS 2021](https://arxiv.org/pdf/2108.13264)
- [Gu, Ignatov and Timofte, closed-loop against single-shot LLM architecture design, 2026](https://arxiv.org/html/2603.12091v1)
