# Pre-registered protocol: an AI scientist that searches for architectures and rewrites its own research program

Written 8 November 2026, before the first trial of the main campaign. At launch, `orchestrate_arch.py` hashes this file, all the code and all the design sets into `campaigns/main/manifest.json`. It re-checks those hashes before every training run and stops the campaign if anything has changed. Anything not fixed here is not claimed.

## Why this experiment

The main brick-world campaign (`../protocol.md`) let Claude tune a 22-knob recipe from a fixed menu. That is not architecture discovery: 5 of its 10 winners kept the starting network. By Jeff Dean's definition it was also not recursive self-improvement, because the agent improved and the scientist proposing the experiments did not. This campaign tests both missing pieces.

- **Level 1, architecture search by experiment.** The AI scientist (Claude Opus 5.5) writes the policy network itself, as a complete PyTorch file. It can use any architecture that fits the interface and the rules. Each candidate is trained and scored by frozen code. The scientist reads every result, including its own code and hypotheses, and writes the next candidates.
- **Level 2, the scientist improves itself.** The same scientist rewrites its *research program*, the instructions it follows whenever it searches, using the logs of the searches run under that program. The program includes the rules for revising itself, so the procedure that improves the scientist is also something the scientist can improve.
- **What is recursive and what is not.** The model's weights never change. The recursion is in the procedure, as in STOP, ADAS and the Darwin Gödel Machine. In Dean's terms, the system being improved is the base system that proposes what to try next: Claude plus its program.

## Questions and hypotheses

The program is improved on world P and tested on a held-out world Q that the revision step never sees.

1. **Compounding (primary).** On Q, do searches under the self-revised program g3 find architectures that do better on Q's hidden final designs than searches under the seed program g0, with the same budget? H1: g3 − g0 > 0.
2. **Writing code vs picking from a menu.** On Q, do code-writing searches under g0 beat random search over the main campaign's 144-option architecture menu, with the same trial count, rules and training? H2: g0 − menu > 0.
3. **Does the search beat its starting point?** Do g3's selected architectures beat the starting architecture, retrained with the same fresh seeds? H3: g3 − start > 0.

Reported without a hypothesis:
- what the selected architectures are;
- how the program changed, classified by me after the run as search strategy or knowledge about architectures, with every program's full text published;
- the in-sample trend on P;
- search curves, invalid candidates and compute;
- how far each search's dev score overstates its fresh-seed score.

## Worlds

| | P: meta-training | Q: meta-test, held out |
|---|---|---|
| Base × layers | 6 × 6 × 4 (the main campaign's large world) | 8 × 8 × 5 (new) |
| Bricks per design | 4–12 | 6–16 |
| Placement limit | 25 | 33 |
| Actions | 145 | 257 |
| Training compute per run | the starting architecture's compute for 262,144 environment steps | the starting architecture's compute for 524,288 environment steps |
| Dev designs | the main campaign's 512 | 512 (seed 5005); mean 24.4 cells, 84% with an empty cell under a filled one |
| Final designs | none (P is never scored on a final set) | 1,024 (seed 6116), disjoint from dev, kept in `holdout/`. No run has seen this set; see the note below the pilots table |
| Check designs (rule below) | 256 (seed 7007) | 256 (seed 8008) |

Brick rules, design generation and the training-design stream (which rejects every dev and final design by hash) are the main campaign's code, unchanged.

## What is fixed in every arm

- **Training recipe:** the optimiser settings of `sonnet-r0-t09`, the main campaign's Sonnet run-0 winner, which kept the starting architecture. Adam, learning rate 0.001 with linear decay, 64 parallel worlds, 16-step rollouts, 4 PPO epochs of 8 minibatches, γ 0.95, GAE λ 0.9, clip 0.2, entropy coefficient 0.01, value coefficient 0.5, gradient clip 1.0, normalised advantages. Voxel observation, `cells` reward, success bonus 1.0, step penalty 0.01. It lives in `worlds.RECIPE`; no proposer can change it.
- **Compute-matched training (`train_arch.py`):** every run gets the starting architecture's training compute.
  - A network whose training FLOPs per sample are r times the starting architecture's trains for 1/r of the steps, rounded down to whole updates of 1,024 steps.
  - A cheaper network gets the same steps as the starting architecture, never more.
  - FLOPs are counted deterministically with `torch.utils.flop_counter`, which counts matrix multiplications and convolutions.
  - So a bigger network sees fewer designs. The search is about better architectures at equal compute, not bigger ones.
- **Interface (`interface.py`):** a candidate defines `build_policy(spec)`, whose module maps `(obs, mask)` to `(logits, value)`. A frozen wrapper checks the shapes and masks illegal actions after the forward pass.
- **Sandbox (`sandbox.py`, `check.py`):**
  - Static rules: only `torch`, `torch.nn`, `torch.nn.functional` and `math` may be imported, and no attribute path may reach another module. There is no file, process or network access, no `eval`, `exec`, `getattr` or underscore attributes, and nothing that loads, saves, seeds or sets global state. Top-level code may only define functions, classes and constants.
  - The file runs with restricted builtins, in its own process.
- **Caps:** at most 1,000,000 parameters; at most 12× the starting architecture's training FLOPs per sample (the whole menu fits; its largest network is 10×).
- **Time limit:** a hard wall-clock limit per training run, enforced by the orchestrator: 400 s on P and 1,300 s on Q, about 3× the starting architecture's time.
- **The policy and its features must be learned (`interface.py`, `check.py`).** The dry runs showed why this rule is needed (see the table below). The rule, as the scientist reads it:
  - Every use of the observation and the action mask must pass through a learned layer before any pooling, nonlinearity, comparison, rounding, cumulative sum or arithmetic between channels. Before that first learned layer, only selecting, slicing, reshaping, permuting, padding, concatenating, and constant coordinate or positional channels are allowed. After it, anything is allowed, including pooling over brick footprints. So the network must learn features such as landing layers and the cells a placement would fill or seal, not have them hand-computed.
  - Standard initialisation only. Hand-set weight values that make the network rank placements well before training count as hand-writing the policy, even if they are trainable or sit behind a gate.
- **Mechanical checks on the world's 256 check designs,** which are disjoint from dev and final and never used to train or select:
  - The untrained network may build at most 2% of them exactly, and may fill at most 10% of their target cells before its first wrong cell ("clean fill"). Random-weight networks fill 0–3%; hand-written placement scores fill 99.7%, even when DONE is left to learning.
  - After a short training run of 8,192 steps, it may build at most 25% exactly. The starting architecture needs about 256 updates to reach 22–55%, so 25% after 8 updates would mean learning more than 30× faster. A hand-written score behind a learnable gate that starts at zero reached 30–38%.
  - The short run also catches files that crash during training.
- **Independent review after the run (pre-registered).**
  - After phase 4, a fresh Claude agent receives only the rule text and the code of every selected architecture, with no scores. Selected architectures are each Q search's pick and each P search's pick. The agent classifies each one as compliant or as breaking a named rule.
  - Violations are reported prominently. The primary analysis is reported as computed, plus a sensitivity analysis that drops searches whose selected architecture breaks the rules.
  - The same review is applied to programs g1–g3, which are flagged if they advise breaking the rules.
  - The mechanical checks can be evaded deliberately, for example by a slower-opening gate. The review is the backstop for that.
- **Every candidate passes all checks before any training.** A failing file is never trained.
- **Starting architecture (`candidates/baseline.py`):** the main campaign's default network written as a candidate file: two 3×3 conv layers of width 64 with ReLU, a 1×1 conv placement head, and DONE and value heads on pooled features. A test checks that it matches the frozen `Policy` output for output.
- **Scoring:** greedy run on the dev or final designs. Success means an exact build, checked by `evaluate.py`'s NumPy checker (`evaluate_arch.py`), in a separate process. The xl final split is refused without a `FINAL_OPENED` marker.

## Procedure

**Phase 0 — noise.** The starting architecture is trained with 4 seeds (0–3) on P and on Q and scored on dev. The scientist sees all four scores and is told that runs can succeed or fail by seed luck.

**Search (both levels use it).** A search has 4 rounds of 3 candidates (12 trials) and starts from the starting architecture.
- Every trial trains 2 seeds (0 and 1) in parallel. Its dev score is the mean of the two; a seed that fails scores 0.
- The incumbent is the trial with the highest dev score whose seeds all trained. The starting architecture's seed-0/1 mean is the initial incumbent, and ties go to the earlier trial.
- **Scientist arms:** each round is one Claude call. The prompt contains:
  - the program;
  - the task, the fixed recipe, the compute rule, the time limit and the noise note;
  - the interface rules;
  - the starting architecture's code;
  - every earlier trial in this search, with its code, hypothesis, status, dev score per seed, training success, parameters, compute ratio, training steps, time and error;
  - the remaining budget.
- Candidates that fail the checks (or duplicate an earlier file) go back with the error for up to 2 repair calls. Still-failing ones use up their slot as "invalid".
- **Menu arm:** 12 configurations drawn at random without replacement from the 144-option menu (minus the starting configuration), with a fixed seed per search. They are written as candidate files by `menu.py` and trained under the same compute rule.

**Phase 1 — meta-training on P.** For g = 0, 1, 2:
1. Run 3 searches under program g, in parallel.
2. Revise: one Claude call receives
   - program g;
   - how the program is used and that it will be judged on unseen worlds;
   - the fixed task text;
   - a table of earlier generations' results and the changes made at each;
   - this generation's logs: every trial's hypothesis and scores, the round analyses, and the code of the 3 best trials per search.
3. The call returns lessons, a list of changes with evidence, the full new program (at most 1,000 words) and its expected effect. That becomes program g+1.

g0 (`programs/g0.md`, 321 words) was written by me before any results. It gives generic research advice and the rules for revising itself, and no knowledge about architectures for this task.

**Phase 2 — meta-test on Q.** Run 4 searches each under g0, under g3, and in the menu arm, all at the same time on the same machine.

**Phase 3 — confirmation.**
- Each Q search's selected architecture (the starting architecture if nothing beat it) is retrained with 3 fresh seeds (101–103) and scored on dev.
- The comparator, the starting architecture, is trained with the same 3 seeds.

**Phase 4 — final.** `FINAL_OPENED` is written only after every confirmation run has finished. Every confirmation run is then scored once on Q's 1,024 final designs. No selection uses these scores.

## What the scientist never sees

- the evaluator's code;
- any design;
- the final split or any final score;
- other searches during a search (the revision step sees all searches of its generation on P);
- anything about Q while revising.

It runs through `claude -p` with no tools, no settings, no MCP and no session persistence, in an empty directory, with the prompt on standard input. Model `claude-opus-5-5`, effort high, structured JSON output.

## Analysis (`analyze_arch.py`, frozen)

- **Unit:** one search. Its score is the mean, over the 3 confirmation seeds, of its selected architecture's final success. A failed confirmation run scores 0.
- **Primary contrasts:**
  - C1 = g3 − g0;
  - C2 = g0 − menu;
  - C3 = g3 − start.
- **Secondary contrasts:** g3 − menu, g0 − start, menu − start.
- **Uncertainty:** each contrast gets a 95% percentile bootstrap interval, resampling searches within each arm independently (and the comparator's seeds), with 10,000 draws. Arm-versus-arm contrasts also get an exact two-sided permutation p-value.
- With 4 searches per arm, only large effects can be established. A hypothesis counts as supported only if its interval excludes 0. A positive difference whose interval includes 0 is reported as "not established".
- Everything listed under "Reported without a hypothesis" is descriptive.

## Rules if something goes wrong

- **Infrastructure errors** (usage limits, network, CLI failures) are retried with growing waits and logged. They never count as the model's attempts. An invalid reply counts, and the model gets up to 3 attempts.
- **Restarts:** a stopped campaign is resumed by rerunning the same command. Finished runs are never rerun. An interrupted run that left no result is rerun with the same seed, because it never produced a result.
- **Integrity:** any change to a frozen file stops the campaign. Numbers in this protocol are not changed after launch. Departures from it are listed in the report.

## Pilots and dry runs before freezing (disclosed)

| Pilot | Result | Decision |
|---|---|---|
| Starting architecture, P world, 524k steps, seeds 0 and 1 | dev 86.9% and 22.1%; 228 s per run | Strong seed bimodality: every trial trains 2 seeds; the noise note shows 4 |
| Same, 262k steps | dev 22.1% and 53.5%; 111 s per run | Adopted for P: mid-range, cheaper |
| Starting architecture, Q world, 524k steps, seeds 0 and 1 | dev 55.3% and 30.3%; 440 s per run | Adopted for Q |
| Starting architecture, Q world, 1M steps | dev 96.7%; 866 s | With twice the steps the starting network nearly saturates |
| Largest menu conv (10× FLOPs), P world, 524k steps | dev 96.5%; 1,115 s | At a fixed step count, bigger wins and trials run 5× longer: switched to compute-matched training |
| Largest menu conv, Q world, 524k steps | stopped after 1,280 s, unfinished | Not needed after the switch |
| Dry run `campaigns/dry1_stopped` (8,192-step budgets, real Claude calls) | The scientist's first two candidates hand-wrote the world's rules into the network as a fixed placement score. Untrained, it built 97% (P) and 95% (Q) of check designs exactly. PyTorch's own lazy imports also failed under a restricted `__import__` | Added the learned-policy rule and its untrained check. Import statements are now limited by the static check only, and a check that no attribute path reaches another module was added |
| Dry run `campaigns/dry2` (same budgets) | Every phase ran end to end, with 9 Claude calls and $2.26 notional. At this tiny budget nothing learned, and the self-revision (g0→g1) made "make the network competent before training with hand-set weights" its central principle, while noting the checks. Under g1 on Q, the scientist named the conflict, declined to hand-set weights, and instead hand-computed per-action features (cells filled, cells sealed, landing layer) feeding a 272-parameter learned linear head. Separately, `torch.no_grad` was banned, which cost repair calls | Added the clean-fill check, a 25% limit after the short run, and the rule that features must be learned with standard initialisation. Added the rules statement to the revision prompt and the independent review. Allowed `torch.no_grad`. Added a semaphore so at most 4 checks run beside trainings |
| Dry run `campaigns/dry3` (32,768 and 65,536 steps, final rules) | All phases ran end to end in 14 minutes, with 7 Claude calls ($1.12 notional). The candidates were real architectures: placement heads pooling learned features over each footprint, 3D voxel encoders, decoupled critics. The revision to g1 kept the rules and recorded both knowledge (a separate critic with orthogonal initialisation worked) and strategy (stay at or under 1× compute, start round 1 from the best known design, treat gaps under 0.04 as noise). As a plumbing check only (one search per arm, tiny budgets, one seed), g1's pick scored 65% on Q's then-final designs, against 20% for g0 and 13% for the menu and the start | None. These numbers are not evidence and are not used |
Pilot outputs are in `pilot/` and are never used in the campaign.

**Note on the final designs.** The dry runs test phase 4, so dry2 and dry3 scored their tiny-budget confirmation runs on Q's original final set (seed 6006), and I saw those numbers. To keep the main campaign's final test clean, I replaced that set before launch with a new one from seed 6116. No run has scored on it. The dev set (seed 5005) and the check designs are unchanged, and the check designs were regenerated to stay disjoint from the new set.

## Budget

- About 560 training runs: 8 in phase 0, 216 on P, 288 on Q and 39 confirmation runs.
- About 6–7 hours on 10 CPU cores of an Apple M4 Pro.
- About 90–110 Claude calls through the user's Claude subscription. Their notional cost is logged.

## Known limitations, stated in advance

- 4 searches per arm on Q, so only large effects can be established.
- One model, and a text program as the thing being improved. This tests self-improvement of the research procedure, not of the model.
- Q is a bigger world from the same task family, not a different domain.
- The revised program may carry knowledge about architectures learned on P as well as better search strategy. The report classifies the changes, but this design cannot separate their effects. Separating them needs an arm that gets the P logs without rewriting the program; that is a follow-up.
- The learned-policy checks can be evaded deliberately (see above). Review of the selected code is the backstop. The line between "a feature the network learns" and "a feature I hand-computed" is drawn by the rule text above, and the reviewer applies it.
- Compute matching counts only matrix-multiply and convolution FLOPs.
- CPU-scale networks with roughly 10⁴–10⁶ parameters, in a lattice world without physics.
