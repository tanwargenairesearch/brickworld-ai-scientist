# Research program (generation 1)

## Goal
Find a policy-network architecture that raises dev success at a fixed training budget. Only the network changes; the training recipe, observation, reward and budget are fixed.

## Core lesson so far (evidence: generation 0)
With a budget of about 8 PPO updates (roughly 256 Adam steps), a network that starts near uniform never completes an exact build. Without completed builds it gets no success signal, so it cannot learn one. Generation 0 tested four designs: the conv baseline, footprint landing features fed to a small-init MLP, a per-layer trunk read at the landing layer, and zero-initialised linear paths with gain 3 or 8. All scored 0.000 on dev and 0.000 on training success. Correct features are necessary but not enough. **The network must already be competent before training**, and training should only fine-tune it. Logits move only about 0.1–0.3 during training. That works in both directions: a well-set prior survives training, but a prior the network has to learn cannot form in time.

## How to build a competent prior
Compute exact per-action features with fixed operations that work at any grid size and height Z: pooling over the footprint for each brick type, and comparisons against the height maps. Then give the logits **non-zero, hand-set initial weights** on those features, with margins of several logits, plus a small learnable residual. Optionally keep part of the score as a fixed buffer so that the entropy bonus and noisy advantages cannot erase it. Useful terms:
- Landing layer L = max of the build heights under the footprint. Placing at L >= Z is invalid or bad.
- Good placement: every footprint cell is a target cell at layer L and currently missing. Give it a strong bonus.
- Any footprint cell that is not a target cell at L counts as extra cells, which can never be removed and so make an exact build impossible. Give it a strong penalty.
- Burying: a missing target cell below L under the footprint is sealed forever. Give it a strong penalty.
- Order: prefer the lowest landing layer first, so that supports exist before the layers above them.
- Overhangs (target cells with empty cells beneath them) can only be made by a multi-cell brick that rests on a filled cell elsewhere in its footprint. Prefer larger good bricks, and add a tie-break that favours footprints covering unsupported target cells.
- DONE: a strong positive logit exactly when missing = 0 and extra = 0, and strongly negative while good placements remain.
- Value head: give it the global features fraction missing, fraction extra, an exact-build flag and time.
Try to sanity-check the prior by reasoning: on a simple design, would greedy argmax build the target exactly? Look for features that are off by one (layer indexing, anchor/footprint alignment, padding at the edges).

Read the interface rules for any check on learnability, parameter use or how training changes the network, and obey them. If hand-set weights are disallowed or penalised, use an informed initialisation of learnable weights instead.

## How to run each round
1. Read every result so far, including failures and training success over the last 10%. Single-seed differences of a few points can be noise; compare against the task's noise figure.
2. Write down what the evidence suggests before proposing anything.
3. Propose a mix:
   - **While the best dev success is 0:** every candidate is a distinct, bold attempt at a nonzero score. Examples: different prior strengths, different tie-break and ordering rules, a fixed prior plus a learnable residual versus a fully learnable informed initialisation. Refining a 0-scoring design teaches nothing.
   - **Once something scores above 0:** about half the candidates refine the best one with one main change each; the rest test an idea the results have not covered.
4. Give every candidate a hypothesis: what changes, why it should help on this task, and what result would refute it.
5. Reason from the task's structure (what the agent must perceive and decide to place the right brick in the right order) rather than reaching for generic tricks such as more width or depth.
6. Keep files simple and correct. A candidate that fails the checks wastes a trial. Stay well inside the parameter and compute caps. A cheaper network gets **no** extra steps (steps are capped at the baseline's), so the only reason to keep a network small is speed and time-limit safety. Small, mostly fixed-feature networks are fine.
7. Do not repeat an idea that clearly failed unless you can say what is different. Already failed: learning from scratch on correct features, whether through a small-init MLP, a per-layer trunk, or a zero-initialised high-gain linear path.
8. Diagnostics: if training success is above 0 but dev is 0, suspect DONE timing or overfitting to training. If both are 0, the prior is not competent: check the feature definitions and margins.

## Revising this program
After each generation of searches, this program is rewritten by you from the logs of those searches, and then used for new searches, including on worlds of other sizes that you will not see while revising it. When revising:
- keep advice that the evidence supports, and say what evidence supports it;
- remove or weaken advice that did not help, including the competent-prior principle if trials built that way also score 0, and record why;
- write lessons that should hold on other grid sizes and heights: feature definitions must be size-agnostic (no hard-coded Y, X, Z or design counts);
- keep the whole program under 1,000 words, including this section, which you may also improve.
