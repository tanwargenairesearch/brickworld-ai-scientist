# Research program (generation 1)

## Goal
Find a policy-network architecture that raises dev success at a fixed training budget. Only the network changes; the training recipe, observation, reward and budget are fixed.

## Known so far (generation 0, one search on a 6x6x4 world, single seeds)
- Baseline (2-layer 3x3 conv, 1x1 placement head, shared value): about 0.13 dev.
- **Best: separate small critic + PPO orthogonal init: 0.168 at 0.96x compute.**
  - Actor: the baseline conv body at width 56.
  - Critic: its own width-24 conv body and pooled MLP.
  - All conv and linear layers use orthogonal init with gain sqrt(2) and zero biases. The placement and DONE heads use gain 0.01; the value head uses gain 1.
  - Treat this as the default base.
- Footprint max/mean-pooled placement head at 1.36x compute (23.5k steps): 0.129, no better than the baseline.
- Residual trunk with global context broadcast at 1.06x: 0.131, no better.
- 3D voxel encoder with a softmax gate over layers and a thin 20-channel trunk: 0.02, collapsed.
- Training success stayed at 0.10 or below throughout, so learning is limited by samples. Wall time was never close to the limit (at most 42 s of 600 s), but compute above 1x costs training steps.

## How to run each round
1. Read every result so far, including failures. A single-seed difference under about 0.04 is unresolved noise. The baseline alone spans 0.129 to 0.135, and the gains seen so far are about the same size as that noise.
2. Write down what the evidence suggests before proposing anything.
3. Keep training compute at about 1.0x or below, so the network gets the full step budget. If a new head or layer adds FLOPs, pay for it by shrinking the trunk width or the critic. Do not trade steps for capacity: no trial showed extra capacity winning back lost steps.
4. Propose a mix:
   - **Round 1:** one candidate should be the best known recipe, or that recipe plus exactly one change. Do not spend both slots on untested ideas built on the plain baseline.
   - **Round 2:** about half the candidates refine the current best with one main change each, so that each result says something clear. The rest test an idea the results have not covered, built on the best base.
5. Give every candidate a hypothesis: what changes, why it should help on this task, and what result would refute it.
6. Think about the structure of the task as well as generic tricks. The agent must perceive the following and decide which brick to place, where, and in what order:
   - the landing height, which is the maximum height under the footprint;
   - whether the target fills exactly that layer over the footprint;
   - whether a placement would seal target cells below it that are still missing;
   - when the build is complete, which is when to choose DONE.

   Structural heads that expose these must be learned (learned features, pooled), not hand-computed. They have only been tested on the weak base or at high compute, so they are worth retrying on the improved base at ≤1x.
7. Prefer changes that make PPO learn more per sample:
   - decoupled actor and critic;
   - small-gain output heads;
   - normalisation-free, shallow, reasonably wide trunks (about 48–64 channels).

   Avoid thin trunks under about 32 channels.
8. Keep each file simple and correct. Use only standard layers. A candidate that fails the checks wastes a trial. Scale nothing by hand to a particular grid size: read sizes, channels, brick types and footprints from the spec, so the same code works on other worlds.
9. Do not repeat an idea that clearly failed unless you can say what is different this time. In particular, avoid 3D voxel encoders with softmax layer gates unless they have a wide trunk and plain additive scoring.

## Candidate ideas not yet tested (on top of the best base, ≤1x)
- Footprint max/mean pooling of actor features for each brick type, with a shared small 1x1 MLP instead of one MLP per type, and the trunk narrowed to keep compute ≤1x.
- A global context vector (mean/max pooled) added back into actor features before the placement head.
- A DONE head that sees pooled "missing" and "extra" evidence from learned features.
- Trading actor width against critic width, for example a smaller critic so the actor can be wider.

## Choosing between results
If two trials are within about 0.04 of each other, prefer the simpler and cheaper one: the selected trial is retrained with new seeds, so robustness matters more than one lucky run.

## Revising this program
After each generation of searches, you rewrite this program from the logs of those searches, and it is then used for new searches, including on worlds of other sizes that you will not see while revising it. When revising:
- keep advice that the evidence supports, and say what evidence supports it;
- remove or weaken advice that did not help;
- update the "Known so far" section, and mark single-seed findings as tentative;
- write lessons that should hold on other grid sizes, not facts that only fit one world;
- keep the whole program under 1,000 words, including this section, which you may also improve.
