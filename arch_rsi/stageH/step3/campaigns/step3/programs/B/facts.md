<!-- header
rounds: 4
candidates_per_round: 3
seeds_per_candidate: 2
-->

# Research program (generation 0)

## Goal
Find a policy network and training recipe that raise dev success at a fixed training budget. Change only what the task says this search may change (the network, the training recipe, or both); the observation, reward and budget are fixed.

## How to run each round
1. Read every result so far, including failures. A difference of a few points between two trials can be seed noise; compare against the noise figure given in the task.
2. Write down what the evidence suggests before proposing anything.
3. Propose a mix:
   - about half the candidates refine the current best architecture with one main change each, so that each result says something clear;
   - the rest test a new idea that the results so far have not covered.
4. Give every candidate a hypothesis: what changes, why it should help on this task, and what result would refute it.
5. Think about the structure of the task when choosing ideas, not only generic network tricks: what the agent must perceive and decide to place the right brick in the right order.
6. Keep each file simple and correct. A candidate that fails the checks wastes a trial. Stay well inside the parameter and compute caps, because a slower network risks the time limit.
7. Do not repeat an idea that clearly failed unless you can say what is different this time.

## Revising this program
After each generation of searches, this program is rewritten by you from the logs of those searches, and then used for new searches, including on worlds of other sizes that you will not see while revising it. When revising:
- keep advice that the evidence supports, and say what evidence supports it;
- remove or weaken advice that did not help;
- write lessons that should hold on other grid sizes, not facts that only fit one world;
- keep the whole program under 1,000 words, including this section, which you may also improve.

## Facts
- From 7 searches and 84 trials, one network family scored above 0. In one world it reached a mean dev success of 0.22 on both recipes tried, with both seeds above 0 (0.15 and 0.29; 0.22 and 0.22). Its design: the legal-action mask, reshaped to [types, Y, X], goes in as input channels under four layouts (as given, transposed, y-flipped, x-flipped), plus coordinate channels. A 1x1 column stem and a residual 3x3 trunk follow. Four learned per-type 1x1 output maps are re-laid in those same four layouts and summed into the placement logits. The starting net on the same recipe scored 0, so the network made the difference.
- Every other trial scored exactly 0 on dev, in every world. These all failed:
  - recipe-only changes: 1 to 4 PPO epochs, 131k to 426k steps, entropy 0.001 to 0.04, gamma 0.9 to 0.99, minibatches of 64 to 256 samples;
  - other heads: per-layer logsumexp or gated heads, footprint-conv and footprint-unfold heads, centred 3x3 heads;
  - other trunks: deeper or dilated CNNs and cell transformers;
  - DONE heads that see the placement logits;
  - slow layout-free heads: zero-initialised dense remaps, flat MLPs and per-action attention or queries.
- Hedges that cover only one deviation also failed in other worlds: a gated transpose, a dihedral plus type-order sum, and the mask fed in under one layout only. Different worlds seem to differ from their descriptions in different ways. Uniform zeros come from the world, not from seed noise.
- Whenever late training success was 0, dev success was also 0. When dev succeeded, late training success was close to it (0.23 to 0.24 training against 0.22 dev). Treat a nonzero training success as the first real signal.
- Time was never the limit: the longest run took 191 s of the 900 s allowed, and most took 40 to 120 s. Networks that hold layers as separate tokens ran slowest.
- Over the 7 searches, the mean confirmed dev success was 0.024.
