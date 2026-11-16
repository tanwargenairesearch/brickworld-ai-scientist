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
- Generation 0 ran 7 searches in 7 worlds, 84 trials in all. Every trial scored 0.000 dev success on both seeds, and 0.000 training success in the last 10% of training. Nothing beat the starting architecture, so it was selected every time and the confirmed mean was 0.000. With all scores at zero, seed noise could not be measured and no idea could be ranked against another.
- Training success was also 0, so the failure is in learning at all, not in generalising to held-out designs.
- These recipe changes never produced a nonzero score: ppo_epochs 1, 2, 3, 4 and 8 (65k to 426k steps); about 32 to 800 policy updates; lr 3e-4 to 2e-3; ent_coef 0 to 0.04; gamma 0.9 to 0.99; gae_lambda 0.9 to 0.95; clip 0.15; a DONE logit bias started at -2; a separate critic.
- These network families never produced a nonzero score: the starting pooling head, deeper heads, stem skips, per-voxel column encoders, per-layer heads combined with logsumexp or gates, deeper and dilated residual CNNs with global context, cell and column transformers, the legality mask as input, 3x3 per-type heads with no anchor assumption, flip/transpose (dihedral) and layout-permutation ensembles, dense heads, pure MLPs, action-query attention, and footprint raw-column MLPs.
- Heads that assume no action layout (pure MLP, dense head) also scored 0. So a mismatch between action indices and grid cells is not shown to be the cause, though their weak priors may simply learn too slowly.
- No trial failed the checks or the time limit. Train times ranged from 26 to 237 s. Per-layer heads were the slowest (about 2 to 3 times the starting network's time).
- Searches in different worlds repeated the same order of ideas: first the recipe, then per-layer heads, then layout-free heads, then exploration settings. Repeating this sequence gave no new information.
- Never tried: lr above 2e-3, n_envs above 256, gae_lambda below 0.9, and networks much smaller than the starting one.
