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
- One search (an 8x8 world, 6 layers, 4 rounds x 3 candidates x 2 seeds): all 12 trials and the starting architecture scored 0.000 dev success on every seed, and 0.000 training success (last 10%). The failure was in learning itself, not in generalisation. The starting architecture was selected and confirmed at 0.000.
- Covered without any non-zero result: the starting conv + footprint-pool head with 1 or 2 PPO epochs (about 210k-426k steps); entropy weight 0.003-0.03; gamma 0.9-0.97 with lambda 0.85-0.95; lr 1.5e-3-2e-3; n_envs 64-128; a per-layer landing-gated head; a cell transformer; a dilated CNN with a global summary; flat linear and MLP heads with no assumed action layout; a plain MLP; mask and coordinate input channels; a 5x5 per-type output head.
- None of these can be ranked against the others: equal zeros give no evidence for or against any of them.
- Not tried: ppo_epochs above 2, n_envs above 256, lr below 1e-3, ent_coef 0, gae_lambda below 0.85, clip_eps other than 0.2.
- Run times were 45-245 s against the 900 s cap. The per-layer head (batch x Z) was the slowest. Networks of 0.3-0.7M parameters with flat heads ran fine. No trial was invalid or failed.
