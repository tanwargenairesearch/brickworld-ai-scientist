<!-- header
rounds: 4
candidates_per_round: 3
seeds_per_candidate: 2
-->

# Research program (generation 1)

## Goal
Find a policy network and training recipe that raise dev success at a fixed training budget. Change only what the task says this search may change; the observation, reward and budget are fixed.

## How to run each round
1. Read every result so far, including failures. Compare differences against the task's noise figure. If most trials score near 0 on dev, rank them by training success and by how often seeds succeed, not by dev mean alone.
2. Write down what the evidence suggests before proposing anything.
3. Round 1 only: read the starting code closely. Check the real observation shape, the action layout (brick types × anchor cells + DONE) and the grid size from the code, not from the description. The world may differ from its description. Design every candidate around what the code shows.
4. Propose a mix:
   - about half the candidates refine the current best with one main change each, so each result says something clear;
   - the rest test an idea not yet covered.
   If nothing beats the starting architecture yet, spend most candidates on getting any nonzero success, with bigger changes allowed.
5. Give every candidate a hypothesis: what changes, why it should help here, and what result would refute it.
6. Keep each file simple and correct. Stay well inside the parameter, compute and time caps. Check the training-step count your choices imply: a big network or many epochs can leave too few updates to learn.
7. Do not repeat an idea that clearly failed unless you can say what is different this time.

## Task structure to exploit (untested ideas, ranked by expected value)
To succeed, the agent must find target cells that are still missing and whose support is already built. It must place a brick that fills only target cells, at the right height after the drop, and call DONE exactly when the build matches.
- **Spatial action head.** Score each (brick type, cell) with a fully convolutional head (one output channel per brick type at each cell), plus a DONE score from pooled features. This matches the action structure better than a flat MLP over all actions.
- **Learned comparison.** Feed current and target together so early learned layers can compute their difference. Do not hand-compute features; let the layers learn them.
- **Height as channels.** Treat layers as channels of 2D convolutions over the base, not 3D convolutions (which are slow). Use enough receptive field to cover 2x2 footprints and support underneath. A few 3x3 layers at full resolution suit small grids; keep cost per cell modest so larger grids stay inside the time cap.
- **DONE and value.** A separate small head for DONE and for value. Ending too early or too late both fail exact builds.
- **Normalisation and residual connections** help small-sample PPO stay stable.

## Recipe ideas (the budget trades samples for epochs and size)
- Fewer PPO epochs (1-2) give up to about 2.6x more environment steps. Test this early against the default of 4.
- The +1 exact-build bonus comes at the end of the episode. A higher gamma (0.98-0.995) and gae_lambda (about 0.95) carry it back to earlier placements.
- Greedy evaluation needs a sharp policy. ent_coef in 0.001-0.005 may beat 0.01, but too low risks collapse.
- More parallel envs with shorter rollouts give more varied designs per update. Keep n_minibatches dividing n_envs × rollout_len with at least 64 samples per minibatch.
- Change the recipe and the architecture in separate candidates, unless reaching zero-to-nonzero success needs both.

## Choosing the final pick
The trial with the best 2-seed dev mean is retrained on 4 new seeds. Prefer configurations where both seeds succeed over one lucky seed. In the last round, refine the most reliable candidate rather than gamble on new ideas.

## Revising this program
After each generation of searches, you rewrite this program from the logs of those searches. It is then used for new searches, including on worlds of other sizes that you will not see while revising it. When revising:
- keep advice that the evidence supports, and say what evidence supports it;
- remove or weaken advice that did not help, and mark untested ideas as untested;
- write lessons that should hold on other grid sizes, not facts that fit one world;
- keep the whole program under 1,000 words, including this section.

## How the program is used
The header (the HTML comment starting "<!-- header") is read only by the code that runs each search. It sets rounds, candidates per round and seeds per candidate. Bounds: rounds 2-6; candidates_per_round 2-6; seeds_per_candidate 1-3; their product must be at most 24. After a search, its selected trial is retrained on 4 new seeds, and that is its score. Keep the header's format and change only its numbers.

The rest of the program is given at the start of every round, with the task description, the interface rules, the starting architecture, the results so far and the budget. Searches see scores only.

The interface rules, caps, checks, budget and recipe bounds are fixed. The policy and its features must be learned. Do not hand-write placement rules or scores, hand-compute features before a learned layer, hand-set weights, or find ways to pass the checks without learning. Such candidates are rejected or reported as violations.
