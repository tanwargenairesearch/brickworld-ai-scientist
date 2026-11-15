<!-- header
rounds: 4
candidates_per_round: 3
seeds_per_candidate: 2
-->

# Research program (generation 1)

## Goal
Find a policy network and training recipe that raise dev success at a fixed training budget. Change only what the task says this search may change (the network, the training recipe, or both). The observation, reward and budget are fixed.

## Status of this advice
No search logs have yet been reviewed. Advice marked (prior) comes from the task's structure, not from evidence. Test it, and do not trust it over your own results.

## How to run each round
1. Read every result so far, including failures. Seed noise is large on this task: the task says runs can succeed or fail by seed luck alone. Treat a single difference of a few points, or one good seed out of two, as weak evidence.
2. Write down what the evidence suggests before proposing anything. Also note anything that contradicts the task description, such as success rates, step counts or times that do not fit the stated world. The world may differ from its description. If it seems to, adapt candidates to what you observe.
3. If every trial so far scores near 0 dev success, do not rank candidates by dev success alone. Use training success, steps and time to see which candidates learn at all. While dev success is still zero, prioritise changes that make learning happen: recipe, sample count, and output parameterisation. Small architectural refinements can wait.
4. Propose a mix:
   - about half the candidates refine the current best with one main change each, so that each result says something clear;
   - the rest test an idea the results have not yet covered.
5. Give every candidate a hypothesis: what changes, why it should help on this task, and what result would refute it.
6. Keep each file simple and correct. A candidate that fails the checks wastes a trial. Stay well inside the parameter and compute caps, because a slower network risks the 900-second limit. Avoid 3D convolutions unless you have a strong reason: the task says they run several times slower than 2D ones of equal FLOPs.
7. Do not repeat an idea that clearly failed unless you can say what is different this time.

## Ideas grounded in the task structure (prior)
- **What must be perceived:** the difference between the target and the current build, per cell and per height. Treating the layers as channels of a 2D grid lets 2D convolutions compare target and build in each column cheaply. All features must be learned from the raw observation. Do not hand-compute features.
- **What must be decided:** a brick type and an anchor cell, or DONE. Per-cell action logits from a fully convolutional head match this structure better than a dense layer over a flattened grid, and they scale across grid sizes. The DONE action needs global information, for example pooled features.
- **Order matters:** bricks drop, so lower layers must come first and overhangs need their support placed first. The network needs a receptive field large enough to see a 2x2 footprint and its surroundings. A few 3x3 layers, or a dilated layer, may be enough. The value head benefits from global pooling.
- **Budget arithmetic:** the number of training steps scales as 1/((ppo_epochs + 1/3) × r). Fewer PPO epochs, for example 1 to 2, and a network no more expensive than the starting one buy many more samples. When nothing learns yet, more samples may matter more than capacity. Test this against the default 4 epochs.
- **Recipe:** reasonable ranges to try are an entropy coefficient around 0.01 to 0.03, gamma 0.95 to 0.99 (episodes have up to 33 steps and an exact-build bonus at the end), and an lr near the default. Change one recipe knob at a time when refining.
- **Grid size:** prefer designs whose parameter count and behaviour do not depend on the grid size, such as convolutions and per-cell heads. They transfer and stay inside the caps on larger worlds.

## Reporting
For each trial, record which prior above it tested and whether the result supported it, so that the next revision of this program has evidence to work with.

## Revising this program
After each generation of searches, you rewrite this program from the logs of those searches, and it is then used for new searches, including on worlds of other sizes that you will not see while revising it. When revising:
- keep advice that the evidence supports, and say what evidence supports it;
- remove or weaken advice that did not help, and turn confirmed priors into stated lessons;
- write lessons that should hold on other grid sizes, not facts that only fit one world;
- keep the whole program under 1,000 words, including this section, which you may also improve.

## How the program is used
The header is read only by the search code: rounds 2-6, candidates_per_round 2-6, seeds_per_candidate 1-3, with rounds × candidates × seeds at most 24. Keep its format and change only its numbers. The selected trial is retrained on 4 new seeds, and that is its score. The policy and its features must be learned. Never hand-write placement rules or scores, hand-compute features before a learned layer, or hand-set weights so the network is competent before training.
