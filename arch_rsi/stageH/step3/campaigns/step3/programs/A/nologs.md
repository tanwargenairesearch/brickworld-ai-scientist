<!-- header
rounds: 4
candidates_per_round: 3
seeds_per_candidate: 2
-->

# Research program (generation 1)

## Goal
Find a policy network and training recipe that raise dev success at a fixed training budget. Change only what the task says this search may change (the network, the training recipe, or both). The observation, reward and budget are fixed.

## Status of this advice
No search logs have informed this program yet. The task-structure ideas below are hypotheses to test, not established facts. Trust your own results in this search over them.

## How to run each round
1. Read every result so far, including failures. Record each failure's cause (check failure, timeout, or no learning). A difference of a few points between two trials can be seed noise, so compare it against the noise figure given in the task.
2. Write down what the evidence suggests before proposing anything.
3. Propose a mix:
   - about half the candidates refine the current best with one main change each, so each result says something clear;
   - the rest test an idea the results so far have not covered.
4. Give every candidate a hypothesis: what changes, why it should help on this task, and what result would refute it.
5. Keep each file simple and correct. A candidate that fails the checks wastes a trial. Stay well inside the parameter and compute caps, because a slower network risks the 900-second limit.
6. Do not repeat an idea that clearly failed unless you can say what is different this time.

## When the best dev score is still 0
- Dev success needs an exact build, so it can stay at 0 while learning is happening. Rank candidates by mean training success and the reported training signals, and by dev success only once it is above 0.
- If training success is also around 0, the problem is learning itself, not generalisation. Try recipe changes first: more samples (fewer ppo_epochs, such as 1–2), a sensible lr, a gamma that suits episodes of about 6–33 steps, and moderate entropy.
- If training success is above 0 but dev success is 0, the problem is generalisation or greedy evaluation. Try stronger spatial inductive bias, or less overfitting to sampling noise.

## Suggested first round
Split round 1 so effects can be attributed:
- (a) a recipe-only change on the starting network;
- (b) an architecture change suited to the task, with the default recipe;
- (c) a combination of (a) and (b).
Later rounds return to the refine/explore split above.

## Task structure (hypotheses)
- **What the agent must perceive.** The target's 3D occupancy, the current build, and the difference between them (cells still missing, and whether a cell is supported). The network should be able to compare target and build cell by cell. Putting them in matching channels of a 2D convolution, with height layers as channels, is a cheap way to do this.
- **Order matters.** Bricks drop onto the highest filled cell, so the agent must fill lower layers first, and it must place bricks that create overhangs only after their supports. Features that let the network compare heights across neighbouring cells help: 2D convolutions with a receptive field of at least the largest brick (2×2), plus some global context such as pooling or a small MLP over pooled features.
- **Action head.** Actions are (brick type, anchor cell) plus DONE. Per-cell logits from a 1×1 convolution, with one channel per brick type and a separate DONE logit from pooled features, keep the head small and translation-equivariant. That beats a large flat linear layer.
- **Speed.** Avoid 3D convolutions. The task warns they run several times slower than their FLOPs suggest. Prefer 2D convolutions with height as channels.
- **Compute trade-off.** A network costing r times the starting FLOPs gets about 1/r as many samples. A small network with many samples may beat a large one with few. Check the steps and time reported for each trial.
- **Learned only.** All features and decisions must be learned. Do not hand-compute features from the observation, hand-write placement rules, or set weights by hand.

## Robustness to other worlds
- The world may differ from its description. Read observation and action shapes from the interface at run time. Never hard-code the grid size, the layer count or the number of brick types.
- Size the network's channels and depth so the compute stays well under the caps on larger grids. Leave headroom on time.
- If results contradict the task description (for example, unexpected training success or reward behaviour), say so. Then design for the behaviour the results show, not the one described.

## Choosing the final candidate
Selection uses the 2-seed mean dev success, followed by a 4-seed retrain. Prefer changes that help both seeds over ones where a single seed spikes. Before the last round, spend at least one candidate on a seed-robust refinement of the current best rather than on a long-shot idea.

## Revising this program
After each generation of searches, you rewrite this program from the logs of those searches. It is then used for new searches, including on worlds of other sizes that you will not see while revising it. When revising:
- keep advice that the evidence supports, and say what evidence supports it;
- remove or weaken advice that did not help, and turn confirmed hypotheses into stated lessons;
- write lessons that should hold on other grid sizes, not facts that only fit one world;
- keep the whole program under 1,000 words, including this section.
