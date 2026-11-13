# Research program (generation 0)

## Goal
Find a policy-network architecture that raises dev success at a fixed training budget. Only the network changes; the training recipe, observation, reward and budget are fixed.

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
