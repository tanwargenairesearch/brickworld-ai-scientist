# Physics check: plan, written before any physics was run

Written 8 Nov 2026, 08:16 GMT. This is an exploratory follow-up to the main campaign; it was not part of its pre-registration (`../protocol.md`). Nothing below changes any result of the main campaign.

## Question

The brick world uses lattice rules with no physics: a placed brick drops onto the highest filled cell under its footprint and stays there, even when it rests on one cell at a corner. The exact-match verifier rewards a build when its cells equal the target's. **Would the builds the agents got exactly right still stand if the bricks were real, loose rigid bodies?** This checks how much the verifier leaves out, not how good the agents are.

## What has been done before this plan

- `rollout.py` replayed trained agents greedily on 512 fresh designs per scale (seed 3003, none in dev or final) and recorded every placement. The agents: the 5 baseline seeds, the 15 k = 32 incumbents of random, Sonnet and Opus (original-seed checkpoints) at small scale, and at large scale the baseline and the best transfer recipe (`random-r4-t22`), 2 seeds each. Exact rates on these fresh designs matched the main campaign's final scores within a few points (for example baseline 32–33%, Sonnet 99–100%). That is the only result seen so far.
- No stability computation or simulation has been run.

## Two models of a brick

1. **Welded ("snap-like").** Each brick is fixed to whatever it lands on. By the lattice rule every brick rests on something, so every build stands. Used only for the illustration video, and labelled so. Real bricks hold by stud clutch, which is not simulated here.
2. **Loose rigid bricks.** Used for the check. Boxes on an 8 mm grid pitch, 9.6 mm tall, footprints 0.2 mm smaller than the grid so neighbours don't touch sideways; ABS density 1,050 kg/m³; friction 0.4.

## Measures

**A. Static analysis (primary, exact).** For a set of bricks, a linear program asks whether non-negative contact forces at the corners of every contact patch can hold every brick in force and torque balance under gravity. Each build is classified:
- *stable*: feasible when every corner force must be at least ε (ε = 1% of the lightest brick's weight per corner), so the load lies strictly inside the support;
- *marginal*: feasible only with some corner forces at zero (a knife-edge: the centre of mass lies on the edge of its support);
- *unstable*: infeasible even with zero allowed.
"Robustly stable" means stable. Marginal builds fall under any perturbation toward the overhang.

**B. Construction order (secondary).** The same classification for every prefix of the build order; a build is "buildable in this order" if every intermediate state is stable.

**C. MuJoCo replay (secondary, dynamic).** Each exact build is replayed brick by brick in MuJoCo with loose bricks. Each brick starts 0.5 mm above its lattice rest position with a uniform random lateral offset of up to ±0.3 mm in x and y, at rest, then settles for 0.25 s before the next; after the last brick, 1 s more. A replay "stands" if every brick ends within 1.0 mm of its lattice position and tilted less than 5°. Three offset seeds per build. If the projected run time exceeds 1 hour, a fixed-seed random subset of 2,000 exact builds per scale is simulated instead, and that is disclosed.

**Units and intervals.** Shares are over exact builds; 95% intervals are bootstrap intervals over designs (resampling designs, keeping each design's builds from all agents in a group together). Group comparisons are paired on the designs both groups built exactly.

## Predictions (made before running)

- **P1.** Every exact build whose design has no hidden cells is robustly stable. (Each brick then sits fully on cells below; this is a check of the solver, not a finding.)
- **P2.** At least a quarter of exact builds of designs with hidden cells are not robustly stable (marginal or unstable).
- **P3.** On designs both built exactly, the share that is robustly stable differs by less than 5 pp between the baseline and the Claude k = 32 incumbents, because the verifier ignores stability.
- **P4.** MuJoCo agrees with the static analysis: at least 99% of replays of builds stable in every prefix stand, and at least 95% of replays of builds unstable in some prefix fall. Marginal builds split, depending on the random offsets.

## Validation before the check

- A fully supported three-layer stack must stay within 0.1 mm for 2 s in MuJoCo (solver drift).
- A 2×2 brick on two diagonal cells is stable; on two adjacent cells, marginal; on one corner cell, marginal; a brick placed on the overhanging half of a marginal brick, unstable (static analysis).
- A marginal case in MuJoCo falls when its offset points toward the overhang and stands when it points away.
If any validation fails, the check stops and the failure is reported.

## Reporting

Every number with its interval, including results against these predictions. Outputs: `REPORT.md`, `results.json`, a replay video, and a figure. Labelled exploratory everywhere it appears.
