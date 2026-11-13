# Physics check: results (exploratory)

Run 8 Nov 2026, 08:14–08:40 GMT, on the same laptop as the main campaign. MuJoCo 4.0.0, CPU only. The plan was written and hashed before any physics ran (`PLAN.md`, `PLAN.sha256`). This was not part of the main campaign's pre-registration and changes none of its results.

## What was done

- **Rollouts.** `rollout.py` replayed trained agents greedily on 512 fresh designs per world. The designs used seed 3003, and none are in dev or final. Every placement was recorded. Exact rates on these designs match the campaign's final scores:
  - starting recipe: 32–33%;
  - Sonnet and Opus k = 32 winners: 99–100%.
- **Static analysis** (`brickphys.classify`). A linear program classifies each exact build as *stable*, *knife-edge* or *unstable* for loose rigid bricks under gravity. "Knife-edge" means feasible only with some corner force at zero, so the centre of mass is exactly over a support edge. Every prefix of the build order is classified too.
- **MuJoCo replays.** Each exact build is replayed brick by brick as loose boxes:
  - 8 mm pitch, 9.6 mm tall, ABS density, friction 0.4;
  - each brick released 0.5 mm above its lattice position, with a random ±0.3 mm sideways offset;
  - 3 offset seeds per build;
  - 8,905 exact builds and 26,715 replays in total.
- **Validation.** All 13 validation scenes in `PLAN.md` passed before the check (`validation.json`):
  - stack drift: 0.001 mm over 2 s;
  - knife-edge cases fall or stand depending on the offset's direction;
  - a welded corner brick stands.

## Results

95% intervals are bootstrap intervals over designs.

| | Exact builds | Stable | Knife-edge | Unstable | Stable at every step | Stood in MuJoCo (planned criterion) |
|---|---|---|---|---|---|---|
| **Small world**: the target designs, as generated | 512 designs | 37.3% | 52.5% | 10.2% | 35.7% | — |
| Starting recipe (5 seeds) | 834 | 100% | 0 | 0 | 100% | 98.4% [97.9, 99.0] |
| Random-search k = 32 winners (5) | 1,818 | 48.6% [43.8, 53.0] | 43.8% | 7.6% | 47.5% [42.8, 52.0] | 54.3% [50.3, 58.3] |
| Claude k = 32 winners (10) | 5,088 | 37.5% [33.5, 41.7] | 52.6% | 9.9% | 36.1% [32.0, 40.4] | 45.3% [41.6, 49.1] |
| **Larger world**: the target designs, as generated | 512 designs | 23.4% | 56.4% | 20.1% | 23.0% | — |
| Starting recipe (2 seeds) | 173 | 100% | 0 | 0 | 100% | 99.0% [98.1, 99.8] |
| Best transfer recipe (2 seeds) | 992 | 24.3% [20.5, 28.0] | 56.5% | 19.3% | 23.9% [20.2, 27.8] | 30.5% [27.2, 34.2] |

### What explains the table

- **The starting recipe never built an overhang exactly.**
  - Small world: 0 of 1,695 attempts on designs with a hidden cell, against 834 of 865 (96.4%) on designs without one.
  - Larger world: 0 of 802, against 173 of 222 (77.9%).
  - Designs without overhangs are 34% of the small designs. That explains the starting recipe's 29–33% score, and why all its exact builds are stable: every brick in them sits fully on something.
- **The search winners learned overhangs.**
  - Exact on overhang designs: Sonnet 99.2%, Opus 98.9% and random-search winners 58.2% (small world); best transfer recipe 96.0% (larger world).
- **Overhangs are what wouldn't stand.**
  - Of the overhang designs as generated, only 18 of 339 (5.3%, small) and 9 of 401 (2.2%, larger) are stable as loose bricks.
  - The agents' exact overhang builds match the targets closely. Not robustly stable: Claude 94.7% [92.2, 97.0], random search 94.7% [92.1, 97.0], best transfer 97.5% [95.8, 99.0].
  - 65% of Claude's exact builds use the same bricks the generator used.
- **Every exact build of a design without overhangs is stable**, in all groups.

## Predictions from the plan

| # | Prediction | Result |
|---|---|---|
| P1 | Every exact build of a design without hidden cells is robustly stable | **Supported.** 100% in every group (3,790 builds). |
| P2 | At least a quarter of exact builds of designs with hidden cells are not robustly stable | **Supported, far beyond the threshold:** 94.7% (Claude), 94.7% (random), 97.5% (larger world, best recipe). |
| P3 | On designs both built exactly, the robustly-stable share differs by under 5 pp between the starting recipe and Claude's winners | **Supported but uninformative.** The difference is 0.0 pp on 169 shared designs, because the starting recipe built no overhang design exactly, so every shared design is one without overhangs. |
| P4 | MuJoCo agrees with the static analysis: ≥ 99% of builds stable at every step stand; ≥ 95% of builds unstable at some step fall | **Partly met.** Unstable at some step: 98.4% (small) and 99.5% (larger) fell, which meets it. Stable at every step: 98.7% (small) and 98.3% (larger) stood, just under 99%, which misses it. Knife-edge builds stood in 17.2% (small) and 12.6% (larger) of replays. |

### Why P4 missed: a flaw in the plan's offsets

Some "stable" builds failed the planned MuJoCo criterion: 155 replays. A sample of 60 was re-run first, then the independent fact-check covered all 155. Every one was the same artefact:
- a brick was pushed 1.0–5.9 mm sideways, with tilt under 0.002° and no drop;
- the cause was the plan's ±0.3 mm offsets, which exceed the 0.2 mm gap between neighbours (footprints are 0.2 mm under the grid, so 0.1 mm clearance per side), so a brick could be released overlapping its neighbour and get shoved;
- with zero offsets, the sampled builds stood.

(`sensitivity.py`'s docstring says "0.1 mm side gap", meaning the clearance per side.)

`sensitivity.py` re-ran every replay with the same seeds and a "toppled" criterion: a brick tilted more than 5° or more than 2 mm below its lattice height. It is post hoc and labelled as such:

- **Determinism.** The re-run reproduced all 26,715 planned outcomes exactly (0 mismatches).
- **Builds stable at every step.** None toppled: 100% of 10,611 replays (small) and 1,230 (larger).
- **Knife-edge builds.** 17.4% (small) and 12.9% (larger) of replays didn't topple. The rest did, whenever an offset nudged a brick toward its overhang.
- **Builds unstable at some step.** 98.3% (small) and 99.5% (larger) of replays toppled.
- **Didn't topple, by group:**

  | Group | Didn't topple |
  |---|---|
  | Starting recipe, small | 100% |
  | Random-search winners | 55.0% |
  | Claude winners | 45.8% |
  | Starting recipe, larger | 100% |
  | Best transfer recipe | 31.3% |

So, apart from the shove artefact, MuJoCo agrees with the static analysis.

## Limitations

- **This checks loose bricks, a pessimistic bound.** Real bricks grip each other through their studs. With that grip, many knife-edge and unstable builds would hold. Whether they hold, and under what load, is what a snap-fit model such as BrickSim's computes; it is not simulated here.
- **The static analysis is exact for rigid bricks with vertical contact forces at patch corners.** It ignores friction-driven support from side contacts, which don't exist here because neighbours have a gap.
- **The MuJoCo replays use a fixed release protocol.** No robot or gripper is simulated, so placement forces are absent.
- **The agents were never trained or selected on physics.** This measures what the exact-match verifier leaves out, not a failure of the agents.

## Files

- `PLAN.md` and `PLAN.sha256` (hashes of the plan and inputs, recorded before running).
- Code: `rollout.py`, `brickphys.py`, `validate.py`, `run_check.py`, `sensitivity.py`, `pick.py`, `render.py`, `figure.py`.
- Data: `rollouts.json`, `records.jsonl`, `results.json`, `sensitivity.json`, `validation.json`, `picks.json`.
- Media (`media/`):
  - `brickworld_replay.mp4`, 46 s, three clips:
    1. the starting recipe against Sonnet's winner on the same design;
    2. that same exact build welded, then loose;
    3. the larger world;
  - per-clip MP4s, GIFs and final frames;
  - `05-physics.png`, the article figure.
- The video's designs were chosen by fixed rules in `pick.py`: the first design by index meeting each rule. Clip 2's rule was tightened after the offset artefact was found, so that it shows a real topple rather than a shove.
