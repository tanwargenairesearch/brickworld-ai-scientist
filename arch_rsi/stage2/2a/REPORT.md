# Stage 2a: calibrating world W. Result

**When and how:** 18:53:21 to 19:07:52 UTC on 9 November 2026 (15 minutes), under `PROTOCOL_2A.md` (SHA-256 `d6bb1c1d…`, recorded in `runs/manifest.json` before the first run).

**Sources:** every number below comes from `runs/decisions.json` and `runs/summary.json`, written by `run_2a.py`. The per-design counts come from each run's `dev.json`.

**What ran:**
- 50 training runs, all finished, with none failed or timed out.
- No Claude calls were made.
- Every score is exact-build success on W's 512 dev designs. W's final designs were never read, and `FINAL_OPENED_W` doesn't exist.

## Outcome: the chosen budget is 131,072 steps

The protocol's ladder:

| Budget | Start, Stage G's recipe (6 seeds) | Start, frozen recipe (2 seeds) | Reference A (2 seeds) | Reference B (2 seeds) | Reading |
|---|---|---|---|---|---|
| 262,144 | 33.4% (4 of 6 learned) | 9.8% | 91.6% | 86.9% | too easy: a reference reached 90% |
| 131,072 (fallback) | 16.3% (2 of 6 learned) | 13.4% | 66.5% | 10.1% | passes |

**How the rules played out:**
- 262,144 was "too easy" only because reference A reached 91.6%. The start was at 33%.
- The protocol then ran its one fallback, 131,072 steps. That passed, so it's the chosen budget.
- 524,288 and above weren't run, because a larger budget is only easier.

**The networks:**
- **Reference A:** no-feedback search 9's pick.
- **Reference B:** no-feedback search 5's pick.

**"Learned"** means the seed builds at least 10% of the 446 dev designs that have hidden cells.

## Noise at 131,072 steps

| Network | Seeds | Mean | SD | Learned the hidden-cell skill |
|---|---|---|---|---|
| Start, Stage G's recipe | 20 (2000–2009, 3000–3009) | 12.3% | 0.072 | 2 of 20 |
| — seeds 2000–2009 | 10 | 13.7% | 0.094 | |
| — seeds 3000–3009 | 10 | 10.9% | 0.042 | |
| Start, frozen recipe | 6 | 9.1% | 0.053 | 0 of 6 |
| Reference A | 6 | 55.2% | 0.140 | 6 of 6 |
| Reference B | 6 | 4.1% | 0.059 | 1 of 6 |

- **No seed-range effect on W.** The two ranges average 13.7% and 10.9%. An exact permutation test gives p = 0.42. The gap seen on Q doesn't repeat here.
- **The start is stuck on a floor.** 7 of the 20 seeds built exactly the same 35 designs, all without hidden cells. Only 2 of 20 learned the hidden-cell skill.
- **Headroom:** the start leaves 88 points, and the best known network (reference A) is at 55%. So a search has room above both.

## What this means for Stage 2b

**1. The start is weak again.**
- At the chosen budget, the start with Stage G's recipe averages 12.3% and learns the skill on 2 of 20 seeds. That's close to Stage 1's start (22.7%, 2 of 12).
- Decision 6 meant every search to start from "a strong, known-good point". On W at this budget, the recipe is known-good but the network isn't.
- So 2b's second primary contrast, g0 − (start with Stage G's recipe), would almost certainly pass and would show little.

**2. Architecture matters a lot on W.**
- Reference A (55.2%) and reference B (4.1%) are both strong on Q (95.4% and 94.1% in the retest).
- On W they're 51 points apart.
- Reference B's FLOP ratio is 1.42 on W (1.33 on Q), so compute matching trains it for 90,112 steps instead of 131,072. It learned the skill on 1 of 6 seeds. Whether the shorter training explains the gap hasn't been tested.

**3. Machine time for 2b: about 18 hours.**
- The projection is 17.9 h for 3,064 runs, below the 48-hour flag.
- It assumes Claude's networks cost like the references (about 21 s per run with 4 at once).
- Networks built on 3D convolutions could cost far more (`../../gpu/profile/REPORT.md`). That's why 2b needs a wall-time limit per run.

## Departures and disclosures

1. **The fallback was used.** That's within the protocol, but it means W at 131,072 steps is the only budget that passed. The protocol made no choice between budgets.
2. **The reference networks were chosen from the retest's results,** and g0 search 1's pick was replaced for speed. Both are disclosed in the protocol, and neither touches a claim.
3. **A floor of 35 designs.** I counted after the run which designs the seeds on that floor built. It's descriptive only.

## Costs

- **Machine time:** 15 minutes start to finish, 4 runs at once.
- **Claude calls:** none, so $0.
