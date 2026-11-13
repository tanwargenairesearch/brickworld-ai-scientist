# Stage 2e: interruption on 10 November 2026

The user asked to stop the run because the GPU was needed for other work. Stage 2e is paused, not finished.
It must not be resumed without the user's go-ahead.

## What happened

| Time (UTC) | Event |
|---|---|
| 21:14:33 | Run started (`run_2e.py --preset 2e`, ceilings 50 hours and $230). |
| 23:29:42 | Last line the runner logged: `pairs/p02/hidden round 6`. `clock.json` holds 8108.6 s up to this line. |
| 23:31:48 | Interrupt (SIGINT) sent to the job's process group, the same as Ctrl-C in its terminal tab. It ended the training processes, but the runner stayed alive and started four new ones. |
| 23:32:14 | Runner ended with SIGTERM, then its remaining training processes. Nothing from the run is left running. |

Two sessions acted on the same request. The parent session reports that it also sent SIGINT to the runner
(PID 49007) and to `caffeinate` (PID 49010) at about 23:31. The four failure records carry the time 23:31:48, the
second of the group signal in the table, which is the one that reached the training processes.

## State at the stop

- Pairs started: p01 to p05. None finished. No pick, no confirmation run and no review exists.
- Rounds done (seen, hidden): p01 5, 5; p02 6, 6; p03 5, 5; p04 5, 5; p05 5, 5.
- Model calls saved: 59, none failed. Notional cost $16.39.
- Training results in the campaign: 302, all with status `ok`.
- Every `state.json` is readable. `PROTOCOL_2E.md` still has the approved hash. Nothing on the held-out split was opened.

## What was moved here, and why

The first interrupt killed four training runs of `pairs/p05/seen` round 6. The runner recorded them as failed
candidates: three `crashed (exit -2)` and one `eval_failed`, each with `KeyboardInterrupt` as the error.

`train_one` reuses any saved `result.json`, whatever its status. On a resume those four records would have counted
as real failures, and each seed would have scored 0. That would have changed pair p05's seen branch because of the
stop, not because of the candidates.

So the four seed folders were moved here, whole and unchanged. `moved.json` lists them with the hash of each
`result.json`. In the campaign those runs now look as they should: not yet run.

- The rule used: move a record only if its error is the operator's interrupt. No score was read to apply it.
- Round 6 of `pairs/p05/seen` was not written to `state.json`, so no state was edited.
- `r6c3/seed0` had finished training. Only its scoring was interrupted. Its checkpoint is here too. A resume trains it again.
- To undo: move each folder back to the same path under `campaigns/stage2e/`.

## What a resume has to disclose

1. This interruption, and that four records were moved by hand. The protocol says the runner needs no hand rule after
   an interruption. That held for model replies, not for training runs killed by an interrupt.
2. The clock is about 152 s short. The runner was alive from 23:29:42 to 23:32:14 after its last clock update.
   `clock.json` was not edited.
3. Other round-6 training runs of p04 and p05 were under way and have no result. They hold only `run.json`, and
   some a copy of `candidate.py`. A resume runs them again from the start.

## How to stop this runner next time

Don't press Ctrl-C and don't signal the process group first. End the runner alone (`kill -TERM <runner pid>`), then
its training processes. That leaves no failure records behind.
