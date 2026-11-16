# Brickworld AI Scientist: can an AI scientist learn to discover?

An AI scientist designs networks and training recipes for an agent that builds 3D brick designs in worlds with hidden rules, then rewrites its own research procedure; every test was fixed before its run. All training runs locally on one Apple Silicon laptop (its GPU through PyTorch MPS from Stage 2 on, its CPU for Stage 1), and the AI scientist is a hosted model.

Article: https://tanwargenairesearch.github.io/brickworld-ai-scientist/

## Three loops

- **Loop 1, train and score:** frozen, hashed code checks each candidate, trains it with PPO on a fixed budget and scores exact builds on 512 dev designs.
- **Loop 2, design the learner:** in each search the AI scientist writes candidate networks and recipes (by default 4 rounds of 3) and reads their results. Its pick is retrained on fresh seeds.
- **Loop 3, rewrite the procedure:** after 7 searches in 7 training worlds, it rewrites its research program, which is kept only if its next 7 searches score higher. Held-out worlds never feed back.

## The steps

| Step | Question | Protocol | Report | Result |
|---|---|---|---|---|
| Stage 1 (CPU) | Does seeing its scores help its search? | `arch_rsi/stage1/protocol_stage1.md` | `arch_rsi/stage1/REPORT.md` | No measurable help: +1.5 points over scores hidden (95% interval −9.1 to +12.0) |
| Stage 2b (GPU) | The same, when it also sets the recipe | `arch_rsi/stage2/2b/PROTOCOL_2B.md` | `arch_rsi/stage2/2b/REPORT.md` | No measurable help: +1.5 points (−10.1 to +12.2), so Stage 2c was not run |
| Stage H1 | Do 4 hidden rules break a strong network, with one allowed change repairing each? | `arch_rsi/stageH/PROTOCOL_H1.md` | `arch_rsi/stageH/REPORT.md` | 2 of 4 pass (shifted booklet, mirrored hands) |
| Step 2 | Without being told, does it find a rule's fix from readouts of its own runs? | `arch_rsi/stageH/step2/PROTOCOL_STEP2.md` | `arch_rsi/stageH/step2/REPORT_STEP2.md` | Readouts: 23 of 24 searches. Scores only and scores hidden: 0 of 24. P1 and P2 pass (Holm p 0.000002) |
| Step 3a | Do 8 more rules, in 4 families, pass? | `arch_rsi/stageH/step3a/PROTOCOL_STEP3A.md` | `arch_rsi/stageH/step3a/REPORT_STEP3A.md` | 8 of 8 pass |
| Step 3 | Does rewriting its procedure on 7 rules help it find fixes from scores alone in a held-out family, against the original program and facts-only and no-logs rewrites? | `arch_rsi/stageH/step3/PROTOCOL_STEP3.md` | `arch_rsi/stageH/step3/REPORT_STEP3.md` | Ending D: P1 +3.1 points (95% interval −0.4 to +8.4, Holm p 0.65); the check reached 8 of 12, with 9 needed. So this test can't say whether a better procedure would help |

Step 3's departures are in `arch_rsi/stageH/step3/RUN_NOTES.md`. Lineage B's final program is its second rewrite (g2), because its third rewrite never fit the 1,000-word limit in nine attempts. A small script, `arch_rsi/stageH/step3/finish_step3_b_final_g2.py`, finished the run.

## What's in the repo

- `arch_rsi/`: the frozen harness, `scientist.py` (every AI-scientist call), and one folder per stage with its protocol, runner, analysis, tests and report: `stage1/`, `stage2/` and `stageH/` (Steps 2, 3a and 3 in `step2/`, `step3a/`, `step3/`). `PLAN.md`, `pilot/stage0/` and `gpu/` hold the plan and calibrations. Stage 2c, Stage 2e (paused) and Step 2S have no result.
- `arch_rsi/campaigns/`: Stage 1's and 2b's records, including reference A (`stage1/confirm/1c9c0e6b521fc0bb/candidate.py`), where every Stage H search starts.
- Step 2: each search in `arch_rsi/stageH/step2/campaigns/step2/<world>/<arm>/s<i>/` keeps `round_<r>.json` (the exact prompt, reply, rejected attempts and notional cost), its repair calls, every network written (`cands/`) and the run scores (`trials/`).
- Step 3: `arch_rsi/stageH/step3/campaigns/step3/programs/<A|B>/` keeps every program (`g0.md`–`g3.md`, `facts.md`, `nologs.md`) and rewrite call (`call_*.json`, with rejected attempts in `call_*.failed.json`). Its searches are in `meta/` and `test/`, laid out like Step 2's.
- `energy_check/` (the AI scientist's second suggestion, prompted by Prefrontal's CRAM paper), `physics/` (an exploratory check), `site/` and `docs/` (the article), and `tools/release/` (which made this history). The root holds the first experiment.

## What the published history leaves out, and why

- **Sealed held-out splits** (`holdout/`, `arch_rsi/holdout/`, `arch_rsi/stage2/holdout/`) and **every file with "final" in its name** (except images and the script above). These sets are generated from seeds in the code, so they're sealed by procedure, not by secrecy: no step opens a held-out set before its single planned evaluation, and the release process never reads them.
- **Model weights and design sets** (`*.pt`, about 3.1 GB). The design sets can be regenerated from their seeds (below); weights come from retraining.
- **The author's working notes and ledger** (`notes/`, `LEDGER.md`).

The history was filtered before publication, so commit hashes differ from the originals and from the IDs quoted in reports. Commit dates were kept. `COMMIT_MAP.md` gives the published ID for every original one. Paths in records read `/path/to/brickworld-ai-scientist`.

## Reproduce

The runs used Python 3.14.3 on macOS; the GPU steps need Apple Silicon (MPS):

```bash
uv venv --python 3.14.3 .venv
uv pip install --python .venv/bin/python torch==2.12.1 numpy==2.4.2 scipy==1.18.0 matplotlib==3.10.8
```

Regenerate the seeded design sets. The runners check their hashes before starting, so a set that doesn't match byte for byte stops the run.

```bash
.venv/bin/python make_targets.py
.venv/bin/python -m arch_rsi.make_targets && .venv/bin/python -m arch_rsi.make_check_designs
.venv/bin/python -I arch_rsi/stage2/make_targets_w.py
.venv/bin/python -I arch_rsi/stage2/2c/make_targets_ws.py
.venv/bin/python -I arch_rsi/stageH/make_targets_h.py
.venv/bin/python -I -B arch_rsi/stageH/step2/make_diag.py
```

Tests (none calls the AI scientist; some `test_identity_*.py` files also need checkpoints that aren't in the history):

```bash
.venv/bin/python arch_rsi/tests/test_arch.py
.venv/bin/python arch_rsi/stage1/tests/test_stage1.py
for t in arch_rsi/stageH/test_analyze_h1.py arch_rsi/stageH/step2/test_{prompts,run,analyze}_step2.py \
         arch_rsi/stageH/step3a/test_analyze_3a.py arch_rsi/stageH/step3/test_{prog3,prompts3,run_step3,analyze_step3}.py; do
  .venv/bin/python -I -B "$t" || break
done
```

Each step, on the GPU after Stage 1, with its protocol's estimate of time and notional AI-scientist cost (actual in brackets):

```bash
# Stage 1, CPU: 6–8 h, $18–25 (6 h 7 min, $18.26)
caffeinate -i .venv/bin/python -I arch_rsi/stage1/run_stage1.py --preset stage1
.venv/bin/python -I arch_rsi/stage1/review_stage1.py --campaign arch_rsi/campaigns/stage1
.venv/bin/python -I arch_rsi/stage1/analyze_stage1.py --campaign arch_rsi/campaigns/stage1
# Stage 2b: about 18 h, about $60 (18 h 37 min, $58.35)
caffeinate -i .venv/bin/python -I arch_rsi/stage2/2b/run_2b.py --preset 2b
.venv/bin/python -I arch_rsi/stage2/2b/review_2b.py --campaign arch_rsi/campaigns/stage2b
.venv/bin/python -I arch_rsi/stage2/2b/analyze_2b.py --campaign arch_rsi/campaigns/stage2b
# Stage H1, no AI scientist: allow 1 h (36.5 min)
nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/run_h1.py
# Step 2: 13–16 h, about $141 (14.0 h, $143.58)
nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/step2/run_step2.py --preset step2
# Step 3a, no AI scientist: 1–1.5 h (46 min)
nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/step3a/run_3a.py
# Step 3: 25–30 h, $250–350 (29.7 h elapsed with stops, $424.08)
nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/step3/run_step3.py --preset step3
```

From Stage H1 on, each runner also runs its analysis (and, in Steps 2 and 3, the reviews). Stage 1, Stage 2b and Steps 2 and 3 have `_dry` presets. Runners resume from disk, reusing finished runs and calls. To rerun a step from scratch, move its campaign folder aside first (later steps read reference A from Stage 1's).

**AI-scientist settings (Stages 1 to 3):** model `claude-opus-5-5` (Anthropic), effort `high`, through the signed-in `claude -p` command-line tool with no tools, MCP servers or settings, in an empty directory, replying in JSON to a schema. No temperature or output-token limit is set. Calls time out at 2,700 s and get 3 attempts, with the validation error fed back. Rate-limit, server and network errors are retried for about 6 hours without using attempts. Files that fail the checks get up to 2 repair calls per round.

**The AI scientist's replies are not reproducible.** The hosted model can't be seeded, so a rerun writes different networks and programs, and the analyses use the saved replies. To re-check a result, rerun its analysis on the saved campaign, for example `analyze_step2.py --campaign arch_rsi/stageH/step2/campaigns/step2`.

`arch_rsi/gpu/GPU_PLAN.md` didn't assume that training on MPS is deterministic; it scheduled a check. In that check (Stage G), repeating a seed gave bit-identical trained weights for both seeds tried, and Step 3a's rerun of Stage H1's plain cell matched design by design. Both were on one machine and one PyTorch version, so compare results over seeds.

## The first experiment

The repo root holds the first experiment (CPU, 7–8 November 2026, `protocol.md`). AI-scientist arms and seeded random search tuned a 22-knob PPO recipe, testing transfer across scale, design variety and a planted checker flaw. The AI scientist's picks reached 96–97% final exact at 32 trials against random search's 60.2%, but hit the ceiling by 8 trials, so the scaling questions stayed open (`campaigns/main/REPORT.md`).

```bash
.venv/bin/python tests/test_world.py      # verifier tests (after make_targets.py above)
.venv/bin/python tests/test_proposer.py   # proposer retry rules, offline
.venv/bin/python orchestrate.py --campaign campaigns/rerun --backend cli   # recorded run: 5 h 6 min, 10 CPU workers
.venv/bin/python analyze.py --campaign campaigns/rerun
```

Its per-trial final-set scores went with the "final" files, so only a fresh run reproduces them. In its pilots, CPU training was bit-identical across thread counts.
