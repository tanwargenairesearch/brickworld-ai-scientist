# A toy AI scientist in a brick-assembly world

A small, pre-registered experiment behind the article "Scalable RSI: can an AI scientist discover architectures that hold up at scale?". An agent learns to build target designs in a lattice brick world. An AI scientist (Claude Sonnet 5.5 or Opus 5.5) proposes training recipes and architectures for that agent, compared against seeded random search on the same budget.

Read `protocol.md` first. It fixes the questions, arms, budgets, selection rules and analyses before the main campaign, and lists every pilot that came before it.

## Layout

| Path | What it is | Who may change it |
|---|---|---|
| `brickworld/env.py` | Batched lattice brick world and design generator (PyTorch; CPU, MPS or CUDA) | frozen |
| `brickworld/model.py` | Observation encodings and the conv/MLP policy | frozen |
| `brickworld/space.py` | The 22-knob search space and the baseline recipe | frozen |
| `train.py` | PPO trainer: one recipe JSON in, one checkpoint out | frozen |
| `evaluate.py` | Evaluator with its own NumPy checkers (`exact`, and the planted `topview` flaw) | frozen |
| `make_targets.py`, `targets/`, `holdout/` | Held-out dev and final designs | frozen |
| `proposer.py` | The AI scientist: prompt, schema, API and `claude -p` backends | frozen |
| `orchestrate.py` | Campaign runner: arms, worker pool, fresh seeds, transfer, variety, final opening | frozen |
| `analyze.py` | Pre-registered analyses and figures | hash recorded at launch; any later change is disclosed |
| `tests/test_world.py`, `tests/test_proposer.py` | Verifier and world tests; proposer retry rules | frozen |
| `pilot/` | Pilot runs used to choose the world size and budget (never used in the campaign) | — |
| `campaigns/dry1`, `campaigns/dry3` | Dry runs of every phase at 8,192 steps per trial, with real proposer calls | — |
| `campaigns/dry2_integrity_stop` | A dry run that stopped itself when `protocol.md` was edited mid-run (the integrity check working) | — |
| `campaigns/main/` | Manifest, journal, every trial, every proposer prompt and reply, final scores | written by the runner |

## Reproduce

```bash
uv venv --python 3.14.3 .venv
uv pip install --python .venv/bin/python torch==2.12.1 numpy==2.4.2 scipy==1.18.0 'anthropic>=2.6,<3' pydantic matplotlib==3.10.8
.venv/bin/python make_targets.py          # regenerates identical design sets (seeded)
.venv/bin/python tests/test_world.py      # 10 verifier tests
.venv/bin/python tests/test_proposer.py   # proposer retry rules, offline
.venv/bin/python orchestrate.py --campaign campaigns/main      # about 9–12 h on 10 CPU cores
.venv/bin/python analyze.py --campaign campaigns/main
```

Any single trial can be re-run from its `campaigns/main/trials/<id>/spec.json` with `train.py` and scored with `evaluate.py`. On CPU, training is bit-reproducible for a given seed. The AI scientist's replies are not reproducible, because model sampling is not seedable, so every prompt and reply is saved in `campaigns/main/proposer/`.
