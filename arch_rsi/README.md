# arch_rsi: architecture search by an AI scientist that rewrites its own research program

Read `protocol.md` first: it is the pre-registration, hashed at launch.

| File | What it does | Frozen at launch |
|---|---|---|
| `worlds.py` | World specs (P = 6×6×4, Q = 8×8×5), the fixed PPO recipe, and the caps | yes |
| `make_targets.py`, `make_check_designs.py` | Q's dev and final designs; check designs for both worlds | outputs hashed |
| `interface.py` | What a candidate file must implement; the frozen wrapper that masks actions | yes |
| `sandbox.py` | Static rules, plus loading a candidate with restricted builtins | yes |
| `check.py` | Pre-training check: rules, caps, FLOPs, learned-policy rule (untrained and short-run) | yes |
| `train_arch.py` | PPO with the fixed recipe and compute-matched steps; trains one candidate and seed | yes |
| `evaluate_arch.py` | Greedy scoring with the main campaign's NumPy checker; final split locked | yes |
| `menu.py` | The main campaign's 144-option architecture menu as candidate files (control arm) | yes |
| `candidates/baseline.py` | Starting architecture as a candidate file (matches the frozen `Policy`) | yes |
| `programs/g0.md` | The seed research program (written before any results) | yes |
| `scientist.py` | Level 1 (write candidates, repair) and Level 2 (rewrite the program) via `claude -p` | yes |
| `orchestrate_arch.py` | Phases 0–4, integrity checks, resume | yes |
| `analyze_arch.py` | Pre-registered analysis | yes |
| `tests/test_arch.py` | Sandbox, equivalence, caps, learned-policy rule, compute matching, splits, final lock | no |
| `pilot/` | Pilot runs (disclosed in the protocol, never used in the campaign) | no |
| `campaigns/` | Dry runs and the main campaign | no |

```
.venv/bin/python arch_rsi/tests/test_arch.py
.venv/bin/python -I arch_rsi/orchestrate_arch.py --campaign arch_rsi/campaigns/main --preset main
.venv/bin/python -I arch_rsi/analyze_arch.py --campaign arch_rsi/campaigns/main
```
