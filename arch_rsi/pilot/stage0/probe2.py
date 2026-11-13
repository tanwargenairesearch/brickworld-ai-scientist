"""Stage 0, step 5: a second ceiling probe, one g0 search on Q at a quarter of the budget (131,072 steps), because at
the full budget the probe's best design scored 96%. Same code and rules as the main preset; only the step budgets and
time limits differ (P also at 131,072 for phase 0's noise note). About 4 Claude calls. Dev only.
usage (project root): caffeinate -i .venv/bin/python -I arch_rsi/pilot/stage0/probe2.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arch_rsi import orchestrate_arch as O  # noqa: E402

O.PRESETS["stage0_q131k"] = dict(O.PRESETS["main"], steps={"large": 131072, "xl": 131072}, timeout={"large": 400, "xl": 600})
cdir = Path(__file__).resolve().parent / "probe2"
c = O.Campaign(cdir, "stage0_q131k")
c.log("probe 2 start")
c.base = c.baselines()
st = c.search(cdir / "Q" / "g0" / "s1", "xl", "g0", (O.HERE / "programs" / "g0.md").read_text(), 101)
c.log(f"probe 2 done: incumbent {st['incumbent']}")
c.log("PROBE2_DONE")
