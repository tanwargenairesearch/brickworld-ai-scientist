"""Stage 0 of arch_rsi/PLAN.md: measure noise, floor and ceiling. A pilot, disclosed in the protocol and never used
in the campaign's analysis. It never reads or scores the final split, and never writes FINAL_OPENED.

usage (from the project root; resumable):
  caffeinate -i .venv/bin/python -I arch_rsi/pilot/stage0/stage0.py

Runs, on the main preset's budgets and code:
  1. the starting architecture on P and Q, seeds 0-3 exactly as the campaign's phase 0 would;
  2. six more seeds (200-205) per world, for a 10-seed noise figure;
  3. one real search on Q under program g0, at full budget, as the ceiling probe (about 4 Claude calls).
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arch_rsi.orchestrate_arch import HERE, Campaign  # noqa: E402

EXTRA_SEEDS = [200, 201, 202, 203, 204, 205]


def main():
    cdir = Path(__file__).resolve().parent / "campaign"
    c = Campaign(cdir, "main")
    c.log("stage 0 start")
    c.base = c.baselines()
    worlds = (c.cfg["P"], c.cfg["Q"])
    extra = [(w, s) for w in worlds for s in EXTRA_SEEDS]

    def noise(j):
        w, s = j
        return j, c.train_one(cdir / "noise" / w / f"seed{s}", cdir / "baseline" / w / "candidate.py", w, s)

    def probe():
        g0 = (HERE / "programs" / "g0.md").read_text()
        return c.search(cdir / "Q" / "g0" / "s1", c.cfg["Q"], "g0", g0, 101)

    with ThreadPoolExecutor(len(extra) + 1) as ex:
        noise_jobs = [ex.submit(noise, j) for j in extra]
        probe_job = ex.submit(probe)
        res = [f.result() for f in noise_jobs]
        st = probe_job.result()
    bad = [(j, r["status"]) for j, r in res if r["status"] != "ok"]
    c.log(f"noise seeds done, failures: {bad}")
    c.log(f"ceiling probe done: incumbent {st['incumbent']}")
    c.log("STAGE0_DONE")


if __name__ == "__main__":
    main()
