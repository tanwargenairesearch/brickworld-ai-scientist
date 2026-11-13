"""Check that the code on disk reproduces the scored dev run of the learned energy, design by design.

It replays results/learned_dev.json: the same fit, settings, batches and random numbers. For every dev design of
every world it compares the search moves, the brick count and whether the build was exact with the record. It
writes nothing and reports no new score: the only outcome is "reproduced" or a failed assertion.

Why: run_learned.py and metal_search.py were edited after the dev run (the speed benchmark's timing), so this
shows the kernel and the pipeline on disk are the ones that were scored.

usage (project root; needs an Apple GPU, for about a second):
  .venv/bin/python -I -B energy_check/verify_replay.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import run_check as RC  # noqa: E402
import run_learned as RL  # noqa: E402


def main():
    torch.set_num_threads(4)
    rec = json.loads((RL.RESULTS / "learned_dev.json").read_text())
    s = rec["setting"]
    assert s == json.loads(RL.SETTINGS.read_text())["setting"]
    tabs, fit = RL.fitted_tables(s["n_train"])
    for k, v in rec["fit"].items():
        if k != "fit_seconds":
            assert fit[k] == v, f"the fit differs from the record: {k}"
    RL.warm_up()
    rows = {r["world"]: r for r in rec["rows"]}
    for name, (spec, dev, _) in RC.worlds().items():
        r = RL.solve(spec, torch.load(dev), tabs, s, per_design=True)
        want = rows[spec.name]
        for k in ("exact", "stopped", "moves", "bricks"):
            assert r["per_design"][k] == want["per_design"][k], f"{name}: per-design {k} differs from the record"
        for k in ("exact", "stopped", "not_stopped", "stopped_over_step_limit", "illegal_placements", "moves_max"):
            assert r[k] == want[k], f"{name}: {k} differs from the record"
        print(f"{name}: reproduced ({len(want['per_design']['moves'])} designs: moves, bricks, stopped and exact all equal)")


if __name__ == "__main__":
    main()
