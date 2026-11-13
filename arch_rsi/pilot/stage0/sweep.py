"""Stage 0, step 4: Q's budget was too generous (the ceiling probe's best design scored 96% on dev), so retrain the
starting architecture and the probe's two best designs (fixed rule: top two by 2-seed dev mean) at lower budgets.
No Claude calls. Dev only; never touches the final split.
usage (project root): caffeinate -i .venv/bin/python -I arch_rsi/pilot/stage0/sweep.py
"""
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "bin" / "python")
A = ROOT / "arch_rsi"
probe = json.loads((HERE / "campaign" / "Q" / "g0" / "s1" / "state.json").read_text())
top = sorted([t for t in probe["trials"] if t["result"]["status"] == "ok"], key=lambda t: -t["result"]["dev_exact"])[:2]
CANDS = {"start": A / "candidates" / "baseline.py"} | {t["label"]: HERE / "campaign" / "Q" / "g0" / "s1" / "trials" / t["label"] / "candidate.py" for t in top}
BUDGETS = [131072, 262144, 393216]
SEEDS = {"start": [300, 301, 302, 303, 304, 305]} | {k: [300, 301, 302, 303] for k in CANDS if k != "start"}


def run(job):
    name, budget, seed = job
    d = HERE / "sweep" / f"{name}_{budget}_s{seed}"
    if (d / "result.json").exists():
        return json.loads((d / "result.json").read_text())
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"candidate": str(CANDS[name]), "world": "xl", "seed": seed, "budget_steps": budget, "threads": 1}))
    t0 = time.time()
    p = subprocess.run([PY, "-I", str(A / "train_arch.py"), "--spec", str(d / "run.json"), "--out", str(d / "run")], capture_output=True, text=True, timeout=2400)
    tl = json.loads((d / "run" / "train_log.json").read_text()) if (d / "run" / "train_log.json").exists() else {}
    res = {"name": name, "budget": budget, "seed": seed, "status": tl.get("status", f"crashed {p.returncode}"), "steps": tl.get("steps"),
           "flops_ratio": tl.get("flops_ratio"), "dev_exact": None}
    if res["status"] == "ok":
        subprocess.run([PY, "-I", str(A / "evaluate_arch.py"), "--trial", str(d / "run"), "--split", "dev", "--out", str(d / "dev.json")],
                       capture_output=True, text=True, timeout=1800)
        res["dev_exact"] = json.loads((d / "dev.json").read_text())["exact"]
    res["wall_s"] = round(time.time() - t0, 1)
    (d / "result.json").write_text(json.dumps(res))
    print(json.dumps(res), flush=True)
    return res


jobs = [(n, b, s) for b in BUDGETS for n in CANDS for s in SEEDS[n]]
print(f"candidates {list(CANDS)}; {len(jobs)} runs", flush=True)
with ThreadPoolExecutor(10) as ex:
    out = list(ex.map(run, jobs))
print("SWEEP_DONE", flush=True)
