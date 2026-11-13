"""Stage 0, step 6: P's budget probably has the same ceiling problem. Train the starting architecture and the Q probe's
two best designs on P at 65,536, 131,072 and 262,144 steps. No Claude calls. Dev only.
usage (project root): caffeinate -i .venv/bin/python -I arch_rsi/pilot/stage0/psweep.py
"""
import json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; HERE = Path(__file__).resolve().parent; A = ROOT / "arch_rsi"
PY = str(ROOT / ".venv" / "bin" / "python")
T = HERE / "campaign" / "Q" / "g0" / "s1" / "trials"
CANDS = {"start": A / "candidates" / "baseline.py", "r4c3": T / "r4c3" / "candidate.py", "r1c1": T / "r1c1" / "candidate.py"}
def run(job):
    name, budget, seed = job
    d = HERE / "psweep" / f"{name}_{budget}_s{seed}"
    if (d / "result.json").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"candidate": str(CANDS[name]), "world": "large", "seed": seed, "budget_steps": budget, "threads": 1}))
    t0 = time.time()
    p = subprocess.run([PY, "-I", str(A / "train_arch.py"), "--spec", str(d / "run.json"), "--out", str(d / "run")], capture_output=True, text=True, timeout=2400)
    tl = json.loads((d / "run" / "train_log.json").read_text()) if (d / "run" / "train_log.json").exists() else {}
    res = {"name": name, "budget": budget, "seed": seed, "status": tl.get("status", f"crashed {p.returncode}"), "error": (tl.get("error") or p.stderr[-300:]) if tl.get("status") != "ok" else None,
           "steps": tl.get("steps"), "dev_exact": None}
    if res["status"] == "ok":
        subprocess.run([PY, "-I", str(A / "evaluate_arch.py"), "--trial", str(d / "run"), "--split", "dev", "--out", str(d / "dev.json")], capture_output=True, text=True, timeout=1800)
        res["dev_exact"] = json.loads((d / "dev.json").read_text())["exact"]
    res["wall_s"] = round(time.time() - t0, 1)
    (d / "result.json").write_text(json.dumps(res)); print(json.dumps(res), flush=True)
jobs = [("start", b, s) for b in (65536, 131072) for s in range(300, 306)] + [(n, b, s) for n in ("r4c3", "r1c1") for b in (65536, 131072, 262144) for s in range(300, 304)]
print(len(jobs), "runs", flush=True)
with ThreadPoolExecutor(4) as ex:
    list(ex.map(run, jobs))
print("PSWEEP_DONE", flush=True)
