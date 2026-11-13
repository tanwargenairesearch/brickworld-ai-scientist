"""Extra check after Stage G (not in PROTOCOL_G.md; disclosed): is the start's jump under the GPU recipe caused by the
recipe or by the device? Trains the start with Stage G's chosen recipe (256 worlds, lr 0.001, 32 minibatches) on the
CPU, on Q at 262,144 steps, seeds 300-302: the same seeds and budget as G3's GPU runs and Stage 0's CPU sweep.
No Claude calls; dev designs only.
usage (project root): .venv/bin/python -I arch_rsi/gpu/stageG/extra_recipe_on_cpu.py
"""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
A = ROOT / "arch_rsi"
PY = str(ROOT / ".venv" / "bin" / "python")
OUT = HERE / "extra_cpu_recipe"
RECIPE = {"n_envs": 256, "lr": 0.001, "n_minibatches": 32}


def run(seed):
    d = OUT / f"start_b262144_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"candidate": str(A / "candidates" / "baseline.py"), "world": "xl", "seed": seed,
                                            "budget_steps": 262144, "threads": 1, "device": "cpu", "recipe": RECIPE}))
    subprocess.run([PY, "-I", str(A / "gpu" / "train_gpu.py"), "--spec", str(d / "run.json"), "--out", str(d / "run")],
                   capture_output=True, text=True, timeout=3600)
    tl = json.loads((d / "run" / "train_log.json").read_text())
    subprocess.run([PY, "-I", str(A / "evaluate_arch.py"), "--trial", str(d / "run"), "--split", "dev", "--out", str(d / "dev.json")],
                   capture_output=True, text=True, timeout=1800)
    res = {"seed": seed, "status": tl["status"], "device": tl["device"], "recipe": tl["recipe"], "steps": tl["steps"],
           "wall_s": tl["wall_s"], "dev_exact": json.loads((d / "dev.json").read_text())["exact"]}
    (d / "result.json").write_text(json.dumps(res))
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    with ThreadPoolExecutor(3) as ex:
        out = list(ex.map(run, [300, 301, 302]))
    print("EXTRA_DONE", [round(r["dev_exact"], 4) for r in out])
