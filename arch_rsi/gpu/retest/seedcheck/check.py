"""Outside the retest protocol: rerun the start on one Stage G seed (300) and one retest seed (1001), alone on the GPU,
and compare trained-weight hashes and dev scores with the originals. Asks only whether the two pipelines are identical."""
import hashlib, json, subprocess, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[4]
A = ROOT / "arch_rsi"
PY = str(ROOT / ".venv" / "bin" / "python")
HERE = Path(__file__).resolve().parent
CAND = A / "campaigns/stage1/confirm/764c7fb529694085/candidate.py"
ORIG = {300: A / "gpu/stageG/runs/cal/xl/start_b262144_s300", 1001: A / "gpu/retest/runs/764c7fb529694085/seed1001"}
def wsha(p):
    sd = torch.load(p, map_location="cpu")["state_dict"]; h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].contiguous().numpy().tobytes())
    return h.hexdigest()[:16]
def dev(p):
    return json.loads(Path(p).read_text())["per_design"]["exact"]
out = {}
for seed, od in ORIG.items():
    d = HERE / f"seed{seed}"; d.mkdir(exist_ok=True)
    (d / "run.json").write_text(json.dumps({"candidate": str(CAND), "world": "xl", "seed": seed, "budget_steps": 262144,
                                            "threads": 1, "device": "mps", "recipe": {"n_envs": 256, "lr": 0.001, "n_minibatches": 32}}))
    subprocess.run([PY, "-I", str(A / "gpu/train_gpu.py"), "--spec", str(d / "run.json"), "--out", str(d / "run")], check=True, capture_output=True)
    subprocess.run([PY, "-I", str(A / "evaluate_arch.py"), "--trial", str(d / "run"), "--split", "dev", "--out", str(d / "dev.json")], check=True, capture_output=True)
    new_dev, old_dev = dev(d / "dev.json"), dev(od / "dev.json")
    out[seed] = {"orig_weights": wsha(od / "run" / "ckpt.pt"), "rerun_weights": wsha(d / "run" / "ckpt.pt"),
                 "orig_dev": sum(old_dev) / len(old_dev), "rerun_dev": sum(new_dev) / len(new_dev), "same_per_design": new_dev == old_dev}
    out[seed]["same_weights"] = out[seed]["orig_weights"] == out[seed]["rerun_weights"]
    print(seed, out[seed], flush=True)
(HERE / "result.json").write_text(json.dumps(out, indent=1))
