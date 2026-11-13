"""Stage 2c pre-training check of one candidate file on world W_s (PROTOCOL_2C.md), run in its own process.

Stage 2b's check (stage2/check_2b.py: check(), unchanged and imported, not copied) with three things pointed at W_s:
- the world name ("ws", registered by ws_world.py in this process only);
- the check designs: W_s's 256 check designs (targets/ws_check.pt) instead of W's;
- the learned-policy rule's short run: the frozen CPU trainer through ws_world.py, with W_s's dev designs held out.
check_metrics and learned_rule below are check_2b's, with only those lines changed.

usage (project root): .venv/bin/python -I arch_rsi/stage2/2c/check_2c.py --candidate FILE --mode both|fixed|recipe_only
                      --budget STEPS --out OUT.json
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import ws_world  # noqa: E402

ws_world.register()
_spec = importlib.util.spec_from_file_location("check_2b", ROOT / "arch_rsi" / "stage2" / "check_2b.py")
C2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C2)

WORLD = "ws"


def check_metrics(pol) -> dict:
    """check_2b.check_metrics, on W_s's check designs."""
    spec = C2.SPECS[WORLD]
    designs = torch.load(ws_world.TARGETS / "ws_check.pt")
    pol.eval()
    w = C2.BrickWorld(spec, len(designs), "cpu", targets=designs, auto_reset=False)
    n = len(designs)
    clean = torch.zeros(n)
    dirty = torch.zeros(n, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(spec.max_steps + 1):
            if bool(w.ended.all()):
                break
            logits, _ = pol(C2.encode(w, "voxels"), w.valid_mask())
            active = ~w.ended
            info = w.step(logits.argmax(dim=1))
            dirty |= (info["cells_out"] > 0) & active
            clean += torch.where(~dirty & active, info["cells_in"].float(), torch.zeros(n))
    exact = float(C2.np_check(w.occ.numpy(), designs.numpy())["exact"].mean())
    return {"exact": exact, "clean_fill": float((clean / designs.flatten(1).sum(1).float()).mean())}


def learned_rule(src: str) -> tuple:
    """check_2b.learned_rule on W_s: the short run goes through ws_world.py train_cpu."""
    aspec = C2.arch_spec(C2.SPECS[WORLD])
    torch.manual_seed(0)
    u0 = check_metrics(C2.CandidatePolicy(C2.load_candidate(src)(aspec), aspec))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "c.py").write_text(src)
        (d / "run.json").write_text(json.dumps({"candidate": str(d / "c.py"), "world": WORLD, "seed": 0,
                                                "budget_steps": C2.CAPS["short_steps"], "threads": 1}))
        p = subprocess.run([sys.executable, "-I", str(HERE / "ws_world.py"), "train_cpu", "--spec", str(d / "run.json"),
                            "--out", str(d / "t")], capture_output=True, text=True, timeout=900)
        logp = d / "t" / "train_log.json"
        log = json.loads(logp.read_text()) if logp.exists() else {"status": "crashed", "error": (p.stderr or "")[-400:]}
        if log["status"] != "ok":
            return u0, None, (f"a short training run ({C2.CAPS['short_steps']:,} steps) failed: {log['status']}: "
                              f"{str(log.get('error'))[:400]}")
        ck = torch.load(d / "t" / "ckpt.pt", map_location="cpu")
        torch.manual_seed(0)
        pol = C2.CandidatePolicy(C2.load_candidate(src)(aspec), aspec)
        pol.load_state_dict(ck["state_dict"])
        u1 = check_metrics(pol)
    return u0, u1, None


C2.WORLD = WORLD
C2.check_metrics = check_metrics
C2.learned_rule = learned_rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--mode", default="both", choices=["both", "fixed", "recipe_only"])
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    res = C2.check(Path(args.candidate).read_text(), args.mode, args.budget)
    Path(args.out).write_text(json.dumps(res))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
