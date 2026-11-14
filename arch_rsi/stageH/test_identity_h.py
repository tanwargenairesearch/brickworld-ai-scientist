"""Identity tests for Stage H1 (PROTOCOL_H1.md, "Copied trainer and evaluator"), all training on the GPU:
1. train_h.py is still exactly its derivation from the frozen train_2b.py (make_train_h.py --check);
2. with rule "plain", train_h.py trains reference A to the same final weights and training curve as the frozen
   Stage 2b trainer on the same spec (32,768 steps, seed 0, GPU);
3. reference A's fix for "shift" in the shift world trains to the same weights as reference A in the plain world
   (H1 is the plain world by construction once fixed);
4. eval_h.py (GPU, rule "plain") gives the same per-design results as the frozen evaluator (which is CPU-only) on
   hplain_check, for a checkpoint that builds: Stage 2c's calibration run of reference A (b131072, seed 2000), read
   only; and the shift fix carrying the same weights gives the same results through the shift rule.
   (Test 2's 32,768-step checkpoint builds nothing, so it can't test the evaluator.)
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/test_identity_h.py   (exits 1 on any failure)
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

import torch  # noqa: E402

PY = [sys.executable, "-I", "-B", "-W", "ignore"]
FAILED = []


def run(args):
    p = subprocess.run(PY + [str(a) for a in args], capture_output=True, text=True, cwd=common.ROOT, timeout=900)
    if p.returncode != 0:
        raise SystemExit(f"{args[0]} failed:\n{p.stderr[-1500:]}")
    return p.stdout


def wsha(trial):
    sd = torch.load(Path(trial) / "ckpt.pt", map_location="cpu")["state_dict"]
    m = hashlib.sha256()
    for k in sorted(sd):
        m.update(k.encode())
        m.update(sd[k].numpy().tobytes())
    return m.hexdigest()


def report(name, ok, detail=""):
    print("PASS" if ok else "FAIL", name, detail)
    if not ok:
        FAILED.append(name)


def spec(d, cand, rule):
    p = Path(d) / f"run_{rule}_{Path(cand).stem}.json"
    p.write_text(json.dumps({"candidate": str(cand), "world": "ws", "rule": rule, "seed": 0, "budget_steps": 32768,
                             "device": "mps", "threads": 1}))
    return p


if __name__ == "__main__":
    p = subprocess.run(PY + [str(HERE / "make_train_h.py"), "--check"], capture_output=True, text=True, cwd=common.ROOT)
    report("train_h_is_its_derivation", p.returncode == 0, p.stdout.strip())
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        run([HERE / "frozen_ref.py", "train", "--spec", spec(d, common.REF_A, "plain"), "--out", d / "frozen"])
        run([HERE / "train_h.py", "--spec", spec(d, common.REF_A, "plain"), "--out", d / "plain"])
        run([HERE / "train_h.py", "--spec", spec(d, common.FIXES["shift"], "shift"), "--out", d / "shift"])
        lf, lp = (json.loads((d / k / "train_log.json").read_text()) for k in ("frozen", "plain"))
        report("plain_trains_like_frozen", wsha(d / "frozen") == wsha(d / "plain") and lf["curve"] == lp["curve"]
               and lf["status"] == lp["status"] == "ok",
               f"(weights {wsha(d / 'frozen')[:12]} vs {wsha(d / 'plain')[:12]}, {lp['n_updates']} updates)")
        report("shift_fix_trains_like_plain", wsha(d / "shift") == wsha(d / "plain"),
               f"(weights {wsha(d / 'shift')[:12]} vs {wsha(d / 'plain')[:12]})")
        cal = common.S2C / "calib" / "b131072" / "refA_g_s2000" / "run"
        from arch_rsi.sandbox import sha
        f1 = common.FIXES["shift"].read_text()
        (d / "cal_shift").mkdir()
        (d / "cal_shift" / "candidate.py").write_text(f1)
        ck = torch.load(cal / "ckpt.pt", map_location="cpu")
        torch.save({**ck, "candidate_sha": sha(f1)}, d / "cal_shift" / "ckpt.pt")
        run([HERE / "frozen_ref.py", "eval", "hplain_check", "--trial", cal, "--split", "dev", "--out", d / "ef.json"])
        run([HERE / "eval_h.py", "--trial", cal, "--rule", "plain", "--designs", "hplain_check", "--out", d / "ep.json"])
        run([HERE / "eval_h.py", "--trial", d / "cal_shift", "--rule", "shift", "--designs", "hplain_check", "--out", d / "es.json"])
        ef, ep, es = (json.loads((d / f).read_text()) for f in ("ef.json", "ep.json", "es.json"))
        same = sum(a == b for a, b in zip(ef["per_design"]["exact"], ep["per_design"]["exact"]))
        report("gpu_eval_matches_frozen_eval", ef["per_design"] == ep["per_design"] and ef["n"] == ep["n"] == 256,
               f"(frozen CPU {ef['exact']:.4f}, GPU {ep['exact']:.4f}; {same}/256 designs agree; iou {ef['iou']:.6f} vs {ep['iou']:.6f})")
        report("shift_fix_scores_like_plain", es["per_design"] == ep["per_design"], f"({es['exact']:.4f} vs {ep['exact']:.4f})")
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
