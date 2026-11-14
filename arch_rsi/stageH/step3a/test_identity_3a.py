"""Identity tests for Step 3a (PROTOCOL_STEP3A.md, "Copied trainer, evaluator and check"), on the GPU.
1. The copies and the fixes are exactly their derivations (make_copies_3a.py and make_fixes_3a.py --check).
2. Under the rules plain, shift and mirror, train_3a.py trains to the same final weights and training curve as Stage H's
   train_h.py on the same spec (reference A, F1_shift and F2_mirror; 32,768 steps, seed 0).
3. Each booklet rule's fix, trained in its rule world by train_3a.py, gets the same weights as reference A in the plain
   world (the fix gives the network exactly the plain input).
4. eval_3a.py gives the same per-design results as eval_h.py under plain and shift, on hplain_check, for a checkpoint
   that builds (Stage 2c's calibration run of reference A, b131072 seed 2000, read only). Each new fix carrying those
   weights, scored by eval_3a.py in its rule world, gives the plain world's per-design results (not "vmirror": there
   the mirror image is built, which test_rules_3a.py checks).
5. check_3a.py gives the same result as check_h.py for F1_shift under shift and F2_mirror under mirror, apart from
   the forward/backward timing.
It refuses to run while Step 2's full run is going or a Step 2/2S runner is alive (run_3a.gpu_free).
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3a/test_identity_3a.py   (exits 1 on any failure)
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common  # noqa: E402

import torch  # noqa: E402

import rules3a as R3  # noqa: E402
from make_fixes_3a import FIXES  # noqa: E402
from run_3a import gpu_free  # noqa: E402

PY = [sys.executable, "-I", "-B", "-W", "ignore"]
STAGEH = HERE.parent
TIMING = ("fb_ms",)
FAILED = []


def run(args):
    p = subprocess.run(PY + [str(a) for a in args], capture_output=True, text=True, cwd=common.ROOT, timeout=1800)
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
    print("PASS" if ok else "FAIL", name, detail, flush=True)
    if not ok:
        FAILED.append(name)


def spec(d, cand, rule):
    p = Path(d) / f"run_{rule}_{Path(cand).stem}.json"
    p.write_text(json.dumps({"candidate": str(cand), "world": "ws", "rule": rule, "seed": 0, "budget_steps": 32768,
                             "device": "mps", "threads": 1}))
    return p


def log_of(d):
    return json.loads((Path(d) / "train_log.json").read_text())


if __name__ == "__main__":
    g = gpu_free()
    if not g["ok"]:
        raise SystemExit(f"refusing to load the GPU: {g}")
    for script in ("make_copies_3a.py", "make_fixes_3a.py"):
        p = subprocess.run(PY + [str(HERE / script), "--check"], capture_output=True, text=True, cwd=common.ROOT)
        report(f"{script}_check", p.returncode == 0, p.stdout.strip())
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for rule, cand in (("plain", common.REF_A), ("shift", common.FIXES["shift"]), ("mirror", common.FIXES["mirror"])):
            run([STAGEH / "train_h.py", "--spec", spec(d, cand, rule), "--out", d / f"h_{rule}"])
            run([HERE / "train_3a.py", "--spec", spec(d, cand, rule), "--out", d / f"n_{rule}"])
            lh, ln = log_of(d / f"h_{rule}"), log_of(d / f"n_{rule}")
            report(f"train_3a_matches_train_h_{rule}", wsha(d / f"h_{rule}") == wsha(d / f"n_{rule}")
                   and lh["curve"] == ln["curve"] and lh["status"] == ln["status"] == "ok",
                   f"(weights {wsha(d / f'h_{rule}')[:12]} vs {wsha(d / f'n_{rule}')[:12]}, {ln['n_updates']} updates)")
        for r in R3.BOOKLET:
            run([HERE / "train_3a.py", "--spec", spec(d, FIXES[r], r), "--out", d / f"fix_{r}"])
            report(f"{R3.IDS[r]}_{r}_fix_trains_like_plain", wsha(d / f"fix_{r}") == wsha(d / "n_plain"),
                   f"(weights {wsha(d / f'fix_{r}')[:12]} vs {wsha(d / 'n_plain')[:12]})")

        from arch_rsi.sandbox import sha
        cal = common.S2C / "calib" / "b131072" / "refA_g_s2000" / "run"
        ck = torch.load(cal / "ckpt.pt", map_location="cpu")

        def carrying(name, src):
            t = d / f"cal_{name}"
            t.mkdir()
            (t / "candidate.py").write_text(src)
            torch.save({**ck, "candidate_sha": sha(src)}, t / "ckpt.pt")
            return t

        def ev(script, trial, rule, out):
            run([script, "--trial", trial, "--rule", rule, "--designs", "hplain_check", "--out", d / out])
            return json.loads((d / out).read_text())

        cs = carrying("shift", common.FIXES["shift"].read_text())
        hp, np_ = ev(STAGEH / "eval_h.py", cal, "plain", "hp.json"), ev(HERE / "eval_3a.py", cal, "plain", "np.json")
        hs, ns = ev(STAGEH / "eval_h.py", cs, "shift", "hs.json"), ev(HERE / "eval_3a.py", cs, "shift", "ns.json")
        report("eval_3a_matches_eval_h_plain", hp["per_design"] == np_["per_design"] and hp["n"] == np_["n"] == 256,
               f"({hp['exact']:.4f} vs {np_['exact']:.4f})")
        report("eval_3a_matches_eval_h_shift", hs["per_design"] == ns["per_design"], f"({hs['exact']:.4f} vs {ns['exact']:.4f})")
        for r in R3.NEW:
            if r == "vmirror":
                continue
            e = ev(HERE / "eval_3a.py", carrying(r, FIXES[r].read_text()), r, f"e_{r}.json")
            report(f"{R3.IDS[r]}_{r}_fix_scores_like_plain", e["per_design"] == np_["per_design"],
                   f"({e['exact']:.4f} vs {np_['exact']:.4f})")

        for rule, cand in (("shift", common.FIXES["shift"]), ("mirror", common.FIXES["mirror"])):
            run([STAGEH / "check_h.py", "--candidate", cand, "--rule", rule, "--out", d / f"ch_{rule}.json"])
            run([HERE / "check_3a.py", "--candidate", cand, "--rule", rule, "--out", d / f"cn_{rule}.json"])
            a, b = (json.loads((d / f"{k}_{rule}.json").read_text()) for k in ("ch", "cn"))
            strip = lambda x: {k: v for k, v in x.items() if k not in TIMING}  # noqa: E731
            report(f"check_3a_matches_check_h_{rule}", strip(a) == strip(b) and a["ok"],
                   f"(ok {a['ok']} vs {b['ok']}; untrained {a.get('check_untrained')} vs {b.get('check_untrained')}; "
                   f"short {a.get('check_short_training')} vs {b.get('check_short_training')})")
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
