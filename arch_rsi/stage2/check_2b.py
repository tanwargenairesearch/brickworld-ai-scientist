"""Stage 2b pre-training check of one candidate file (PROTOCOL_2B.md), run in its own process.

usage (project root): .venv/bin/python -I arch_rsi/stage2/check_2b.py --candidate FILE --mode both|fixed|recipe_only --out OUT.json
Stage 1's checks (check.py's functions, unchanged: sandbox rules, parameter and FLOP caps, the learned-policy rule)
on world W, with W's 256 check designs, plus:
- the RECIPE block: bounds, minibatch rules, and at least 4 updates within the fixed-FLOP budget;
- mode "fixed" (network-only arm): no RECIPE block allowed;
- mode "recipe_only" (recipe-only arm): the network must be the starting architecture's, unchanged.
The learned-policy rule's short run uses the frozen CPU trainer and recipe (8,192 steps, seed 0), as in Stage 1.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from arch_rsi.stage2 import w2b, w_world  # noqa: E402

w_world.register()
import arch_rsi.check as C  # noqa: E402
from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import CandidateError, load_candidate, runtime_check, sha, static_check  # noqa: E402
from arch_rsi.worlds import CAPS, SPECS  # noqa: E402
from brickworld.env import BrickWorld  # noqa: E402
from brickworld.model import encode  # noqa: E402
from evaluate import check as np_check  # noqa: E402

WORLD = "w"
BUDGET = 131072
BASELINE = ROOT / "arch_rsi" / "candidates" / "baseline.py"


def check_metrics(pol) -> dict:
    """check.check_metrics, on W's check designs."""
    spec = SPECS[WORLD]
    designs = torch.load(w_world.TARGETS / "w_check.pt")
    pol.eval()
    w = BrickWorld(spec, len(designs), "cpu", targets=designs, auto_reset=False)
    n = len(designs)
    clean = torch.zeros(n)
    dirty = torch.zeros(n, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(spec.max_steps + 1):
            if bool(w.ended.all()):
                break
            logits, _ = pol(encode(w, "voxels"), w.valid_mask())
            active = ~w.ended
            info = w.step(logits.argmax(dim=1))
            dirty |= (info["cells_out"] > 0) & active
            clean += torch.where(~dirty & active, info["cells_in"].float(), torch.zeros(n))
    exact = float(np_check(w.occ.numpy(), designs.numpy())["exact"].mean())
    return {"exact": exact, "clean_fill": float((clean / designs.flatten(1).sum(1).float()).mean())}


def learned_rule(src: str) -> tuple:
    aspec = arch_spec(SPECS[WORLD])
    torch.manual_seed(0)
    u0 = check_metrics(CandidatePolicy(load_candidate(src)(aspec), aspec))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "c.py").write_text(src)
        (d / "run.json").write_text(json.dumps({"candidate": str(d / "c.py"), "world": WORLD, "seed": 0,
                                                "budget_steps": CAPS["short_steps"], "threads": 1}))
        p = subprocess.run([sys.executable, "-I", str(HERE / "w2b.py"), "train_cpu", "--spec", str(d / "run.json"),
                            "--out", str(d / "t")], capture_output=True, text=True, timeout=900)
        logp = d / "t" / "train_log.json"
        log = json.loads(logp.read_text()) if logp.exists() else {"status": "crashed", "error": (p.stderr or "")[-400:]}
        if log["status"] != "ok":
            return u0, None, f"a short training run ({CAPS['short_steps']:,} steps) failed: {log['status']}: {str(log.get('error'))[:400]}"
        ck = torch.load(d / "t" / "ckpt.pt", map_location="cpu")
        torch.manual_seed(0)
        pol = CandidatePolicy(load_candidate(src)(aspec), aspec)
        pol.load_state_dict(ck["state_dict"])
        u1 = check_metrics(pol)
    return u0, u1, None


def check(src: str, mode: str, budget_steps: int = BUDGET) -> dict:
    spec = arch_spec(SPECS[WORLD])
    errs = static_check(src)
    if errs:
        return {"ok": False, "error": "; ".join(errs[:12])}
    try:
        recipe = w2b.parse_recipe(src)
    except (ValueError, SyntaxError) as e:
        return {"ok": False, "error": str(e)}
    if mode == "fixed" and recipe is not None:
        return {"ok": False, "error": "this search's training recipe is fixed: remove the RECIPE block"}
    if mode == "recipe_only" and w2b.strip_recipe(src) != w2b.strip_recipe(BASELINE.read_text()):
        return {"ok": False, "error": "in this search the network must be the starting architecture, unchanged"}
    rerr = w2b.validate(recipe)
    if rerr:
        return {"ok": False, "error": "; ".join(rerr)}
    try:
        rt = runtime_check(src, spec, CAPS["max_params"], max_fb_ms=1e12)
        base = C.train_flops(BASELINE.read_text(), spec)
        fl = C.train_flops(src, spec)
    except CandidateError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}
    ratio = fl / base
    bud = w2b.budget(budget_steps, ratio, recipe)
    out = {"ok": True, "params": rt["params"], "fb_ms": rt["fb_ms"], "flops_per_sample": fl,
           "flops_ratio": round(ratio, 2), "sha": sha(src), "recipe": w2b.full_recipe(recipe),
           "recipe_from_file": recipe, "budget": bud}
    if ratio > CAPS["flops_ratio"]:
        out.update(ok=False, error=f"training compute is {ratio:.1f}x the starting architecture's per sample; "
                                   f"the cap is {CAPS['flops_ratio']}x")
        return out
    if not bud["ok"]:
        out.update(ok=False, error=f"with this network and recipe the fixed budget allows only {bud['n_updates']} "
                                   f"updates of n_envs * rollout_len = {bud['batch']} steps; at least {w2b.MIN_UPDATES} "
                                   f"are required (fewer epochs, a smaller batch or a cheaper network allow more)")
        return out
    try:
        u0, u1, err = learned_rule(src)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"running the network on check designs failed: {type(e).__name__}: {str(e)[:300]}"}
    out["check_untrained"] = {k: round(v, 4) for k, v in u0.items()}
    out["check_short_training"] = None if u1 is None else {k: round(v, 4) for k, v in u1.items()}
    if err:
        out.update(ok=False, error=err)
    elif u0["exact"] > CAPS["max_untrained_exact"] or u0["clean_fill"] > CAPS["max_untrained_clean"]:
        out.update(ok=False, error=f"the untrained network already builds {u0['exact']:.0%} of the check designs exactly and "
                                   f"fills {u0['clean_fill']:.0%} of their target cells before its first wrong cell (limits "
                                   f"{CAPS['max_untrained_exact']:.0%} and {CAPS['max_untrained_clean']:.0%}): the policy must be "
                                   f"learned, not hand-written or hand-initialised")
    elif u1["exact"] > CAPS["max_short_exact"]:
        out.update(ok=False, error=f"after only {CAPS['short_steps']:,} training steps the network builds {u1['exact']:.0%} of the "
                                   f"check designs exactly (limit {CAPS['max_short_exact']:.0%}), far faster than learning from "
                                   f"scratch allows: the policy must be learned, not hand-written")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--mode", default="both", choices=["both", "fixed", "recipe_only"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=BUDGET)
    args = ap.parse_args()
    torch.set_num_threads(1)
    res = check(Path(args.candidate).read_text(), args.mode, args.budget)
    Path(args.out).write_text(json.dumps(res))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
