"""Pre-training check of one candidate file, run in its own process before any training.

usage: python -I arch_rsi/check.py --candidate FILE --world large|xl --out OUT.json
Checks: static rules (sandbox.static_check), build, parameter cap, output shapes and finiteness, and a
deterministic compute cap: training FLOPs per sample (one forward and backward pass, counted by
torch.utils.flop_counter) at most CAPS["flops_ratio"] times the starting architecture's on the same world.
Forward+backward time on a batch of 128 is recorded for information only.
The policy must be learned (thresholds in worlds.CAPS), on the world's 256 check designs: the untrained network
may build at most 2% exactly and fill at most 10% of target cells before its first wrong cell; after a short
training run of 8,192 steps (seed 0, the fixed recipe) it may build at most 25% exactly. The short run also
catches files that crash during training. Check designs are disjoint from dev and final and are never used
to train or select.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import CandidateError, load_candidate, runtime_check, sha, static_check  # noqa: E402
from arch_rsi.worlds import CAPS, SPECS  # noqa: E402
from brickworld.env import BrickWorld  # noqa: E402
from brickworld.model import encode  # noqa: E402
from evaluate import check as np_check  # noqa: E402

HERE = Path(__file__).resolve().parent


def train_flops(src: str, spec) -> int:
    torch.manual_seed(0)
    pol = CandidatePolicy(load_candidate(src)(spec), spec)
    obs = (torch.rand(2, spec.in_channels, spec.Y, spec.X) < 0.3).float()
    mask = torch.ones(2, spec.n_actions, dtype=torch.bool)
    with FlopCounterMode(display=False) as fc:
        logits, value = pol(obs, mask)
        (torch.log_softmax(logits, -1).mean() + value.pow(2).mean()).backward()
    return fc.get_total_flops() // 2


def check_metrics(pol, world: str) -> dict:
    """Greedy run on the world's check designs: share built exactly, and clean fill = mean share of target cells
    filled before the first placement that puts any cell outside the target."""
    spec = SPECS[world]
    designs = torch.load(HERE / "targets" / f"{world}_check.pt")
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


def learned_rule(src: str, world: str) -> tuple:
    """(untrained metrics, metrics after a short training run, error or None)."""
    aspec = arch_spec(SPECS[world])
    torch.manual_seed(0)
    u0 = check_metrics(CandidatePolicy(load_candidate(src)(aspec), aspec), world)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "c.py").write_text(src)
        (d / "run.json").write_text(json.dumps({"candidate": str(d / "c.py"), "world": world, "seed": 0,
                                                "budget_steps": CAPS["short_steps"], "threads": 1}))
        p = subprocess.run([sys.executable, "-I", str(HERE / "train_arch.py"), "--spec", str(d / "run.json"),
                            "--out", str(d / "t")], capture_output=True, text=True, timeout=900)
        logp = d / "t" / "train_log.json"
        log = json.loads(logp.read_text()) if logp.exists() else {"status": "crashed", "error": (p.stderr or "")[-400:]}
        if log["status"] != "ok":
            return u0, None, f"a short training run ({CAPS['short_steps']:,} steps) failed: {log['status']}: {str(log.get('error'))[:400]}"
        ck = torch.load(d / "t" / "ckpt.pt", map_location="cpu")
        torch.manual_seed(0)
        pol = CandidatePolicy(load_candidate(src)(aspec), aspec)
        pol.load_state_dict(ck["state_dict"])
        u1 = check_metrics(pol, world)
    return u0, u1, None


def check(src: str, world: str) -> dict:
    spec = arch_spec(SPECS[world])
    errs = static_check(src)
    if errs:
        return {"ok": False, "error": "; ".join(errs[:12])}
    try:
        rt = runtime_check(src, spec, CAPS["max_params"], max_fb_ms=1e12)
        base = train_flops((HERE / "candidates" / "baseline.py").read_text(), spec)
        fl = train_flops(src, spec)
    except CandidateError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}
    ratio = fl / base
    out = {"ok": True, "params": rt["params"], "fb_ms": rt["fb_ms"], "flops_per_sample": fl,
           "flops_ratio": round(ratio, 2), "sha": sha(src)}
    if ratio > CAPS["flops_ratio"]:
        out.update(ok=False, error=f"training compute is {ratio:.1f}x the starting architecture's per sample; "
                                   f"the cap is {CAPS['flops_ratio']}x")
        return out
    try:
        u0, u1, err = learned_rule(src, world)
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
    ap.add_argument("--world", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    res = check(Path(args.candidate).read_text(), args.world)
    Path(args.out).write_text(json.dumps(res))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
