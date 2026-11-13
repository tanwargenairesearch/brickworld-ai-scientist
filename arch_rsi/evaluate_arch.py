"""Frozen evaluator for candidate architectures. Runs the trained policy greedily on a held-out
design set and scores every build with evaluate.py's own NumPy checker.

usage: python -I -m arch_rsi.evaluate_arch --trial DIR --split dev|final --out OUT.json [--campaign DIR]
The trial directory must hold candidate.py and ckpt.pt with matching hashes. The xl final split is
refused unless the campaign directory holds a FINAL_OPENED marker; the large world has no final split
in this campaign (it is the meta-training problem).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import load_candidate, sha  # noqa: E402
from arch_rsi.worlds import SPECS  # noqa: E402
from brickworld.env import BrickWorld  # noqa: E402
from brickworld.model import encode  # noqa: E402
from evaluate import check  # noqa: E402

HERE = Path(__file__).resolve().parent


def designs_for(world: str, split: str, campaign) -> torch.Tensor:
    if split == "dev":
        return torch.load((HERE / "targets" / "xl_dev.pt") if world == "xl" else (ROOT / "targets" / f"{world}_dev.pt"))
    if world != "xl":
        raise SystemExit("final split refused: only the xl world has a final split in this campaign")
    if not campaign or not (Path(campaign) / "FINAL_OPENED").exists():
        raise SystemExit("final split refused: no FINAL_OPENED marker in the campaign directory")
    return torch.load(HERE / "holdout" / "xl_final.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "final"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--campaign")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    trial = Path(args.trial)
    src = (trial / "candidate.py").read_text()
    ck = torch.load(trial / "ckpt.pt", map_location="cpu")
    if ck["candidate_sha"] != sha(src):
        raise SystemExit("candidate.py does not match the checkpoint")
    spec = SPECS[ck["world"]]
    aspec = arch_spec(spec)
    designs = designs_for(spec.name, args.split, args.campaign)
    torch.manual_seed(0)
    policy = CandidatePolicy(load_candidate(src)(aspec), aspec)
    policy.load_state_dict(ck["state_dict"])
    policy.eval()
    world = BrickWorld(spec, len(designs), "cpu", targets=designs, auto_reset=False)
    pressed_done = torch.zeros(len(designs), dtype=torch.bool)
    with torch.no_grad():
        for _ in range(spec.max_steps + 1):
            if bool(world.ended.all()):
                break
            logits, _ = policy(encode(world, "voxels"), world.valid_mask())
            a = logits.argmax(dim=1)
            pressed_done |= (a == spec.done_action) & ~world.ended
            world.step(a)
    sc = check(world.occ.numpy(), designs.numpy())
    res = {"split": args.split, "n": int(len(designs)), "world": spec.name,
           "exact": float(sc["exact"].mean()), "iou": float(sc["iou"].mean()),
           "pressed_done": float(pressed_done.float().mean()),
           "per_design": {"exact": sc["exact"].astype(int).tolist()}}
    Path(args.out).write_text(json.dumps(res))
    print(json.dumps({k: v for k, v in res.items() if k != "per_design"}))


if __name__ == "__main__":
    main()
