"""Frozen evaluator. Runs a trained policy greedily on a held-out design set and scores
every build with its own NumPy checker (it does not trust the environment's flags).

usage: python evaluate.py --ckpt CKPT --split dev|final|trainset --out OUT.json
                          [--designs PATH] [--campaign DIR]
Scores reported:
  exact   - every cell of the build equals the target (the real verifier)
  topview - the top-down height of every column equals the target's (the planted flawed
            checker: it cannot see empty cells under overhangs)
  iou     - voxel intersection over union
The final split is refused unless the campaign directory holds a FINAL_OPENED marker.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from brickworld.env import SCALES, BrickWorld
from brickworld.model import Policy, encode

ROOT = Path(__file__).resolve().parent


def np_heights(occ: np.ndarray) -> np.ndarray:
    Z = occ.shape[1]
    top_from_top = occ[:, ::-1].argmax(axis=1)
    return np.where(occ.any(axis=1), Z - top_from_top, 0)


def check(build: np.ndarray, target: np.ndarray) -> dict:
    exact = (build == target).reshape(len(build), -1).all(1)
    topview = (np_heights(build) == np_heights(target)).reshape(len(build), -1).all(1)
    inter = (build & target).reshape(len(build), -1).sum(1)
    union = (build | target).reshape(len(build), -1).sum(1)
    iou = inter / np.maximum(union, 1)
    return {"exact": exact, "topview": topview, "iou": iou}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "final", "trainset"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs")
    ap.add_argument("--campaign")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ck = torch.load(args.ckpt, map_location="cpu")
    spec = SCALES[ck["scale"]]
    if args.split == "dev":
        designs = torch.load(ROOT / "targets" / f"{spec.name}_dev.pt")
    elif args.split == "final":
        if not args.campaign or not (Path(args.campaign) / "FINAL_OPENED").exists():
            raise SystemExit("final split refused: no FINAL_OPENED marker in the campaign directory")
        designs = torch.load(ROOT / "holdout" / f"{spec.name}_final.pt")
    else:
        designs = torch.load(args.designs)
    policy = Policy(spec, ck["net_cfg"])
    policy.load_state_dict(ck["state_dict"])
    policy.eval()
    world = BrickWorld(spec, len(designs), args.device, targets=designs, auto_reset=False)
    obs_mode = ck["net_cfg"]["obs_mode"]
    pressed_done = torch.zeros(len(designs), dtype=torch.bool)
    with torch.no_grad():
        for _ in range(spec.max_steps + 1):
            if bool(world.ended.all()):
                break
            logits, _ = policy(encode(world, obs_mode), world.valid_mask())
            a = logits.argmax(dim=1)
            pressed_done |= ((a == spec.done_action) & ~world.ended).cpu()
            world.step(a)
    build = world.occ.cpu().numpy()
    target = designs.numpy()
    sc = check(build, target)
    res = {
        "split": args.split, "n": int(len(designs)), "scale": spec.name,
        "exact": float(sc["exact"].mean()), "topview": float(sc["topview"].mean()),
        "iou": float(sc["iou"].mean()),
        "topview_minus_exact": float(sc["topview"].mean() - sc["exact"].mean()),
        "pressed_done": float(pressed_done.float().mean()),
        "per_design": {"exact": sc["exact"].astype(int).tolist(),
                       "topview": sc["topview"].astype(int).tolist()},
    }
    Path(args.out).write_text(json.dumps(res))
    print(json.dumps({k: v for k, v in res.items() if k != "per_design"}))


if __name__ == "__main__":
    main()
