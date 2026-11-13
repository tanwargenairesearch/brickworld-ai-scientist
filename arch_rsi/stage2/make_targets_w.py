"""Designs for world W (Stage 2), made once before any Stage 2 training: dev 512, final 1,024, check 256.

Uses make_targets.unique unchanged. The three sets are disjoint. The training stream rejects dev and final
(their hashes go in w_heldout_hashes.pt), as for xl. Refuses to overwrite existing files.
usage (project root): .venv/bin/python -I arch_rsi/stage2/make_targets_w.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import torch  # noqa: E402

from arch_rsi.make_targets import unique  # noqa: E402
from arch_rsi.stage2.w_world import FINAL, TARGETS, W  # noqa: E402
from brickworld.env import has_hidden_cells  # noqa: E402

DEV_SEED, FINAL_SEED, CHECK_SEED = 9009, 9119, 9229  # new seeds, not used by any earlier set

if __name__ == "__main__":
    outs = [TARGETS / "w_dev.pt", TARGETS / "w_heldout_hashes.pt", TARGETS / "w_check.pt", FINAL]
    if any(p.exists() for p in outs):
        raise SystemExit("W designs already exist; refusing to overwrite")
    TARGETS.mkdir(exist_ok=True)
    FINAL.parent.mkdir(exist_ok=True)
    dev, dh = unique(W, DEV_SEED, 512, [])
    fin, fh = unique(W, FINAL_SEED, 1024, dh.tolist())
    chk, ch = unique(W, CHECK_SEED, 256, dh.tolist() + fh.tolist())
    assert not set(dh.tolist()) & set(fh.tolist())
    assert not set(ch.tolist()) & (set(dh.tolist()) | set(fh.tolist()))
    torch.save(dev, TARGETS / "w_dev.pt")
    torch.save(chk, TARGETS / "w_check.pt")
    torch.save(fin, FINAL)
    torch.save(torch.cat([dh, fh]), TARGETS / "w_heldout_hashes.pt")
    stats = {"w": {"X": W.X, "Y": W.Y, "Z": W.Z, "k_min": W.k_min, "k_max": W.k_max,
                   "dev_n": 512, "final_n": 1024, "check_n": 256,
                   "dev_mean_cells": float(dev.flatten(1).sum(1).float().mean()),
                   "dev_share_hidden": float(has_hidden_cells(dev).float().mean())}}
    (TARGETS / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
