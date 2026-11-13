"""Designs for world W_s (Stage 2c meta-training): dev 512 and check 256, made once before any 2c training.

Uses make_targets.unique unchanged. The two sets are disjoint. W_s has no final split: it is used only to meta-train
programs, and every claim is tested on W. The training stream rejects W_s's dev designs (ws_heldout_hashes.pt).
New seeds, not used by any earlier set (xl 5005/6116, check 7007/8008, W 9009/9119/9229).
Refuses to overwrite existing files.
usage (project root): .venv/bin/python -I arch_rsi/stage2/2c/make_targets_ws.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from arch_rsi.make_targets import unique  # noqa: E402
from brickworld.env import has_hidden_cells  # noqa: E402
from ws_world import TARGETS, WS  # noqa: E402

DEV_SEED, CHECK_SEED = 9339, 9449

if __name__ == "__main__":
    outs = [TARGETS / "ws_dev.pt", TARGETS / "ws_check.pt", TARGETS / "ws_heldout_hashes.pt"]
    if any(p.exists() for p in outs):
        raise SystemExit("W_s designs already exist; refusing to overwrite")
    TARGETS.mkdir(exist_ok=True)
    dev, dh = unique(WS, DEV_SEED, 512, [])
    chk, ch = unique(WS, CHECK_SEED, 256, dh.tolist())
    assert not set(ch.tolist()) & set(dh.tolist())
    torch.save(dev, TARGETS / "ws_dev.pt")
    torch.save(chk, TARGETS / "ws_check.pt")
    torch.save(dh, TARGETS / "ws_heldout_hashes.pt")
    stats = {"ws": {"X": WS.X, "Y": WS.Y, "Z": WS.Z, "k_min": WS.k_min, "k_max": WS.k_max,
                    "dev_n": 512, "check_n": 256, "seeds": {"dev": DEV_SEED, "check": CHECK_SEED},
                    "dev_mean_cells": float(dev.flatten(1).sum(1).float().mean()),
                    "dev_share_hidden": float(has_hidden_cells(dev).float().mean()),
                    "check_share_hidden": float(has_hidden_cells(chk).float().mean())}}
    (TARGETS / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
