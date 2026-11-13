"""Held-out designs for the xl world (dev 512, final 1024), made once before any search.
The large world reuses the main campaign's dev set and held-out hashes."""
import json
from pathlib import Path

import torch

from brickworld.env import design_hash, generate_designs, has_hidden_cells, zobrist_codes
from arch_rsi.worlds import SPECS

HERE = Path(__file__).resolve().parent
DEV_SEED, FINAL_SEED = 5005, 6116  # 6006 was the final seed used by the dry runs; replaced before the main launch


def unique(spec, seed, n, exclude):
    gen = torch.Generator().manual_seed(seed)
    codes = zobrist_codes(spec)
    keep, seen = [], set(exclude)
    while len(keep) < n:
        d = generate_designs(spec, 4 * n, gen)
        for i, h in enumerate(design_hash(d, codes).tolist()):
            if h not in seen:
                seen.add(h); keep.append(d[i])
                if len(keep) == n:
                    break
    out = torch.stack(keep)
    return out, design_hash(out, codes)


if __name__ == "__main__":
    spec = SPECS["xl"]
    dev, dh = unique(spec, DEV_SEED, 512, [])
    fin, fh = unique(spec, FINAL_SEED, 1024, dh.tolist())
    assert not set(dh.tolist()) & set(fh.tolist())
    torch.save(dev, HERE / "targets" / "xl_dev.pt")
    torch.save(fin, HERE / "holdout" / "xl_final.pt")
    torch.save(torch.cat([dh, fh]), HERE / "targets" / "xl_heldout_hashes.pt")
    stats = {"xl": {"dev_n": 512, "final_n": 1024,
                    "dev_mean_cells": float(dev.flatten(1).sum(1).float().mean()),
                    "dev_share_hidden": float(has_hidden_cells(dev).float().mean()),
                    "final_share_hidden": float(has_hidden_cells(fin).float().mean())}}
    (HERE / "targets" / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
