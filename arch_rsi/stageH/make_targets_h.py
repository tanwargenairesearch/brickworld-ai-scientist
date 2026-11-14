"""Stage H1 design sets (PROTOCOL_H1.md), made once before any Stage H run. Refuses to overwrite.

  hplain_dev 512, hplain_check 256: the plain W_s generator (make_targets.unique, unchanged), seeds 4141 and 4242
  htorus_dev 512, htorus_check 256: the wraparound generator (rules.generate_designs_torus), seeds 4343 and 4444
Each set excludes W_s's dev and check designs and the sets made before it. h_heldout_hashes.pt holds W_s's dev
hashes plus all four sets; every Stage H training stream rejects them. No final split.
usage (project root): .venv/bin/python -I arch_rsi/stageH/make_targets_h.py
"""
import json

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from arch_rsi.make_targets import unique
from brickworld.env import design_hash, zobrist_codes
from rules import designs_hidden, generate_designs_torus

SEEDS = {"hplain_dev": 4141, "hplain_check": 4242, "htorus_dev": 4343, "htorus_check": 4444}
SIZES = {"hplain_dev": 512, "hplain_check": 256, "htorus_dev": 512, "htorus_check": 256}


def unique_torus(spec, seed, n, exclude):
    """make_targets.unique with the wraparound generator."""
    gen = torch.Generator().manual_seed(seed)
    codes = zobrist_codes(spec)
    keep, seen = [], set(exclude)
    while len(keep) < n:
        d = generate_designs_torus(spec, 4 * n, gen)
        for i, h in enumerate(design_hash(d, codes).tolist()):
            if h not in seen:
                seen.add(h); keep.append(d[i])
                if len(keep) == n:
                    break
    out = torch.stack(keep)
    return out, design_hash(out, codes)


if __name__ == "__main__":
    WS = common.register()
    outs = [common.TARGETS / f"{k}.pt" for k in SEEDS] + [common.TARGETS / "h_heldout_hashes.pt"]
    if any(p.exists() for p in outs):
        raise SystemExit("Stage H designs already exist; refusing to overwrite")
    common.TARGETS.mkdir(exist_ok=True)
    codes = zobrist_codes(WS)
    ws_dev_h = common.ws_world.ws_hashes()
    ws_check_h = design_hash(torch.load(common.ws_world.TARGETS / "ws_check.pt"), codes)
    excl = ws_dev_h.tolist() + ws_check_h.tolist()
    sets, hashes, stats = {}, {}, {}
    for k in SEEDS:
        make = unique_torus if k.startswith("htorus") else unique
        d, h = make(WS, SEEDS[k], SIZES[k], excl)
        assert not set(h.tolist()) & set(excl)
        excl += h.tolist()
        sets[k], hashes[k] = d, h
        stats[k] = {"n": SIZES[k], "seed": SEEDS[k], "mean_cells": round(float(d.flatten(1).sum(1).float().mean()), 3),
                    "share_hidden": round(float(designs_hidden(d).float().mean()), 4)}
    for k, d in sets.items():
        torch.save(d, common.TARGETS / f"{k}.pt")
    torch.save(torch.cat([ws_dev_h] + [hashes[k] for k in SEEDS]), common.TARGETS / "h_heldout_hashes.pt")
    (common.TARGETS / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
