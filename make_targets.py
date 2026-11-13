"""Generate the held-out design sets once, before any training.

dev   (512 unique designs per scale): what every search sees scores on.
final (1024 unique designs per scale): stored in holdout/, scored once after all search ends.
Both are disjoint from each other; training streams reject any design in either (by hash).
"""
import json
from pathlib import Path

import torch

from brickworld.env import SCALES, design_hash, generate_designs, has_hidden_cells, zobrist_codes

ROOT = Path(__file__).resolve().parent
DEV_SEED, FINAL_SEED = 1001, 2002
N_DEV, N_FINAL = 512, 1024


def unique_designs(spec, seed, n, exclude: torch.Tensor):
    gen = torch.Generator().manual_seed(seed)
    codes = zobrist_codes(spec)
    keep, seen = [], set(exclude.tolist())
    while len(keep) < n:
        d = generate_designs(spec, 4 * n, gen)
        for i, h in enumerate(design_hash(d, codes).tolist()):
            if h not in seen:
                seen.add(h)
                keep.append(d[i])
                if len(keep) == n:
                    break
    out = torch.stack(keep)
    return out, design_hash(out, codes)


def main():
    (ROOT / "targets").mkdir(exist_ok=True)
    (ROOT / "holdout").mkdir(exist_ok=True)
    stats = {}
    for name, spec in SCALES.items():
        dev, dev_h = unique_designs(spec, DEV_SEED, N_DEV, torch.zeros(0, dtype=torch.long))
        fin, fin_h = unique_designs(spec, FINAL_SEED, N_FINAL, dev_h)
        assert not set(dev_h.tolist()) & set(fin_h.tolist())
        torch.save(dev, ROOT / "targets" / f"{name}_dev.pt")
        torch.save(fin, ROOT / "holdout" / f"{name}_final.pt")
        torch.save(torch.cat([dev_h, fin_h]), ROOT / "targets" / f"{name}_heldout_hashes.pt")
        stats[name] = {
            "spec": {"X": spec.X, "Y": spec.Y, "Z": spec.Z, "k_min": spec.k_min, "k_max": spec.k_max,
                     "max_steps": spec.max_steps, "n_actions": spec.n_actions},
            "dev": {"n": len(dev), "mean_cells": float(dev.flatten(1).sum(1).float().mean()),
                    "share_with_hidden_cells": float(has_hidden_cells(dev).float().mean())},
            "final": {"n": len(fin), "mean_cells": float(fin.flatten(1).sum(1).float().mean()),
                      "share_with_hidden_cells": float(has_hidden_cells(fin).float().mean())},
        }
    (ROOT / "targets" / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
