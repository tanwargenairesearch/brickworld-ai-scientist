"""Check designs for the "the policy must be learned" rule: 256 per world, disjoint from dev and final
(and the training stream rejects them too, because they are added to no stream; they are only used to
run untrained or randomised networks inside check.py, never to train or select)."""
import torch

from arch_rsi.make_targets import HERE, unique
from arch_rsi.train_arch import heldout_hashes
from arch_rsi.worlds import SPECS

SEEDS = {"large": 7007, "xl": 8008}
for w, s in SEEDS.items():
    d, h = unique(SPECS[w], s, 256, heldout_hashes(w).tolist())
    torch.save(d, HERE / "targets" / f"{w}_check.pt")
    print(w, tuple(d.shape))
