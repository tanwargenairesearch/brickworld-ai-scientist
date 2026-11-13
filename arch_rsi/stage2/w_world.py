"""World W for Stage 2 (10x10x6, 8-20 bricks), added without editing any frozen file.

register() adds W to the frozen modules' lookups, in this process only:
- arch_rsi.worlds.SPECS gains "w" (the trainer and evaluator read this same dict);
- the GPU trainer's heldout_hashes returns W's dev and final hashes for "w", so the training stream rejects them;
- evaluate_arch.designs_for returns W's dev designs, and refuses W's final split unless stage2/FINAL_OPENED_W exists.
Every other world behaves exactly as before.

usage (project root):
  .venv/bin/python -I arch_rsi/stage2/w_world.py train --spec run.json --out DIR     (= gpu/train_gpu.py)
  .venv/bin/python -I arch_rsi/stage2/w_world.py eval --trial DIR --split dev --out OUT.json   (= evaluate_arch.py)
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import arch_rsi.worlds as worlds  # noqa: E402
from brickworld.env import WorldSpec  # noqa: E402

W = WorldSpec("w", X=10, Y=10, Z=6, k_min=8, k_max=20)
TARGETS = HERE / "targets"
FINAL = HERE / "holdout" / "w_final.pt"
MARKER = HERE / "FINAL_OPENED_W"


def w_hashes() -> torch.Tensor:
    return torch.load(TARGETS / "w_heldout_hashes.pt")


def register():
    worlds.SPECS["w"] = W


def load_trainer():
    register()
    spec = importlib.util.spec_from_file_location("train_gpu", ROOT / "arch_rsi" / "gpu" / "train_gpu.py")
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)
    orig = tg.heldout_hashes
    tg.heldout_hashes = lambda world: w_hashes() if world == "w" else orig(world)
    return tg


def load_evaluator():
    register()
    import arch_rsi.evaluate_arch as ea
    orig = ea.designs_for

    def designs_for(world, split, campaign):
        if world != "w":
            return orig(world, split, campaign)
        if split == "dev":
            return torch.load(TARGETS / "w_dev.pt")
        if not MARKER.exists():
            raise SystemExit("final split refused: no FINAL_OPENED_W marker in arch_rsi/stage2")
        return torch.load(FINAL)

    ea.designs_for = designs_for
    return ea


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    mod = {"train": load_trainer, "eval": load_evaluator}[mode]()
    mod.main()
