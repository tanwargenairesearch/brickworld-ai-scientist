"""World W_s for Stage 2c meta-training (PROTOCOL_2C.md): 8x8x6, 6-16 bricks, dev and check designs only.

Added without editing any frozen file (Stage 2b's frozen files stay byte-identical):
- register() adds "ws" (and W, via w_world) to arch_rsi.worlds.SPECS, in this process only;
- load_trainer() imports the frozen Stage 2b trainer and makes its training stream reject W_s's dev designs;
- load_evaluator() makes the frozen evaluator return W_s's dev designs for "ws". W_s has no final split.

usage (project root):
  .venv/bin/python -I arch_rsi/stage2/2c/ws_world.py train --spec run.json --out DIR     (= stage2/train_2b.py)
  .venv/bin/python -I arch_rsi/stage2/2c/ws_world.py eval --trial DIR --split dev --out OUT.json
  .venv/bin/python -I arch_rsi/stage2/2c/ws_world.py train_cpu --spec run.json --out DIR (frozen CPU trainer; the checks' short run)
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import arch_rsi.worlds as worlds  # noqa: E402
from arch_rsi.stage2 import w_world  # noqa: E402
from brickworld.env import WorldSpec  # noqa: E402

WS = WorldSpec("ws", X=8, Y=8, Z=6, k_min=6, k_max=16)
TARGETS = HERE / "targets"


def ws_hashes() -> torch.Tensor:
    return torch.load(TARGETS / "ws_heldout_hashes.pt")


def register():
    w_world.register()
    worlds.SPECS["ws"] = WS


def load_trainer():
    register()
    spec = importlib.util.spec_from_file_location("train_2b", ROOT / "arch_rsi" / "stage2" / "train_2b.py")
    t2b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t2b)
    orig = t2b.heldout_hashes
    t2b.heldout_hashes = lambda world: ws_hashes() if world == "ws" else orig(world)
    return t2b


def load_cpu_trainer():
    """The frozen CPU trainer (arch_rsi/train_arch.py), as w2b.load_cpu_trainer, with W_s's dev designs held out."""
    register()
    import arch_rsi.train_arch as ta
    orig = ta.heldout_hashes
    ta.heldout_hashes = lambda world: ws_hashes() if world == "ws" else orig(world)
    return ta


def load_evaluator():
    register()
    ea = w_world.load_evaluator()
    orig = ea.designs_for

    def designs_for(world, split, campaign):
        if world != "ws":
            return orig(world, split, campaign)
        if split == "dev":
            return torch.load(TARGETS / "ws_dev.pt")
        raise SystemExit("W_s has no final split")

    ea.designs_for = designs_for
    return ea


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    {"train": load_trainer, "eval": load_evaluator, "train_cpu": load_cpu_trainer}[mode]().main()
