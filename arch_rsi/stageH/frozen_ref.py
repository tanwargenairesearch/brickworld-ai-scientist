"""The frozen references for test_identity_h.py, run unchanged: Stage 2b's trainer (stage2/train_2b.py, through
ws_world.load_trainer) and the frozen evaluator (evaluate_arch.py, through ws_world.load_evaluator). Two lookups are
patched the way ws_world.py patches them: the trainer's held-out hashes -> h_heldout_hashes.pt (what train_h.py
rejects), and the evaluator's dev designs -> a Stage H design set. The frozen evaluator runs on the CPU: that is
hard-coded in it, and changing it would mean editing a frozen file.
usage (project root): .venv/bin/python -I arch_rsi/stageH/frozen_ref.py train --spec run.json --out DIR
                      .venv/bin/python -I arch_rsi/stageH/frozen_ref.py eval NAME --trial DIR --split dev --out OUT.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import torch  # noqa: E402

if __name__ == "__main__":
    mode = sys.argv.pop(1)
    if mode == "train":
        t2b = common.ws_world.load_trainer()
        t2b.heldout_hashes = lambda world: torch.load(common.TARGETS / "h_heldout_hashes.pt")
        t2b.main()
    elif mode == "eval":
        name = sys.argv.pop(1)
        ea = common.ws_world.load_evaluator()
        ea.designs_for = lambda world, split, campaign: torch.load(common.TARGETS / f"{name}.pt")
        ea.main()
    else:
        raise SystemExit("mode is train or eval")
