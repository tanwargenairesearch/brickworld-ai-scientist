"""Step 2's two diagnostic designs (PROTOCOL_STEP2.md, "Readouts"): the plain W_s generator, seed 4646, made once.
They are shown in the readouts' traces and never scored. Refuses to overwrite.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2/make_diag.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as C  # noqa: E402

import torch  # noqa: E402

from arch_rsi.make_targets import unique  # noqa: E402

if __name__ == "__main__":
    WS = C.common.register()
    if C.DIAG.exists():
        raise SystemExit("diag.pt exists; refusing to overwrite")
    C.TARGETS2.mkdir(exist_ok=True)
    held = torch.load(C.common.TARGETS / "h_heldout_hashes.pt").tolist()
    d, h = unique(WS, 4646, 2, held)
    torch.save(d, C.DIAG)
    print("diagnostic designs:", [int(x.sum()) for x in d], "cells")
