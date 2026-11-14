"""Step 2S's four probe designs (PROTOCOL_STEP2S.md, "The sandbox"): the plain W_s generator, seed 4747, made once.
Excluded: Stage H's held-out hashes (dev and check designs) and Step 2's two diagnostic designs. They are run only in
probes and never scored. Refuses to overwrite.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2s/make_probe_designs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2s_common as C  # noqa: E402

import torch  # noqa: E402

from arch_rsi.make_targets import unique  # noqa: E402
from brickworld.env import design_hash, zobrist_codes  # noqa: E402


def make():
    WS = C.common.register()
    held = torch.load(C.common.TARGETS / "h_heldout_hashes.pt").tolist()
    diag = design_hash(torch.load(C.C2.DIAG), zobrist_codes(WS)).tolist()
    return unique(WS, C.PROBE_SEED, C.N_PROBE_DESIGNS, held + diag)


if __name__ == "__main__":
    if C.PROBE_DESIGNS.exists():
        raise SystemExit("probe.pt exists; refusing to overwrite")
    C.TARGETS2S.mkdir(exist_ok=True)
    d, _ = make()
    torch.save(d, C.PROBE_DESIGNS)
    print("probe designs:", [int(x.sum()) for x in d], "cells")
