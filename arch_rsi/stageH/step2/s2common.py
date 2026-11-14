"""Shared paths and settings for Step 2 (PROTOCOL_STEP2.md)."""
import sys
from pathlib import Path

STEP2 = Path(__file__).resolve().parent
STAGEH = STEP2.parent
for p in (str(STAGEH.parents[1]), str(STAGEH), str(STEP2)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402  (Stage H1's paths and W_s registration, unchanged)

ROOT = common.ROOT
TARGETS2 = STEP2 / "targets"
DIAG = TARGETS2 / "diag.pt"
PROGRAM = ROOT / "arch_rsi" / "stage2" / "programs" / "g0r.md"
WORLDS = ("shift", "mirror")
ARMS = ("told", "readouts", "scores", "hidden")
READOUT_ARMS = ("told", "readouts")
FOUND = 0.383  # half of reference A's 76.6% on the plain world (Stage H1)
