"""Shared paths and settings for Step 3 (PROTOCOL_STEP3.md). Step 2's and Step 3a's settings are imported, not copied."""
import sys
from pathlib import Path

STEP3 = Path(__file__).resolve().parent
STAGEH = STEP3.parent
STEP2 = STAGEH / "step2"
STEP2S = STAGEH / "step2s"
STEP3A = STAGEH / "step3a"
for _p in (str(STEP2), str(STEP3A), str(STEP3)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import s2common as C2  # noqa: E402  (Step 2's paths and settings; it puts Stage H and the project root on the path)
import rules3a as R3  # noqa: E402  (Step 3a's rules, frozen)

ROOT, common = C2.ROOT, C2.common
PROGRAM_G0R = C2.PROGRAM                     # arch_rsi/stage2/programs/g0r.md, unchanged
FOUND = C2.FOUND                             # 38.3%, Step 2's threshold
CHECK_FOUND, CHECK_N = 9, 12                 # the readouts check: at least 9 of 12 searches score >= 38.3%

# meta-training worlds (reflection, transposition, rotation) and the held-out family (translation)
TRAIN_WORLDS = ("mirror", "vmirror", "bookmirror", "transpose", "antitranspose", "rot90", "rot270")
TEST_WORLDS = ("shift", "rowshift", "handshift")
assert {R3.FAMILY[w] for w in TEST_WORLDS} == {"translation"}
assert not {R3.FAMILY[w] for w in TRAIN_WORLDS} & {"translation"}

LINEAGES = ("A", "B")
TEST_ARMS = ("g0", "g1", "final", "facts", "nologs", "check")
PROGRAM_ARMS = ("g1", "final", "facts", "nologs")      # one program per lineage
START_SEEDS = [0, 1, 2]                      # covers every header's trial seeds (seeds_per_candidate 1-3)
CONFIRM_SEEDS = [5001, 5002, 5003, 5004]
MAX_WORDS = 1000

STEP2_RESULTS = STEP2 / "campaigns" / "step2" / "results.json"
STEP3A_RESULTS = STEP3A / "runs" / "results.json"
