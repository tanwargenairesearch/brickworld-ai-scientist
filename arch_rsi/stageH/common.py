"""Shared paths and the W_s registration for Stage H1 (PROTOCOL_H1.md)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S2C = ROOT / "arch_rsi" / "stage2" / "2c"
for p in (str(ROOT), str(HERE), str(S2C)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ws_world  # noqa: E402  (arch_rsi/stage2/2c/ws_world.py, unchanged)

TARGETS = HERE / "targets"
REF_A = ROOT / "arch_rsi" / "campaigns" / "stage1" / "confirm" / "1c9c0e6b521fc0bb" / "candidate.py"
FIXES = {"shift": HERE / "fixes" / "F1_shift.py", "mirror": HERE / "fixes" / "F2_mirror.py",
         "torus": HERE / "fixes" / "F3_torus.py", "late": HERE / "fixes" / "F4_late.py"}
DESIGNS = {"plain": "hplain", "shift": "hplain", "mirror": "hplain", "late": "hplain", "torus": "htorus"}
WORLD = "ws"


def register():
    ws_world.register()
    return ws_world.WS
