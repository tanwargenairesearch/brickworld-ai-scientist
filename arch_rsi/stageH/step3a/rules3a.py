"""Step 3a's hidden rules (PROTOCOL_STEP3A.md): Stage H's rules plus eight more, all on W_s (8x8x6).

Stage H's rules (plain, shift, mirror, torus, late) run stageH/rules.py's own code: Rule here subclasses rules.Rule
(frozen, imported unchanged) and hands every old name to it. The new rules:
  H5  rowshift       translation    booklet shown ROW_SHIFT rows down, wrapping
  H6  handshift      translation    a brick chosen at anchor x lands at anchor (x + HAND_SHIFT) mod X, same row and type
  H7  vmirror        reflection     a brick chosen at anchor y (dy tall) lands at Y - dy - y, same column and type
  H8  bookmirror     reflection     booklet mirrored left to right: column x shown at X-1-x
  H9  transpose      transposition  booklet cell (y, x) shown at (x, y)
  H10 antitranspose  transposition  booklet cell (y, x) shown at (X-1-x, Y-1-y)
  H11 rot90          rotation       booklet cell (y, x) shown at (X-1-x, y)
  H12 rot270         rotation       booklet cell (y, x) shown at (x, Y-1-y)
The booklet is the target, missing, extra and target-height channels (rules.booklet_channels). For the hands rules the
valid mask is given in the agent's own action terms, as for "mirror". No new rule changes the world or its designs.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402  (stageH/common.py)
import torch  # noqa: E402

import rules as R0  # noqa: E402  (stageH/rules.py, frozen)
from brickworld.env import BRICK_TYPES  # noqa: E402
from rules import booklet_channels, designs_hidden  # noqa: E402,F401  (re-exported for eval_3a.py)

ROW_SHIFT = 4
HAND_SHIFT = 4
OLD = R0.RULES
NEW = ("rowshift", "handshift", "vmirror", "bookmirror", "transpose", "antitranspose", "rot90", "rot270")
RULES = OLD + NEW
BOOKLET = ("rowshift", "bookmirror", "transpose", "antitranspose", "rot90", "rot270")
HANDS = ("handshift", "vmirror")
IDS = {"shift": "H1", "mirror": "H2", "torus": "H3", "late": "H4",
       **{r: f"H{i}" for i, r in enumerate(NEW, start=5)}}
FAMILIES = ("translation", "reflection", "transposition", "rotation")
FAMILY = {"shift": "translation", "rowshift": "translation", "handshift": "translation",
          "mirror": "reflection", "vmirror": "reflection", "bookmirror": "reflection",
          "transpose": "transposition", "antitranspose": "transposition",
          "rot90": "rotation", "rot270": "rotation"}
PASSED_BEFORE = ("shift", "mirror")  # H1 and H2 passed Step 1 (stageH/REPORT.md)
DESIGNS = {**common.DESIGNS, **{r: "hplain" for r in NEW}}
NAMES = {"rowshift": "H5, booklet shifted 4 rows", "handshift": "H6, hands shifted 4 columns",
         "vmirror": "H7, hands mirrored top to bottom", "bookmirror": "H8, booklet mirrored left to right",
         "transpose": "H9, booklet transposed", "antitranspose": "H10, booklet transposed on the other diagonal",
         "rot90": "H11, booklet turned a quarter turn", "rot270": "H12, booklet turned three quarters"}


def booklet_view(name: str, b: torch.Tensor) -> torch.Tensor:
    """What the rule shows of the booklet channels b [B, C, Y, X] (dim 2 rows, dim 3 columns)."""
    if name == "rowshift":
        return torch.roll(b, ROW_SHIFT, dims=2)
    if name == "bookmirror":
        return b.flip(3)
    if name == "transpose":
        return b.transpose(2, 3)
    if name == "antitranspose":
        return b.transpose(2, 3).flip(2).flip(3)
    if name == "rot90":
        return torch.rot90(b, 1, dims=(2, 3))
    if name == "rot270":
        return torch.rot90(b, 3, dims=(2, 3))
    raise ValueError(f"{name!r} is not a booklet rule")


def hands_perm(name: str, spec) -> torch.Tensor:
    """[n_actions] agent action -> world action. DONE maps to itself. For "vmirror", anchors that don't fit map to
    themselves (as rules.mirror_perm); for "handshift" every anchor moves, and the mask hides the ones that don't fit."""
    T, Y, X = spec.n_types, spec.Y, spec.X
    p = torch.arange(spec.n_actions)
    for t, (dy, dx) in enumerate(BRICK_TYPES):
        for y in range(Y):
            for x in range(X):
                a = t * Y * X + y * X + x
                if name == "handshift":
                    p[a] = t * Y * X + y * X + (x + HAND_SHIFT) % X
                elif name == "vmirror":
                    if y <= Y - dy and x <= X - dx:
                        p[a] = t * Y * X + (Y - dy - y) * X + x
                else:
                    raise ValueError(f"{name!r} is not a hands rule")
    return p


class Rule(R0.Rule):
    def __init__(self, name: str, spec, device):
        if name not in RULES:
            raise ValueError(f"unknown rule {name!r}")
        if name in OLD:
            super().__init__(name, spec, device)
            return
        if spec.Y != spec.X:
            raise ValueError("Step 3a's rules need a square board")
        self.name, self.spec, self.device = name, spec, torch.device(device)
        self.perm = hands_perm(name, spec).to(self.device) if name in HANDS else None
        self.ch = torch.tensor(booklet_channels(spec.Z), device=self.device) if name in BOOKLET else None
        self.held = None

    def obs(self, o):
        if self.name in OLD:
            return super().obs(o)
        if self.name not in BOOKLET:
            return o
        o = o.clone()
        o[:, self.ch] = booklet_view(self.name, o[:, self.ch])
        return o

    def act(self, a):
        if self.name in OLD:
            return super().act(a)
        return self.perm[a] if self.name in HANDS else a

    def mask(self, m):
        if self.name in OLD:
            return super().mask(m)
        return m[:, self.perm] if self.name in HANDS else m
