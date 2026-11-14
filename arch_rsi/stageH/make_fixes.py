"""Writes Stage H1's four fixes (PROTOCOL_H1.md): reference A with one change each, by exact text replacement.

  F1_shift.py   roll the booklet channels 4 columns back before the stem
  F2_mirror.py  flip the whole input left to right before the stem
  F3_torus.py   circular padding in the two 3x3 convolutions and in the footprint pooling
  F4_late.py    a RECIPE block with gamma 0.995 and gae_lambda 0.99 (the rest is Stage G's recipe)
Each replacement must match exactly once. Refuses to overwrite.
usage (project root): .venv/bin/python -I arch_rsi/stageH/make_fixes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


def sub(src, old, new, count=1):
    if src.count(old) != count:
        raise SystemExit(f"expected {count} match(es) of {old!r}, found {src.count(old)}")
    return src.replace(old, new)


REF = common.REF_A.read_text()
STEM = "        s = self.stem(obs)\n"
F1 = sub(REF, STEM,

         "        b = list(range(self.Z, 4 * self.Z)) + [4 * self.Z + 1]\n"
         "        obs = obs.clone()\n"
         "        obs[:, b] = torch.roll(obs[:, b], -4, dims=3)\n" + STEM)
F1 = sub(F1, "        super().__init__()\n        self.stem", "        super().__init__()\n        self.Z = spec.Z\n        self.stem")
F2 = sub(REF, STEM, "        s = self.stem(obs.flip(3))\n")
F3 = sub(REF, "nn.Conv2d(COL, WIDTH, 3, padding=1)", 'nn.Conv2d(COL, WIDTH, 3, padding=1, padding_mode="circular")')
F3 = sub(F3, "nn.Conv2d(WIDTH, WIDTH, 3, padding=1)", 'nn.Conv2d(WIDTH, WIDTH, 3, padding=1, padding_mode="circular")')
F3 = sub(F3, "q = F.pad(p, (0, dx - 1, 0, dy - 1))", 'q = F.pad(p, (0, dx - 1, 0, dy - 1), mode="circular")')
F4 = REF.rstrip() + "\n\n\nRECIPE = {'gamma': 0.995, 'gae_lambda': 0.99}\n"

if __name__ == "__main__":
    files = dict(zip(common.FIXES.values(), (F1, F2, F3, F4)))
    if any(p.exists() for p in files):
        raise SystemExit("fixes already exist; refusing to overwrite")
    for p, s in files.items():
        p.parent.mkdir(exist_ok=True)
        p.write_text(s)
        print(p.name, len(s))
