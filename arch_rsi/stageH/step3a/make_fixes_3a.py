"""Writes Step 3a's eight fixes (PROTOCOL_STEP3A.md): reference A with one change each, by exact text replacement.

Booklet rules: undo the rule on the booklet channels before the stem (as Stage H's F1_shift.py does):
  F5_rowshift.py       roll the booklet 4 rows back
  F8_bookmirror.py     flip the booklet left to right
  F9_transpose.py      transpose the booklet
  F10_antitranspose.py transpose the booklet on the other diagonal
  F11_rot90.py         turn the booklet back a quarter turn
  F12_rot270.py        turn the booklet back three quarters
Hands rules:
  F6_handshift.py      roll the placement logits back 4 columns (the DONE logit is unchanged)
  F7_vmirror.py        flip the whole input top to bottom before the stem (as F2_mirror.py flips it left to right)
Each replacement must match exactly once. Refuses to overwrite.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3a/make_fixes_3a.py [--check]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

FIXES = {"rowshift": HERE / "fixes" / "F5_rowshift.py", "handshift": HERE / "fixes" / "F6_handshift.py",
         "vmirror": HERE / "fixes" / "F7_vmirror.py", "bookmirror": HERE / "fixes" / "F8_bookmirror.py",
         "transpose": HERE / "fixes" / "F9_transpose.py", "antitranspose": HERE / "fixes" / "F10_antitranspose.py",
         "rot90": HERE / "fixes" / "F11_rot90.py", "rot270": HERE / "fixes" / "F12_rot270.py"}
UNDO = {"rowshift": "torch.roll(obs[:, b], -4, dims=2)",
        "bookmirror": "obs[:, b].flip(3)",
        "transpose": "obs[:, b].transpose(2, 3)",
        "antitranspose": "obs[:, b].transpose(2, 3).flip(2).flip(3)",
        "rot90": "torch.rot90(obs[:, b], -1, dims=(2, 3))",
        "rot270": "torch.rot90(obs[:, b], 1, dims=(2, 3))"}
STEM = "        s = self.stem(obs)\n"
PLACE = "        place = self.head(f)\n"


def sub(src, old, new, count=1):
    if src.count(old) != count:
        raise SystemExit(f"expected {count} match(es) of {old!r}, found {src.count(old)}")
    return src.replace(old, new)


def booklet_fix(undo: str) -> str:
    s = sub(REF, STEM, "        b = list(range(self.Z, 4 * self.Z)) + [4 * self.Z + 1]\n"
                       "        obs = obs.clone()\n"
                       f"        obs[:, b] = {undo}\n" + STEM)
    return sub(s, "        super().__init__()\n        self.stem", "        super().__init__()\n        self.Z = spec.Z\n        self.stem")


REF = common.REF_A.read_text()


def build() -> dict:
    out = {r: booklet_fix(u) for r, u in UNDO.items()}
    out["handshift"] = sub(REF, PLACE, PLACE + "        place = place.view(place.shape[0], -1, f.shape[2], f.shape[3]).roll(-4, dims=3).flatten(1)\n")
    out["vmirror"] = sub(REF, STEM, "        s = self.stem(obs.flip(2))\n")
    return {FIXES[r]: out[r] for r in FIXES}


if __name__ == "__main__":
    files = build()
    if "--check" in sys.argv:
        bad = [p.name for p, s in files.items() if not p.exists() or p.read_text() != s]
        print("the fixes match their derivation" if not bad else f"differ from their derivation: {bad}")
        sys.exit(1 if bad else 0)
    if any(p.exists() for p in files):
        raise SystemExit("fixes already exist; refusing to overwrite")
    for p, s in files.items():
        p.parent.mkdir(exist_ok=True)
        p.write_text(s)
        print(p.name, len(s))
