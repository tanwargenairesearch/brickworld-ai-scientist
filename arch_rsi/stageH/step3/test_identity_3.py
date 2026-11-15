"""GPU tests of readouts_3.py (Step 2's readouts with Step 3a's rules). Reads saved runs only; trains nothing.
1. On Step 2's start in the shift world (seed 0), it reproduces Step 2's saved readouts exactly.
2. In the two new held-out worlds, the readouts show each rule's evidence, as Step 2's test did for its worlds:
   H5 (rowshift): the target height map as observed is the design's rolled 4 rows;
   H6 (handshift): in the untrained trace, every placement fills cells 4 columns (mod 8) right of the chosen anchor.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3/test_identity_3.py   (needs the GPU free)
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3common as C  # noqa: E402

FAILED = []
TMP = Path(tempfile.mkdtemp())
PY = sys.executable


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except Exception as e:  # noqa: BLE001
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, type(e).__name__, e)
    return fn


def readouts(trial: Path, rule: str, seed: int) -> dict:
    out = TMP / f"{rule}_{seed}.json"
    p = subprocess.run([PY, "-I", "-B", "-W", "ignore", str(C.STEP3 / "readouts_3.py"), "--trial", str(trial), "--rule", rule,
                        "--seed", str(seed), "--out", str(out)], capture_output=True, text=True, cwd=C.ROOT, timeout=900)
    assert p.returncode == 0, p.stderr[-500:]
    return json.loads(out.read_text())


@test
def shift_start_matches_step2():
    d = C.STEP2 / "campaigns" / "step2" / "start" / "shift" / "seed0"
    saved = json.loads((d / "readouts.json").read_text())
    got = readouts(d / "run", "shift", 0)
    assert got["text_full"] == saved["text_full"] and got["text_short"] == saved["text_short"]


def a_run(cell: str) -> Path:
    tries = sorted((C.STEP3A / "runs" / "b131072" / cell).glob("try*"))
    return tries[-1] / "run"


@test
def rowshift_shows_the_shifted_booklet():
    ro = readouts(a_run("rowshift_A_s4000"), "rowshift", 0)
    for t in ro["traces"] + [ro["untrained"]]:
        assert t["seen"] == t["design"][-4:] + t["design"][:-4], (t["design"], t["seen"])
        assert t["seen"] != t["design"]


@test
def handshift_shows_the_shifted_hands():
    ro = readouts(a_run("handshift_A_s4000"), "handshift", 0)
    n = 0
    for s in ro["untrained"]["steps"]:
        m = re.fullmatch(r"(\dx\d) at row (\d), col (\d)", s["action"])
        if not m or not s["filled"]:
            continue
        dx = int(m.group(1)[2])
        x = int(m.group(3))
        cols = sorted({c[1] for c in s["filled"]})
        assert cols == sorted({(x + 4 + k) % 8 for k in range(dx)}), (s, cols)
        n += 1
    assert n >= 1, "no placement in the untrained trace"


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
