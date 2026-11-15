"""Writes step3/manifest.json: the SHA-256 of every Step 3 file (protocol and code), plus the Stage 2c files and the
program it imports (prog2c.py, review_2c.py, g0r.md), before the dry run. run_step3.py stops if any listed file changes.
Refuses to overwrite once the full run's campaign exists.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3/make_manifest3.py
"""
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IMPORTED = [ROOT / "arch_rsi" / "stage2" / "2c" / "prog2c.py", ROOT / "arch_rsi" / "stage2" / "2c" / "review_2c.py",
            ROOT / "arch_rsi" / "stage2" / "programs" / "g0r.md"]

if __name__ == "__main__":
    if (HERE / "campaigns" / "step3").exists():
        raise SystemExit("the full run's campaign exists; refusing to rewrite the manifest")
    files = sorted([HERE / "PROTOCOL_STEP3.md"] + list(HERE.glob("*.py"))) + IMPORTED
    out = {"files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
           "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    (HERE / "manifest.json").write_text(json.dumps(out, indent=1))
    print(len(out["files"]), "files hashed")
