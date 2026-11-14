"""Writes step2/manifest.json: the SHA-256 of every Step 2 file (protocol, code, tests, diagnostic designs).
run_step2.py refuses to run if any listed file changed. Refuses to overwrite once the full campaign exists.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2/make_manifest2.py
"""
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

if __name__ == "__main__":
    if (HERE / "campaigns" / "step2").exists():
        raise SystemExit("the full campaign exists; refusing to rewrite the manifest")
    files = sorted([HERE / "PROTOCOL_STEP2.md"] + list(HERE.glob("*.py")) + list((HERE / "targets").glob("*")))
    out = {"files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
           "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    (HERE / "manifest.json").write_text(json.dumps(out, indent=1))
    print(len(out["files"]), "files hashed")
