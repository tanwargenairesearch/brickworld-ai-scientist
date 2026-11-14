"""Writes step2s/manifest.json: the SHA-256 of every Step 2S file (protocol, code, tests, probe designs).
run_step2s.py refuses to run if any listed file changed or the manifest is missing. Refuses to overwrite once the full
campaign exists.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2s/make_manifest2s.py
"""
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

if __name__ == "__main__":
    if (HERE / "campaigns" / "step2s").exists():
        raise SystemExit("the full campaign exists; refusing to rewrite the manifest")
    files = sorted([HERE / "PROTOCOL_STEP2S.md"] + list(HERE.glob("*.py")) + list((HERE / "targets").glob("*")))
    out = {"files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
           "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    (HERE / "manifest.json").write_text(json.dumps(out, indent=1))
    print(len(out["files"]), "files hashed")
