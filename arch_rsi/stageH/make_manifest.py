"""Writes stageH/manifest.json: the SHA-256 of every Stage H1 file (protocol, code, fixes, design sets), before any run.
run_h1.py refuses to start if any listed file changed. Refuses to overwrite once main runs exist.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/make_manifest.py
"""
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

if __name__ == "__main__":
    if (HERE / "runs" / "b131072").exists():
        raise SystemExit("main runs exist; refusing to rewrite the manifest")
    files = sorted([HERE / "PROTOCOL_H1.md"] + list(HERE.glob("*.py")) + list(HERE.glob("fixes/*.py"))
                   + list((HERE / "targets").glob("*")))
    out = {"files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
           "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    (HERE / "manifest.json").write_text(json.dumps(out, indent=1))
    print(len(out["files"]), "files hashed")
