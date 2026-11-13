"""Rule review of Stage 2e's picks (PROTOCOL_2E.md): Stage 2b's reviewer, prompt, schema and validator, imported
unchanged from review_2b.py. One fresh call per distinct pick. The reviewer sees only the rule text and the code,
never the branch, the pair or a score. It is blind, not independent: the same model family wrote the picks.

usage (project root): .venv/bin/python -I arch_rsi/stage2/2e/review_2e.py --campaign arch_rsi/campaigns/stage2e
run_2e.py calls review() once after the first stage (the second-stage rule needs the verdicts) and this command
reviews whatever is still unreviewed at the end. Verdicts are kept: a pick is never reviewed twice.
Writes review.json and one log per pick in review/.
"""
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S2 = HERE.parent
ROOT = S2.parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("review_2b", S2 / "2b" / "review_2b.py")
RV2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RV2)            # frozen, unchanged
S = RV2.S


def letter(i: int) -> str:
    return chr(ord("A") + i % 26) + (str(i // 26) if i >= 26 else "")


def review(cdir: Path, entries: list) -> dict:
    """Review every distinct pick in entries that has no verdict yet. Returns all verdicts, keyed by code hash."""
    f = cdir / "review.json"
    verdicts = json.loads(f.read_text())["verdicts"] if f.exists() else {}
    picks = {e["sha"] for e in entries if e["kind"] == "search" and e["label"] != "start"}
    todo = sorted(picks - set(verdicts))
    random.Random(2026 + len(verdicts)).shuffle(todo)
    out_dir = cdir / "review"
    out_dir.mkdir(exist_ok=True)
    for h in todo:
        code = (cdir / "confirm" / h / "candidate.py").read_text()
        name = letter(len(verdicts))
        val, _ = S.call_with_retries(RV2.SYSTEM, RV2.prompt(name, code), RV2.SCHEMA, RV2.validate, out_dir / f"{h}.json")
        verdicts[h] = val | {"letter": name}
        f.write_text(json.dumps({"model": S.MODEL, "effort": S.EFFORT, "verdicts": verdicts}, indent=1))
        print(h, name, val["verdict"], val["rule"][:80], flush=True)
    if not f.exists():
        f.write_text(json.dumps({"model": S.MODEL, "effort": S.EFFORT, "verdicts": verdicts}, indent=1))
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    cdir = Path(args.campaign).resolve()
    review(cdir, json.loads((cdir / "confirm" / "summary.json").read_text()))


if __name__ == "__main__":
    main()
