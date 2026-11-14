"""Writes train_3a.py, eval_3a.py and check_3a.py from Stage H's train_h.py, eval_h.py and check_h.py (frozen) by the
exact replacements below. Each must match the stated number of times. Kept so the derivation can be re-run and diffed.

What changes, and nothing else:
- the docstring;
- where the file finds its folders: it lives one level deeper (step3a/), so it puts stageH/ and step3a/ on sys.path
  and train_3a.py's ROOT is parents[3];
- the rules come from rules3a.py instead of rules.py, and so does the table of design sets per rule (DESIGNS);
- check_3a.py's short training run calls train_3a.py instead of train_h.py.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3a/make_copies_3a.py [--check]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGEH = HERE.parent
PATHS = ("sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
         "sys.path.insert(0, str(Path(__file__).resolve().parent))\n")

DOCS = {
    "train_3a.py": '"""Step 3a trainer (PROTOCOL_STEP3A.md): stageH/train_h.py with the rules taken from rules3a.py, nothing else\n'
                   'changed (make_copies_3a.py, checked by test_identity_3a.py). Every Stage H rule runs rules.py\'s own code.\n\n'
                   'usage (project root): .venv/bin/python -I arch_rsi/stageH/step3a/train_3a.py --spec run.json --out DIR\n'
                   'run.json: {"candidate", "world": "ws", "rule": any name in rules3a.RULES, "seed", "budget_steps", "device", "threads"}.\n"""',
    "eval_3a.py": '"""Step 3a evaluator (PROTOCOL_STEP3A.md): stageH/eval_h.py with the rules and their design sets taken from\n'
                  'rules3a.py, nothing else changed (make_copies_3a.py, checked by test_identity_3a.py).\n\n'
                  'usage (project root): .venv/bin/python -I arch_rsi/stageH/step3a/eval_3a.py --trial DIR --rule R --designs SET --out OUT.json\n"""',
    "check_3a.py": '"""Step 3a legality check (PROTOCOL_STEP3A.md): stageH/check_h.py with the rules and their design sets taken from\n'
                   'rules3a.py and the short run made by train_3a.py, nothing else changed (make_copies_3a.py, checked by\n'
                   'test_identity_3a.py).\n\n'
                   'usage (project root): .venv/bin/python -I arch_rsi/stageH/step3a/check_3a.py --candidate FILE --rule R --out OUT.json\n"""',
}
REPS = {
    "train_3a.py": ("train_h.py", [
        ("ROOT = Path(__file__).resolve().parents[2]\n", "ROOT = Path(__file__).resolve().parents[3]\n", 1),
        ("sys.path.insert(0, str(Path(__file__).resolve().parent))\nimport common  # noqa: E402\nfrom rules import Rule  # noqa: E402\n",
         PATHS + "import common  # noqa: E402\nfrom rules3a import Rule  # noqa: E402\n", 1),
    ]),
    "eval_3a.py": ("eval_h.py", [
        ("sys.path.insert(0, str(Path(__file__).resolve().parent))\nimport common  # noqa: E402\n",
         PATHS + "import common  # noqa: E402\n", 1),
        ("from rules import RULES, Rule, designs_hidden  # noqa: E402\n",
         "from rules3a import DESIGNS, RULES, Rule, designs_hidden  # noqa: E402\n", 1),
        ("common.DESIGNS[args.rule]", "DESIGNS[args.rule]", 2),
    ]),
    "check_3a.py": ("check_h.py", [
        ("HERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(HERE))\n",
         "HERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(HERE.parent))\nsys.path.insert(0, str(HERE))\n", 1),
        ("from rules import RULES, Rule  # noqa: E402\n", "from rules3a import DESIGNS, RULES, Rule  # noqa: E402\n", 1),
        ("common.DESIGNS[RULE['name']]", "DESIGNS[RULE['name']]", 1),
        ('str(HERE / "train_h.py")', 'str(HERE / "train_3a.py")', 1),
    ]),
}


def build(name: str) -> str:
    src_name, reps = REPS[name]
    src = (STAGEH / src_name).read_text()
    if not src.startswith('"""'):
        raise SystemExit(f"{src_name} doesn't start with a docstring")
    body = src[src.index('"""', 3) + 3:]
    for old, new, n in reps:
        if body.count(old) != n:
            raise SystemExit(f"{name}: {old!r}: expected {n} match(es), found {body.count(old)}")
        body = body.replace(old, new)
    return DOCS[name] + body


if __name__ == "__main__":
    if "--check" in sys.argv:
        bad = [n for n in REPS if not (HERE / n).exists() or (HERE / n).read_text() != build(n)]
        print("the copies match their derivations" if not bad else f"differ from their derivations: {bad}")
        sys.exit(1 if bad else 0)
    if any((HERE / n).exists() for n in REPS):
        raise SystemExit("a copy exists; refusing to overwrite")
    for n in REPS:
        (HERE / n).write_text(build(n))
        print("wrote", n)
