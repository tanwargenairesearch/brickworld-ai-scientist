"""Step 2's readouts (step2/readouts.py, imported unchanged) with Step 3a's rules: the only change is that its name
Rule is rebound to rules3a.Rule, so the translation worlds H5 (rowshift) and H6 (handshift) work. For Stage H's own
rules (shift, mirror) rules3a.Rule runs rules.py's code, so the output is Step 2's (test_identity_3.py checks it).
The command line is readouts.py's own.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3/readouts_3.py --trial DIR --rule R --seed S --out OUT.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3common as C  # noqa: E402,F401  (puts Step 2 and Step 3a on the path)

import readouts as RO  # noqa: E402  (Step 2's, frozen)
import rules3a  # noqa: E402

RO.Rule = rules3a.Rule
compute = RO.compute

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    RO.torch.set_num_threads(1)
    Path(args.out).write_text(json.dumps(compute(Path(args.trial), args.rule, args.seed)))
