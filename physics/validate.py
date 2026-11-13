"""Validation scenes from PLAN.md. Run before the check: .venv/bin/python validate.py"""
import json
import sys

import numpy as np

from brickphys import classify, replay

SPEC = {"X": 5, "Y": 5, "Z": 3}


def B(y, x, z, dy, dx):
    return {"t": 0, "y": y, "x": x, "z": z, "dy": dy, "dx": dx}


results, ok = {}, True


def check(name, got, want):
    global ok
    good = got == want
    ok &= good
    results[name] = {"got": got, "want": want, "pass": good}
    print(f"{'PASS' if good else 'FAIL'}  {name}: got {got}, want {want}")


# static analysis
check("full support", classify([B(0, 0, 0, 2, 2), B(0, 0, 1, 2, 2)]), "stable")
check("2x2 on two diagonal cells", classify([B(0, 0, 0, 1, 1), B(1, 1, 0, 1, 1), B(0, 0, 1, 2, 2)]), "stable")
check("2x2 on two adjacent cells", classify([B(0, 0, 0, 1, 1), B(0, 1, 0, 1, 1), B(0, 0, 1, 2, 2)]), "marginal")
check("2x2 on one corner cell", classify([B(0, 0, 0, 1, 1), B(0, 0, 1, 2, 2)]), "marginal")
check("2x2 on three cells", classify([B(0, 0, 0, 1, 1), B(0, 1, 0, 1, 1), B(1, 0, 0, 1, 1), B(0, 0, 1, 2, 2)]), "stable")
check("load on the overhanging half", classify([B(0, 0, 0, 1, 1), B(0, 0, 1, 1, 2), B(0, 1, 2, 1, 1)]), "unstable")
check("load on the supported half", classify([B(0, 0, 0, 1, 1), B(0, 0, 1, 1, 2), B(0, 0, 2, 1, 1)]), "stable")

# MuJoCo
stack = [B(1, 1, 0, 2, 2), B(1, 1, 1, 2, 2), B(1, 1, 2, 2, 2)]
_, errs, _ = replay(stack, SPEC, offsets=np.zeros((3, 2)), lift=0.0, final_settle=2.0)
drift = max(e for e, _ in errs)
check("stack drift <= 0.1 mm over 2 s", bool(drift <= 1e-4), True)
results["stack drift <= 0.1 mm over 2 s"]["max_error_mm"] = drift * 1000

cant = [B(1, 1, 0, 1, 1), B(1, 1, 1, 1, 2)]  # 1x2 on a 1x1 tower, overhang toward +x
s_toward, _, _ = replay(cant, SPEC, offsets=np.array([[0, 0], [0.0003, 0]]))
s_away, _, _ = replay(cant, SPEC, offsets=np.array([[0, 0], [-0.0003, 0]]))
check("marginal: offset toward overhang falls", s_toward, False)
check("marginal: offset away from overhang stands", s_away, True)

unst = [B(1, 1, 0, 1, 1), B(1, 1, 1, 1, 2), B(1, 2, 2, 1, 1)]
s_un, _, _ = replay(unst, SPEC, offsets=np.zeros((3, 2)))
check("unstable scene falls", s_un, False)

corner = [B(1, 1, 0, 1, 1), B(1, 1, 1, 2, 2)]
s_w, errs_w, _ = replay(corner, SPEC, welded=True, offsets=np.array([[0, 0], [0.0003, 0.0003]]))
check("welded corner brick stands", s_w, True)
results["welded corner brick stands"]["max_error_mm"] = max(e for e, _ in errs_w) * 1000
s_l, _, _ = replay(corner, SPEC, offsets=np.array([[0, 0], [0.0003, 0.0003]]))
check("loose corner brick with outward offset falls", s_l, False)

json.dump({"all_pass": bool(ok), "tests": results}, open("validation.json", "w"), indent=1)
print("ALL PASS" if ok else "VALIDATION FAILED")
sys.exit(0 if ok else 1)
