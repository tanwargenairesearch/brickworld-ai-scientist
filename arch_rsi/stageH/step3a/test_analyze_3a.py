"""Tests of analyze_3a.py and run_3a.py's job lists on made-up runs (no GPU, no training): the pass rules, Holm over
eight rules, the fallback trigger, the families and the decision, and the identity checks.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3a/test_analyze_3a.py   (exits 1 on any failure)
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import analyze_3a as A  # noqa: E402
import run_3a as RN  # noqa: E402

FAILED = []
GOOD = {"ok": True}
PD = [1, 0, 1, 1, 0, 1, 1, 0]


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except Exception as e:  # noqa: BLE001
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, type(e).__name__, e)
    return fn


def write(runs: Path, budget, cell, seed, score, status="ok", per_design=None):
    d = runs / f"b{budget}" / f"{cell}_s{seed}"
    (d / "try1").mkdir(parents=True, exist_ok=True)
    rec = {"cell": cell, "seed": seed, "budget": budget, "status": status, "error": None if status == "ok" else "x",
           "dev_exact": score, "attempts": 1, "dir": "try1", "train_wall_s": 60.0, "exact_hidden": score,
           "wrong_cells_yx": [[1] * 8 for _ in range(8)]}
    (d / "result.json").write_text(json.dumps(rec))
    (d / "try1" / "dev.json").write_text(json.dumps({"per_design": {"exact": per_design or PD}}))


def campaign(means: dict, budget=A.BUDGET, fail=(), pd=None) -> Path:
    """means: {rule: (pF, rA, rF)}; plain_A is 0.76. Seed s adds 0.002*(s-4000) so the six runs differ."""
    runs = Path(tempfile.mkdtemp())
    for s in A.SEEDS:
        e = 0.002 * (s - 4000)
        write(runs, budget, "plain_A", s, 0.76 + e)
        for r, (pf, ra, rf) in means.items():
            for cell, v in ((f"plain_F_{r}", pf), (f"{r}_A", ra), (f"{r}_F", rf)):
                bad = (cell, s) in fail
                write(runs, budget, cell, s, v + e, status="timeout" if bad else "ok",
                      per_design=(pd or {}).get((cell, s)))
    return runs


PASS = (0.0, 0.0, 0.76)    # plain-fix 0, rule-A 0, rule-fix 0.76: repair 76, specific 152
NODAMAGE = (0.0, 0.70, 0.76)


def ev(runs, checks=None, fb=()):
    return A.evaluate(A.load_results(runs), checks if checks is not None else {r: GOOD for r in A.RULE_NAMES}, fb)


@test
def all_eight_pass_and_every_family_is_usable():
    e = ev(campaign({r: PASS for r in A.RULE_NAMES}))
    assert e["n_pass"] == 8 and e["decision"] == "step3_can_hold_out_a_family", e["decision"]
    for r in A.RULE_NAMES:
        d = e["rules"][r]
        assert abs(d["p_raw"] - 2 / 924) < 1e-12 and abs(d["p_holm"] - 8 * 2 / 924) < 1e-9, (r, d["p_raw"], d["p_holm"])
        assert abs(d["repair"] - 0.76) < 1e-9 and abs(d["interaction"] - 1.52) < 1e-9 and not d["fallback_trigger"]
    assert all(v["usable"] for v in e["families"].values())
    assert e["families"]["translation"]["passing"] == ["H1", "H5", "H6"]


@test
def only_two_families_usable_goes_back_to_the_user():
    m = {r: (PASS if r in ("rowshift", "handshift", "vmirror", "bookmirror") else NODAMAGE) for r in A.RULE_NAMES}
    e = ev(campaign(m))
    assert e["n_pass"] == 4 and e["usable_families"] == ["translation", "reflection"]
    assert e["decision"] == "options_back_to_user"
    assert not any(e["rules"][r]["fallback_trigger"] for r in A.RULE_NAMES)   # no damage, so no fallback


@test
def three_usable_families():
    m = {r: (PASS if r in ("rowshift", "vmirror", "transpose", "antitranspose") else NODAMAGE) for r in A.RULE_NAMES}
    e = ev(campaign(m))
    assert e["usable_families"] == ["translation", "reflection", "transposition"]
    assert e["decision"] == "step3_can_hold_out_a_family"


@test
def one_new_rule_passing_stops():
    m = {r: (PASS if r == "rowshift" else NODAMAGE) for r in A.RULE_NAMES}
    e = ev(campaign(m))
    assert e["n_pass"] == 1 and e["decision"] == "stop" and e["usable_families"] == ["translation"]


@test
def damage_without_repair_triggers_the_fallback_and_doubles_p():
    m = {r: PASS for r in A.RULE_NAMES}
    m["handshift"] = (0.0, 0.0, 0.10)   # damage 76, repair 10
    e = ev(campaign(m))
    d = e["rules"]["handshift"]
    assert not d["pass_repair"] and d["fallback_trigger"] and not d["passes"]
    runs = campaign(m)
    for s in A.SEEDS:   # fallback runs at 262,144 steps that do repair it
        e_ = 0.002 * (s - 4000)
        for cell, v in (("plain_A", 0.95), ("plain_F_handshift", 0.0), ("handshift_A", 0.0), ("handshift_F", 0.95)):
            write(runs, A.FALLBACK, cell, s, v + e_)
    e3 = ev(runs, fb=("handshift",))
    d3 = e3["rules"]["handshift"]
    assert d3["budget"] == A.FALLBACK and abs(d3["p_used"] - 4 / 924) < 1e-12 and d3["passes"]
    assert not e3["rules"]["rowshift"]["fallback"]


@test
def illegal_or_incomplete_fails():
    runs = campaign({r: PASS for r in A.RULE_NAMES}, fail={("transpose_F", 4003)})
    checks = {r: GOOD for r in A.RULE_NAMES}
    checks["rot90"] = {"ok": False, "error": "x"}
    e = ev(runs, checks)
    assert not e["rules"]["transpose"]["complete"] and e["rules"]["transpose"]["p_raw"] == 1.0
    assert not e["rules"]["transpose"]["passes"]
    assert not e["rules"]["rot90"]["legal"] and not e["rules"]["rot90"]["passes"]
    assert e["families"]["transposition"]["passing"] == ["H10"] and e["families"]["rotation"]["passing"] == ["H12"]
    assert e["decision"] == "options_back_to_user"


@test
def identity_checks():
    other = [0] * 8
    runs = campaign({r: PASS for r in A.RULE_NAMES}, pd={("transpose_F", 4002): other})
    h1 = Path(tempfile.mkdtemp())
    for s in A.SEEDS:
        write(h1, A.BUDGET, "plain_A", s, 0.76 + 0.002 * (s - 4000), per_design=other if s == 4005 else PD)
    res = A.analyze(runs, (), h1)
    idt = res["identity"]
    rows = idt["plain_reproduces_step1"]["rows"]
    assert [r["same_per_design"] for r in rows] == [True] * 5 + [False] and not idt["plain_reproduces_step1"]["all_identical"]
    b = idt["booklet_fix_matches_plain"]
    assert b["rowshift"]["all_identical"] and not b["transpose"]["all_identical"]
    assert set(b) == {"rowshift", "bookmirror", "transpose", "antitranspose", "rot90", "rot270"}
    assert res["cells"]["b131072/plain_A"]["wrong_by_row"] == [48] * 8


@test
def runner_job_lists():
    jobs = RN.main_jobs()
    assert len(jobs) == 150 and len({(j[0], j[3]) for j in jobs}) == 150
    assert {j[0] for j in jobs if j[2] == "A"} == {"plain_A"} | {f"{r}_A" for r in A.RULE_NAMES}
    fb = RN.fallback_jobs(["handshift", "vmirror"])
    assert len(fb) == 42 and all(j[4] == A.FALLBACK for j in fb)
    assert {j[0] for j in fb} == {"plain_A", "plain_F_handshift", "handshift_A", "handshift_F",
                                  "plain_F_vmirror", "vmirror_A", "vmirror_F"}


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
