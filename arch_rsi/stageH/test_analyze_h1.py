"""Tests of analyze_h1.py on fake data (no runs needed).
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/test_analyze_h1.py   (exits 1 on any failure)
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_h1 as A  # noqa: E402
from arch_rsi.stage1 import analyze_stage1  # noqa: E402

FAILED = []


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except AssertionError as e:
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, e)
    return fn


def fake(means: dict, budget=A.BUDGET, jitter=0.01, status=None):
    """{(budget, cell, seed): result} with per-seed scores mean + small spread."""
    out = {}
    for cell, m in means.items():
        for i, s in enumerate(A.SEEDS):
            st = (status or {}).get((cell, s), "ok")
            out[(budget, cell, s)] = {"budget": budget, "cell": cell, "seed": s, "status": st, "attempts": 1,
                                      "dev_exact": round(m + jitter * (i - 2.5), 4), "dir": "try1"}
    return out


BASE = {"plain_A": 0.69,
        "plain_F_shift": 0.05, "shift_A": 0.10, "shift_F": 0.69,       # passes
        "plain_F_mirror": 0.84, "mirror_A": 0.30, "mirror_F": 0.60,    # repair 0.30, plain gain 0.15 -> interaction 0.15: fails 2
        "plain_F_torus": 0.69, "torus_A": 0.59, "torus_F": 0.64,       # damage 0.10, repair 0.05: fails, no fallback
        "plain_F_late": 0.70, "late_A": 0.39, "late_F": 0.49}          # damage 0.30, repair 0.10: fails 1, fallback
LEGAL = {r: {"ok": True} for r in A.RULE_NAMES}


@test
def uses_stage1_functions_unchanged():
    assert A.perm_p is analyze_stage1.perm_p and A.holm is analyze_stage1.holm and A.boot_diff is analyze_stage1.boot_diff


@test
def pass_rules_and_fallback_trigger():
    ev = A.evaluate(fake(BASE), LEGAL)
    R = ev["rules"]
    assert R["shift"]["passes"] and R["shift"]["repair"] == 0.59
    assert R["mirror"]["pass_repair"] and not R["mirror"]["pass_specific"] and not R["mirror"]["passes"]
    assert abs(R["mirror"]["interaction"] - 0.15) < 1e-6
    assert not R["torus"]["passes"] and not R["torus"]["fallback_trigger"]
    assert not R["late"]["pass_repair"] and R["late"]["fallback_trigger"]
    assert not R["shift"]["fallback_trigger"]
    assert ev["n_pass"] == 1 and ev["decision"] == "stop"
    assert R["shift"]["p_raw"] == 2 / 924  # complete separation, 6 against 6, two-sided


@test
def holm_over_four():
    ev = A.evaluate(fake(BASE), LEGAL)
    ps = {r: ev["rules"][r]["p_used"] for r in A.RULE_NAMES}
    assert {r: ev["rules"][r]["p_holm"] for r in A.RULE_NAMES} == analyze_stage1.holm(ps)


@test
def decisions_by_count():
    three = dict(BASE, plain_F_mirror=0.69, torus_A=0.20, torus_F=0.69)
    ev = A.evaluate(fake(three), LEGAL)
    assert ev["passing"] == ["shift", "mirror", "torus"] and ev["decision"] == "step2_and_step3_possible"
    two = dict(three, torus_F=0.30)
    assert A.evaluate(fake(two), LEGAL)["decision"] == "step2_only"


@test
def illegal_or_incomplete_fails():
    ev = A.evaluate(fake(BASE), dict(LEGAL, shift={"ok": False}))
    assert not ev["rules"]["shift"]["passes"] and not ev["rules"]["shift"]["pass_legal"]
    res = fake(BASE, status={("shift_F", 4003): "timeout"})
    ev = A.evaluate(res, LEGAL)
    d = ev["rules"]["shift"]
    assert not d["complete"] and d["p_used"] == 1.0 and not d["passes"] and len(d["per_seed"]["rF"]) == 5


@test
def fallback_runs_replace_main_and_double_p():
    res = fake(BASE)
    fb = {"plain_A": 0.92, "plain_F_late": 0.92, "late_A": 0.40, "late_F": 0.90}
    res.update(fake(fb, budget=A.FALLBACK))
    ev = A.evaluate(res, LEGAL, fallback_rules=("late",))
    d = ev["rules"]["late"]
    assert d["budget"] == A.FALLBACK and d["fallback"] and abs(d["repair"] - 0.50) < 1e-6
    assert d["p_used"] == min(1.0, 2 * d["p_raw"]) and d["passes"] and not d["fallback_trigger"]
    assert ev["rules"]["shift"]["budget"] == A.BUDGET


@test
def threshold_is_inclusive():
    exact = dict(BASE, shift_A=0.44, shift_F=0.69, plain_F_shift=0.69)  # repair exactly 0.25, interaction 0.25
    d = A.evaluate(fake(exact, jitter=0.0), LEGAL)["rules"]["shift"]
    assert d["pass_repair"] and d["pass_specific"], (d["repair"], d["interaction"])


@test
def bootstrap_interval_brackets_estimate():
    d = A.evaluate(fake(BASE, jitter=0.05), LEGAL)["rules"]["mirror"]
    lo, hi = d["interaction_ci95_boot"]
    assert lo <= d["interaction"] <= hi


@test
def describe_and_h1_identity_on_a_fake_runs_folder():
    with tempfile.TemporaryDirectory() as t:
        runs = Path(t)
        res = fake({"plain_A": 0.69, "shift_F": 0.69, "torus_A": 0.5})
        for (b, cell, s), r in res.items():
            d = runs / f"b{b}" / f"{cell}_s{s}" / "try1"
            (d / "run").mkdir(parents=True)
            r.update(rule="plain", cand="A", wrong_cells_yx=[[1] * 8] + [[0] * 8] * 7, exact_hidden=0.5, train_wall_s=60.0)
            (d.parent / "result.json").write_text(json.dumps(r))
            (d / "dev.json").write_text(json.dumps({"per_design": {"exact": [1, 0, s % 2]}}))
            (d / "run" / "train_log.json").write_text(json.dumps({"curve": [{"steps": 4096, "train_exact": 0.1}]}))
        loaded = A.load_results(runs)
        assert len(loaded) == 18
        h = A.h1_identity(loaded, runs)
        assert h["all_identical"] and all(x["same_score"] for x in h["rows"])
        cells = A.describe(loaded, runs, [A.BUDGET])
        e = cells[f"b{A.BUDGET}/torus_A"]["edge"]
        assert e["edge_share"] == 1.0 and e["board_edge_share"] == round(28 / 64, 4) and e["wrong_total"] == 48
        assert cells[f"b{A.BUDGET}/plain_A"]["learning_curve"] == [{"steps": 4096, "train_exact": 0.1}]


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
