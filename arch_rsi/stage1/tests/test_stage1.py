"""Tests for Stage 1. Run: .venv/bin/python arch_rsi/stage1/tests/test_stage1.py (exits non-zero on failure).
No training and no Claude calls."""
import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arch_rsi" / "stage1"))

import analyze_stage1 as A  # noqa: E402
import run_stage1 as R  # noqa: E402
from arch_rsi import orchestrate_arch as O  # noqa: E402
from arch_rsi import scientist as S  # noqa: E402

BASE = (O.HERE / "candidates" / "baseline.py").read_text()
G0 = (O.HERE / "programs" / "g0.md").read_text()


def fake_campaign(preset="stage1"):
    c = R.Stage1.__new__(R.Stage1)  # no manifest, no hashing: only the prompt-building methods are used
    c.cfg = R.PRESETS[preset]
    c.base = {"xl": {"devs": [0.541015625, 0.162109375, 0.16015625, 0.51953125], "mean": 0.3457, "shown_dev": 0.3515625}}
    return c


TRIALS = [
    {"label": "r1c1", "round": 1, "name": "footprint head", "hypothesis": "HYPO-ONE", "code": "def build_policy(spec):\n    pass",
     "result": {"status": "ok", "dev_exact": 0.734567, "dev_per_seed": [0.912345, 0.556789], "train_exact": 0.645321,
                "params": 12345, "flops_ratio": 1.5, "train_steps": 87040, "wall_s": 177.7, "sha": "aaa"}},
    {"label": "r1c2", "round": 1, "name": "bad file", "hypothesis": "HYPO-TWO", "code": "x",
     "result": {"status": "invalid", "error": "CHECK-ERROR-TEXT", "dev_exact": None, "sha": "bbb"}},
    {"label": "r1c3", "round": 1, "name": "timeout net", "hypothesis": "HYPO-THREE", "code": "y",
     "result": {"status": "timeout", "error": "killed after 600 s", "dev_exact": 0.0, "dev_per_seed": [0.0, 0.0],
                "train_exact": 0.0, "params": 999, "flops_ratio": 11.0, "train_steps": 0, "wall_s": 600.2, "sha": "ccc"}},
]
SCORE_STRINGS = ["0.734567", "0.912345", "0.556789", "0.645321", "0.541", "0.352", "0.346", "0.162", "177.7", "600.2"]


def test_hidden_prompt_has_no_scores():
    c = fake_campaign()
    for trials in ([], TRIALS):
        p = R.l1_prompt_hidden(G0, c.task_for("xl", True), c.interface(), BASE, trials, 2, 4, 3)
        for s in SCORE_STRINGS + ["dev_success", "train_success", "train_seconds", "Current best", "dev success 0"]:
            assert s not in p, f"hidden prompt leaks {s!r}"
        assert R.HIDDEN_NOTE in p
    p = R.l1_prompt_hidden(G0, c.task_for("xl", True), c.interface(), BASE, TRIALS, 2, 4, 3)
    for s in ["HYPO-ONE", "HYPO-TWO", "HYPO-THREE", "CHECK-ERROR-TEXT", "killed after 600 s", '"status": "invalid"',
              '"status": "timeout"', '"status": "ok"', '"params": 12345', "def build_policy", "round 2 of 4"]:
        assert s in p, f"hidden prompt lacks {s!r}"


def test_g0_prompt_is_the_frozen_one_with_scores():
    c = fake_campaign()
    task = c.task_for("xl", False)
    p = S.l1_prompt(G0, task, c.interface(), BASE, c.base["xl"]["shown_dev"], TRIALS, TRIALS[0], 2, 4, 3)
    for s in ["0.734567", "0.912345", "dev success 0.352", "Current best: trial r1c1", "0.541, 0.162, 0.160, 0.520"]:
        assert s in p, f"g0 prompt lacks {s!r}"
    assert R.HIDDEN_NOTE not in p


def test_task_text_states_the_stage1_selection():
    c = fake_campaign()
    for hidden in (False, True):
        t = c.task_for("xl", hidden)
        assert R.FROZEN_SELECTION not in t
        assert "the 3 trials with the highest dev success each train 2 more seeds" in t
        assert "highest mean over its 4 seeds" in t
        assert "131,072" in t and "600 seconds" in t
    d = fake_campaign("stage1_dry")
    d.base = {"xl": {"devs": [0.1, 0.2, 0.3], "mean": 0.2, "shown_dev": 0.15}}
    assert "each train 1 more seed, and the one with the highest mean over its 3 seeds" in d.task_for("xl", False)


def test_choose_pick():
    rows = [{"label": "a", "order": 0, "all_ok": True, "mean": 0.60},
            {"label": "b", "order": 1, "all_ok": True, "mean": 0.70},
            {"label": "c", "order": 2, "all_ok": False, "mean": 0.90}]
    assert R.choose_pick(rows, 0.35) == "b"                       # best eligible mean; c had a failed seed
    assert R.choose_pick(rows, 0.70) is None                      # must beat the start strictly
    assert R.choose_pick(rows, 0.80) is None
    tie = [{"label": "x", "order": 5, "all_ok": True, "mean": 0.5}, {"label": "y", "order": 2, "all_ok": True, "mean": 0.5}]
    assert R.choose_pick(tie, 0.1) == "y"                         # ties go to the earlier trial
    assert R.choose_pick([], 0.3) is None


def test_top_for_retest():
    tr = [{"label": l, "result": {"status": s, "dev_exact": d}} for l, s, d in
          [("r1c1", "ok", 0.5), ("r1c2", "invalid", None), ("r1c3", "timeout", 0.9), ("r2c1", "ok", 0.7),
           ("r2c2", "ok", 0.5), ("r2c3", "ok", 0.2)]]
    assert [t["label"] for t in R.top_for_retest(tr, 3)] == ["r2c1", "r1c1", "r2c2"]


def test_search_order_interleaves_arms():
    o = R.search_order(["g0", "nofb", "menu"], 2)
    assert o == [("g0", 1), ("nofb", 1), ("menu", 1), ("g0", 2), ("nofb", 2), ("menu", 2)]


def test_stage1_files_exist_and_are_hashed():
    for p in R.STAGE1_FILES:
        assert p.exists(), p
    assert R.PRESETS["stage1"]["confirm_seeds"] == list(range(1001, 1009))
    assert R.PRESETS["stage1"]["comparator_seeds"][:8] == R.PRESETS["stage1"]["confirm_seeds"]
    assert R.PRESETS["stage1"]["base_seeds"] == R.PRESETS["stage1"]["trial_seeds"] + R.PRESETS["stage1"]["retest_seeds"]
    assert R.PRESETS["stage1_dry"]["base_seeds"] == R.PRESETS["stage1_dry"]["trial_seeds"] + R.PRESETS["stage1_dry"]["retest_seeds"]
    assert R.PRESETS["stage1"]["steps"]["xl"] == 131072 and R.PRESETS["stage1"]["n_searches"] == 10


def test_permutation_and_holm():
    assert abs(A.perm_p([1, 2, 3], [4, 5, 6]) - 0.1) < 1e-12      # 2 of 20 splits are as extreme
    assert A.perm_p([1, 1], [1, 1]) == 1.0
    h = A.holm({"a": 0.01, "b": 0.04})
    assert abs(h["a"] - 0.02) < 1e-12 and abs(h["b"] - 0.04) < 1e-12
    h = A.holm({"a": 0.03, "b": 0.02})
    assert abs(h["b"] - 0.04) < 1e-12 and abs(h["a"] - 0.04) < 1e-12


def test_analysis_end_to_end_and_violation_rule():
    H = A.hidden_flags()
    seeds = [1001, 1002]

    def entry(kind, arm, i, label, sha, devs, status=None):
        return {"kind": kind, "arm": arm, "search": i, "label": label, "name": label, "sha": sha, "mean_selected": 0.9,
                "incumbent_2seed": label, "seeds": seeds, "dev": {str(s): d for s, d in zip(seeds, devs)},
                "status": {str(s): (status or "ok") for s in seeds}, "wall_s": {str(s): 1.0 for s in seeds}}
    ents = [entry("search", "g0", i, f"r4c{i}", f"g{i}", [0.9, 0.8]) for i in (1, 2, 3)]
    ents += [entry("search", "nofb", i, f"r2c{i}", f"n{i}", [0.3, 0.4]) for i in (1, 2, 3)]
    ents += [entry("search", "menu", i, "r1c1", f"m{i}", [0.2, 0.2]) for i in (1, 2, 3)]
    ents += [entry("comparator", "start", 0, "start", "st", [0.1, 0.3]),
             entry("comparator", "critic_ortho", 0, "critic_ortho", "co", [0.5, 0.0], "timeout")]
    with tempfile.TemporaryDirectory() as d:
        pending = A.analyse(Path(d), ents, None, H)
        assert pending["primary_family"]["status"] == "pending review" and pending["primary_family"]["gate_passed"] is None
        assert pending["per_arm"]["critic_ortho"]["per_seed"] == [0.0, 0.0]      # failed seeds score 0
        rv = {"g1": {"verdict": "violation"}, "g2": {"verdict": "compliant"}, "g3": {"verdict": "compliant"}}
        res = A.analyse(Path(d), ents, rv, H)
    g0 = res["per_arm"]["g0"]
    assert g0["violations"] == 1 and abs(g0["mean"] - 0.85) < 1e-9
    assert abs(g0["primary_mean"] - (0.2 + 0.85 + 0.85) / 3) < 1e-4       # the violator scores the start's 0.2
    pf = res["primary_family"]["contrasts"]
    assert set(pf) == {"g0 - nofb", "g0 - start"} and res["primary_family"]["status"] == "final"
    assert all("p_holm" in v and "ci95_boot" in v for v in pf.values())


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok    {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}\n{traceback.format_exc()}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
