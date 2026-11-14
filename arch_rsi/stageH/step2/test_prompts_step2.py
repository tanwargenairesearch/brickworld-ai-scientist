"""Tests of Step 2's prompts (PROTOCOL_STEP2.md: arms, readouts, keeping the rule hidden).
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2/test_prompts_step2.py   (exits 1 on any failure)
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as C  # noqa: E402
import prompts as P  # noqa: E402
import readouts as RO  # noqa: E402

from arch_rsi import scientist as S  # noqa: E402

FAILED = []
H1 = C.STAGEH / "runs" / "b131072"


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except AssertionError as e:
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, e)
    return fn


# real readouts from Stage H1's trained runs (read only): reference A in each rule world, and in the plain world
RO_REAL = {w: RO.compute(H1 / f"{w}_A_s4000" / "try1" / "run", w, 0) for w in C.WORLDS}
PROGRAM = C.PROGRAM.read_text()
REF = C.common.REF_A.read_text()


def start(w):
    return {"code": REF, "dev": 0.0, "devs": [0.0, 0.0], "readouts": {"text_full": RO_REAL[w]["text_full"],
                                                                      "text_short": RO_REAL[w]["text_short"]}}


def trials(w):
    ro = {"text_full": RO_REAL[w]["text_full"], "text_short": RO_REAL[w]["text_short"]}
    out = []
    for r in (1, 2):
        for k in (1, 2, 3):
            out.append({"label": f"r{r}c{k}", "round": r, "name": f"variant {r}{k}", "hypothesis": "a neutral hypothesis",
                        "code": REF + f"\n# variant {r}{k}\n",
                        "result": {"status": "ok", "dev_exact": 0.01 * k, "dev_per_seed": [0.0, 0.02 * k], "train_exact": 0.0,
                                   "params": 55846, "flops_ratio": 1.04, "train_steps": 122880, "wall_s": 60.0,
                                   "sha": f"{r}{k}", "readouts": ro}})
    return out


def prompts_for(w):
    tr = trials(w)
    ros = {t["label"]: t["result"]["readouts"] for t in tr}
    return {a: P.build(a, w, PROGRAM, start(w), tr, tr[2], 3, 4, 3, readouts=ros if a in C.READOUT_ARMS else None)
            for a in C.ARMS}


PR = {w: prompts_for(w) for w in C.WORLDS}


@test
def readout_prompt_without_readouts_is_2b_prompt():
    tr = trials("shift")
    for t in tr:
        t["result"].pop("readouts")
    task = P.task_text("scores", "shift", start("shift"))
    a = P.l1_prompt_readouts(PROGRAM, task, P.interface(), REF, 0.0, tr, tr[2], 3, 4, 3)
    b = S.l1_prompt(PROGRAM, task, P.interface(), REF, 0.0, tr, tr[2], 3, 4, 3)
    assert a == b


@test
def no_rule_words_in_harness_text_of_untold_arms():
    for w in C.WORLDS:
        for a in ("readouts", "scores", "hidden"):
            found = P.rule_words(PR[w][a])
            assert not found, (w, a, found[:5])
            assert P.NOTE in PR[w][a] and P.RULE_TEXT[w] not in PR[w][a]


@test
def told_arm_states_the_rule_and_only_that_differs():
    for w in C.WORLDS:
        t, r = PR[w]["told"], PR[w]["readouts"]
        assert P.RULE_TEXT[w] in t and P.NOTE not in t
        assert t.replace("## This world's hidden rule\n" + P.RULE_TEXT[w], "") == r.replace("## A note on this world\n" + P.NOTE, "")


@test
def readouts_only_in_readout_arms():
    for w in C.WORLDS:
        for a in C.ARMS:
            has = "readouts (seed 0)" in PR[w][a] and "Readouts of the starting architecture" in PR[w][a]
            assert has == (a in C.READOUT_ARMS), (w, a)


@test
def full_readouts_for_latest_round_and_best_only():
    p = PR["shift"]["readouts"]
    blocks = p.split('{"trial": ')[1:]
    full = {b.split('"')[1]: "greedy trace, diagnostic design 1" in b for b in blocks}
    assert full == {"r1c1": False, "r1c2": False, "r1c3": True, "r2c1": True, "r2c2": True, "r2c3": True}, full


@test
def hidden_arm_shows_no_scores():
    for w in C.WORLDS:
        p = PR[w]["hidden"]
        assert "dev_success" not in p and "Starting architecture (dev success" not in p and "scored 0.000" not in p
        assert "## Scores are hidden in this search" in p


@test
def selection_sentence_is_step2s():
    for a in C.ARMS:
        t = P.task_text(a, "mirror", start("mirror"))
        assert P.NEW_SEL in t and P.OLD_SEL not in t and "8 x 8 base, up to 6 layers" in t and "131,072" in t


@test
def readouts_show_each_rules_evidence():
    s = RO_REAL["shift"]["traces"][0]
    assert s["seen"] == [row[-4:] + row[:-4] for row in s["design"]]
    m = RO_REAL["mirror"]["untrained"]["steps"]
    put = [x for x in m if x["action"] != "DONE"][0]
    t, rest = put["action"].split(" at row ")
    y, x = (int(v) for v in rest.split(", col "))
    dx = {"1x1": 1, "1x2": 2, "2x1": 1, "2x2": 2}[t]
    assert sorted({c[1] for c in put["filled"]}) == list(range(8 - dx - x, 8 - x)), (put, x, dx)
    assert P.rule_words(RO.LEGEND) == []


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
