"""Tests of prompts3.py (no GPU, no model).
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3/test_prompts3.py"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3common as C  # noqa: E402
import prog3 as PG  # noqa: E402
import prompts3 as Q  # noqa: E402

import prompts as P  # noqa: E402  (Step 2's)
import run_step2 as RS  # noqa: E402
import rules3a  # noqa: E402

FAILED = []
REF = C.common.REF_A.read_text()
G0R = C.PROGRAM_G0R.read_text()
START = {"raw": [0.0, 0.0, 0.0], "code": REF, "readouts": {"text_full": "FULL READOUT", "text_short": "SHORT READOUT"}}
TRIAL = {"label": "r1c1", "round": 1, "name": "wider", "hypothesis": "more channels", "code": REF,
         "result": {"status": "ok", "dev_exact": 0.0, "dev_per_seed": [0.0, 0.0], "sha": "x",
                    "readouts": {"text_full": "TRIAL FULL", "text_short": "TRIAL SHORT"}}}


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except Exception as e:  # noqa: BLE001
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, type(e).__name__, e)
    return fn


def literal_2c(name: str) -> str:
    tree = ast.parse((C.ROOT / "arch_rsi" / "stage2" / "2c" / "run_2c.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


@test
def two_seed_prompts_are_step2s():
    v = Q.start_view(START, 2)
    for arm in ("scores", "readouts"):
        for world in ("shift", "mirror"):
            for trials, r in (([], 1), ([TRIAL], 2)):
                ro = {"r1c1": TRIAL["result"]["readouts"]} if arm == "readouts" else None
                a = Q.build3(arm, world, G0R, v, trials, None, r, 4, 3, 2, readouts=ro)
                b = P.build(arm, world, G0R, v, trials, None, r, 4, 3, readouts=ro)
                assert a == b, (arm, world, r)
    assert P.NOTE in Q.build3("scores", "rowshift", G0R, v, [], None, 1, 4, 3, 2)


@test
def other_seed_counts_change_only_two_sentences():
    t2 = Q.task_text3("scores", "shift", Q.start_view(START, 2), 2)
    for n, words in ((1, "every trial trains 1 seed and reports its dev success"), (3, "every trial trains 3 seeds")):
        t = Q.task_text3("scores", "shift", Q.start_view(START, n), n)
        assert words in t and f"highest {n}-seed dev success" in t and f"own {n}-seed score" in t and "2-seed" not in t
        a, b = t2.split("\n"), t.split("\n")
        assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 2, n
    p = Q.build3("scores", "shift", G0R, Q.start_view(START, 1), [], None, 1, 6, 4, 1)
    assert "This is round 1 of 6. Each round trains 4 candidates in parallel; 24 trials remain" in p


@test
def start_view_rounds_like_step2():
    s = dict(START, raw=[0.12344, 0.5, 0.25])
    assert Q.start_view(s, 2)["devs"] == [0.1234, 0.5] and Q.start_view(s, 2)["dev"] == round((0.12344 + 0.5) / 2, 4)
    assert Q.start_view(s, 3)["dev"] == round((0.12344 + 0.5 + 0.25) / 3, 4)


@test
def rewrite_prompt_keeps_2cs_text_where_stated():
    assert Q.facts_restriction() == literal_2c("FACTS_RESTRICTION")
    l2c = literal_2c("L2C_PROMPT")
    rule_par = l2c[l2c.index("The interface rules, caps"):l2c.index("{restriction}")]
    assert rule_par in Q.L3_PROMPT
    tail = l2c[l2c.index("## What to return"):]
    assert tail.replace("{max_words:,}", "X") == Q.L3_PROMPT[Q.L3_PROMPT.index("## What to return"):].replace("{max_words:,}", "X")


@test
def nologs_differs_only_in_table_and_logs():
    task = Q.task_text3("scores", "mirror", Q.start_view(START, 2), 2)
    real = Q.l3_prompt("real", "g0", PG.g0_program(G0R), task, "TABLE", "generation 0 (program g0)", "LOGS")
    nolg = Q.l3_prompt("nologs", "g0", PG.g0_program(G0R), task, "TABLE", "generation 0 (program g0)", "LOGS")
    facts = Q.l3_prompt("facts", "g0", PG.g0_program(G0R), task, "TABLE", "generation 0 (program g0)", "LOGS")
    assert "TABLE" not in nolg and "LOGS" not in nolg and nolg.count(Q.NONE) == 2
    assert real.replace("TABLE", Q.NONE).replace(Q.LOGS_INTRO + "\n\nLOGS", Q.NONE) == nolg
    assert facts.replace(Q.facts_restriction(), "").replace("Follow the restriction above.",
                                                            "Revise the program by following its own section on revising it.") == real


@test
def harness_text_names_no_rule():
    texts = [Q.task_text3(arm, w, Q.start_view(START, n), n) for arm in ("scores", "readouts") for w in C.TEST_WORLDS
             for n in (1, 2, 3)]
    texts += [Q.build3("scores", "shift", G0R, Q.start_view(START, n), [], None, 1, 4, 3, n) for n in (1, 2, 3)]
    texts += [Q.l3_prompt(k, "g0", PG.g0_program(G0R), texts[0], "TABLE", "generation 0 (program g0)", "LOGS")
              for k in ("real", "facts", "nologs")]
    texts += [Q.LOGS_INTRO, Q.NONE, G0R, RS.REV.prompt("A", REF)]
    for t in texts:
        assert Q.rule_words(t) == [], Q.rule_words(t)
    assert Q.rule_words("turned rows transposed 4 rows diagonal quarter") != []


@test
def judge_prompt_is_step2s():
    an = ["a1", "a2", "a3", "a4"]
    assert Q.judge_prompt("shift", an, "n", "h") == RS.judge_prompt("shift", an, "n", "h")
    for w in C.TEST_WORLDS:
        assert Q.RULE_TEXT[w] in Q.judge_prompt(w, an, "n", "h")


@test
def rule_sentences_match_the_rules():
    assert rules3a.ROW_SHIFT == 4 and rules3a.HAND_SHIFT == 4
    assert "rolled 4 rows toward higher y" in Q.RULE_TEXT["rowshift"]
    assert "(x + 4) mod 8" in Q.RULE_TEXT["handshift"]


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
