"""Tests of Step 2S's prompts (PROTOCOL_STEP2S.md, "Arms", "The sandbox", "Integrity rules": keeping the rule hidden).
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2s/test_prompts_2s.py   (exits 1 on any failure)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2s_common as C  # noqa: E402

import probes_2s as PB  # noqa: E402
import prompts as P  # noqa: E402  (Step 2's)
import prompts_2s as PR  # noqa: E402

from arch_rsi import scientist as S  # noqa: E402

FAILED = []
REF = C.common.REF_A.read_text()
PROGRAM = C.C2.PROGRAM.read_text()
START = {"code": REF, "dev": 0.0, "devs": [0.0, 0.0], "readouts": {"text_full": "FULL READOUT", "text_short": "SHORT"}}
TRIALS = [{"label": "r1c1", "round": 1, "name": "t", "hypothesis": "h", "code": REF + "\n# a\n",
           "result": {"status": "ok", "dev_exact": 0.1, "dev_per_seed": [0.1, 0.1], "train_exact": 0.1, "params": 1,
                      "flops_ratio": 1.0, "train_steps": 1, "wall_s": 1.0, "sha": "x"}}]
PL = [{"brick": "1x2", "row": 3, "col": 0}, {"brick": "1x1", "row": 0, "col": 0}]
HIST = {q: {"probes": [{"design": q, "purpose": f"purpose {q}", "placements": PL}],
            "results": [{"seen": [[q] * 8] * 8, "final": [[0] * 8] * 8, "stopped": None,
                         "steps": [dict(PL[0], filled=[[3, 6, 0], [3, 7, 0]], cells_in=1, cells_out=1, reward=0.02, illegal=False),
                                   dict(PL[1], filled=[[0, 0, 0]], cells_in=0, cells_out=1, reward=-0.05, illegal=False)]}]}
        for q in (1, 2, 3)}


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except AssertionError as e:
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, e)
    return fn


def base(arm, w="shift", r=2):
    return PR.base_prompt(arm, w, PROGRAM, START, TRIALS, None, r, 4, 3)


@test
def base_prompts_are_step2s_own():
    for w in C.C2.WORLDS:
        for r in (1, 2):
            want = P.build("scores", w, PROGRAM, START, TRIALS[:r - 1], None, r, 4, 3)
            assert PR.base_prompt("scores", w, PROGRAM, START, TRIALS[:r - 1], None, r, 4, 3) == want
            assert PR.base_prompt("sandbox", w, PROGRAM, START, TRIALS[:r - 1], None, r, 4, 3) == want
        told = base("told_sandbox", w)
        assert told == S.l1_prompt(PROGRAM, P.task_text("told", w, START), P.interface(), REF, 0.0, TRIALS, None, 2, 4, 3)
        assert P.RULE_TEXT[w] in told and P.NOTE not in told and "READOUT" not in told
        for a in ("scores", "sandbox"):
            b = base(a, w)
            assert P.NOTE in b and not any(P.RULE_TEXT[x] in b for x in P.RULE_TEXT) and "READOUT" not in b


@test
def probe_prompt_has_the_sandbox_text_and_the_probe_format():
    b = base("sandbox")
    q = PR.probe_prompt(b, {}, 1)
    head = b.split(PR.RETURN)[0]
    assert q.startswith(head) and q.count(PR.SANDBOX_TEXT) == 1 and PR.HOW in q and q.endswith(PR.PROBE_RETURN)
    assert "exactly 3 objects" not in q and "No probes yet in this search." in q
    assert q.count(PR.RETURN) == 1


@test
def proposal_prompt_is_the_base_with_results_before_what_to_return():
    b = base("sandbox")
    p = PR.proposal_prompt(b, {1: HIST[1]}, 1)
    head, tail = b.split(PR.RETURN)
    assert p.startswith(head + "\n## Sandbox results\n") and p.endswith(PR.RETURN + tail)
    assert p.index("## Budget") < p.index("## Sandbox results") < p.index("## What to return")
    assert "final build column heights" in p and "(row 3, col 6, layer 0)" in p


@test
def latest_round_in_full_earlier_rounds_as_requests():
    b = base("sandbox", r=3)
    p = PR.proposal_prompt(b, HIST, 3)
    assert "Round 3 probes and what the world did:" in p and "Round 2 probes (requests only):" in p
    assert "Round 1 probes (requests only):" in p and p.count("final build column heights") == 1
    assert "purpose 1" in p and "purpose 2" in p
    q = PR.probe_prompt(b, {1: HIST[1], 2: HIST[2]}, 3)
    assert "Round 2 probes and what the world did:" in q and "Round 1 probes (requests only):" in q
    assert q.count("final build column heights") == 1 and "Round 3" not in q
    assert PR.proposal_prompt(b, {3: {"probes": [], "results": []}}, 3).count("No probes were run.") == 1


@test
def no_rule_word_in_any_text_the_harness_writes():
    texts = PR.harness_texts() + [PB.render_results(HIST[1]["probes"], HIST[1]["results"])]
    for w in C.C2.WORLDS:
        texts.append(P.task_text("scores", w, START))
    for t in texts:
        assert not P.rule_words(t), (P.rule_words(t), t[:80])


@test
def sandbox_text_is_the_protocols_exactly():
    proto = (C.STEP2S / "PROTOCOL_STEP2S.md").read_text().replace("\n", " ")
    assert f'"{PR.SANDBOX_TEXT}"' in proto


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
