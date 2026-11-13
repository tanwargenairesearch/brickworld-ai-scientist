"""Tests for prog2c.py. No pytest in the venv, so plain asserts; exits non-zero on the first failure.
Uses Stage 1's g0 search 2 (read-only) as a real fixture for the rewrite logs.
usage (project root): .venv/bin/python -I arch_rsi/stage2/2c/test_prog2c.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import prog2c as P  # noqa: E402
from arch_rsi.scientist import fmt_trial  # noqa: E402

G0R = (ROOT / "arch_rsi" / "stage2" / "programs" / "g0r.md").read_text()
FIX = json.loads((ROOT / "arch_rsi" / "campaigns" / "stage1" / "Q" / "g0" / "s2" / "state.json").read_text())


def raises(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"no ValueError containing {needle!r}")


def test_header():
    g0 = P.g0_program(G0R)
    h, body = P.parse_program(g0)
    assert h == P.G0_HEADER and body.lstrip("\n") == G0R
    assert P.runs_per_search(h) == 30 == P.MAX_RUNS
    bad = dict(P.G0_HEADER, seeds_per_candidate=3)             # 4*3*3 + 6 = 42 > 30
    raises(lambda: P.parse_program(P.render_header(bad) + G0R), "over 30")
    bad = dict(P.G0_HEADER, split={"network": 1, "recipe": 1, "both": 2})
    raises(lambda: P.parse_program(P.render_header(bad) + G0R), "sum to candidates_per_round")
    bad = dict(P.G0_HEADER, rounds=7)
    raises(lambda: P.parse_program(P.render_header(bad) + G0R), "rounds must be 2-6")
    raises(lambda: P.parse_program(G0R), "must start with")
    raises(lambda: P.parse_program(P.render_header(P.G0_HEADER).replace("rounds: 4", "rounds: four") + G0R), "whole number")
    ok = dict(P.G0_HEADER, rounds=3, candidates_per_round=4, split={"network": 1, "recipe": 1, "both": 2})  # 24 + 6
    assert P.parse_program(P.render_header(ok) + G0R)[0] == ok


def test_facts_only():
    g0 = P.g0_program(G0R)
    good = g0.rstrip("\n") + "\n\n## Facts\n- On W_s, footprint-aware heads learned the hidden-cell skill more often.\n"
    assert P.facts_only_problems(g0, good) == [], P.facts_only_problems(g0, good)
    edited = good.replace("Read every result so far", "Read the best results so far")
    assert any("must be copied exactly" in p and "How to run each round" in p for p in P.facts_only_problems(g0, edited))
    assert P.strip_header(g0) == G0R and P.strip_header("<!-- header\nrounds: x\n-->\nbody\n") == "body\n"
    assert P.strip_header("# no header\n") == "# no header\n"
    assert P.parse_program(g0)[1] == G0R                                           # body byte-identical to 2b's g0r
    hdr = P.render_header(dict(P.G0_HEADER, rounds=3, candidates_per_round=4,
                               split={"network": 0, "recipe": 0, "both": 4})) + G0R + "\n## Facts\n- x\n"
    assert "the header changed" in P.facts_only_problems(g0, hdr)
    assert any("exactly one" in p for p in P.facts_only_problems(g0, g0))
    two = good + "\n## Facts\n- again\n"
    assert any("exactly one" in p for p in P.facts_only_problems(g0, two))


def test_shuffle_and_logs():
    trials = FIX["trials"]
    sh = P.shuffle_search(trials, 1)
    assert sh == P.shuffle_search(trials, 1)                                       # deterministic
    ok = [t for t in trials if t["result"]["status"] == "ok"]
    real = sorted(t["result"]["dev_exact"] for t in ok)
    shuf = sorted(t["result"]["dev_exact"] for t in sh if t["result"]["status"] == "ok")
    assert real == shuf                                                            # same scores, moved
    assert [t["result"]["status"] for t in sh] == [t["result"]["status"] for t in trials]
    assert any(a["result"]["dev_exact"] != b["result"]["dev_exact"] for a, b in zip(trials, sh))
    for a, b in zip(trials, sh):                                                   # a bundle moves together
        if b["result"]["status"] == "ok":
            src = next(t for t in ok if t["result"]["dev_exact"] == b["result"]["dev_exact"])
            assert b["result"]["dev_per_seed"] == src["result"]["dev_per_seed"]
            assert b["code"] == a["code"]                                          # code stays with its trial
    assert P._relabel("scored 0.6709 and 67.1%", {"0.6709": "0.1016", "67.1%": "10.2%"}) == "scored 0.1016 and 10.2%"
    assert P._relabel("lr 0.00067", {"0.67": "0.10"}) == "lr 0.00067"               # no partial-number hits
    searches = [{"trials": trials}]
    real_logs = P.format_logs(searches, fmt_trial, 0.3447)
    shuf_logs = P.format_logs(searches, fmt_trial, 0.3447, shuffle_seed=1)
    for a in FIX.get("analyses", {}).values():
        if a and len(a) > 40:
            assert a[:40] not in real_logs and a[:40] not in shuf_logs             # analyses never included
    assert real_logs != shuf_logs and "### Search 1: best dev success" in shuf_logs
    best = max(t["result"]["dev_exact"] for t in ok)
    assert f"best dev success {best}" in real_logs and f"best dev success {best}" in shuf_logs


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all prog2c tests passed")
