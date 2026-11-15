"""Tests of prog3.py (no GPU, no model). usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3/test_prog3.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3common as C  # noqa: E402
import prog3 as PG  # noqa: E402

from arch_rsi import scientist as S  # noqa: E402

FAILED = []
G0R = C.PROGRAM_G0R.read_text()


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except Exception as e:  # noqa: BLE001
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, type(e).__name__, e)
    return fn


@test
def g0_is_g0r_with_steps_header():
    h, body = PG.parse_program(PG.g0_program(G0R))
    assert h == {"rounds": 4, "candidates_per_round": 3, "seeds_per_candidate": 2} and body == G0R
    assert PG.settings(h) == {"R": 4, "K": 3, "trial_seeds": [0, 1]} and PG.trial_runs(h) == 24


@test
def header_bounds():
    ok = [{"rounds": 6, "candidates_per_round": 4, "seeds_per_candidate": 1},
          {"rounds": 2, "candidates_per_round": 4, "seeds_per_candidate": 3},
          {"rounds": 2, "candidates_per_round": 2, "seeds_per_candidate": 1}]
    bad = [({"rounds": 6, "candidates_per_round": 3, "seeds_per_candidate": 2}, "over 24"),
           ({"rounds": 7, "candidates_per_round": 2, "seeds_per_candidate": 1}, "rounds must be"),
           ({"rounds": 4, "candidates_per_round": 3}, "missing seeds_per_candidate"),
           ({"rounds": 4, "candidates_per_round": 3, "seeds_per_candidate": 2, "retest_top": 1}, "unknown header keys")]
    for h in ok:
        assert PG.validate_header(h) == [], h
        assert PG.parse_program(PG.render_header(h) + "\n" + G0R)[0] == h
    for h, msg in bad:
        probs = PG.validate_header(h)
        assert any(msg in p for p in probs), (h, probs)
    for text in ("no header\n" + G0R, "<!-- header\nrounds: four\n-->\n" + G0R):
        try:
            PG.parse_program(text)
            assert False, "parsed a bad header"
        except ValueError:
            pass


@test
def facts_only_is_2cs_logic():
    """With 2c's parser, prog3.facts_only_problems returns exactly 2c's answer; with Step 3's, it applies the same rules."""
    P2C = PG.P2C
    parent2c = P2C.g0_program(G0R)
    kids = [parent2c.rstrip("\n") + "\n\n## Facts\n- a fact\n",
            parent2c.replace("rounds: 4", "rounds: 3").replace("both: 3", "both: 3") + "\n## Facts\n- x\n",
            parent2c.replace("Read every result so far", "Read every result") + "\n## Facts\n- x\n",
            parent2c + "\n## Facts\n- x\n\n## Facts\n- y\n",
            parent2c + "\n## Facts\n" + "word " * 900]
    for k in kids:
        assert PG.facts_only_problems(parent2c, k, parse=P2C.parse_program) == P2C.facts_only_problems(parent2c, k), k[-80:]
    parent = PG.g0_program(G0R)
    assert PG.facts_only_problems(parent, parent.rstrip("\n") + "\n\n## Facts\n- a fact\n") == []
    assert PG.facts_only_problems(parent, parent.replace("rounds: 4", "rounds: 2") + "\n## Facts\n- x\n") == ["the header changed"]
    assert PG.facts_only_problems(parent, parent.replace("Read every result", "Read the results") + "\n## Facts\n- x\n")
    assert P2C.parse_program is not PG.parse_program        # the swap is undone


@test
def labels_are_neutral_and_fixed():
    for L in C.LINEAGES:
        lab = PG.world_labels(L, C.TRAIN_WORLDS)
        assert lab == PG.world_labels(L, C.TRAIN_WORLDS)
        assert sorted(lab.values()) == sorted(f"world {k}" for k in range(1, 8)) and set(lab) == set(C.TRAIN_WORLDS)
    assert PG.world_labels("A", C.TRAIN_WORLDS) != PG.world_labels("B", C.TRAIN_WORLDS)


@test
def logs_name_worlds_neutrally_and_keep_analyses():
    t = {"label": "r1c1", "round": 1, "name": "n", "hypothesis": "h", "code": "def build_policy(): pass",
         "result": {"status": "ok", "dev_exact": 0.5, "dev_per_seed": [0.5, 0.5], "sha": "x"}}
    searches = [{"label": "world 2", "trials": [t], "analyses": {"1": "first look", "2": "second"}, "base_dev": 0.0},
                {"label": "world 5", "trials": [dict(t, label="r1c2")], "analyses": {}, "base_dev": 0.0}]
    out = PG.format_logs(searches, S.fmt_trial)
    assert "### Search 1, in world 2: best dev success 0.5" in out and "### Search 2, in world 5:" in out
    assert "- round 1: first look\n- round 2: second" in out and "(none)" in out
    assert "### Search 1:" not in out


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
