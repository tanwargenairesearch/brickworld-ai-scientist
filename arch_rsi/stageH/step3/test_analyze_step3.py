"""Tests of analyze_step3.py on made-up campaigns (no GPU, no model): each ending A-E is chosen when it should be,
P4's untested case, scoring a violation as the start, and the check.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3/test_analyze_step3.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3common as C  # noqa: E402
import analyze_step3 as AN  # noqa: E402

FAILED = []
TMP = Path(tempfile.mkdtemp())
HG0 = {"rounds": 4, "candidates_per_round": 3, "seeds_per_candidate": 2}
HNEW = {"rounds": 6, "candidates_per_round": 4, "seeds_per_candidate": 1}


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except Exception as e:  # noqa: BLE001
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, type(e).__name__, e)
    return fn


def campaign(name, sc, finals=("g2", "g2"), header_change=True, violation=None):
    """sc: {arm: score for every search}. 8 searches per world for g0 and 4 per lineage per world for the program arms,
    4 per world for the check arm; final "g1" means the final arm is shared with g1."""
    cdir = TMP / name
    cdir.mkdir()
    ents = []
    for w in C.TEST_WORLDS:
        for arm in C.TEST_ARMS:
            lins = [None] if arm in ("g0", "check") else list(C.LINEAGES)
            n = 8 if arm == "g0" else 4
            for L in lins:
                if arm == "final" and finals[C.LINEAGES.index(L)] == "g0":
                    continue
                for i in range(1, n + 1):
                    s = sc[arm]
                    sha = f"{arm}{L}{w}{i}"
                    ents.append({"phase": "test", "world": w, "arm": arm, "lineage": L, "search": i,
                                 "path": f"test/{arm}/{L or '-'}/{w}/s{i}", "label": "start" if s == 0 else "r1c1",
                                 "name": "x", "sha": sha, "score": s, "first_round_found": 1 if s >= C.FOUND else None,
                                 "n_trials": 12, "settings": {"R": 4, "K": 3, "trial_seeds": [0, 1]},
                                 "shared_with": "g1" if arm == "final" and finals[C.LINEAGES.index(L)] == "g1" else None})
    gens = lambda f: [{"generation": 0, "kept": None, "mean": 0.0}, {"generation": 1, "kept": f != "g0", "mean": 0.5,  # noqa: E731
                      "header": HNEW if header_change else HG0, "parent_header": HG0}]
    summ = {"entries": ents, "notes": {}, "starts": {w: {"confirm": {"5001": 0.0, "5002": 0.0}} for w in C.TEST_WORLDS},
            "lineages": {L: {"final": f, "generations": gens(f)} for L, f in zip(C.LINEAGES, finals)}}
    (cdir / "summary.json").write_text(json.dumps(summ))
    verdicts = {e["sha"]: {"verdict": "violation" if e["sha"] == violation else "compliant"} for e in ents if e["label"] != "start"}
    (cdir / "review.json").write_text(json.dumps({"verdicts": verdicts}))
    return AN.analyze(cdir, n_perm=2000)


BASE = {"g0": 0.0, "g1": 0.7, "final": 0.8, "facts": 0.0, "nologs": 0.0, "check": 0.7}


@test
def ending_a():
    r = campaign("a", BASE)
    assert all(v["passes"] for v in r["primary"].values()) and r["ending"] == "A", r["summary_line"]
    assert r["ending_text"].endswith(AN.CAVEAT)


@test
def ending_b_without_compounding_or_header():
    assert campaign("b1", dict(BASE, final=0.7))["ending"] == "B"
    r = campaign("b2", BASE, header_change=False)
    assert r["primary"]["P4"]["passes"] and r["ending"] == "B"


@test
def ending_c_names_what_wasnt_separated():
    for name, sc, which in (("c1", dict(BASE, facts=0.7), "added facts,"), ("c2", dict(BASE, nologs=0.7), "the act of rewriting,"),
                            ("c3", dict(BASE, facts=0.7, nologs=0.7), "added facts or the act of rewriting,")):
        r = campaign(name, sc)
        assert r["ending"] == "C" and f"rather than {which}" in r["ending_text"], (name, r["ending_text"])


@test
def endings_d_and_e_follow_the_check():
    d = campaign("d", dict(BASE, g1=0.0, final=0.0, check=0.0))
    e = campaign("e", dict(BASE, g1=0.0, final=0.0))
    assert d["ending"] == "D" and not d["check"]["passes"] and e["ending"] == "E" and e["check"]["passes"]
    assert e["check"] == {"found": 12, "n": 12, "needed": 9, "passes": True}


@test
def p4_untested_when_nothing_kept():
    r = campaign("p4", BASE, finals=("g0", "g0"))
    assert r["primary"]["P4"]["p"] == 1.0 and r["primary"]["P4"]["diff"] is None and r["primary"]["P4"]["lineages"] == []
    assert r["ending"] == "B"


@test
def shared_final_counts_once_per_lineage():
    r = campaign("sh", dict(BASE, final=0.7), finals=("g1", "g2"))
    assert r["per_arm"]["final"]["n"] == 24 and r["primary"]["P4"]["lineages"] == ["A", "B"]


@test
def violation_scores_as_the_start():
    r = campaign("v", BASE, violation="g1Ashift1")
    row = next(x for x in r["rows"] if x["path"] == "test/g1/A/shift/s1")
    assert row["violation"] and row["score"] == 0.0 and r["per_arm"]["g1"]["violations"] == 1


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
