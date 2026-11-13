"""Tests for review_2c.py and analyze_2c.py on the fake campaign test_run_2c.py leaves in SCRATCH_DIR (run that first).
Fake model calls for the reviews; a fake start-with-Stage-G-recipe score per seed. No GPU, no real model calls.
usage (project root): .venv/bin/python -I arch_rsi/stage2/2c/test_analyze_2c.py SCRATCH_DIR
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import analyze_2c as AZ  # noqa: E402
import review_2c as RV  # noqa: E402

SRC = Path(sys.argv[1]).resolve() / "camp2c_test"
CAMP = Path(sys.argv[1]).resolve() / "camp2c_analyze"
CALLS = []


def fake_call(system, user, schema, validate, log_path, max_model_attempts=3):
    CALLS.append(system)
    if system == RV.PROGRAM_SYSTEM:
        obj = {"verdict": "no rule-breaking advice", "quotes": "none", "explanation": "fine"}
    else:  # pick review: call every third reviewed file a violation, deterministically by its label
        letter = user.split("(call it ")[1].split(")")[0]
        obj = {"verdict": "violation" if ord(letter[0]) % 2 else "compliant", "rule": "r", "evidence": "e"}  # A, C, E, ...
    val = validate(obj)
    log_path.write_text(json.dumps({"prompt": user, "reply": obj, "usage": {"notional_cost_usd": 0.5, "output_tokens": 10}}))
    return val, {}


def test_tiers_and_family():
    g = {"a": [0.6, 0.7, 0.8, 0.65], "b": [0.1, 0.2, 0.15, 0.12], "c": [0.4, 0.5, 0.45, 0.42]}
    with contextlib.redirect_stdout(io.StringIO()):
        f = AZ.family4(g, {"X": ("a", "b"), "Y": ("a", "c"), "Z": ("a", "missing")})
    assert f["Z"]["tested"] is False and f["Z"]["p_holm"] is None
    ps = {"X": AZ.AS1.perm_p(g["a"], g["b"]), "Y": AZ.AS1.perm_p(g["a"], g["c"]), "Z": 1.0}
    hb = AZ.AS1.holm(ps)
    assert f["X"]["p_holm"] == round(hb["X"], 4) and f["Y"]["p_holm"] == round(hb["Y"], 4)   # m = 3, not 2
    yes = {"tested": True, "diff": 0.1, "p_holm": 0.01}
    no = {"tested": True, "diff": 0.1, "p_holm": 0.2}
    untested = {"tested": False, "diff": None, "p_holm": None}
    lin_h = {"A": {"kept_header_change": True}}
    lin_n = {"A": {"kept_header_change": False}}
    t = AZ.tiers({"P1": yes, "P2": yes, "P3": yes, "P4": yes}, lin_h, False)
    assert t["level2"] and t["recursive"] and t["words_allowed"] == "recursive"
    assert AZ.tiers({"P1": yes, "P2": yes, "P3": yes, "P4": yes}, lin_n, False)["words_allowed"] == "Level 2"
    assert AZ.tiers({"P1": yes, "P2": yes, "P3": yes, "P4": untested}, lin_h, False)["words_allowed"] == "Level 2"
    assert AZ.tiers({"P1": yes, "P2": no, "P3": yes, "P4": yes}, lin_h, False)["words_allowed"].startswith("none")
    assert AZ.tiers({"P1": yes, "P2": yes, "P3": yes, "P4": yes}, lin_h, True) == {"status": "pending review"}


def test_on_fake_campaign():
    if CAMP.exists():
        shutil.rmtree(CAMP)
    shutil.copytree(SRC, CAMP)
    # the fake picks differ from the start only by comments (the same network); give every other one a real change
    picks = sorted({e["sha"] for e in json.loads((CAMP / "confirm" / "summary.json").read_text()) if e["label"] != "start"})
    assert picks, "the fake campaign has no picks"
    for h in picks[::2]:
        f = CAMP / "confirm" / "w" / h / "candidate.py"
        f.write_text(f.read_text() + "\n_VARIANT = 1\n")
    RV.S.call_with_retries = fake_call
    with contextlib.redirect_stdout(io.StringIO()):
        rv = RV.review_picks(CAMP)
        pr = RV.review_programs(CAMP)
    entries = json.loads((CAMP / "confirm" / "summary.json").read_text())
    picks = {e["sha"] for e in entries if e["label"] != "start"}
    assert set(rv["verdicts"]) == picks                                             # every distinct pick judged once
    by_c = [h for h, v in rv["verdicts"].items() if v["source"] == "by construction"]
    for h in by_c:
        e = next(e for e in entries if e["sha"] == h)
        src = (CAMP / "confirm" / "w" / h / "candidate.py").read_text()
        assert AZ_strip(src) == AZ_strip(RV.BASE.read_text()), e
    called = [h for h, v in rv["verdicts"].items() if v["source"] == "2c review"]
    assert len([c for c in CALLS if c == RV.RV2.SYSTEM]) == len(called)
    n_prog = len({RV.sha(f.read_text()) for f in CAMP.glob("programs/*/*.md")})
    assert len(pr["programs"]) == n_prog
    # analysis with the review, and a fake start score per seed
    review = rv["verdicts"]
    start = {str(s): 0.05 + 0.01 * (s - 1001) for s in range(1001, 1009)}
    H = AZ.AN2.hidden_flags()
    with contextlib.redirect_stdout(io.StringIO()):
        res = AZ.analyse(CAMP, entries, review, H, start)
        res["programs"] = AZ.programs(CAMP)
        res["spend"] = AZ.spend(CAMP)
    pf = res["primary_family"]
    assert pf["status"] == "final" and set(pf["contrasts"]) == {"P1", "P2", "P3", "P4"}
    assert all(pf["contrasts"][k]["p_holm"] is not None for k in ("P1", "P2", "P3"))
    for arm, d in res["per_arm"].items():
        for s in d["searches"]:
            if s["verdict"] == "violation":
                assert abs(s["primary_score"] - round(sum(start.values()) / 8, 4)) < 1e-4
            else:
                assert s["primary_score"] == s["score"]
    assert sum(d["violations"] for d in res["per_arm"].values()) > 0                  # the fake reviewer flagged some
    assert res["tiers"]["status"] == "final" and "words_allowed" in res["tiers"]
    assert "P1" in pf["contrasts"] and pf["contrasts"]["P1"]["n"] == [len(res["per_arm"]["g1"]["searches"]), len(res["per_arm"]["g0"]["searches"])]
    sec = res["secondary_family"]["contrasts"]
    assert "final - g0" in sec and "g1@A - facts@A" in sec and res["secondary_family"]["best_2b_arm"] in (None,) + tuple(sec)
    assert res["programs"]["A"]["g0"]["kind"] == "start" and res["programs"]["A"]["facts"]["kind"] == "facts"
    assert res["programs"]["A"]["g1"]["diff_lines"] > 0 and res["programs"]["A"]["g0"]["diff_lines"] == 0
    assert res["spend"]["calls"]["reviews"]["calls"] == len(called) + n_prog
    assert "meta-training (W_s)" in res["spend"]["runs_per_search"]
    # P4 when one lineage kept nothing: only the other lineage enters, with g1 from that lineage only
    lin = AZ.lineage_info(CAMP)
    lin["A"]["final"], lin["B"]["final"] = "g0", "g1"
    rows = AZ.score_entries(CAMP, entries, review, H, start)
    g = AZ.groups_for(rows, "_y", lin)
    assert len(g["g1@p4"]) == len([r for r in rows if r["arm"] == "g1" and r["lineage"] == "B"])
    assert all(r["lineage"] == "B" for r in rows if r["arm"] == "final" and r["_y"] in g["final"]) or not g["final"]
    # pending review: no verdicts, as-computed scores, tiers pending
    with contextlib.redirect_stdout(io.StringIO()):
        res_p = AZ.analyse(CAMP, entries, None, H, start)
    assert res_p["primary_family"]["status"] == "pending review" and res_p["tiers"] == {"status": "pending review"}
    # a review missing a pick stops the analysis
    some = next(iter(called))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            AZ.analyse(CAMP, entries, {k: v for k, v in review.items() if k != some}, H, start)
        raise AssertionError("missing verdict not caught")
    except SystemExit as e:
        assert "no verdict" in str(e)


def AZ_strip(src):
    return RV.w2b.strip_recipe(src)


if __name__ == "__main__":
    test_tiers_and_family()
    print("ok test_tiers_and_family")
    test_on_fake_campaign()
    print(f"ok test_on_fake_campaign ({len(CALLS)} fake review calls)")
    print("all analyze_2c / review_2c tests passed")
