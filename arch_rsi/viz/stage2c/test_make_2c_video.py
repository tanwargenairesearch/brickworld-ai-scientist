"""Test for make_2c_video.py on the fake campaign test_run_2c.py leaves in SCRATCH_DIR (run that first).
It adds a fake review and results.json, renders the stills (and the video with --video), and checks that the renderer
stops when a program, a generation's runs or an arm's score no longer match the saved results. Scene 3 is a card
here (--skip-build): the fake campaign has no trained checkpoints. No GPU, no model calls.

usage (project root): nice -n 19 .venv/bin/python -I arch_rsi/viz/stage2c/test_make_2c_video.py SCRATCH_DIR [--video]
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A.parent))
sys.path.insert(0, str(A / "stage2" / "2c"))
sys.path.insert(0, str(HERE))

import analyze_2c as AZ  # noqa: E402
import make_2c_video as MV  # noqa: E402
import review_2c as RV  # noqa: E402

SCRATCH = Path(sys.argv[1]).resolve()
SRC, CAMP, OUT = SCRATCH / "camp2c_test", SCRATCH / "camp2c_video", SCRATCH / "vid2c"
LONG = ("Reserve one candidate per round for a recipe-only change on the current best network, because recipe changes "
        "gave the largest single gains in these searches and were cheap to test against seed noise at this budget")
EVID = ("In 4 of 6 searches the best trial changed only the learning rate or the number of parallel worlds; network "
        "changes without a recipe change stayed within the spread between seeds")


def fake_call(system, user, schema, validate, log_path, max_model_attempts=3):
    obj = ({"verdict": "no rule-breaking advice", "quotes": "none", "explanation": "fine"} if system == RV.PROGRAM_SYSTEM
           else {"verdict": "compliant", "rule": "none", "evidence": "e"})
    log_path.write_text(json.dumps({"prompt": user, "reply": obj, "usage": {}}))
    return validate(obj), {}


def prepare():
    if CAMP.exists():
        shutil.rmtree(CAMP)
    shutil.copytree(SRC, CAMP)
    for f in CAMP.glob("programs/*/revision_g*.json"):          # realistic lengths for the quoted changes
        rv = json.loads(f.read_text())
        rv["result"]["changes"] = [{"change": LONG, "evidence": EVID}, {"change": "Retest 4 candidates instead of 3.",
                                                                       "evidence": "The 2-seed leader lost its retest twice."}]
        f.write_text(json.dumps(rv, indent=1))
    RV.S.call_with_retries = fake_call
    with contextlib.redirect_stdout(io.StringIO()):
        review = RV.review_picks(CAMP)["verdicts"]
        entries = json.loads((CAMP / "confirm" / "summary.json").read_text())
        start = {str(s): 0.1 for s in range(1001, 1009)}
        res = AZ.analyse(CAMP, entries, review, AZ.AN2.hidden_flags(), start)
    (CAMP / "results.json").write_text(json.dumps(res, indent=1))
    return res


def stops(fn, what):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except (AssertionError, SystemExit):
        return
    raise AssertionError(f"the renderer did not stop when {what}")


def test_scene3_real():
    """Scene 3's real path on world W, with Stage 2b's dry-run search (real checkpoints, tiny budgets; not a result)."""
    src = A / "campaigns" / "stage2b_dry" / "W" / "g0" / "s1"
    if not src.exists():
        return "skipped (no stage2b_dry campaign)"
    camp = SCRATCH / "camp2c_scene3"
    if camp.exists():
        shutil.rmtree(camp)
    sdir = camp / "W" / "final" / "A" / "s1"
    shutil.copytree(src, sdir)
    st = json.loads((sdir / "state.json").read_text())
    st["settings"] = {"R": 2, "K": 2, "trial_seeds": [0, 1], "retest_top": 2, "retest_seeds": [2, 3], "roles": ["both", "both"]}
    (sdir / "state.json").write_text(json.dumps(st))
    b = json.loads((A / "campaigns" / "stage2b_dry" / "baseline" / "summary.json").read_text())["w"]
    (camp / "baseline" / "w").mkdir(parents=True)
    (camp / "baseline" / "w" / "summary.json").write_text(json.dumps({"all": {str(q): d for q, d in zip(b["seeds"], b["devs"])}}))
    D = {"cdir": camp, "arms": {"final": {"searches": [{"pick": "r1c2", "verdict": "compliant", "primary_score": 0.1,
                                                        "search": 1, "lineage": "A", "path": "W/final/A/s1"}]}}}
    with contextlib.redirect_stdout(io.StringIO()):
        frames, ch = MV.scene3_frames(D, False)
    assert not ch["none"] and sum(times for _, times in frames) == MV.SECONDS[3] * MV.vk.FPS
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frames[-1][0]).save(OUT / "scene3_real_last.png")
    Image.fromarray(frames[len(frames) // 3][0]).save(OUT / "scene3_real_mid.png")
    return f"ok ({len(frames)} distinct frames, {MV.SECONDS[3]} s)"


if __name__ == "__main__":
    print("scene 3 on real W checkpoints:", test_scene3_real())
    res = prepare()
    args = ["--campaign", str(CAMP), "--out", str(OUT), "--skip-build"] + (["--video"] if "--video" in sys.argv else [])
    stops(lambda: MV.main(args), "the campaign has fewer than the 16 searches per arm the endings state")
    args.append("--allow-small")
    with contextlib.redirect_stdout(io.StringIO()):
        data = MV.main(args)
    assert MV.versus(0.4689, 0.4687) == "46.89% vs 46.87%" and MV.versus(0.5, 0.4) == "50.0% vs 40.0%"
    assert MV.versus(0.5, 0.5) == "50.0% vs 50.0% (equal)" and MV.plural(1, "seed") == "1 seed"
    for n in ("scene1_system", "scene2_program", "scene4_answer", "scene5_limits"):
        assert (OUT / f"{n}.png").exists(), n
    assert data["ending"] in ("recursive", "level2", "transfer_only", "none") and "{" not in data["video_line"]
    if "--video" in sys.argv:
        assert abs(data["video_seconds"] - sum(MV.SECONDS.values())) < 0.5, data["video_seconds"]
    # every ending fills in cleanly, including an untested P4
    c = res["primary_family"]["contrasts"]
    yes = {"tested": True, "diff": 0.21, "p_holm": 0.004, "ci95_boot": [0.1, 0.3]}
    no = {"tested": True, "diff": 0.03, "p_holm": 0.6, "ci95_boot": [-0.1, 0.2]}
    un = {"tested": False, "diff": None, "p_holm": None, "ci95_boot": None}
    for key, tier, cs in (("recursive", "recursive", (yes, yes, yes, yes)), ("level2", "Level 2", (yes, yes, yes, un)),
                          ("transfer_only", "none: report as measured", (yes, no, yes, no)),
                          ("none", "none: report as measured", (no, yes, no, un))):
        fake = {"tiers": {"words_allowed": tier}, "primary_family": {"contrasts": dict(zip(("P1", "P2", "P3", "P4"), cs))}}
        k, v, art = MV.ending(fake)
        assert k == key and "{" not in v + art, (key, k)
        assert ("recursive" in (v + art).lower()) == (key in ("recursive", "level2")), key
    assert "not tested: no lineage kept a rewrite" in MV.ending({"tiers": {"words_allowed": "Level 2"}, "primary_family": {
        "contrasts": dict(zip(("P1", "P2", "P3", "P4"), (yes, yes, yes, un)))}})[2]
    # the scene-3 rule: eligible picks only, by primary score, position floor(m/2)
    D = {"arms": {"final": {"searches": [
        {"pick": "r1c1", "verdict": "compliant", "primary_score": 0.5, "search": 1},
        {"pick": "start", "verdict": "compliant (by construction)", "primary_score": 0.9, "search": 2},
        {"pick": "r2c1", "verdict": "violation", "primary_score": 0.8, "search": 3},
        {"pick": "r3c2", "verdict": "compliant", "primary_score": 0.7, "search": 4},
        {"pick": "r1c3", "verdict": "compliant", "primary_score": 0.6, "search": 5}]}}}
    ch = MV.pick_scene3(D)
    assert ch["m"] == 3 and ch["position"] == 1 and ch["row"]["search"] == 5          # 0.7, [0.6], 0.5
    assert MV.pick_scene3({"arms": {}})["none"]
    # tampering stops the renderer
    prog = next(iter(sorted((CAMP / "programs" / "A").glob("g1.md"))))
    keep = prog.read_text()
    prog.write_text(keep + "\nextra line\n")
    stops(lambda: MV.load(CAMP), "a program file was edited")
    prog.write_text(keep)
    gj = json.loads((CAMP / "meta" / "A" / "gen0" / "generation.json").read_text())
    e = gj["searches"][0]
    rf = CAMP / "confirm" / "ws" / e["sha"] / f"seed{e['seeds'][0]}" / "result.json"
    keep = rf.read_text()
    rf.write_text(json.dumps(json.loads(keep) | {"dev_exact": 0.999}))
    stops(lambda: MV.load(CAMP), "a W_s confirmation run was edited")
    rf.write_text(keep)
    rj = json.loads((CAMP / "results.json").read_text())
    rj["per_arm"]["g1"]["primary_mean"] += 0.01
    (CAMP / "results.json").write_text(json.dumps(rj))
    stops(lambda: MV.load(CAMP), "an arm mean in results.json was edited")
    rj["per_arm"]["g1"]["primary_mean"] -= 0.01
    rj["primary_family"]["status"] = "pending review"
    (CAMP / "results.json").write_text(json.dumps(rj))
    stops(lambda: MV.load(CAMP), "results.json isn't final")
    (CAMP / "results.json").write_text(json.dumps(res, indent=1))
    MV.load(CAMP)
    print(f"all make_2c_video tests passed (ending shown: {data['ending']}; out: {OUT})")
