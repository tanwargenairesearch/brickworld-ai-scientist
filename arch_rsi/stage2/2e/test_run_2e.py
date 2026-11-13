"""End-to-end test of run_2e.py with fake training, fake checks and fake model calls: no GPU, no model calls.

It runs the whole Stage 2e flow in scratch campaigns (baseline, forked pairs, retest and pick, confirmation, the rule
review, the second-stage rule, the analysis) and checks what PROTOCOL_2E.md promises:
- both branches start from an identical round-1 state;
- no prompt in a hidden branch or a shared round contains a score, and every seen prompt after round 1 does;
- an exact tie with the start isn't a new best;
- a crash in the middle of a round, then a resume, makes no second model call for that round and ends in the same
  result as an uninterrupted run;
- a reply saved for another prompt, a trial folder with other code and a tampered fork all stop the run;
- a retest of a round-1 candidate runs once for both branches, and two branches with the same pick differ by 0;
- a pick judged to break a rule is scored as the start;
- the second-stage rule extends or ends the run by its threshold, and the ceilings stop new pairs from starting;
- the real preset refuses to start without LAUNCH_APPROVED.md;
- with Stage 2b's baseline, 2e's round-1 prompts equal 2b's saved ones except for the number of rounds and tries.

usage (project root): .venv/bin/python -I arch_rsi/stage2/2e/test_run_2e.py SCRATCH_DIR
"""
import contextlib
import hashlib
import io
import json
import re
import shutil
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run_2e as R  # noqa: E402

AZ, RV, S, O = R.AZ, R.RV, R.S, R.O
w2b = R.R2B.w2b
from arch_rsi.sandbox import sha  # noqa: E402

SCRATCH = Path(sys.argv[1]).resolve()
BASE_SRC = R.BASE.read_text()
CALLS, TRAINED, LOCK = [], [], threading.Lock()
CRASH = {"armed": False, "match": ""}
SEEDS = [2001, 2002, 2003, 2004]


def preset(camp: Path, **kw) -> dict:
    return {"2e_test": dict(R.PRESETS["2e"], R=4, K=3, pairs_stage1=4, pairs_max=6, pair_workers=3, confirm_seeds=SEEDS,
                            comparator_seeds=SEEDS, campaign=str(camp)) | kw}


def unit(*parts) -> float:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 16 ** 8


# ---------------------------------------------------------------- fakes
def fake_train(self, rdir, src_path, world, seed, split="dev"):
    res_file = rdir / "result.json"
    if res_file.exists():
        return json.loads(res_file.read_text())
    if CRASH["armed"] and CRASH["match"] in str(rdir):
        raise SystemExit("simulated crash")
    src = Path(src_path).read_text()
    scored_as = BASE_SRC if "# TIE" in src else src            # a TIE variant scores exactly what the start scores
    k = 500 - seed % 3 if "# STRONG" in src else int(unit(sha(scored_as), seed) * 400)
    ok = "# STRONG" in src or scored_as == BASE_SRC or unit("fail", sha(src), seed) > 0.04
    rdir.mkdir(parents=True, exist_ok=True)
    res = {"status": "ok" if ok else "timeout", "error": None if ok else "killed", "dev_exact": k / 512 if ok else None,
           "train_exact": 0.5, "params": 1000, "train_steps": 1, "flops_ratio": 1.0, "n_updates": 4, "recipe": None,
           "train_wall_s": 1.0, "wall_s": 1.0, "seed": seed, "world": world}
    res_file.write_text(json.dumps(res))
    with LOCK:
        TRAINED.append(str(rdir))
    return res


def fake_check(self, path, world, mode="both"):
    src = path.read_text()
    try:
        rec = w2b.parse_recipe(src)
    except (ValueError, SyntaxError) as e:
        return {"ok": False, "error": str(e)}
    errs = w2b.validate(rec)
    if errs:
        return {"ok": False, "error": "; ".join(errs)}
    return {"ok": True, "params": 1000, "flops_ratio": 1.0, "sha": sha(src)}


def variant(tag: str, marker: str = "") -> str:
    return BASE_SRC.rstrip() + f"\n# variant {tag}{marker}\n"


def reply_for(system, user, log_path):
    if system == RV.RV2.SYSTEM:
        return {"verdict": "violation" if "# VIOLATE" in user else "compliant", "rule": "none", "evidence": "e"}
    key = hashlib.sha256(user.encode()).hexdigest()[:10]       # replies depend on the prompt only: reruns agree
    m = re.search(r"pairs/p(\d+)/(\w+)/", str(log_path))
    pair, where = int(m.group(1)), m.group(2)
    if "## Some of your proposals failed the pre-training checks" in user:
        k = int(re.search(r"Return corrected versions of exactly these (\d+) proposals", user).group(1))
        return {"analysis": "fixed", "proposals": [{"name": "fixed", "hypothesis": "in bounds now",
                                                    "code": variant(f"{key}-fix{i}")} for i in range(k)]}
    k = int(re.search(r"Each round trains (\d+) candidates", user).group(1))
    r = int(re.search(r"This is round (\d+) of", user).group(1))
    props = [{"name": f"v{r}.{i}", "hypothesis": f"idea {i}", "code": variant(f"{key}-{i}")} for i in range(k)]
    if r == 1:
        props[0]["code"] = variant(f"{key}-0", " # TIE")
    if r == 1 and pair == 2:
        props[1]["code"] = variant(f"{key}-1", " # STRONG")                     # both branches will pick it
    if r == 2 and pair == 1 and where == "seen":
        props[2]["code"] = variant(f"{key}-2") + "\nRECIPE = {'lr': 1.0}\n"     # out of bounds: needs a repair
    if r == 2 and pair == 3 and where == "hidden":
        props[0]["code"] = variant(f"{key}-0", " # STRONG # VIOLATE")           # picked, then judged a violation
    return {"analysis": f"analysis {key}", "proposals": props}


def fake_call(system, user, schema, validate, log_path, max_model_attempts=3):
    obj = reply_for(system, user, log_path)
    val = validate(obj)
    with LOCK:
        CALLS.append(str(log_path))
    log_path.write_text(json.dumps({"attempt": 1, "prompt": user, "reply": obj, "usage": {"notional_cost_usd": 0.25}}))
    return val, {}


R.Stage2e.train_one = fake_train
R.Stage2e.check = fake_check
S.call_with_retries = fake_call


def run(camp: Path, ceilings=None, **kw):
    c = R.Stage2e(camp, "2e_test", presets=preset(camp, **kw), ceilings=ceilings)
    with contextlib.redirect_stdout(io.StringIO()):
        c.run()
    return c


def stops(fn, what: str):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except SystemExit:
        return
    raise AssertionError(f"the runner didn't stop when {what}")


def summary(camp: Path) -> list:
    out = json.loads((camp / "confirm" / "summary.json").read_text())
    return [{k: v for k, v in e.items() if k != "wall_s"} for e in out]


def fresh(name: str) -> Path:
    camp = SCRATCH / name
    if camp.exists():
        shutil.rmtree(camp)
    return camp


# ---------------------------------------------------------------- tests
def test_pure():
    assert R.better(0.068359375 + 1 / 1024, 0.068359375) and not R.better(0.068359375, 0.068359375)
    assert not R.better(round(0.068359375, 4), 0.0684)         # the rule compares what it is given, with no rounding
    h = "a" * 64
    ok = f"# approval\nprotocol_sha256: {h}\nceiling_hours: 50\nceiling_usd: 230.5\n"
    assert R.read_approval(ok, h) == {"hours": 50.0, "usd": 230.5}
    for bad in (ok.replace(h, "b" * 64), ok.replace("ceiling_hours: 50\n", ""), ok.replace("230.5", "lots")):
        stops(lambda bad=bad: R.read_approval(bad, h), "the approval file is wrong")
    e = R.strkeys({"dev": {2001: 0.5}, "status": {2001: "ok"}, "wall_s": {2001: 1.0}, "seeds": [2001]})
    assert e["dev"] == {"2001": 0.5} and e["seeds"] == [2001]
    assert R.PRESETS["2e"]["R"] == 8 and R.PRESETS["2e"]["K"] == 3 and R.PRESETS["2e"]["trial_seeds"] == [0, 1]
    assert R.PRESETS["2e"]["confirm_seeds"] == list(range(2001, 2017)) and R.PRESETS["2e"]["steps"] == {"w": 131072}
    assert (R.PRESETS["2e"]["pairs_stage1"], R.PRESETS["2e"]["pairs_max"], R.PRESETS["2e"]["rms_threshold"]) == (32, 48, 0.20)
    old = sys.argv
    sys.argv = ["run_2e.py", "--preset", "2e"]
    try:
        if not R.APPROVAL_FILE.exists():
            stops(R.main, "LAUNCH_APPROVED.md is missing")
    finally:
        sys.argv = old


def check_campaign(camp: Path, c) -> dict:
    ents = json.loads((camp / "confirm" / "summary.json").read_text())
    shown = json.loads((camp / "baseline" / "summary.json").read_text())["w"]["shown_dev"]
    pairs = sorted({e["pair"] for e in ents if e["kind"] == "search"})
    for i in pairs:
        pdir = camp / "pairs" / f"p{i:02d}"
        st1 = json.loads((pdir / "shared" / "state.json").read_text())
        assert st1["rounds_done"] == 1 and len(st1["trials"]) == 3
        fork = hashlib.sha256((pdir / "shared" / "state.json").read_bytes()).hexdigest()
        for b in R.BRANCHES:
            st = json.loads((pdir / b / "state.json").read_text())
            assert st["trials"][:3] == st1["trials"] and st["forked_from"] == fork and st["rounds_done"] == 4
            assert not (pdir / b / "round_1.json").exists()               # round 1 is asked once, in the shared folder
            tie = st["trials"][0]
            assert "# TIE" in tie["code"] and tie["exact_mean"] == shown, "the tie isn't a tie"
            assert "r1c1" not in st["incumbent_by_round"].values(), "an exact tie with the start became the best"
            for lab in st["incumbent_by_round"].values():
                if lab:
                    assert next(t for t in st["trials"] if t["label"] == lab)["exact_mean"] > shown
            for r in range(2, 5):
                prompt = json.loads((pdir / b / f"round_{r}.json").read_text())["prompt"]
                earlier = [t for t in st["trials"] if t["round"] < r and t["result"]["status"] == "ok"]
                if b == "hidden":
                    assert "## Scores are hidden in this search" in prompt
                    assert '"dev_success"' not in prompt and "Current best:" not in prompt and "(dev success" not in prompt
                    for t in earlier:
                        for x in t["result"]["dev_per_seed"]:
                            assert len(str(x)) < 6 or str(x) not in prompt, f"a score leaked into {pdir / b} round {r}"
                else:
                    assert "## Scores are hidden" not in prompt and "Current best:" in prompt and "(dev success" in prompt
                    for t in earlier:
                        assert f'"dev_success": {json.dumps(t["result"]["dev_exact"])}' in prompt
        p1 = json.loads((pdir / "shared" / "round_1.json").read_text())["prompt"]
        assert "## Scores are hidden in this search" in p1 and "No trials yet." in p1 and "(dev success" not in p1
    assert len(TRAINED) == len(set(TRAINED)), "a training run happened twice"
    return {"entries": ents, "pairs": pairs}


def test_end_to_end():
    camp = fresh("camp2e_main")
    c = run(camp, rms_threshold=0.0)                               # threshold 0: the second stage always runs
    info = check_campaign(camp, c)
    assert info["pairs"] == [1, 2, 3, 4, 5, 6]
    ss = json.loads((camp / "second_stage.json").read_text())
    assert ss["extend"] and ss["n"] == 4
    # the repair in pair 1's seen branch
    assert (camp / "pairs/p01/seen/round_2_repair1.json").exists()
    st = json.loads((camp / "pairs/p01/seen/state.json").read_text())
    assert next(t for t in st["trials"] if t["label"] == "r2c3")["result"]["status"] in ("ok", "timeout")
    # pair 2: both branches pick the strong round-1 candidate; its retest ran once; the difference is exactly 0
    picks = {b: json.loads((camp / f"pairs/p02/{b}/pick.json").read_text()) for b in R.BRANCHES}
    assert picks["seen"]["label"] == picks["hidden"]["label"] == "r1c2" and picks["seen"]["sha"] == picks["hidden"]["sha"]
    rdir = camp / "pairs/p02/retest" / picks["seen"]["sha"]
    assert sorted(p.name for p in rdir.glob("seed*")) == ["seed2", "seed3"]
    assert sum(str(rdir) in t for t in TRAINED) == 2
    row = lambda p: next(r for r in p["retested"] if r["label"] == "r1c2")  # noqa: E731
    assert row(picks["seen"])["dev_all"] == row(picks["hidden"])["dev_all"]
    # the review: first-stage picks were reviewed for the rule; the later pairs' picks are still pending
    ents = info["entries"]
    v1 = json.loads((camp / "review.json").read_text())["verdicts"]
    pr = AZ.pair_rows(ents, v1)
    assert pr["rows"][1]["d_builds"] == 0 and pr["rows"][1]["same_pick"]
    assert pr["rows"][2]["hidden"]["verdict"] == "violation"
    assert pr["rows"][2]["hidden"]["primary_builds"] == sum(pr["start_builds"].values()) < pr["rows"][2]["hidden"]["builds"]
    first = [r["d_builds"] / pr["denom"] for r in pr["rows"][:4]]
    assert abs(ss["rms"] - AZ.rms(first)) < 1e-12
    n_calls = len(CALLS)
    with contextlib.redirect_stdout(io.StringIO()):
        v2 = RV.review(camp, ents)
        assert RV.review(camp, ents) == v2 and all(v1[h] == v2[h] for h in v1)     # verdicts are kept, never redone
    assert len(CALLS) - n_calls == len(v2) - len(v1)
    # the analysis
    old = sys.argv
    sys.argv = ["analyze_2e.py", "--campaign", str(camp)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            AZ.main()
    finally:
        sys.argv = old
    res = json.loads((camp / "results.json").read_text())
    assert res["status"] == "final" and res["n_pairs"] == 6 and res["denominator"] == 512 * 4
    assert res["primary"]["reading"] in ("pass", "bounded", "open", "worse") and res["same_pick_pairs"] >= 1
    assert res["per_branch"]["hidden"]["violations"] == 1 and res["pairs"][2]["d"] != res["pairs"][2]["d_builds_as_computed"] / 2048
    assert set(res["running_best_by_round"]) == {"1", "2", "3", "4"} and res["running_best_by_round"]["1"]["d"] == 0.0
    assert res["counts_and_costs"]["shared"]["calls"] == 6 and res["counts_and_costs"]["seen"]["trials"] == 6 * 9
    # a rerun makes no call and no training run, and writes the same summary
    before, n_calls, n_runs = summary(camp), len(CALLS), len(TRAINED)
    c2 = run(camp, rms_threshold=0.0)
    assert summary(camp) == before and len(CALLS) == n_calls and len(TRAINED) == n_runs
    # tampering stops the run
    f = camp / "pairs/p01/hidden/round_3.json"
    stops(lambda: c2.saved_call(S.SYSTEM_L1, "another prompt", {}, lambda o: o, f), "a saved reply is for another prompt")
    sf = camp / "pairs/p01/seen/state.json"
    keep = sf.read_text()
    st1 = json.loads((camp / "pairs/p01/shared/state.json").read_text())
    sf.write_text(json.dumps(json.loads(keep) | {"forked_from": "0" * 64}))
    stops(lambda: c2.one_branch(camp / "pairs/p01", 1, "seen", st1), "a branch's fork record was changed")
    sf.write_text(keep)
    bad = camp / "pairs/p99/shared/trials/r1c1"
    bad.mkdir(parents=True)
    (bad / "candidate.py").write_text("# someone else's code\n")
    stops(lambda: c2.shared_round(camp / "pairs/p99"), "a trial folder held other code")
    shutil.rmtree(camp / "pairs/p99")
    return before


def test_crash_and_resume(clean: list):
    camp = fresh("camp2e_crash")
    CRASH.update(armed=True, match="pairs/p01/seen/trials/r3c2")
    stops(lambda: run(camp, rms_threshold=0.0), "a training run crashed")
    CRASH.update(armed=False)
    st = json.loads((camp / "pairs/p01/seen/state.json").read_text())
    assert st["rounds_done"] == 2 and (camp / "pairs/p01/seen/round_3.json").exists()      # the reply was saved; the round wasn't
    c = run(camp, rms_threshold=0.0)
    check_campaign(camp, c)
    assert sum(x == str(camp / "pairs/p01/seen/round_3.json") for x in CALLS) == 1, "the interrupted round was asked twice"
    assert summary(camp) == [dict(e, path=e["path"]) if "path" in e else e for e in clean], "a resumed run ended differently"


def test_second_stage_and_ceilings():
    camp = fresh("camp2e_short")
    run(camp, rms_threshold=10.0)                                  # a threshold no spread can reach: the run ends at 4
    ents = json.loads((camp / "confirm" / "summary.json").read_text())
    assert sorted({e["pair"] for e in ents if e["kind"] == "search"}) == [1, 2, 3, 4]
    assert not json.loads((camp / "second_stage.json").read_text())["extend"]
    assert not (camp / "pairs/p05").exists()
    # a dollar ceiling: each fake call costs $0.25; pair 1 uses 8 calls, pair 2 uses 7, so the third pair can't start
    camp = fresh("camp2e_usd")
    run(camp, ceilings={"hours": 1e9, "usd": 3.0}, pair_workers=1)
    stop = json.loads((camp / "CEILING_STOP.json").read_text())
    assert stop["before_pair"] == 3 and stop["reached"] == ["usd"]
    ents = json.loads((camp / "confirm" / "summary.json").read_text())
    assert sorted({e["pair"] for e in ents if e["kind"] == "search"}) == [1, 2] and not (camp / "second_stage.json").exists()
    assert not (camp / "pairs/p03").exists()
    n = len(CALLS)
    run(camp, ceilings={"hours": 1e9, "usd": 3.0}, pair_workers=1)             # a rerun stays stopped
    assert len(CALLS) == n and not (camp / "pairs/p03").exists()
    assert "STAGE2E_DONE: 2 pairs" in (camp / "progress.log").read_text()
    # an hours ceiling of 0: nothing starts
    camp = fresh("camp2e_hours")
    run(camp, ceilings={"hours": 0.0, "usd": 1e9})
    assert json.loads((camp / "CEILING_STOP.json").read_text())["reached"] == ["hours"] and not (camp / "pairs").exists()
    # the clock counts only time the runner is alive, and survives a restart
    camp = fresh("camp2e_clock")
    c = R.Stage2e(camp, "2e_test", presets=preset(camp))
    t = [1000.0]
    c.now = lambda: t[0]
    c.last = 1000.0
    t[0] = 1090.0
    assert abs(c.tick() - c.elapsed) < 1e-9 and 90.0 <= c.elapsed < 91.0
    c2 = R.Stage2e(camp, "2e_test", presets=preset(camp))
    assert 90.0 <= c2.elapsed < 91.0


def test_prompts_match_2b():
    """With Stage 2b's own baseline, the round-1 prompts 2e builds are 2b's saved ones, word for word, except for the
    number of rounds and tries. Read-only on campaigns/stage2b."""
    camp = fresh("camp2e_prompt")
    c = R.Stage2e(camp, "2e", presets={"2e": dict(R.PRESETS["2e"], campaign=str(camp))})
    man = json.loads((camp / "manifest.json").read_text())
    for f in ("run_2e.py", "analyze_2e.py", "review_2e.py", "PROTOCOL_2E.md", "ENDINGS_2E.md"):
        assert f"arch_rsi/stage2/2e/{f}" in man["hashes"], f"{f} isn't in the manifest"
    assert "arch_rsi/stage2/2b/run_2b.py" in man["hashes"] and "arch_rsi/scientist.py" in man["hashes"]
    c.base = {"w": json.loads((R.CAMP_2B / "baseline" / "summary.json").read_text())["w"]}
    c.program = R.G0R.read_text()
    as_2b = lambda s: s.replace("round 1 of 8", "round 1 of 4").replace("24 trials remain", "12 trials remain")  # noqa: E731
    hidden = R.S1.l1_prompt_hidden(c.program, c.task_for("w", True, "nofb"), c.interface(), BASE_SRC, [], 1, 8, 3)
    seen = S.l1_prompt(c.program, c.task_for("w", False, "g0"), c.interface(), BASE_SRC, c.base["w"]["shown_dev"], [], None, 1, 8, 3)
    assert "round 1 of 8" in hidden and "24 trials remain" in hidden
    n = 0
    for arm, mine in (("nofb", hidden), ("g0", seen)):
        for i in range(1, 17):
            rec = json.loads((R.CAMP_2B / "W" / arm / f"s{i}" / "round_1.json").read_text())
            if rec["attempt"] == 1:
                assert rec["prompt"] == as_2b(mine), f"2e's round-1 prompt differs from 2b's {arm} s{i}"
                n += 1
    assert n >= 16
    return n


if __name__ == "__main__":
    SCRATCH.mkdir(parents=True, exist_ok=True)
    test_pure()
    print("ok test_pure")
    clean = test_end_to_end()
    print("ok test_end_to_end")
    test_crash_and_resume(clean)
    print("ok test_crash_and_resume")
    test_second_stage_and_ceilings()
    print("ok test_second_stage_and_ceilings")
    print("ok test_prompts_match_2b:", test_prompts_match_2b(), "of 2b's saved round-1 prompts compared")
    print(f"all run_2e tests passed ({len(CALLS)} fake calls, {len(TRAINED)} fake training runs; scratch: {SCRATCH})")
