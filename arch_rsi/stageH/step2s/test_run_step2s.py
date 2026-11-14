"""Tests of run_step2s.py with fake training and a fake model (the model call itself, scientist.call_with_retries, is
replaced, so both runners' own call paths and Step 2S's no-repeat cache are exercised). Probes run for real on the GPU
(a few world steps each). Covers: the copied search() matching Step 2's for the scores arm, the sandbox round, saved
probes, a resume after a crash between the probe call and the proposal call, which worlds run, the refusals, and the
analysis. Never reads Step 2's real results: STEP2_RESULTS points to a made-up file.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2s/test_run_step2s.py   (exits 1 on any failure)
"""
import json
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2s_common as C  # noqa: E402

import probes_2s as PB  # noqa: E402
import prompts as P  # noqa: E402  (Step 2's)
import prompts_2s as PR  # noqa: E402
import run_step2 as RS  # noqa: E402  (Step 2's)
import run_step2s as RN  # noqa: E402

from arch_rsi import scientist as S  # noqa: E402

FAILED = []
REF = C.common.REF_A.read_text()
FIX = {"shift": "\n# FIX_SHIFT torch.roll\n", "mirror": "\n# FIX_MIRROR .flip(3)\n"}
TMP = Path(tempfile.mkdtemp())
LOCK = threading.Lock()
CALLS, FAIL = [], set()


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except AssertionError as e:
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, e)
    return fn


def step2_results(told_found, scores_found):
    """Made-up Step 2 results: per world, that many of 12 told / scores-only searches at 0.5, the rest at 0.1."""
    rows = []
    for w in C.C2.WORLDS:
        for arm, k in (("told", told_found[w]), ("scores", scores_found[w]), ("readouts", 0), ("hidden", 0)):
            rows += [{"world": w, "arm": arm, "search": i, "score": 0.5 if i < k else 0.1, "found": i < k} for i in range(12)]
    return {"status": "final", "preset": "step2", "rows": rows}


S2 = step2_results({"shift": 10, "mirror": 11}, {"shift": 2, "mirror": 0})
C.STEP2_RESULTS = TMP / "step2_results.json"
C.STEP2_RESULTS.write_text(json.dumps(S2))
QUAL2 = C.qualify(S2)


def fake_call(system, user, schema, validate, log_path, max_model_attempts=3):
    """The model: proposals carry a world's FIX marker at fixed places; probe replies are fixed by arm and round."""
    log_path = Path(log_path)
    with LOCK:
        if str(log_path) in FAIL:
            raise RuntimeError("fake model failure")
        CALLS.append(str(log_path))
    if system == RS.REV.SYSTEM:
        obj = {"verdict": "compliant", "rule": "x", "evidence": "y"}
    elif system == RS.JUDGE_SYSTEM:
        obj = {"verdict": "yes" if "told" in log_path.name else "no", "evidence": "z"}
    else:
        world, arm, s, name = log_path.parts[-4], log_path.parts[-3], log_path.parts[-2], log_path.stem
        r = int(name.split("_")[1])
        if name.endswith("_probes"):
            n = 0 if arm == "told_sandbox" else 2 if r == 1 else 1
            obj = {"probes": [{"design": 1 + (k + r) % 4, "purpose": f"probe {k} of round {r}",
                               "placements": [{"brick": "1x2", "row": 3, "col": 0}, {"brick": "1x1", "row": r, "col": k}]}
                              for k in range(n)]}
        else:
            repair = "repair" in name                      # the one broken proposal is repaired alone
            props = []
            for c in range(3 if not repair else 1):
                code = REF + f"\n# {arm} {s} r{r} c{c}{' repaired' if repair else ''}\n"
                if arm in ("told", "told_sandbox") and r == 1 and c == 0:
                    code += FIX[world]
                if arm in ("readouts", "sandbox") and r == 2 and c == 1:
                    code += FIX[world]
                if arm == "scores" and r == 1 and c == 2 and not repair:
                    code += "# BROKEN\n"
                props.append({"name": f"{arm} r{r}c{c}", "hypothesis": "neutral", "code": code})
            obj = {"analysis": f"analysis {arm} r{r}", "proposals": props}
    val = validate(obj)
    rec = {"prompt": user, "reply": obj, "usage": {"notional_cost_usd": 0.1}}
    log_path.write_text(json.dumps(rec))
    return val, rec


S.call_with_retries = fake_call


class FakeWork:
    """Fake GPU work: a file scores about 0.7 if it carries its world's FIX marker, else 0."""
    def verify(self):
        return {"ok": True}

    def _train_eval(self, rdir, world):
        spec = json.loads((rdir / "run.json").read_text())
        code = Path(spec["candidate"]).read_text()
        good = ("FIX_SHIFT" if world == "shift" else "FIX_MIRROR") in code
        dev = (0.70 + 0.01 * (spec["seed"] % 4)) if good else 0.0
        return {"status": "ok", "error": None, "dev_exact": dev, "train_exact": dev, "params": 1, "train_steps": 1,
                "flops_ratio": 1.0, "n_updates": 30, "recipe": None, "train_wall_s": 1.0}

    def _readouts(self, rdir, world, seed):
        return {"text_full": "FULL READOUT", "text_short": "SHORT READOUT"}

    def check(self, path, world):
        code = path.read_text()
        return {"ok": "BROKEN" not in code, "error": None if "BROKEN" not in code else "broken on purpose",
                "params": 1, "flops_ratio": 1.0}


class F2(FakeWork, RS.Step2):
    pass


class F2S(FakeWork, RN.Step2S):
    pass


def calls_under(d: Path):
    return [c for c in CALLS if c.startswith(str(d) + "/")]


def prompt_of(p: Path):
    return json.loads(p.read_text())["prompt"]


# ---- one fake campaign in both worlds, 2 searches per arm
FULL = F2S(TMP / "c", "step2s_dry", qual=QUAL2)
FULL.n_per_arm = {a: 2 for a in C.ARMS}
FULL.run()
RES = json.loads((TMP / "c" / "results.json").read_text())


@test
def init_sets_every_attribute_step2_sets():
    a = set(vars(RS.Step2(TMP / "a", "step2_dry")))
    b = set(vars(RN.Step2S(TMP / "b", "step2s_dry", qual=QUAL2)))
    assert a <= b, a - b


@test
def copied_search_matches_step2_for_the_scores_arm():
    g2, g2s = F2(TMP / "e2", "step2_dry"), F2S(TMP / "e2s", "step2s_dry", qual=QUAL2)
    for w in C.C2.WORLDS:
        s2, s2s = g2.start(w), g2s.start(w)
        assert s2 == s2s
        st2, st2s = g2.search(w, "scores", 0, s2), g2s.search(w, "scores", 0, s2s)
        assert json.dumps(st2, sort_keys=True) == json.dumps(st2s, sort_keys=True)
        d2, d2s = TMP / "e2" / w / "scores" / "s0", TMP / "e2s" / w / "scores" / "s0"
        n2 = sorted(p.name for p in d2.glob("round_*.json"))
        assert n2 == sorted(p.name for p in d2s.glob("round_*.json")) and "round_1_repair1.json" in n2
        for n in n2:
            assert prompt_of(d2 / n) == prompt_of(d2s / n), n


@test
def every_search_finishes_and_worlds_and_arms_are_right():
    ents = json.loads((TMP / "c" / "summary.json").read_text())["entries"]
    assert len(ents) == 12 and {(e["world"], e["arm"]) for e in ents} == {(w, a) for w in C.C2.WORLDS for a in C.ARMS}
    ents = {(e["world"], e["arm"], e["search"]): e for e in ents}
    for w in C.C2.WORLDS:
        assert ents[(w, "told_sandbox", 0)]["label"] == "r1c1" and ents[(w, "sandbox", 1)]["label"] == "r2c2"
        assert ents[(w, "scores", 0)]["label"] == "start"


@test
def sandbox_rounds_call_probes_first_and_show_results():
    for w in C.C2.WORLDS:
        for arm in C.SANDBOX_ARMS:
            d = TMP / "c" / w / arm / "s0"
            mine = [Path(c).name for c in calls_under(d)]
            for r in range(1, 5):
                assert mine.index(f"round_{r}_probes.json") < mine.index(f"round_{r}.json"), mine
                q, p = prompt_of(d / f"round_{r}_probes.json"), prompt_of(d / f"round_{r}.json")
                assert PR.SANDBOX_TEXT in q and "## Sandbox results" in p and PR.SANDBOX_TEXT not in p
                assert (P.RULE_TEXT[w] in p) == (arm == "told_sandbox") and (P.NOTE in p) == (arm == "sandbox")
                assert "READOUT" not in p
            assert "Round 3 probes (requests only):" in prompt_of(d / "round_4.json")
        d = TMP / "c" / w / "scores" / "s0"
        assert not list(d.glob("round_*_probe*.json")) and "Sandbox" not in prompt_of(d / "round_2.json")


@test
def saved_probe_results_match_a_fresh_run():
    import torch
    from rules import Rule
    for w in C.C2.WORLDS:
        saved = json.loads((TMP / "c" / w / "sandbox" / "s0" / "round_1_probe_results.json").read_text())
        assert len(saved["probes"]) == 2
        rule, designs = Rule(w, PB.SPEC, torch.device("mps")), PB.load_designs()
        fresh = [PB.run(rule, designs[p["design"] - 1], p["placements"]) for p in saved["probes"]]
        assert fresh == saved["results"]


@test
def analysis_probes_and_calls():
    assert RES["status"] == "final" and RES["worlds"] == ["mirror", "shift"]
    pu = RES["probes"]["per_arm"]
    assert pu["sandbox"]["probes_per_search"] == [5, 5, 5, 5] and pu["told_sandbox"]["searches_with_no_probes"] == 4
    assert pu["sandbox"]["mean_placements_per_probe"] == 2
    assert RES["digbench_prediction"]["diff_told_minus_sandbox"] == -5 and RES["digbench_prediction"]["as_predicted"]
    assert RES["calls"]["sandbox"]["calls"] == 32 and RES["calls"]["told_sandbox"]["calls"] == 32
    assert RES["calls"]["scores"]["calls"] == 20 and RES["calls"]["judge"]["calls"] == 12
    assert RES["precondition"] == {"found": 4, "n": 4, "needed": 3, "holds": True}
    assert RES["primary"]["P1S"]["diff"] > 0 and RES["per_arm"]["sandbox"]["found"] == 4
    assert RES["drift"]["n"] == [4, 24] and RES["drift"]["found"] == [0, 2]
    assert RES["ending"] in ("pass", "fail") and RES["ending_text"].endswith("whose weights never changed.")


@test
def resume_after_a_crash_between_probe_call_and_proposal_repeats_no_call():
    d = TMP / "r"
    FAIL.add(str(d / "shift" / "sandbox" / "s0" / "round_3.json"))
    G = F2S(d, "step2s_dry", qual=QUAL2)
    try:
        G.run()
    except SystemExit:
        pass
    FAIL.clear()
    sd = d / "shift" / "sandbox" / "s0"
    assert json.loads((sd / "state.json").read_text())["rounds_done"] == 2
    assert (sd / "round_3_probes.json").exists() and (sd / "round_3_probe_results.json").exists()
    before = len(CALLS)
    F2S(d, "step2s_dry", qual=QUAL2).run()
    again = CALLS[before:]
    mine = [Path(c).name for c in again if c.startswith(str(sd) + "/")]
    assert mine == ["round_3.json", "round_4_probes.json", "round_4.json"], mine
    others = [c for c in again if not c.startswith(str(sd) + "/")]
    assert all("/review/" in c or "/judge/" in c for c in others), others
    assert (d / "results.json").exists()


@test
def a_saved_call_is_reused_only_for_the_same_prompt():
    f = TMP / "cache.json"
    f.write_text(json.dumps({"prompt": "PROMPT A", "reply": {"probes": []}}))
    g = RN.Step2S(TMP / "k", "step2s_dry", qual=QUAL2)
    n = len(CALLS)
    assert g.call_probes("PROMPT A", f)[0] == [] and len(CALLS) == n
    try:
        g.call_probes("PROMPT B", f)
        assert False, "reused a call made for another prompt"
    except SystemExit:
        pass


@test
def which_worlds_run():
    q = C.qualify(step2_results({"shift": 9, "mirror": 8}, {"shift": 8, "mirror": 0}))
    assert q["worlds"] == ["shift"] and q["searches_per_world"] == {"sandbox": 24, "scores": 24, "told_sandbox": 12}
    q = C.qualify(step2_results({"shift": 12, "mirror": 12}, {"shift": 9, "mirror": 3}))
    assert q["worlds"] == ["mirror"] and not q["per_world"]["shift"]["room_left"]
    assert QUAL2["worlds"] == ["shift", "mirror"] and QUAL2["searches_per_world"]["told_sandbox"] == 6
    none = C.qualify(step2_results({"shift": 8, "mirror": 12}, {"shift": 0, "mirror": 9}))
    assert none["worlds"] == []
    for bad in (dict(S2, status="running"), dict(S2, preset="step2_dry"), dict(S2, rows=S2["rows"][1:])):
        try:
            C.qualify(bad)
            assert False, "accepted"
        except SystemExit:
            pass
    try:
        RN.Step2S(TMP / "n", "step2s_dry", qual=none)
        assert False, "ran with no world"
    except SystemExit as e:
        assert "no world qualifies" in str(e)


@test
def refusals():
    real = C.STEP2_RESULTS
    C.STEP2_RESULTS = TMP / "missing.json"
    try:
        RN.Step2S(TMP / "m", "step2s_dry")
        assert False, "ran without Step 2's results"
    except SystemExit as e:
        assert "waits" in str(e)
    finally:
        C.STEP2_RESULTS = real
    real = C.STEP2S
    C.STEP2S = TMP                                          # an empty folder: no LAUNCH_APPROVED.md, no manifest
    try:
        try:
            RN.Step2S(TMP / "x", "step2s", qual=QUAL2).run()
            assert False, "ran without approval"
        except SystemExit as e:
            assert "LAUNCH_APPROVED" in str(e)
        assert not RN.hash_checks()["ok"]                   # no Step 2S manifest
    finally:
        C.STEP2S = real


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
