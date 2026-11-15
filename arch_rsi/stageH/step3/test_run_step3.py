"""Tests of run_step3.py with fake training and a fake model (no GPU, no model calls): the copies of Step 2's code,
the header obeyed, generations with keep and reject, the parent-header fallback, both controls, the meta-test schedule
and shared final arm, the check arm's readouts, neutral labels and separation (no rewrite reads a held-out search),
no rule named in any harness prompt, resume after crashes without repeating a call, the launch guard, and the analysis.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3/test_run_step3.py   (exits 1 on any failure)
"""
import inspect
import json
import re
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3common as C  # noqa: E402
import prog3 as PG  # noqa: E402
import prompts3 as Q  # noqa: E402
import run_step3 as R3  # noqa: E402
import analyze_step3 as AN  # noqa: E402

import run_step2 as RS  # noqa: E402
import rules3a  # noqa: E402
from arch_rsi import scientist as S  # noqa: E402

FAILED = []
REF = C.common.REF_A.read_text()
G0R = C.PROGRAM_G0R.read_text()
IDX = {w: i for i, w in enumerate(rules3a.RULES)}
TMP = Path(tempfile.mkdtemp())
CALLS, LOCK, FAIL = [], threading.Lock(), {"at": None}
AN.analyze.__defaults__ = (2000,)          # fewer shuffles in tests


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except Exception as e:  # noqa: BLE001
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, type(e).__name__, e)
    return fn


def world_of(parts):
    return next(p for p in parts if p in rules3a.RULES)


def proposals(prompt, relp):
    world = world_of(relp)
    r = int(re.search(r"This is round (\d+) of (\d+)", prompt).group(1))
    repair = "repair" in relp[-1]
    K = (int(re.search(r"Return corrected versions of exactly these (\d+) proposals", prompt).group(1)) if repair
         else int(re.search(r"Each round trains (\d+) candidates", prompt).group(1)))
    prog = prompt.split("## Your research program\n", 1)[1].split("\n## Task", 1)[0]
    fix = (1 if "FULL READOUT" in prompt else 2 if "FAKE-PROCEDURE" in prog
           else 3 if "FAKE-FACT" in prog and world == "shift" else None)
    better = "FAKE-BETTER" in prog
    props = []
    for c in range(1, K + 1):
        code = REF + f"\n# w{IDX[world]} r{r} c{c}{' repaired' if repair else ''}\n"
        if fix == r and c == 1 and not repair:
            code += f"# FIX_{IDX[world]}\n" + ("# BETTER\n" if better else "")
        if relp[0] == "meta" and world == "mirror" and r == 1 and c == 2 and not repair:
            code += "# BROKEN\n"
        props.append({"name": f"idea r{r}c{c}", "hypothesis": "a plain idea", "code": code})
    note = f"MARKTEST{IDX[world]}" if world in C.TEST_WORLDS else "meta note"
    return {"analysis": f"analysis r{r} {note}", "proposals": props}


def rewrite_objs(prompt, name, L):
    first = prompt.index("\n")
    parent = prompt[first + 1:prompt.index("\n\n## How the program is used")] + "\n"
    h, body = PG.parse_program(parent)
    base = {"lessons": "l", "changes": [{"change": "c", "evidence": "e"}], "expected_effect": "x"}

    def prog(hdr, b):
        return PG.render_header(hdr) + "\n" + b
    if name == "call_g1":
        hdr = {"rounds": 6, "candidates_per_round": 4, "seeds_per_candidate": 1} if L == "A" else h
        return [dict(base, program=prog(hdr, body.rstrip("\n") + "\n\n## Searching new worlds\nFAKE-PROCEDURE: one idea per candidate.\n"))]
    if name == "call_facts":
        return [dict(base, program=parent.rstrip("\n") + "\n\n## Facts\n- FAKE-FACT: every candidate scored 0.\n")]
    if name == "call_nologs":
        return [dict(base, program=prog(h, body.rstrip("\n") + "\n\n## Extra\nFAKE-NOLOGS: think harder.\n"))]
    if name == "call_g2" and L == "A":            # three bad headers: the parent's header is used
        return [dict(base, program="<!-- header\nrounds: 9\ncandidates_per_round: 3\nseeds_per_candidate: 2\n-->\n\n" + body)] * 3
    if name == "call_g2":
        return [dict(base, program=prog({"rounds": 3, "candidates_per_round": 4, "seeds_per_candidate": 2},
                                         body.rstrip("\n") + "\nFAKE-BETTER.\n"))]
    return [dict(base, program=parent)]


def fake_cwr(system, user, schema, validate, log_path, max_model_attempts=3):
    rel = Path(log_path).relative_to(TMP).parts
    camp, relp = rel[0], rel[1:]
    if FAIL["at"] and "/".join(relp) == FAIL["at"]:
        raise RuntimeError("fake model failure")
    with LOCK:
        CALLS.append((camp,) + tuple(relp))
    if system == S.SYSTEM_L1:
        objs = [proposals(user, relp)]
    elif system == S.SYSTEM_L2:
        objs = rewrite_objs(user, Path(log_path).stem, relp[1])
    elif system == RS.REV.SYSTEM:
        objs = [{"verdict": "compliant", "rule": "none", "evidence": "fine"}]
    elif system == RS.JUDGE_SYSTEM:
        objs = [{"verdict": "no", "evidence": "none"}]
    elif system == R3.RV2C.PROGRAM_SYSTEM:
        objs = [{"verdict": "no rule-breaking advice", "quotes": "none", "explanation": "fine"}]
    else:
        raise AssertionError("unknown system prompt")
    errs = []
    for n, obj in enumerate(objs, 1):
        try:
            val = validate(obj)
        except (ValueError, TypeError, KeyError) as e:
            errs.append(str(e))
            continue
        rec = {"prompt": user, "reply": obj, "usage": {"notional_cost_usd": 0.1}, "attempt": n, "rejected_attempts": errs}
        Path(log_path).write_text(json.dumps(rec))
        return val, rec
    raise RuntimeError("fake model failed: " + "; ".join(errs))


S.call_with_retries = fake_cwr


class FakeIO:
    """Fake training, readouts and checks. A run scores about 0.715 if its file carries its world's FIX marker
    (0.815 with BETTER), else 0."""

    def verify(self):
        return {"ok": True}

    def _train_eval(self, rdir, world):
        spec = json.loads((rdir / "run.json").read_text())
        code = Path(spec["candidate"]).read_text()
        good = f"# FIX_{IDX[spec['rule']]}\n" in code
        dev = ((0.80 if "# BETTER" in code else 0.70) + 0.01 * (spec["seed"] % 4)) if good else 0.0
        return {"status": "ok", "error": None, "dev_exact": dev, "train_exact": dev, "params": 1, "train_steps": 1,
                "flops_ratio": 1.0, "n_updates": 30, "recipe": None, "train_wall_s": 1.0}

    def _readouts(self, rdir, world, seed):
        return {"text_full": "FULL READOUT", "text_short": "SHORT READOUT"}

    def check(self, path, world):
        bad = "BROKEN" in path.read_text()
        return {"ok": not bad, "error": "broken on purpose" if bad else None, "params": 1, "flops_ratio": 1.0}


class Fake(FakeIO, R3.Step3):
    pass


class Fake2(FakeIO, RS.Step2):
    pass


SMALL = dict(R3.BASE, campaign="unused", needs_launch=False, gpu_slots=4, meta_workers=4, test_workers=4)
R3.PRESETS["fake"] = dict(SMALL, lineages=["A", "B"], rewrites=2, train_worlds=["mirror", "rot90"],
                          test_worlds=["shift", "rowshift"], g0_per_world=2, per_lineage_world=1, check_per_world=2)
R3.PRESETS["fake_small"] = dict(SMALL, lineages=["A"], rewrites=1, train_worlds=["mirror"], test_worlds=["shift"],
                                g0_per_world=1, per_lineage_world=1, check_per_world=1)
F = Fake(TMP / "c", "fake")
F.run()
CDIR = TMP / "c"
RES = json.loads((CDIR / "results.json").read_text())
SUMM = json.loads((CDIR / "summary.json").read_text())
ENT = [e for e in SUMM["entries"]]


def by(arm, **kw):
    return [e for e in ENT if e["arm"] == arm and all(e.get(k) == v for k, v in kw.items())]


@test
def copies_differ_from_step2_only_in_script_paths():
    pairs = {"_train_eval": [('C.STAGEH / "train_h.py"', 'C.STEP3A / "train_3a.py"'), ('C.STAGEH / "eval_h.py"', 'C.STEP3A / "eval_3a.py"')],
             "_readouts": [('C.STEP2 / "readouts.py"', 'C.STEP3 / "readouts_3.py"')],
             "check": [('C.STAGEH / "check_h.py"', 'C.STEP3A / "check_3a.py"')]}
    for name, subs in pairs.items():
        src = inspect.getsource(getattr(RS.Step2, name))
        for a, b in subs:
            assert src.count(a) == 1, (name, a)
            src = src.replace(a, b)
        assert src == inspect.getsource(getattr(R3.Step3, name)), name
    assert set(vars(RS.Step2(TMP / "x2", "step2_dry"))) <= set(vars(R3.Step3(TMP / "x3", "step3_dry")))


@test
def g0_search_is_step2s_search():
    start = {"raw": [0.0, 0.0, 0.0], "code": REF, "readouts": None}
    v = Q.start_view(start, 2)
    f2 = Fake2(TMP / "s2", "step2_dry")
    st2 = f2.search("mirror", "scores", 0, v)
    f3 = Fake(TMP / "s3", "fake")
    st3 = f3.search3(TMP / "s3" / "eq" / "mirror", "mirror", "scores", G0R, PG.settings(PG.G0_HEADER), v)

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(x) for k, x in o.items() if k != "wall_s"}
        return [strip(x) for x in o] if isinstance(o, list) else o
    assert strip(st2) == strip(st3)
    for r in range(1, 5):
        p2 = json.loads((TMP / "s2" / "mirror" / "scores" / "s0" / f"round_{r}.json").read_text())["prompt"]
        p3 = json.loads((TMP / "s3" / "eq" / "mirror" / f"round_{r}.json").read_text())["prompt"]
        assert p2 == p3, r


@test
def every_arm_has_its_searches():
    n = {a: len(by(a)) for a in C.TEST_ARMS}
    assert n == {"g0": 4, "check": 4, "g1": 4, "facts": 4, "nologs": 4, "final": 4}, n
    assert len(by("final", lineage="A", shared_with="g1")) == 2 and len(by("final", lineage="B")) == 2
    assert SUMM["notes"] == {"A": "final is g1: searches shared"}


@test
def lineages_keep_reject_and_fall_back():
    la = json.loads((CDIR / "meta" / "A" / "lineage.json").read_text())
    lb = json.loads((CDIR / "meta" / "B" / "lineage.json").read_text())
    ga, gb = la["generations"], lb["generations"]
    assert [g["kept"] for g in ga] == [None, True, False] and la["final"] == "g1"
    assert ga[2]["header_fallback"] and ga[2]["header"] == ga[1]["header"]
    assert [g["kept"] for g in gb] == [None, True, True] and lb["final"] == "g2"
    assert abs(ga[0]["mean"]) < 1e-9 and abs(ga[1]["mean"] - 0.715) < 1e-9 and abs(gb[2]["mean"] - 0.815) < 1e-9
    assert RES["kept_rewrite_changed_header"]


@test
def header_is_obeyed():
    st = json.loads((CDIR / "meta" / "A" / "gen1" / "mirror" / "state.json").read_text())
    assert st["rounds_done"] == 6 and len(st["trials"]) == 24
    assert all(len(t["result"].get("dev_per_seed", [0])) == 1 for t in st["trials"] if t["result"]["status"] == "ok")
    p = json.loads((CDIR / "meta" / "A" / "gen1" / "mirror" / "round_1.json").read_text())["prompt"]
    assert "every trial trains 1 seed" in p and "highest 1-seed dev success" in p and "Each round trains 4 candidates" in p
    st0 = json.loads((CDIR / "meta" / "A" / "gen0" / "mirror" / "state.json").read_text())
    assert st0["rounds_done"] == 4 and len(st0["trials"]) == 12


@test
def scores_by_arm():
    def mean_of(es):
        return sum(e["score"] for e in es) / len(es)
    assert all(e["score"] == 0.0 for e in by("g0")) and all(e["score"] == 0.0 for e in by("nologs"))
    assert all(abs(e["score"] - 0.715) < 1e-9 for e in by("check")) and all(abs(e["score"] - 0.715) < 1e-9 for e in by("g1"))
    assert [e["world"] for e in by("facts") if e["score"] > 0] == ["shift", "shift"]
    assert abs(mean_of(by("final", lineage="B")) - 0.815) < 1e-9
    p = json.loads((CDIR / "test" / "check" / "-" / "shift" / "s1" / "round_1.json").read_text())["prompt"]
    assert "FULL READOUT" in p
    p = json.loads((CDIR / "test" / "g0" / "-" / "shift" / "s1" / "round_2.json").read_text())["prompt"]
    assert "READOUT" not in p and Q.NOTE in p


@test
def rewrites_read_only_meta_training():
    calls = sorted((CDIR / "programs").glob("*/call_*.json"))
    assert {c.stem for c in calls} == {"call_g1", "call_g2", "call_facts", "call_nologs"} and len(calls) == 8
    for c in calls:
        p = json.loads(c.read_text())["prompt"]
        assert "MARKTEST" not in p, c
        for w in ("mirror", "rot90", "shift", "rowshift"):
            assert w not in p, (c, w)
        if c.stem == "call_nologs":
            assert "### Search" not in p and p.count(Q.NONE) == 2
        else:
            assert "### Search 1, in world 1" in p and "meta note" in p and "labelled world 1 to world 2" in p


@test
def no_harness_prompt_names_a_rule():
    seen = 0
    for p in CDIR.rglob("*.json"):
        rel = p.relative_to(CDIR).parts
        if rel[0] == "judge" or not (p.name.startswith(("round_", "call_")) or rel[0] in ("review", "program_review")):
            continue
        rec = json.loads(p.read_text())
        if "prompt" in rec:
            seen += 1
            assert Q.rule_words(rec["prompt"]) == [], (p, Q.rule_words(rec["prompt"]))
    assert seen > 100, seen


@test
def no_call_repeated_and_analysis_final():
    mine = [c for c in CALLS if c[0] == "c"]
    assert len(mine) == len(set(mine))
    assert RES["status"] == "final" and RES["ending"] in "ABCDE" and RES["all_reviewed"]
    assert RES["check"]["found"] == 4 and RES["check"]["passes"]
    assert RES["per_arm"]["g1"]["found"] == 4 and RES["per_arm"]["g0"]["found"] == 0
    assert RES["calls"]["rewrite"]["calls"] == 8 and RES["calls"]["judge"]["calls"] == 22
    assert set(RES["primary"]) == {"P1", "P2", "P3", "P4"} and RES["primary"]["P4"]["lineages"] == ["A", "B"]


@test
def resume_after_crashes_repeats_no_call():
    runs = []
    for at in ("programs/A/call_facts.json", "test/g1/A/shift/s1/round_2.json", None):
        FAIL["at"] = at
        before = len(CALLS)
        try:
            Fake(TMP / "r", "fake_small").run()
            runs.append("done")
        except SystemExit as e:
            runs.append(str(e))
        FAIL["at"] = None
        assert len(CALLS) > before or at is None
    assert runs[-1] == "done" and all("failures" in r for r in runs[:2]), runs
    mine = [c for c in CALLS if c[0] == "r"]
    assert len(mine) == len(set(mine)), [c for c in mine if mine.count(c) > 1][:3]
    assert (TMP / "r" / "results.json").exists()


@test
def full_run_refuses_without_launch_approval():
    assert not (C.STEP3 / "LAUNCH_APPROVED.md").exists()
    try:
        R3.Step3(TMP / "full", "step3").run()
        assert False, "ran without approval"
    except SystemExit as e:
        assert "LAUNCH_APPROVED" in str(e)


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
