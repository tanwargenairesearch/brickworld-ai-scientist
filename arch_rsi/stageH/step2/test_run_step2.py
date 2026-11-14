"""Tests of run_step2.py with fake training and fake calls (no GPU, no model): prompts per arm, readouts, selection,
confirmation, repairs, duplicates, resume after a crash without repeating a call, the review, and the analysis.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2/test_run_step2.py   (exits 1 on any failure)
"""
import json
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as C  # noqa: E402
import prompts as P  # noqa: E402
import run_step2 as RS  # noqa: E402

FAILED = []
REF = C.common.REF_A.read_text()
FIX = {"shift": "\n# FIX_SHIFT torch.roll\n", "mirror": "\n# FIX_MIRROR .flip(3)\n"}


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except AssertionError as e:
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, e)
    return fn


class Fake(RS.Step2):
    """Fake GPU work and fake model calls. A file scores about 0.75 if it carries its world's FIX marker, else 0."""
    fail_at = None

    def __init__(self, cdir, preset):
        super().__init__(cdir, preset)
        self.calls, self.prompts = [], {}
        self.clock = threading.Lock()

    def verify(self):
        return {"ok": True}

    def _train_eval(self, rdir, world):
        src = (rdir / "run.json").read_text()
        spec = json.loads(src)
        code = Path(spec["candidate"]).read_text()
        good = "FIX_SHIFT" in code if world == "shift" else "FIX_MIRROR" in code
        dev = (0.70 + 0.01 * (spec["seed"] % 4)) if good else 0.0
        return {"status": "ok", "error": None, "dev_exact": dev, "train_exact": dev, "params": 1, "train_steps": 1,
                "flops_ratio": 1.0, "n_updates": 30, "recipe": None, "train_wall_s": 1.0}

    def _readouts(self, rdir, world, seed):
        return {"text_full": "FULL READOUT", "text_short": "SHORT READOUT"}

    def check(self, path, world):
        code = path.read_text()
        return {"ok": "BROKEN" not in code, "error": None if "BROKEN" not in code else "broken on purpose", "params": 1,
                "flops_ratio": 1.0}

    def call(self, prompt, k, log_path):
        parts = log_path.relative_to(self.dir).parts
        world, arm, s = parts[0], parts[1], parts[2]
        name = log_path.stem
        with self.clock:
            if self.fail_at and (world, arm, s, name) == self.fail_at:
                raise RuntimeError("fake model failure")
            self.calls.append((world, arm, s, name))
            self.prompts[(world, arm, s, name)] = prompt
        r = int(name.split("_")[1])
        repair = "repair" in name
        props = []
        for c in range(k):
            code = REF + f"\n# {arm} {s} r{r} c{c}{' repaired' if repair else ''}\n"
            if arm == "told" and r == 1 and c == 0:
                code += FIX[world]
            if arm == "readouts" and r == 2 and c == 1:
                code += FIX[world]
            if arm == "scores" and r == 1 and c == 2 and not repair:
                code += "# BROKEN\n"
            if arm == "hidden" and r == 2 and c == 0:
                code = REF + f"\n# {arm} {s} r1 c0\n"          # duplicate of an earlier trial
            props.append({"name": f"{arm} r{r}c{c}", "hypothesis": "neutral", "code": code})
        val = {"analysis": f"analysis {arm} r{r}", "proposals": props}
        log_path.write_text(json.dumps({"prompt": prompt, "reply": val, "usage": {"notional_cost_usd": 0.1}}))
        return val, {}

    def call_simple(self, system, prompt, schema, validate, log_path):
        if system == RS.REV.SYSTEM:
            val = {"verdict": "violation" if "FIX_MIRROR" in prompt else "compliant", "rule": "x", "evidence": "y"}
        else:
            JUDGE_PROMPTS[str(log_path)] = prompt
            val = {"verdict": "yes" if "told" in str(log_path) else "no", "evidence": "z"}
        log_path.write_text(json.dumps({"usage": {"notional_cost_usd": 0.05}}))
        return val, {}


JUDGE_PROMPTS = {}
RS.PRESETS["fake"] = dict(RS.BASE, n_searches=2, campaign="unused", needs_launch=False, search_workers=6)
TMP = Path(tempfile.mkdtemp())
F = Fake(TMP / "c", "fake")
F.run()
RES = json.loads((TMP / "c" / "results.json").read_text())


@test
def every_search_finishes_with_an_entry():
    ents = json.loads((TMP / "c" / "summary.json").read_text())["entries"]
    assert len(ents) == 16 and {(e["world"], e["arm"]) for e in ents} == {(w, a) for w in C.WORLDS for a in C.ARMS}


@test
def picks_and_confirmation():
    ents = {(e["world"], e["arm"], e["search"]): e for e in json.loads((TMP / "c" / "summary.json").read_text())["entries"]}
    for w in C.WORLDS:
        assert ents[(w, "told", 0)]["label"] == "r1c1" and ents[(w, "told", 0)]["first_round_found"] == 1
        assert ents[(w, "readouts", 0)]["label"] == "r2c2" and ents[(w, "readouts", 0)]["first_round_found"] == 2
        assert ents[(w, "scores", 1)]["label"] == "start" and ents[(w, "hidden", 1)]["label"] == "start"
        assert sorted(ents[(w, "told", 0)]["dev"]) == ["5001", "5002", "5003", "5004"]
        assert abs(sum(ents[(w, "told", 0)]["dev"].values()) / 4 - 0.715) < 1e-9


@test
def prompts_by_arm():
    for (w, a, s, n), p in F.prompts.items():
        if n != "round_2":
            continue
        assert (P.RULE_TEXT[w] in p) == (a == "told")
        assert ("FULL READOUT" in p) == (a in C.READOUT_ARMS), (w, a)
        assert (P.NOTE in p) == (a != "told")
        if a == "hidden":
            assert "dev_success" not in p


@test
def broken_file_is_repaired_and_duplicate_is_refused():
    st = json.loads((TMP / "c" / "shift" / "scores" / "s0" / "state.json").read_text())
    t = next(x for x in st["trials"] if x["label"] == "r1c3")
    assert "repaired" in t["code"] and t["result"]["status"] == "ok"
    assert ("shift", "scores", "s0", "round_1_repair1") in F.calls
    st = json.loads((TMP / "c" / "mirror" / "hidden" / "s0" / "state.json").read_text())
    t = next(x for x in st["trials"] if x["label"] == "r2c1")
    assert t["result"]["status"] == "invalid" and "identical to r1c1" in t["result"]["error"]


@test
def review_violation_scores_as_start():
    rows = {(r["world"], r["arm"], r["search"]): r for r in RES["rows"]}
    assert rows[("mirror", "told", 0)]["violation"] and rows[("mirror", "told", 0)]["score"] == 0.0
    assert not rows[("shift", "told", 0)]["violation"] and abs(rows[("shift", "told", 0)]["score"] - 0.715) < 1e-9
    assert RES["all_reviewed"]


@test
def analysis_and_costs():
    assert RES["status"] == "final" and RES["per_arm"]["told"]["found"] == 2 and RES["per_arm"]["readouts"]["found"] == 2  # the mirror fixes are judged violations
    assert RES["check"] == {"found": 2, "n": 4, "needed": 3, "passes": False}
    assert RES["calls"]["told"]["calls"] == 16 and RES["calls"]["judge"]["calls"] == 16
    assert RES["per_arm"]["told"]["judge"]["yes"] == 4


@test
def judge_sees_every_round_analysis():
    assert len(JUDGE_PROMPTS) == 16, len(JUDGE_PROMPTS)
    for path, prompt in JUDGE_PROMPTS.items():
        arm = Path(path).stem.split("_")[1]
        for r in range(1, 5):
            assert f"### Round {r}\nanalysis {arm} r{r}" in prompt, (path, r)


@test
def resume_after_a_crash_repeats_no_call():
    G = Fake(TMP / "r", "fake")
    G.cfg = dict(G.cfg, n_searches=1)
    G.fail_at = ("shift", "readouts", "s0", "round_3")
    try:
        G.run()
    except SystemExit:
        pass
    st = json.loads((TMP / "r" / "shift" / "readouts" / "s0" / "state.json").read_text())
    assert st["rounds_done"] == 2
    H = Fake(TMP / "r", "fake")
    H.cfg = dict(H.cfg, n_searches=1)
    H.run()
    mine = [c for c in H.calls if c[:3] == ("shift", "readouts", "s0")]
    assert [c[3] for c in mine] == ["round_3", "round_4"], mine
    assert len(H.calls) == 2, H.calls                      # nothing else was called again
    assert (TMP / "r" / "results.json").exists()


@test
def full_run_refuses_without_launch_approval():
    real, C.STEP2 = C.STEP2, TMP                           # an empty folder, whatever the real one holds
    try:
        RS.Step2(TMP / "x", "step2").run()
        assert False, "ran without approval"
    except SystemExit as e:
        assert "LAUNCH_APPROVED" in str(e)
    finally:
        C.STEP2 = real


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
