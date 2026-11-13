"""End-to-end test of run_2c.py with fake training, fake checks and fake model calls: no GPU, no model calls.

It runs the whole Stage 2c flow (baselines, 2 lineages x 3 generations on W_s with rewrites, the facts-only and
shuffled controls, the meta-test on W) in a scratch campaign, then reruns it to check that a resume makes no new calls
and gives the same result. It also checks, on Stage 2b's real files (read-only), that the round-1 prompt 2c would send
for g0 on W is byte-identical to the one 2b sent, which decision 5 (reusing 2b's g0 arm) needs.

usage (project root): .venv/bin/python -I arch_rsi/stage2/2c/test_run_2c.py SCRATCH_DIR
"""
import contextlib
import hashlib
import io
import itertools
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

import prog2c as P  # noqa: E402
import run_2c as R  # noqa: E402

S, w2b, O = R.S, R.w2b, R.O
from arch_rsi.sandbox import sha  # noqa: E402

SCRATCH = Path(sys.argv[1]).resolve()
CAMP = SCRATCH / "camp2c_test"
BASE_SRC = R.BASE.read_text()
CALLS, LOCK, UID = [], threading.Lock(), itertools.count(1)
HEADERS = [  # valid headers the fake rewrites cycle through (all within the 30-run limit)
    {"rounds": 3, "candidates_per_round": 4, "seeds_per_candidate": 2, "retest_top": 3, "split": {"network": 1, "recipe": 1, "both": 2}},
    {"rounds": 5, "candidates_per_round": 2, "seeds_per_candidate": 2, "retest_top": 4, "split": {"network": 0, "recipe": 1, "both": 1}},
    {"rounds": 2, "candidates_per_round": 6, "seeds_per_candidate": 1, "retest_top": 3, "split": {"network": 2, "recipe": 2, "both": 2}},
]
PRESET = {"2c_test": dict(R.PRESETS["2c"], steps={R.W: 131072, R.WS: 65536}, timeout={R.W: 900, R.WS: 900}, rewrites=2,
                          searches_per_gen=2, ws_confirm_seeds=[2001, 2002], test_searches=2, campaign=str(CAMP))}


def uid() -> int:
    with LOCK:
        return next(UID)


# ---------------------------------------------------------------- fakes
def fake_train(self, rdir, src_path, world, seed, split="dev"):
    res_file = rdir / "result.json"
    if res_file.exists():
        return json.loads(res_file.read_text())
    rdir.mkdir(parents=True, exist_ok=True)
    u = int(hashlib.sha256(f"{sha(Path(src_path).read_text())}|{seed}|{world}".encode()).hexdigest()[:8], 16) / 16 ** 8
    ok = u > 0.03
    res = {"status": "ok" if ok else "timeout", "error": None if ok else "killed", "dev_exact": round(0.1 + 0.8 * u, 4) if ok else None,
           "train_exact": round(0.7 * u, 4), "params": 1000, "train_steps": 1, "flops_ratio": 1.0, "wall_s": 1.0,
           "seed": seed, "world": world}
    res_file.write_text(json.dumps(res))
    return res


def fake_check(self, path, world, mode="both"):
    src = path.read_text()
    try:
        rec = w2b.parse_recipe(src)
    except (ValueError, SyntaxError) as e:
        return {"ok": False, "error": str(e)}
    if mode == "fixed" and rec is not None:
        return {"ok": False, "error": "this search's training recipe is fixed: remove the RECIPE block"}
    if mode == "recipe_only" and w2b.strip_recipe(src) != w2b.strip_recipe(BASE_SRC):
        return {"ok": False, "error": "in this search the network must be the starting architecture, unchanged"}
    errs = w2b.validate(rec)
    if errs:
        return {"ok": False, "error": "; ".join(errs)}
    return {"ok": True, "params": 1000, "flops_ratio": 1.0, "sha": sha(src)}


def variant(with_recipe: bool) -> str:
    n = uid()
    code = BASE_SRC.rstrip() + f"\n# variant {n}\n"
    return code + (f"\n\nRECIPE = {{'lr': {round(1e-4 + (n % 29) * 1e-4, 6)}}}\n" if with_recipe else "")


def reply_for(system, user, attempt):
    if system == S.SYSTEM_L1:
        if "## Some of your proposals failed the pre-training checks" in user:
            k = int(re.search(r"Return corrected versions of exactly these (\d+) proposals", user).group(1))
            return {"analysis": f"ANALYSIS-{uid()}", "proposals": [{"name": "fixed", "hypothesis": "no recipe",
                                                                    "code": variant(False)} for _ in range(k)]}
        k = int(re.search(r"Each round trains (\d+) candidates", user).group(1))
        return {"analysis": f"ANALYSIS-{uid()}", "proposals": [{"name": f"v{i}", "hypothesis": f"scored 0.5 maybe {i}",
                                                                "code": variant(True)} for i in range(k)]}
    m = re.search(r"## Your current research program \((\w+)\)\n(.*?)\n\n## How the program is used", user, re.S)
    parent = m.group(2) + "\n"
    if "## A restriction on this revision" in user:
        prog = parent.rstrip("\n") + "\n\n## Facts\n- networks that see the whole grid did better in these searches.\n"
        if attempt == 1:                                                   # first try edits a section: must be rejected
            prog = prog.replace("Read every result so far", "Read the results so far")
        return {"lessons": "l", "changes": [{"change": "facts", "evidence": "logs"}], "program": prog, "expected_effect": "e"}
    h = HEADERS[uid() % len(HEADERS)]
    if attempt == 1:                                                       # first try breaks the header: must be sent back
        h = dict(h, rounds=9)
    prog = P.render_header(h) + "\n" + P.strip_header(parent).rstrip("\n") + f"\n\n## Lesson\n- note {uid()}\n"
    return {"lessons": "l", "changes": [{"change": f"header {h['rounds']} rounds", "evidence": "logs"}], "program": prog,
            "expected_effect": "e"}


def fake_call(system, user, schema, validate, log_path, max_model_attempts=3):
    feedback = ""
    for n in range(1, max_model_attempts + 1):
        msg = user + (f"\n\nYour previous reply was rejected: {feedback}. Reply again in full." if feedback else "")
        obj = reply_for(system, user, n)
        try:
            val = validate(obj)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            feedback = str(e)[:600]
            continue
        with LOCK:
            CALLS.append({"path": str(log_path), "attempt": n})
        log_path.write_text(json.dumps({"attempt": n, "prompt": msg, "reply": obj}))
        return val, {}
    raise RuntimeError(f"fake model failed: {feedback}")


R.Stage2c.train_one = fake_train
R.Stage2c.check = fake_check
S.call_with_retries = fake_call


# ---------------------------------------------------------------- tests
def test_pure():
    g0 = R.settings(P.G0_HEADER)
    assert g0 == {"R": 4, "K": 3, "trial_seeds": [0, 1], "retest_top": 3, "retest_seeds": [2, 3], "roles": ["both"] * 3}
    s = R.settings(HEADERS[2])
    assert s["trial_seeds"] == [0] and s["retest_seeds"] == [1, 2] and s["roles"] == ["network"] * 2 + ["recipe"] * 2 + ["both"] * 2
    assert R.settings(P.G0_HEADER, {"rounds": 2, "seeds_per_candidate": 1})["R"] == 2
    t = R.recipe_text(["network", "recipe", "recipe", "both"])
    assert "proposal 1 changes only the network" in t and "proposals 2 to 3 change only the training recipe" in t
    assert "proposal 4 may change both" in t and "{{" not in t
    assert R.recipe_text(["both"] * 3) == R.R2B.RECIPE_TEXT["both"].replace("{{", "{").replace("}}", "}")
    # header fallback: two bad headers are sent back, the third uses the parent's header with the new body
    g0p = P.g0_program(P_G0R)
    v = R.rewrite_validator(g0p, 1000, False)
    bad = {"program": P.render_header(dict(P.G0_HEADER, rounds=9)) + "\n" + P_G0R + "\n## New\n- x\n", "changes": []}
    for _ in range(2):
        try:
            v(bad)
            raise AssertionError("bad header accepted")
        except ValueError as e:
            assert "the header" in str(e)
    out = v(bad)
    assert out["header_fallback"] and P.parse_program(out["program"])[0] == P.G0_HEADER and "## New" in out["program"]
    assert R.rewrite_validator(g0p, 1000, False, start=2)(bad)["header_fallback"]         # the same on resume
    vf = R.rewrite_validator(g0p, 1000, True)
    try:
        vf(bad)
        raise AssertionError("facts-only accepted a changed header")
    except ValueError as e:
        assert "may only add a '## Facts' section" in str(e)
    assert R.conf_mean({"dev": {"1": 0.5, "2": None}, "status": {"1": "ok", "2": "timeout"}}) == 0.25


def run_campaign():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        camp = R.Stage2c(CAMP, "2c_test", PRESET)
        camp.run()
    return camp


def test_campaign():
    camp = run_campaign()
    c = camp.cfg
    summ = json.loads((CAMP / "confirm" / "summary.json").read_text())
    reuse = json.loads((CAMP / "W" / "g0_reuse.json").read_text())
    # identity check for reusing 2b's g0: everything but 2b's completion (and its search count) must match
    allowed = ("2b has not finished", "2b has 16 g0 searches")
    assert all(r.startswith(allowed) for r in reuse["reasons"]), reuse["reasons"]
    # the W baseline's seeds 0-3 are 2b's; the noise note numbers therefore match 2b's
    bw = json.loads((CAMP / "baseline" / "w" / "summary.json").read_text())
    b2 = json.loads((R.CAMP_2B / "baseline" / "summary.json").read_text())["w"]
    assert bw["devs"] == b2["devs"] and bw["mean"] == b2["mean"] and bw["from_2b"] == [0, 1, 2, 3]
    arms = {}
    for e in summ:
        arms.setdefault((e["arm"], e["lineage"]), []).append(e)
    n_g0 = c["test_searches"] * len(c["lineages"])
    assert len(arms.get(("g0", None), [])) == n_g0, {k: len(v) for k, v in arms.items()}
    notes = json.loads((CAMP / "W" / "final_notes.json").read_text())
    for L in c["lineages"]:
        for arm in ("g1", "facts", "shuffled"):
            assert len(arms[(arm, L)]) == c["test_searches"]
        lin = json.loads((CAMP / "meta" / L / "lineage.json").read_text())
        if lin["final"] == "g0":
            assert ("final", L) not in arms and notes[L].startswith("final is g0")
        else:
            assert len(arms[("final", L)]) == c["test_searches"]
            assert (lin["final"] == "g1") == (notes.get(L, "").startswith("final is g1"))
        # test-before-keep: kept iff strictly above the current parent; the final program is the last kept one
        cur, cur_mean = "g0", lin["generations"][0]["mean"]
        for row in lin["generations"][1:]:
            assert row["parent"] == cur and row["parent_mean"] == cur_mean
            assert row["kept"] == (row["mean"] > cur_mean)
            if row["kept"]:
                cur, cur_mean = row["program"], row["mean"]
        assert lin["final"] == cur
        pdir = CAMP / "programs" / L
        g0, facts = (pdir / "g0.md").read_text(), (pdir / "facts.md").read_text()
        assert P.facts_only_problems(g0, facts) == []
        assert json.loads((pdir / "call_facts.json").read_text())["attempt"] == 2          # one rejection, then fixed
        assert json.loads((pdir / "call_g1.json").read_text())["attempt"] == 2             # bad header sent back once
        # g1, facts-only and shuffled got the same input, except the logs (shuffled) and the restriction (facts-only)
        p1, pf, ps = (json.loads((pdir / f"call_{k}.json").read_text())["prompt"] for k in ("g1", "facts", "shuffled"))
        p1, pf, ps = (p.split("\n\nYour previous reply was rejected:")[0] for p in (p1, pf, ps))
        cut = lambda p: re.sub(r"## Logs of the searches run under .*?## What to return", "LOGS", p, flags=re.S)  # noqa: E731
        assert cut(p1) == cut(ps) and p1 != ps
        assert pf.replace(R.FACTS_RESTRICTION, "").replace("Follow the restriction above.",
                                                             "Revise the program by following its own section on revising it.") == p1
        assert "ANALYSIS-" not in p1 + pf + ps                                              # Amendment 1
        for p in (p1, pf, ps):
            assert "## Earlier generations\nA rewritten program is kept only if" in p and "- generation 0: program g0" in p
    # every search obeyed its program's header, role by role
    n_searches = 0
    for st_file in CAMP.glob("**/state.json"):
        sdir = st_file.parent
        st = json.loads(st_file.read_text())
        h, _ = P.parse_program(_program_of(sdir))
        s = R.settings(h)
        assert st["settings"] == s and st["rounds_done"] == s["R"] and len(st["trials"]) == s["R"] * s["K"]
        for t in st["trials"]:
            assert t["role"] == s["roles"][int(t["label"].split("c")[1]) - 1]
            if t["result"]["status"] == "invalid":
                continue
            if t["role"] == "network":
                assert w2b.parse_recipe(t["code"]) is None
            if t["role"] == "recipe":
                assert w2b.strip_recipe(t["code"]) == w2b.strip_recipe(BASE_SRC)
            seeds = sorted(int(p.name[4:]) for p in (sdir / "trials" / t["label"]).glob("seed*"))
            assert seeds[:len(s["trial_seeds"])] == s["trial_seeds"], (sdir, t["label"], seeds)
        pick = json.loads((sdir / "pick.json").read_text())
        assert len(pick["retested"]) <= s["retest_top"]
        sel = [camp.base[st_world(sdir)]["all"][str(x)] for x in s["trial_seeds"] + s["retest_seeds"]]
        assert abs(pick["start_mean"] - sum(sel) / len(sel)) < 1e-12
        n_searches += 1
    assert n_searches >= 2 * 3 * 2 + 2 * 3 * 2
    man = json.loads((CAMP / "manifest.json").read_text())
    assert any(k.endswith("2c/run_2c.py") for k in man["hashes"]) and any(k.endswith("2b/run_2b.py") for k in man["hashes"])
    return summ


def st_world(sdir: Path) -> str:
    return R.WS if "meta" in sdir.relative_to(CAMP).parts else R.W


def _program_of(sdir: Path) -> str:
    """The full program (header included) a search ran under, from the campaign's files."""
    rel = sdir.relative_to(CAMP).parts
    if rel[0] == "meta":
        L, g = rel[1], int(rel[2][3:])
        gen = json.loads((CAMP / "meta" / L / f"gen{g}" / "generation.json").read_text())
        return (CAMP / "programs" / L / f"{gen['program']}.md").read_text()
    arm = rel[1]
    if arm == "g0":
        return P.g0_program(P_G0R)
    L = rel[2]
    name = json.loads((CAMP / "meta" / L / "lineage.json").read_text())["final"] if arm == "final" else arm
    return (CAMP / "programs" / L / f"{name}.md").read_text()


def test_resume(first):
    n = len(CALLS)
    run_campaign()
    assert len(CALLS) == n, f"resume made {len(CALLS) - n} new model calls"
    assert json.loads((CAMP / "confirm" / "summary.json").read_text()) == first


def test_final_routing():
    """The final arm on W: skipped when nothing was kept (final is g0), run on its own when final is a later rewrite."""
    camp2 = CAMP.parent / "camp2c_routing"
    if camp2.exists():
        shutil.rmtree(camp2)
    pre = {"2c_test": dict(PRESET["2c_test"], test_searches=1, reuse_g0=False, campaign=str(camp2))}
    pa = {k: (CAMP / "programs" / "A" / f"{k}.md").read_text() for k in ("g0", "g1", "g2", "facts", "shuffled")}
    lins = {"A": {k: pa[k] for k in ("g0", "g1", "facts", "shuffled")} | {"final": pa["g0"]},
            "B": {k: pa[k] for k in ("g0", "g1", "facts", "shuffled")} | {"final": pa["g2"]}}
    with contextlib.redirect_stdout(io.StringIO()):
        c2 = R.Stage2c(camp2, "2c_test", pre)
        c2.base = {R.W: c2.baseline(R.W)}
        entries = c2.meta_test(lins)
    got = sorted((e["arm"], e["lineage"]) for e in entries)
    assert ("final", "A") not in got and got.count(("final", "B")) == 1 and got.count(("g0", None)) == 2, got
    fb = next(e for e in entries if e["arm"] == "final")
    assert "shared_with" not in fb and fb["path"] == "W/final/B/s1" and fb["program_sha"] == sha(pa["g2"])
    notes = json.loads((camp2 / "W" / "final_notes.json").read_text())
    assert notes == {"A": "final is g0: nothing was kept"}


P_G0R = R.G0R.read_text()

if __name__ == "__main__":
    if CAMP.exists():
        shutil.rmtree(CAMP)
    test_pure()
    print("ok test_pure")
    first = test_campaign()
    print(f"ok test_campaign ({len(first)} meta-test entries, {len(CALLS)} fake model calls)")
    test_resume(first)
    print("ok test_resume (no new calls, same summary)")
    test_final_routing()
    print("ok test_final_routing")
    print("all run_2c tests passed")
