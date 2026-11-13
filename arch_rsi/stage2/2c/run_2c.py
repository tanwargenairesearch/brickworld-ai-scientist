"""Stage 2c (PROTOCOL_2C.md): does the procedure improve itself? Meta-train programs on W_s, meta-test them on W.

Reuses Stage 2b's runner (run_2b.Stage2b, imported, not edited: GPU training and checks on W, the task text) and
Stage 1's retest-and-pick, and adds:
- programs with a machine-read header (prog2c.py) that sets each search's rounds, candidates per round, seeds per
  candidate, retest count and network/recipe/both split. The runner obeys it, and the checks enforce each role;
- world W_s for meta-training (ws_world.py, check_2c.py), with its budget from the calibration (ws_budget.json);
- two lineages of 3 rewrites each with test-before-keep on W_s, and the facts-only and shuffled controls;
- saved proposals: a round whose model reply is on disk reuses it on resume (after checking the prompt is the same),
  so no round is redrawn;
- the meta-test on W: g1, final, facts-only and shuffled, 8 searches per lineage each; g0 is reused from 2b only
  if the identity check passes, and otherwise rerun with 16 searches.

usage (project root):
  caffeinate -i .venv/bin/python -I arch_rsi/stage2/2c/run_2c.py --preset 2c_dry   (tiny budgets; after 2b ends)
  caffeinate -i .venv/bin/python -I arch_rsi/stage2/2c/run_2c.py --preset 2c       (needs ws_budget.json and
                                                                                    LAUNCH_APPROVED.md)
Resumes from disk if rerun with the same preset. W's final split is never opened here.
"""
import argparse
import copy
import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
S2 = HERE.parent
A = S2.parent
ROOT = A.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import prog2c as P  # noqa: E402
import ws_world  # noqa: E402

ws_world.register()
_spec = importlib.util.spec_from_file_location("run_2b", S2 / "2b" / "run_2b.py")
R2B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R2B)
O, S, S1, w2b = R2B.O, R2B.S, R2B.S1, R2B.w2b
from arch_rsi.sandbox import sha  # noqa: E402

W, WS = "w", "ws"
PY = R2B.PY
BASE = R2B.BASE
G0R = S2 / "programs" / "g0r.md"
CAMP_2B = A / "campaigns" / "stage2b"
BUDGET_FILE = HERE / "ws_budget.json"
APPROVAL_FILE = HERE / "LAUNCH_APPROVED.md"
ROLE_MODE = {"network": "fixed", "recipe": "recipe_only", "both": "both"}   # check_2b modes
ROLE_ARM = {"network": "netonly", "recipe": "reconly", "both": "g0"}         # 2b arm with that uniform task text
TEST_ARMS = ("g1", "final", "facts", "shuffled")
SHUFFLE_SEED = {"A": 1, "B": 2}
FILES_2C = [HERE / p for p in ("run_2c.py", "prog2c.py", "ws_world.py", "check_2c.py", "PROTOCOL_2C.md",
                               "targets/ws_dev.pt", "targets/ws_check.pt", "targets/ws_heldout_hashes.pt")]

PRESETS = {
    "2c": dict(R2B.PRESETS["2b"], Q=W, steps={W: 131072, WS: None}, timeout={W: 900, WS: None},
               lineages=["A", "B"], rewrites=3, searches_per_gen=6, ws_confirm_seeds=[2001, 2002, 2003],
               test_searches=8, start_seeds=[0, 1, 2, 3, 4], reuse_g0=True, header_cap=None, lineage_workers=2,
               campaign="arch_rsi/campaigns/stage2c"),
}
PRESETS["2c_dry"] = dict(PRESETS["2c"], steps={W: 16384, WS: 16384}, timeout={W: 600, WS: 600}, lineages=["A"],
                         rewrites=1, searches_per_gen=1, ws_confirm_seeds=[2001], test_searches=1,
                         confirm_seeds=[1001], reuse_g0=False, search_workers=6,
                         header_cap={"rounds": 2, "seeds_per_candidate": 1, "retest_top": 1},
                         campaign="arch_rsi/campaigns/stage2c_dry")

L2C_PROMPT = """## Your current research program ({name})
{program}

## How the program is used
The program has two parts.
- The header at the top (the HTML comment that starts with "<!-- header") is read only by the code that runs each search. It sets the number of rounds, the candidates per round, the training seeds per candidate, how many of the best candidates train {n_retest} more seeds before the final selection, and the split: how many of each round's candidates change only the network, only the training recipe, or both. Bounds: rounds 2-6; candidates_per_round 2-6; seeds_per_candidate 1-3; retest_top 1-4; the split's three counts must sum to candidates_per_round; and rounds x candidates_per_round x seeds_per_candidate + retest_top x {n_retest} must be at most {max_runs}, so every program gets the same number of training runs per search. Keep the header's format and change only its numbers.
- The rest of the program is what you receive at the start of every round of every architecture search, together with a fixed task description (which states the header's settings in words), the interface rules, the starting architecture, the search's results so far (every trial's code, hypothesis and scores) and the budget.

The program is the only part of the procedure you can change. Next it will be used for new searches, including on brick worlds of other sizes that you will not see while revising it. It will be judged by how well the architectures found under it do on hidden designs, at the same training budget.

The interface rules, caps, checks, the training budget and the recipe bounds are fixed and are not part of the program. In particular the policy and its features must be learned: a program that advises hand-writing a placement rule or score, hand-computing features from the observation before a learned layer, hand-setting weights so the network is competent before training, or ways to pass the checks without learning, is advising rule violations. Candidates built that way are rejected by the checks or reported as violations after the campaign.
{restriction}
## The fixed task description, as given in the searches below
{task_example}

## Earlier generations
{gen_table}

## Logs of the searches run under {logs_label}
{logs}

## What to return
{instruction} Return a JSON object with:
- "lessons": at most 200 words: what these searches show about which architectures, recipes and ways of searching worked.
- "changes": the changes you are making to the program, each with the evidence for it.
- "program": the complete new program in Markdown, starting with the header, at most {max_words:,} words.
- "expected_effect": one sentence: what should improve, and how one could tell."""

FACTS_RESTRICTION = """
## A restriction on this revision
In this revision you may only add facts. Return the current program exactly as it is, character for character, including its header and every section, and add one section at the end headed "## Facts": a short list of what these searches showed (for example which kinds of network or recipe did well or badly, and how large the differences were compared with seed noise). Do not change, move or remove anything else. A program that changes anything other than adding that section is rejected.
"""


# ---------------------------------------------------------------- pure helpers (tested in test_run_2c.py)
def settings(h: dict, cap: dict = None) -> dict:
    """What a search runs under a header: rounds, candidates, trial and retest seeds, and each candidate's role."""
    h = {k: (min(v, cap[k]) if cap and k in cap else v) for k, v in h.items()}
    n, sp = h["seeds_per_candidate"], h["split"]
    return {"R": h["rounds"], "K": h["candidates_per_round"], "trial_seeds": list(range(n)),
            "retest_top": h["retest_top"], "retest_seeds": [n + i for i in range(P.RETEST_SEEDS)],
            "roles": ["network"] * sp["network"] + ["recipe"] * sp["recipe"] + ["both"] * sp["both"]}


def recipe_text(roles: list) -> str:
    """The task's recipe paragraph: 2b's own text for a uniform split, otherwise each proposal's role in order."""
    if len(set(roles)) == 1:
        return R2B.RECIPE_TEXT[ROLE_MODE[roles[0]]].replace("{{", "{").replace("}}", "}")

    def which(role):
        idx = [i + 1 for i, r in enumerate(roles) if r == role]
        if not idx:
            return None, False
        return (f"proposal {idx[0]}" if len(idx) == 1 else f"proposals {idx[0]} to {idx[-1]}"), len(idx) == 1
    parts = []
    w, one = which("network")
    if w:
        parts.append(f"{w} {'changes' if one else 'change'} only the network, with no RECIPE constant, so the default "
                     "recipe below is used")
    w, one = which("recipe")
    if w:
        parts.append(f"{w} {'changes' if one else 'change'} only the training recipe: return the starting "
                     "architecture's file unchanged with one top-level constant added, `RECIPE = {...}`; only that "
                     "constant is used and the network is always the starting architecture")
    w, one = which("both")
    if w:
        parts.append(f"{w} may change both the network and the recipe")
    return ("- In this search each round's proposals have fixed roles, in this order: " + "; ".join(parts) +
            ". To set a recipe, add one top-level constant to the file, `RECIPE = {...}`, a literal dict with any of "
            "these keys and bounds: " + R2B.BOUNDS_TEXT + ". Keys you leave out take the default recipe: " +
            R2B.DEFAULTS_TEXT + ". A file without RECIPE trains with the default.")


def conf_mean(entry: dict) -> float:
    """A search's confirmed score: mean dev success over its confirmation seeds, a failed run counting 0."""
    vals = [(entry["dev"][s] if entry["status"][s] == "ok" else 0.0) for s in entry["dev"]]
    return sum(vals) / len(vals)


def l2c_prompt(name, program, task_example, gen_table, logs_label, logs, max_words, facts_only=False) -> str:
    return L2C_PROMPT.format(
        name=name, program=program.rstrip("\n"), n_retest=P.RETEST_SEEDS, max_runs=P.MAX_RUNS,
        restriction=FACTS_RESTRICTION if facts_only else "", task_example=task_example, gen_table=gen_table,
        logs_label=logs_label, logs=logs, max_words=max_words,
        instruction=("Follow the restriction above." if facts_only
                     else "Revise the program by following its own section on revising it."))


def rewrite_validator(parent: str, max_words: int, facts_only: bool, start: int = 0):
    """validate() for a rewrite reply. A full rewrite with a bad header is sent back twice; on the third bad header
    the parent's header is used with the new body (disclosed as header_fallback). A facts-only rewrite must pass the
    facts-only check; there is no fallback, so 3 failures stop the run (call_with_retries raises)."""
    ph, _ = P.parse_program(parent)
    count = {"n": start}

    def validate(obj):
        count["n"] += 1
        val = S.parse_program(obj, max_words)
        if facts_only:
            probs = P.facts_only_problems(parent, val["program"], max_words)
            if probs:
                raise ValueError("this revision may only add a '## Facts' section: " + "; ".join(probs))
            return dict(val, header_fallback=False)
        try:
            P.parse_program(val["program"])
            return dict(val, header_fallback=False)
        except ValueError as e:
            if count["n"] < 3:
                raise ValueError(f"the header: {e}. The program must start with the header block in the format "
                                 f"shown, with every value inside its bounds") from e
            prog = P.render_header(ph) + "\n" + P.strip_header(val["program"])
            if len(prog.split()) > max_words:
                raise ValueError(f"program has {len(prog.split())} words; the limit is {max_words}") from e
            return dict(val, program=prog, header_fallback=True, header_error=str(e))
    return validate


# ---------------------------------------------------------------- the campaign
class Stage2c(R2B.Stage2b):
    def __init__(self, cdir: Path, preset: str, presets: dict = None):
        cfg = dict((presets or PRESETS)[preset])
        files = list(FILES_2C)
        if cfg["steps"][WS] is None:
            if not BUDGET_FILE.exists():
                raise SystemExit(f"{BUDGET_FILE.name} is missing: calibrate W_s first (PROTOCOL_2C.md)")
            b = json.loads(BUDGET_FILE.read_text())
            cfg["steps"] = dict(cfg["steps"], **{WS: int(b["steps"])})
            cfg["timeout"] = dict(cfg["timeout"], **{WS: int(b["timeout"])})
            files.append(BUDGET_FILE)
        O.PRESETS[preset] = cfg
        O.FROZEN = list(O.FROZEN) + [p for p in R2B.FILES_2B + files if p not in O.FROZEN]
        O.Campaign.__init__(self, cdir, preset)
        self.run_locks = {}
        self.base = {}

    # ---- training and checks: W through Stage 2b's code, W_s through ws_world.py and check_2c.py
    def train_one(self, rdir: Path, src_path: Path, world: str, seed: int, split: str = "dev") -> dict:
        if world == W:
            return R2B.Stage2b.train_one(self, rdir, src_path, world, seed, split)
        res_file = rdir / "result.json"
        if res_file.exists():
            return json.loads(res_file.read_text())
        rdir.mkdir(parents=True, exist_ok=True)
        spec = rdir / "run.json"
        spec.write_text(json.dumps({"candidate": str(src_path), "world": world, "seed": seed,
                                    "budget_steps": self.cfg["steps"][world], "threads": 1, "device": "mps"}))
        with self.slots:
            self.verify()
            t0 = time.time()
            try:
                p = subprocess.run([PY, "-I", str(HERE / "ws_world.py"), "train", "--spec", str(spec), "--out", str(rdir / "run")],
                                   capture_output=True, text=True, timeout=self.cfg["timeout"][world])
                tl = json.loads((rdir / "run" / "train_log.json").read_text()) if (rdir / "run" / "train_log.json").exists() else {}
                status = tl.get("status", f"crashed (exit {p.returncode})")
                err = tl.get("error") or (None if status == "ok" else (p.stderr or "")[-400:])
            except subprocess.TimeoutExpired:
                tl, status, err = {}, "timeout", f"killed after {self.cfg['timeout'][world]} s"
            train_wall = round(time.time() - t0, 1)
            dev = None
            if status == "ok":
                q = subprocess.run([PY, "-I", str(HERE / "ws_world.py"), "eval", "--trial", str(rdir / "run"),
                                    "--split", "dev", "--out", str(rdir / "dev.json")], capture_output=True, text=True, timeout=1800)
                if q.returncode == 0:
                    dev = json.loads((rdir / "dev.json").read_text())["exact"]
                else:
                    status, err = "eval_failed", (q.stderr or "")[-400:]
        res = {"status": status, "error": err, "dev_exact": dev, "train_exact": tl.get("train_exact_tail"),
               "params": tl.get("params"), "train_steps": tl.get("steps"), "flops_ratio": tl.get("flops_ratio"),
               "n_updates": tl.get("n_updates"), "recipe": tl.get("recipe"), "train_wall_s": train_wall,
               "wall_s": round(time.time() - t0, 1), "seed": seed, "world": world}
        res_file.write_text(json.dumps(res))
        return res

    def check(self, path: Path, world: str, mode: str = "both") -> dict:
        out = path.with_suffix(".check.json")
        meta = path.with_suffix(".check_for.json")
        key = {"sha": sha(path.read_text()), "world": world, "mode": mode}
        if out.exists() and meta.exists() and json.loads(meta.read_text()) == key:
            return json.loads(out.read_text())                       # resume: same file, world and mode
        if world == W:
            res = R2B.Stage2b.check(self, path, world, mode)
        else:
            with self.check_slots:
                p = subprocess.run([PY, "-I", str(HERE / "check_2c.py"), "--candidate", str(path), "--mode", mode,
                                    "--budget", str(self.cfg["steps"][world]), "--out", str(out)],
                                   capture_output=True, text=True, timeout=1200)
            if p.returncode != 0 or not out.exists():
                return {"ok": False, "error": f"check process failed: {(p.stderr or '')[-300:]}"}
            res = json.loads(out.read_text())
        if out.exists():
            meta.write_text(json.dumps(key))
        return res

    # ---- the starting architecture on each world
    def baseline(self, world: str) -> dict:
        c = self.cfg
        d = self.dir / "baseline" / world
        d.mkdir(parents=True, exist_ok=True)
        if (d / "summary.json").exists():
            return json.loads((d / "summary.json").read_text())
        src = BASE.read_text()
        (d / "candidate.py").write_text(src)
        from2b = {}
        if world == W and c["reuse_g0"]:
            # W's seeds 0-3 come from 2b (same trainer, budget, file and seeds), so the scientist's noise note on W
            # is 2b's, word for word; only seed 4 (for 3-seed headers) is trained here.
            m2 = json.loads((CAMP_2B / "manifest.json").read_text())["config"]
            b2 = json.loads((CAMP_2B / "baseline" / "summary.json").read_text())[W]
            if (m2["steps"][W], m2["timeout"][W]) != (c["steps"][W], c["timeout"][W]) or b2["check"]["sha"] != sha(src):
                raise SystemExit("2b's W baseline was run under other conditions")
            from2b = dict(zip(b2["seeds"], b2["devs"]))
        chk = self.check(d / "candidate.py", world)
        if not chk.get("ok"):
            raise SystemExit(f"the starting architecture failed the checks on {world}: {chk}")
        todo = [s for s in c["start_seeds"] if s not in from2b]
        with ThreadPoolExecutor(max(1, len(todo))) as ex:
            res = dict(zip(todo, ex.map(lambda s: self.train_one(d / f"seed{s}", d / "candidate.py", world, s), todo)))
        allv = {**from2b, **{s: r["dev_exact"] for s, r in res.items()}}
        if any(allv[s] is None for s in c["start_seeds"]):
            raise SystemExit(f"starting architecture failed on {world}: {res}")
        devs = [allv[s] for s in c["base_seeds"]]
        summ = {"devs": devs, "mean": sum(devs) / len(devs), "min": min(devs), "max": max(devs),
                "all": {str(s): allv[s] for s in c["start_seeds"]}, "from_2b": sorted(from2b), "check": chk,
                "seeds": c["base_seeds"]}
        (d / "summary.json").write_text(json.dumps(summ, indent=1))
        self.log(f"baseline {world}: {[round(allv[s], 3) for s in c['start_seeds']]} (from 2b: {sorted(from2b)})")
        return summ

    def view(self, world: str, st: dict, for_selection: bool = False):
        """A shallow copy whose cfg and baseline match one search's settings, so Stage 1 and 2b methods (task text,
        noise note, retest-and-pick) run unchanged. The noise note always shows the start's 4 base seeds, as in 2b;
        for selection the start's mean is over the same seed ids as each candidate's (trial + retest seeds)."""
        V = copy.copy(self)
        V.cfg = dict(self.cfg, R=st["R"], K=st["K"], trial_seeds=st["trial_seeds"], retest_top=st["retest_top"],
                     retest_seeds=st["retest_seeds"], Q=world)
        b = self.base[world]
        shown = [b["all"][str(s)] for s in st["trial_seeds"]]
        sel = [b["all"][str(s)] for s in st["trial_seeds"] + st["retest_seeds"]]
        V.base = dict(self.base)
        V.base[world] = dict(b, shown_dev=sum(shown) / len(shown))
        if for_selection:
            V.base[world]["mean"] = sum(sel) / len(sel)
        return V

    def task_text(self, V, world: str, roles: list) -> str:
        """Stage2b.task_for itself for a uniform split; for a mixed split the same text with the roles paragraph."""
        if len(set(roles)) == 1:
            return R2B.Stage2b.task_for(V, world, False, ROLE_ARM[roles[0]])
        sp = O.SPECS[world]
        t = R2B.TASK_2B.format(X=sp.X, Y=sp.Y, Z=sp.Z, max_steps=sp.max_steps, k_min=sp.k_min, k_max=sp.k_max,
                               steps=V.cfg["steps"][world], timeout=V.cfg["timeout"][world], n_dev=512,
                               noise=V.noise_note(world), recipe_text=recipe_text(roles))
        if S1.FROZEN_SELECTION not in t:
            raise SystemExit("task text changed: selection sentence not found")
        return t.replace(S1.FROZEN_SELECTION, S1.selection_text(V.cfg))

    def saved_call(self, system, prompt, schema, validate, path: Path):
        """S.call_with_retries, unless this exact prompt's reply is already on disk (resume without redrawing)."""
        if path.exists():
            rec = json.loads(path.read_text())
            sp = rec.get("prompt", "")
            if not (sp == prompt or sp.startswith(prompt + "\n\nYour previous reply was rejected:")):
                raise SystemExit(f"{path} was written for a different prompt; refusing to reuse or redraw it")
            return validate(rec["reply"])
        val, _ = S.call_with_retries(system, prompt, schema, validate, path)
        return val

    # ---- one search under a program's header
    def search_2c(self, sdir: Path, world: str, body: str, st_set: dict) -> dict:
        sdir.mkdir(parents=True, exist_ok=True)
        st_file = sdir / "state.json"
        st = json.loads(st_file.read_text()) if st_file.exists() else {"trials": [], "rounds_done": 0, "incumbent": None,
                                                                        "settings": st_set}
        if st.get("settings") != st_set:
            raise SystemExit(f"{sdir}: settings differ from the saved state")
        if (sdir / "program.md").exists():
            if (sdir / "program.md").read_text() != body:
                raise SystemExit(f"{sdir}: program differs from the one this search started with")
        else:
            (sdir / "program.md").write_text(body)
        V = self.view(world, st_set)
        R, K, seeds, roles = st_set["R"], st_set["K"], st_set["trial_seeds"], st_set["roles"]
        base_src = BASE.read_text()
        base_dev = V.base[world]["shown_dev"]
        task = self.task_text(V, world, roles)

        def normalise(p, role):
            if role != "recipe":
                return p
            try:
                rec = w2b.parse_recipe(p["code"]) or {}
            except (ValueError, SyntaxError):
                rec = {}
            return dict(p, code=w2b.with_recipe(base_src, rec) if rec else base_src)
        for r in range(st["rounds_done"] + 1, R + 1):
            trials = st["trials"]
            seen = {sha(base_src): "the starting architecture"} | {t["result"]["sha"]: t["label"] for t in trials if t.get("result", {}).get("sha")}
            inc = next((t for t in trials if t["label"] == st["incumbent"]), None)
            prompt = S.l1_prompt(body, task, self.interface(), base_src, base_dev, trials, inc, r, R, K)
            val = self.saved_call(S.SYSTEM_L1, prompt, S.l1_schema(K), lambda o: S.parse_proposals(o, K), sdir / f"round_{r}.json")
            props, analysis = [normalise(p, roles[i]) for i, p in enumerate(val["proposals"])], val["analysis"]
            checks = [None] * K
            for j in range(3):
                todo = [i for i in range(K) if checks[i] is None or not checks[i]["ok"]]

                def run_check(i, j=j):
                    p = sdir / "cands" / f"r{r}c{i + 1}_v{j}.py"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(props[i]["code"])
                    return self.check(p, world, ROLE_MODE[roles[i]])
                with ThreadPoolExecutor(max(1, len(todo))) as ex:
                    got = dict(zip(todo, ex.map(run_check, todo)))
                for i in todo:
                    c = got[i]
                    h = sha(props[i]["code"])
                    batch_dup = [k for k in range(K) if k < i and sha(props[k]["code"]) == h]
                    if c.get("ok") and (h in seen or batch_dup):
                        c = {"ok": False, "error": f"identical to {seen.get(h) or f'proposal {batch_dup[0] + 1} in this round'}"}
                    checks[i] = c
                failed = [i for i in range(K) if not checks[i]["ok"]]
                if not failed or j == 2:
                    break
                fl = [{"name": props[i]["name"], "error": checks[i]["error"], "code": props[i]["code"]} for i in failed]
                try:
                    fix = self.saved_call(S.SYSTEM_L1, S.repair_prompt(prompt, fl), S.l1_schema(len(failed)),
                                          lambda o, n=len(failed): S.parse_proposals(o, n), sdir / f"round_{r}_repair{j + 1}.json")
                except RuntimeError as e:
                    self.log(f"{sdir.relative_to(self.dir)} round {r} repair {j + 1} failed: {e}")
                    break
                for i, f in zip(failed, fix["proposals"]):
                    props[i] = normalise({"name": f["name"] or props[i]["name"], "hypothesis": f["hypothesis"], "code": f["code"]}, roles[i])
            new, jobs = [], []
            for i in range(K):
                t = {"label": f"r{r}c{i + 1}", "round": r, "role": roles[i], "name": props[i]["name"],
                     "hypothesis": props[i]["hypothesis"], "code": props[i]["code"], "check": checks[i]}
                if not checks[i]["ok"]:
                    t["result"] = {"status": "invalid", "error": checks[i]["error"], "dev_exact": None, "sha": sha(props[i]["code"])}
                else:
                    jobs.append(t)
                new.append(t)
            with ThreadPoolExecutor(max(1, len(jobs))) as ex:
                outs = list(ex.map(lambda t: self.run_candidate(sdir / "trials" / t["label"], t["code"], world, seeds, t["check"]), jobs))
            for t, o in zip(jobs, outs):
                t["result"] = o
            st["trials"] = trials + new
            st.setdefault("analyses", {})[str(r)] = analysis
            best = max([0.0 if inc is None else inc["result"]["dev_exact"], base_dev])
            for t in new:
                if t["result"]["status"] == "ok" and t["result"]["dev_exact"] > best:
                    best, st["incumbent"] = t["result"]["dev_exact"], t["label"]
            st["rounds_done"] = r
            st_file.write_text(json.dumps(st, indent=1))
            self.log(f"{sdir.relative_to(self.dir)} round {r}: " + ", ".join(
                f"{t['label']} {t['result']['status']} {t['result'].get('dev_exact')}" for t in new) + f" | incumbent {st['incumbent']} {best:.3f}")
        return st

    def confirm_on(self, src: str, seeds: list, world: str) -> dict:
        """Stage1.confirm, with one directory per world (a pick may be confirmed on both worlds)."""
        h = sha(src)
        d = self.dir / "confirm" / world / h
        d.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if not (d / "candidate.py").exists():
                (d / "candidate.py").write_text(src)
        with ThreadPoolExecutor(len(seeds)) as ex:
            res = list(ex.map(lambda s: self.locked_train(d / f"seed{s}", d / "candidate.py", world, s), seeds))
        return dict(zip(seeds, res))

    def one_search_2c(self, sdir: Path, world: str, program: str, confirm_seeds: list) -> dict:
        h, body = P.parse_program(program)
        st_set = settings(h, self.cfg["header_cap"])
        st = self.search_2c(sdir, world, body, st_set)
        pick = S1.Stage1.retest_and_pick(self.view(world, st_set, for_selection=True), sdir, st, world)
        conf = self.confirm_on(pick["src"], confirm_seeds, world)
        rel = sdir.relative_to(self.dir)
        self.log(f"{rel} confirmed: {[round(r['dev_exact'], 3) if r['dev_exact'] is not None else r['status'] for r in conf.values()]}")
        return {"kind": "search", "world": world, "path": str(rel), "program_sha": sha(program), "settings": st_set,
                "label": pick["label"], "name": pick["name"], "sha": pick["sha"], "mean_selected": pick["mean_selected"],
                "incumbent_2seed": pick["incumbent_2seed"], "seeds": confirm_seeds,
                "dev": {str(s): r["dev_exact"] for s, r in conf.items()},
                "status": {str(s): r["status"] for s, r in conf.items()},
                "wall_s": {str(s): r["wall_s"] for s, r in conf.items()}}

    # ---- meta-training: one generation of searches on W_s, rewrites, and lineages
    def generation(self, L: str, g: int, name: str, program: str) -> dict:
        c = self.cfg
        gdir = self.dir / "meta" / L / f"gen{g}"
        n = c["searches_per_gen"]
        with ThreadPoolExecutor(n) as ex:
            entries = list(ex.map(lambda i: self.one_search_2c(gdir / f"s{i}", WS, program, c["ws_confirm_seeds"]), range(1, n + 1)))
        states = [json.loads((gdir / f"s{i}" / "state.json").read_text()) for i in range(1, n + 1)]
        means = [conf_mean(e) for e in entries]
        row = {"generation": g, "program": name, "program_sha": sha(program), "settings": entries[0]["settings"],
               "search_means": means, "mean": sum(means) / len(means), "stats": self.stats(states), "searches": entries}
        (gdir / "generation.json").write_text(json.dumps(row, indent=1))
        return row

    def gen_table(self, hist: list, current: str) -> str:
        lines = ["A rewritten program is kept only if the mean confirmed dev success of its searches (each search's pick "
                 "retrained on fresh seeds) is strictly higher than that of the program it was written from; otherwise "
                 "that program stays, and the next rewrite starts from it."]
        for row in hist:
            g, s = row["generation"], row["stats"]
            line = f"- generation {g}: program {row['program']}" + (" (the starting program)" if g == 0 else f", written from {row['parent']}")
            line += (f"; mean confirmed dev success {row['mean']:.3f} over {len(row['search_means'])} searches; best dev "
                     f"success per search {s['best']}; invalid candidates {s['invalid']}, failed trainings {s['failed']}")
            if g > 0:
                line += f"; {'kept' if row['kept'] else 'not kept'} (compared with {row['parent_mean']:.3f} for {row['parent']})"
            lines.append(line)
            if g > 0 and row.get("changes"):
                lines.append("  changes made when writing it: " + "; ".join(str(ch.get("change", ""))[:200] for ch in row["changes"]))
        lines.append(f"The current program is {current}.")
        return "\n".join(lines)

    def rewrite(self, L: str, kind: str, g: int, parent: dict, latest: dict, hist: list) -> str:
        """kind: real (g{g+1} from the parent), facts or shuffled (written once, from g0 and generation 0's logs)."""
        pdir = self.dir / "programs" / L
        name = {"real": f"g{g + 1}", "facts": "facts", "shuffled": "shuffled"}[kind]
        out_prog = pdir / f"{name}.md"
        if out_prog.exists():
            return out_prog.read_text()
        gdir = self.dir / "meta" / L / f"gen{g}"
        states = [json.loads((gdir / f"s{i}" / "state.json").read_text()) for i in range(1, self.cfg["searches_per_gen"] + 1)]
        V = self.view(WS, latest["settings"])
        logs = P.format_logs(states, S.fmt_trial, V.base[WS]["shown_dev"],
                             shuffle_seed=SHUFFLE_SEED[L] if kind == "shuffled" else None)
        prompt = l2c_prompt(parent["name"], parent["text"], self.task_text(V, WS, latest["settings"]["roles"]),
                            self.gen_table(hist, parent["name"]), f"generation {g} (program {latest['program']})", logs,
                            self.cfg["max_words"], facts_only=kind == "facts")
        call = pdir / f"call_{name}.json"
        start = json.loads(call.read_text())["attempt"] - 1 if call.exists() else 0
        val = self.saved_call(S.SYSTEM_L2, prompt, S.L2_SCHEMA,
                              rewrite_validator(parent["text"], self.cfg["max_words"], kind == "facts", start), call)
        prog = val["program"]
        (pdir / f"revision_{name}.json").write_text(json.dumps({"kind": kind, "parent": parent["name"], "from_generation": g,
                                                                 "result": val, "sha": sha(prog)}, indent=1))
        out_prog.write_text(prog)
        self.log(f"lineage {L}: program {name} written from {parent['name']} ({len(prog.split())} words"
                 f"{', parent header used' if val.get('header_fallback') else ''})")
        return prog

    def lineage(self, L: str) -> dict:
        c = self.cfg
        pdir = self.dir / "programs" / L
        pdir.mkdir(parents=True, exist_ok=True)
        g0 = P.g0_program(G0R.read_text())
        if (pdir / "g0.md").exists() and (pdir / "g0.md").read_text() != g0:
            raise SystemExit(f"{pdir / 'g0.md'} differs from g0")
        (pdir / "g0.md").write_text(g0)
        parent, cur = {"name": "g0", "text": g0}, {"name": "g0", "text": g0}
        parent_row, hist, out = None, [], {"g0": g0}
        for g in range(c["rewrites"] + 1):
            row = self.generation(L, g, cur["name"], cur["text"])
            if g == 0:
                row["kept"], parent_row = None, row
            else:
                rv = json.loads((pdir / f"revision_{cur['name']}.json").read_text())
                row |= {"parent": parent["name"], "parent_mean": parent_row["mean"], "kept": row["mean"] > parent_row["mean"],
                        "changes": rv["result"]["changes"], "header_fallback": rv["result"].get("header_fallback", False)}
                if row["kept"]:
                    parent, parent_row = cur, row
            hist.append(row)
            self.log(f"lineage {L} generation {g} ({cur['name']}): mean confirmed {row['mean']:.3f}"
                     + ("" if g == 0 else f" vs {row['parent_mean']:.3f} for {row['parent']}: {'kept' if row['kept'] else 'not kept'}"))
            if g == c["rewrites"]:
                break
            if g == 0:
                with ThreadPoolExecutor(3) as ex:
                    fs = {k: ex.submit(self.rewrite, L, k, 0, parent, row, hist) for k in ("real", "facts", "shuffled")}
                    texts = {k: f.result() for k, f in fs.items()}
                out["facts"], out["shuffled"] = texts["facts"], texts["shuffled"]
                cur = {"name": "g1", "text": texts["real"]}
            else:
                cur = {"name": f"g{g + 1}", "text": self.rewrite(L, "real", g, parent, row, hist)}
        out["g1"] = (pdir / "g1.md").read_text()
        out["final"], out["final_name"] = parent["text"], parent["name"]
        summary = {"lineage": L, "final": parent["name"],
                   "programs": {k: sha(v) for k, v in out.items() if k != "final_name"},
                   "generations": [{k: v for k, v in r.items() if k != "searches"} for r in hist]}
        (self.dir / "meta" / L / "lineage.json").write_text(json.dumps(summary, indent=1))
        return out

    # ---- meta-test on W
    def g0_reuse(self):
        """Decision 5: reuse 2b's g0 arm only if the conditions are identical. Returns (ok, reasons, entries)."""
        if not self.cfg["reuse_g0"]:
            return False, ["reuse is off in this preset"], []
        reasons = []
        m2 = json.loads((CAMP_2B / "manifest.json").read_text())
        if (m2["model"], m2["effort"]) != (S.MODEL, S.EFFORT):
            reasons.append(f"model or effort differs: 2b {m2['model']}/{m2['effort']}")
        reasons += [f"{rel} changed since 2b" for rel, h in m2["hashes"].items() if O.file_hash(ROOT / rel) != h]
        g0set = settings(P.G0_HEADER)
        cf = m2["config"]
        for k in ("R", "K", "trial_seeds", "retest_top", "retest_seeds"):
            if cf[k] != g0set[k]:
                reasons.append(f"2b's {k} {cf[k]} differs from g0's header ({g0set[k]})")
        for k in ("confirm_seeds", "base_seeds"):
            if cf[k] != self.cfg[k]:
                reasons.append(f"2b's {k} differs")
        if (cf["steps"][W], cf["timeout"][W]) != (self.cfg["steps"][W], self.cfg["timeout"][W]):
            reasons.append("2b's W budget or time limit differs")
        if g0set["roles"] != ["both"] * 3:
            reasons.append("g0's split is not 3 x both")
        body = P.parse_program(P.g0_program(G0R.read_text()))[1]
        V = self.view(W, g0set)
        prompt = S.l1_prompt(body, self.task_text(V, W, g0set["roles"]), self.interface(), BASE.read_text(),
                             V.base[W]["shown_dev"], [], None, 1, g0set["R"], g0set["K"])
        saved = json.loads((CAMP_2B / "W" / "g0" / "s1" / "round_1.json").read_text())["prompt"]
        if not (saved == prompt or saved.startswith(prompt + "\n\nYour previous reply was rejected:")):
            reasons.append("the round-1 prompt 2c would send differs from 2b's")
        log = (CAMP_2B / "progress.log").read_text()
        summ = CAMP_2B / "confirm" / "summary.json"
        entries = []
        if "STAGE2B_DONE" not in log or not summ.exists():
            reasons.append("2b has not finished")
        else:
            entries = [e for e in json.loads(summ.read_text()) if e["kind"] == "search" and e["arm"] == "g0"]
            if len(entries) != self.cfg["test_searches"] * len(self.cfg["lineages"]):
                reasons.append(f"2b has {len(entries)} g0 searches")
        return not reasons, reasons, entries

    def meta_test(self, lins: dict) -> list:
        c = self.cfg
        ok, reasons, reused = self.g0_reuse()
        self.log(f"g0 on W: {'reused from 2b' if ok else 'rerun here'} ({'; '.join(reasons) or 'identity check passed'})")
        (self.dir / "W").mkdir(exist_ok=True)
        (self.dir / "W" / "g0_reuse.json").write_text(json.dumps({"reused": ok, "reasons": reasons}, indent=1))
        jobs, notes = [], {}
        for i in range(1, c["test_searches"] + 1):
            for L in c["lineages"]:
                for arm in TEST_ARMS:
                    text = lins[L][arm]
                    if arm == "final" and sha(text) in (sha(lins[L]["g0"]), sha(lins[L]["g1"])):
                        notes[L] = "final is g0: nothing was kept" if sha(text) == sha(lins[L]["g0"]) else "final is g1: searches shared"
                        continue
                    jobs.append((arm, L, i, text))
                if not ok:
                    jobs.append(("g0", None, (i - 1) * len(c["lineages"]) + c["lineages"].index(L) + 1, lins[L]["g0"]))

        def run(job):
            arm, L, i, text = job
            sdir = self.dir / "W" / arm / (L or "") / f"s{i}"
            return {"arm": arm, "lineage": L, "search": i} | self.one_search_2c(sdir, W, text, c["confirm_seeds"])
        with ThreadPoolExecutor(c["search_workers"]) as ex:
            entries = list(ex.map(run, jobs))
        for L, note in notes.items():
            if note.startswith("final is g1"):
                entries += [e | {"arm": "final", "shared_with": "g1"} for e in entries if e["arm"] == "g1" and e["lineage"] == L]
        if ok:
            entries += [e | {"arm": "g0", "lineage": None, "reused_from": "stage2b", "world": W} for e in reused]
        (self.dir / "W" / "final_notes.json").write_text(json.dumps(notes, indent=1))
        return entries

    def run(self):
        c = self.cfg
        self.log(f"start preset={self.preset}")
        with ThreadPoolExecutor(2) as ex:
            fs = {w: ex.submit(self.baseline, w) for w in (WS, W)}
            self.base = {w: f.result() for w, f in fs.items()}
        with ThreadPoolExecutor(c["lineage_workers"]) as ex:
            lins = dict(zip(c["lineages"], ex.map(self.lineage, c["lineages"])))
        self.log("meta-training done: final programs " + ", ".join(f"{L} {lins[L]['final_name']}" for L in c["lineages"]))
        entries = self.meta_test(lins)
        (self.dir / "confirm").mkdir(exist_ok=True)
        (self.dir / "confirm" / "summary.json").write_text(json.dumps(entries, indent=1))
        self.verify()
        self.log("STAGE2C_DONE: run review_2c.py, then analyze_2c.py. W's final split was not opened.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    args = ap.parse_args()
    if "STAGE2B_DONE" not in (CAMP_2B / "progress.log").read_text():
        raise SystemExit("Stage 2b is still running (no STAGE2B_DONE): 2c would compete for the GPU")
    if args.preset == "2c" and not APPROVAL_FILE.exists():
        raise SystemExit(f"{APPROVAL_FILE.name} is missing: 2c runs only after 2b's gate passes and the user approves")
    Stage2c((ROOT / PRESETS[args.preset]["campaign"]).resolve(), args.preset).run()


if __name__ == "__main__":
    main()
