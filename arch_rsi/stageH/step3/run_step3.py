"""Step 3 runner (PROTOCOL_STEP3.md). Resumable: everything finished is kept on disk and reused, and no finished
call is repeated (a saved reply is validated again and reused).

Step 2's runner (run_step2.Step2, frozen) is imported and reused for training runs, candidates, confirmation, the pick
and the rule review. Copied and changed, each shown by test_run_step3.py to differ from Step 2's only as stated:
- _train_eval, _readouts and check: Step 3a's train_3a.py, eval_3a.py and check_3a.py, and readouts_3.py, so every
  Step 3a world runs (Stage H's own rules run their own frozen code there);
- search: the header's rounds, candidates and seeds, the program body, and prompts3.build3. With g0 and scores-only
  feedback it writes the same prompts and state as Step 2's search().
Added, following Stage 2c's runner (stage2/2c/run_2c.py): generations, the rewrite calls (real, facts-only, no-logs),
test-before-keep and lineages; plus neutral world labels, the meta-test and its schedule, the check arm, the
diagnosis judge for the held-out worlds and the descriptive program review.

Schedule: g0 and the check arm run on the held-out worlds from the start; each lineage's g1, facts-only and no-logs arms
start once those programs exist, and its final arm after its last generation. Meta-training and meta-test searches
share the GPU (4 runs at once) through separate search pools, so the sequential generations aren't queued behind
the meta-test. Rewrites read only meta-training logs.
usage (project root):
  nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/step3/run_step3.py --preset step3_dry
  nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/step3/run_step3.py --preset step3     (needs LAUNCH_APPROVED.md)
"""
import argparse
import importlib.util
import json
import random
import subprocess
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3common as C  # noqa: E402
import prog3 as PG  # noqa: E402
import prompts3 as Q  # noqa: E402

import run_step2 as RS  # noqa: E402  (Step 2's runner, frozen)
from arch_rsi import scientist as S  # noqa: E402
from arch_rsi.sandbox import sha  # noqa: E402

_spec = importlib.util.spec_from_file_location("review_2c", C.ROOT / "arch_rsi" / "stage2" / "2c" / "review_2c.py")
RV2C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RV2C)

PY = RS.PY
BASE = dict(steps=131072, timeout=900, gpu_slots=4, check_slots=2, meta_workers=14, test_workers=6,
            lineages=list(C.LINEAGES), rewrites=3, train_worlds=list(C.TRAIN_WORLDS), test_worlds=list(C.TEST_WORLDS),
            g0_per_world=8, per_lineage_world=4, check_per_world=4, confirm_seeds=list(C.CONFIRM_SEEDS),
            start_seeds=list(C.START_SEEDS), max_words=C.MAX_WORDS)
PRESETS = {"step3": dict(BASE, campaign="campaigns/step3", needs_launch=True),
           "step3_dry": dict(BASE, campaign="campaigns/step3_dry", needs_launch=False, lineages=["A"], rewrites=1,
                             train_worlds=[C.TRAIN_WORLDS[0]], test_worlds=[C.TEST_WORLDS[0]], g0_per_world=1,
                             per_lineage_world=1, check_per_world=1, meta_workers=2, test_workers=6)}
ARM_FEEDBACK = {"g0": "scores", "g1": "scores", "final": "scores", "facts": "scores", "nologs": "scores", "check": "readouts"}


def hash_checks() -> dict:
    """Step 2's checks (2b's 39 hashes, Stage H's 26 files, Step 2's 12) plus Step 2S's, Step 3a's and Step 3's manifests."""
    h = RS.hash_checks()
    for key, m in (("step2s", C.STEP2S / "manifest.json"), ("step3a", C.STEP3A / "manifest.json"),
                   ("step3", C.STEP3 / "manifest.json")):
        own = json.loads(m.read_text())["files"] if m.exists() else {}
        bad = [p for p, x in own.items() if not (C.ROOT / p).exists() or RS.sha_file(C.ROOT / p) != x]
        h[key] = len(own)
        h["changed"] = h["changed"] + bad + ([] if own else [str(m.relative_to(C.ROOT))])
    h["ok"] = not h["changed"]
    return h


def conf_mean(e: dict) -> float:
    """A search's confirmed score: the mean over its confirmation seeds, a failed run counting 0 (2c's conf_mean)."""
    vals = [(e["dev"][s] if e["status"][s] == "ok" else 0.0) for s in e["dev"]]
    return sum(vals) / len(vals)


def rewrite_validator(parent: str, max_words: int, facts_only: bool, start: int = 0):
    """2c's rewrite_validator with Step 3's header: a full rewrite with a bad header is sent back twice; on the third
    the parent's header is used with the new body (header_fallback). A facts-only rewrite must pass the facts-only
    check, with no fallback, so 3 failures stop the run."""
    ph, _ = PG.parse_program(parent)
    count = {"n": start}

    def validate(obj):
        count["n"] += 1
        val = S.parse_program(obj, max_words)
        if facts_only:
            probs = PG.facts_only_problems(parent, val["program"], max_words)
            if probs:
                raise ValueError("this revision may only add a '## Facts' section: " + "; ".join(probs))
            return dict(val, header_fallback=False)
        try:
            PG.parse_program(val["program"])
            return dict(val, header_fallback=False)
        except ValueError as e:
            if count["n"] < 3:
                raise ValueError(f"the header: {e}. The program must start with the header block in the format "
                                 f"shown, with every value inside its bounds") from e
            prog = PG.render_header(ph) + "\n" + PG.strip_header(val["program"])
            if len(prog.split()) > max_words:
                raise ValueError(f"program has {len(prog.split())} words; the limit is {max_words}") from e
            return dict(val, program=prog, header_fallback=True, header_error=str(e))
    return validate


class Step3(RS.Step2):
    def __init__(self, cdir: Path, preset: str):
        # Step2.__init__'s attributes, with this step's presets (test_run_step3.py checks none is missing)
        self.cfg = PRESETS[preset]
        self.preset = preset
        self.dir = Path(cdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.gpu = threading.Semaphore(self.cfg["gpu_slots"])
        self.chk = threading.Semaphore(self.cfg["check_slots"])
        self.lock = threading.Lock()
        self.sha_locks = {}
        self.program = C.PROGRAM_G0R.read_text()
        self.ref_src = C.common.REF_A.read_text()
        # this step's
        self.g0 = PG.g0_program(self.program)
        self.labels = {L: PG.world_labels(L, self.cfg["train_worlds"]) for L in self.cfg["lineages"]}
        self.starts = {}
        self.test_futs, self.notes = [], {}

    def verify(self):
        h = hash_checks()
        if not h["ok"]:
            (self.dir / "INTEGRITY_STOP").write_text(json.dumps(h))
            raise SystemExit(f"integrity stop: {h['changed']}")
        return h

    # ---- calls: a finished call is never repeated (Step 2S's rule)
    def cached(self, system, prompt, schema, validate, log_path: Path):
        if log_path.exists():
            rec = json.loads(log_path.read_text())
            if "reply" in rec:
                if not str(rec.get("prompt", "")).startswith(prompt):
                    raise SystemExit(f"{log_path}: the saved call has a different prompt; refusing to reuse it")
                return validate(rec["reply"]), rec
        return S.call_with_retries(system, prompt, schema, validate, log_path)

    def call(self, prompt: str, k: int, log_path: Path):
        return self.cached(S.SYSTEM_L1, prompt, S.l1_schema(k), lambda o: S.parse_proposals(o, k), log_path)

    def call_simple(self, system, prompt, schema, validate, log_path: Path):
        return self.cached(system, prompt, schema, validate, log_path)

    # ---- Step 2's training, readouts and check, with Step 3a's code (copies; only the script paths differ)
    def _train_eval(self, rdir: Path, world: str) -> dict:
        if (rdir / "result.json").exists():
            old = json.loads((rdir / "result.json").read_text())
            if old["status"] == "ok":
                return {k: v for k, v in old.items() if k != "readouts"}
        try:
            p = subprocess.run([PY, "-I", "-B", "-W", "ignore", str(C.STEP3A / "train_3a.py"), "--spec", str(rdir / "run.json"),
                                "--out", str(rdir / "run")], capture_output=True, text=True, cwd=C.ROOT, timeout=self.cfg["timeout"])
            lp = rdir / "run" / "train_log.json"
            tl = json.loads(lp.read_text()) if lp.exists() else {}
            status = tl.get("status", f"crashed (exit {p.returncode})")
            err = tl.get("error") or (None if status == "ok" else (p.stderr or "")[-400:])
        except subprocess.TimeoutExpired:
            tl, status, err = {}, "timeout", f"killed after {self.cfg['timeout']} s"
        dev = None
        if status == "ok":
            q = subprocess.run([PY, "-I", "-B", "-W", "ignore", str(C.STEP3A / "eval_3a.py"), "--trial", str(rdir / "run"),
                                "--rule", world, "--designs", "hplain_dev", "--out", str(rdir / "dev.json")],
                               capture_output=True, text=True, cwd=C.ROOT, timeout=1800)
            if q.returncode == 0:
                dev = json.loads((rdir / "dev.json").read_text())["exact"]
            else:
                status, err = "eval_failed", (q.stderr or "")[-400:]
        return {"status": status, "error": err, "dev_exact": dev, "train_exact": tl.get("train_exact_tail"),
                "params": tl.get("params"), "train_steps": tl.get("steps"), "flops_ratio": tl.get("flops_ratio"),
                "n_updates": tl.get("n_updates"), "recipe": tl.get("recipe"), "train_wall_s": tl.get("wall_s")}

    def _readouts(self, rdir: Path, world: str, seed: int):
        out = rdir / "readouts.json"
        q = subprocess.run([PY, "-I", "-B", "-W", "ignore", str(C.STEP3 / "readouts_3.py"), "--trial", str(rdir / "run"),
                            "--rule", world, "--seed", str(seed), "--out", str(out)], capture_output=True, text=True,
                           cwd=C.ROOT, timeout=900)
        if q.returncode != 0 or not out.exists():
            self.log(f"readouts failed for {rdir}: {(q.stderr or '')[-300:]}")
            return None
        ro = json.loads(out.read_text())
        return {"text_full": ro["text_full"], "text_short": ro["text_short"]}

    def check(self, path: Path, world: str) -> dict:
        out = path.with_suffix(".check.json")
        if out.exists():
            return json.loads(out.read_text())
        with self.chk:
            p = subprocess.run([PY, "-I", "-B", "-W", "ignore", str(C.STEP3A / "check_3a.py"), "--candidate", str(path),
                                "--rule", world, "--out", str(out)], capture_output=True, text=True, cwd=C.ROOT, timeout=1800)
        if p.returncode != 0 or not out.exists():
            return {"ok": False, "error": f"check process failed: {(p.stderr or '')[-300:]}"}
        return json.loads(out.read_text())

    # ---- the start in each world: seeds 0-2 (seed 0 with readouts in the held-out worlds) and 5001-5004
    def start3(self, world: str) -> dict:
        f = self.dir / "start" / world / "start.json"
        if f.exists():
            return json.loads(f.read_text())
        sdir = self.dir / "start" / world
        sdir.mkdir(parents=True, exist_ok=True)
        sp = sdir / "candidate.py"
        if not sp.exists():
            sp.write_text(self.ref_src)
        want_ro = world in self.cfg["test_worlds"]
        seeds = self.cfg["start_seeds"]
        with ThreadPoolExecutor(len(seeds)) as ex:
            per = list(ex.map(lambda s: self.train_seed(sdir / f"seed{s}", sp, world, s, want_ro and s == seeds[0]), seeds))
        conf = self.confirm(world, self.ref_src)
        st = {"world": world, "code": self.ref_src, "seeds": seeds,
              "raw": [r["dev_exact"] if r["status"] == "ok" else 0.0 for r in per], "status": [r["status"] for r in per],
              "readouts": per[0].get("readouts"),
              "confirm": {str(s): v["dev_exact"] if v["status"] == "ok" else 0.0 for s, v in conf.items()}}
        if any(s != "ok" for s in st["status"]) or (want_ro and not st["readouts"]):
            raise SystemExit(f"the start failed in {world}: {st['status']}")
        f.write_text(json.dumps(st))
        self.log(f"start {world}: seeds {st['raw']} confirm {st['confirm']}")
        return st

    # ---- one search: Step 2's search() under a program's header and body (copy; see the module docstring)
    def search3(self, sdir: Path, world: str, arm: str, body: str, sets: dict, start: dict) -> dict:
        sdir.mkdir(parents=True, exist_ok=True)
        st_file = sdir / "state.json"
        st = json.loads(st_file.read_text()) if st_file.exists() else {"trials": [], "rounds_done": 0, "incumbent": None}
        R, K, seeds = sets["R"], sets["K"], sets["trial_seeds"]
        want_ro = arm == "readouts"
        tag = str(sdir.relative_to(self.dir))
        for r in range(st["rounds_done"] + 1, R + 1):
            self.verify()
            trials = st["trials"]
            seen = {sha(self.ref_src): "the starting architecture"} | {t["result"]["sha"]: t["label"] for t in trials
                                                                         if t.get("result", {}).get("sha")}
            inc = next((t for t in trials if t["label"] == st["incumbent"]), None)
            ros = {t["label"]: t["result"].get("readouts") for t in trials if t.get("result", {}).get("readouts")}
            prompt = Q.build3(arm, world, body, start, trials, inc, r, R, K, len(seeds), readouts=ros if want_ro else None)
            val, _ = self.call(prompt, K, sdir / f"round_{r}.json")
            props, analysis = val["proposals"], val["analysis"]
            checks = [None] * K
            for j in range(3):
                todo = [k for k in range(K) if checks[k] is None or not checks[k]["ok"]]

                def run_check(k, j=j):
                    p = sdir / "cands" / f"r{r}c{k + 1}_v{j}.py"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(props[k]["code"])
                    return self.check(p, world)
                with ThreadPoolExecutor(max(1, len(todo))) as ex:
                    got = dict(zip(todo, ex.map(run_check, todo)))
                for k in todo:
                    c = got[k]
                    h = sha(props[k]["code"])
                    dup = [m for m in range(K) if m < k and sha(props[m]["code"]) == h]
                    if c.get("ok") and (h in seen or dup):
                        c = {"ok": False, "error": f"identical to {seen.get(h) or f'proposal {dup[0] + 1} in this round'}"}
                    checks[k] = c
                failed = [k for k in range(K) if not checks[k]["ok"]]
                if not failed or j == 2:
                    break
                fl = [{"name": props[k]["name"], "error": checks[k]["error"], "code": props[k]["code"]} for k in failed]
                try:
                    fix, _ = self.call(S.repair_prompt(prompt, fl), len(failed), sdir / f"round_{r}_repair{j + 1}.json")
                except RuntimeError as e:
                    self.log(f"{tag} round {r} repair {j + 1} failed: {e}")
                    break
                for k, f in zip(failed, fix["proposals"]):
                    props[k] = {"name": f["name"] or props[k]["name"], "hypothesis": f["hypothesis"], "code": f["code"]}
            new, jobs = [], []
            for k in range(K):
                t = {"label": f"r{r}c{k + 1}", "round": r, "name": props[k]["name"], "hypothesis": props[k]["hypothesis"],
                     "code": props[k]["code"], "check": checks[k]}
                if not checks[k]["ok"]:
                    t["result"] = {"status": "invalid", "error": checks[k]["error"], "dev_exact": None, "sha": sha(props[k]["code"])}
                else:
                    jobs.append(t)
                new.append(t)
            with ThreadPoolExecutor(max(1, len(jobs))) as ex:
                outs = list(ex.map(lambda t: self.run_candidate(sdir / "trials" / t["label"], t["code"], world, seeds,
                                                                t["check"], want_ro), jobs))
            for t, o in zip(jobs, outs):
                t["result"] = o
            st["trials"] = trials + new
            st.setdefault("analyses", {})[str(r)] = analysis
            best = max([0.0 if inc is None else inc["result"]["dev_exact"], start["dev"]])
            for t in new:
                if t["result"]["status"] == "ok" and t["result"]["dev_exact"] > best:
                    best, st["incumbent"] = t["result"]["dev_exact"], t["label"]
            st["rounds_done"] = r
            st_file.write_text(json.dumps(st, indent=1))
            self.log(f"{tag} round {r}: " + ", ".join(f"{t['label']} {t['result']['status']} {t['result'].get('dev_exact')}"
                                                      for t in new) + f" | best {st['incumbent']} {best:.3f}")
        return st

    def one_search3(self, sdir: Path, world: str, arm: str, program: str, meta: dict) -> dict:
        """A search under a program (header + body), its pick confirmed on seeds 5001-5004. Cached in entry.json."""
        ef = sdir / "entry.json"
        if ef.exists():
            return json.loads(ef.read_text())
        h, body = PG.parse_program(program)
        sets = PG.settings(h)
        view = Q.start_view(self.starts[world], len(sets["trial_seeds"]))
        feedback = ARM_FEEDBACK[arm] if arm in ARM_FEEDBACK else "scores"
        st = self.search3(sdir, world, feedback, body, sets, view)
        pk = self.pick(st, view)
        conf = self.confirm(world, pk["src"])
        first = next((t["round"] for t in st["trials"] if t["result"]["status"] == "ok" and t["result"]["dev_exact"] >= C.FOUND), None)
        e = dict(meta, world=world, arm=arm, path=str(sdir.relative_to(self.dir)), program_sha=sha(program), settings=sets,
                 feedback=feedback, label=pk["label"], name=pk["name"], sha=pk["sha"], hypothesis=pk["hypothesis"],
                 mean_selected=pk["mean_selected"], dev={str(s): r["dev_exact"] for s, r in conf.items()},
                 status={str(s): r["status"] for s, r in conf.items()}, first_round_found=first,
                 n_trials=len(st["trials"]), n_invalid=sum(t["result"]["status"] == "invalid" for t in st["trials"]))
        e["score"] = conf_mean(e)
        ef.write_text(json.dumps(e, indent=1))
        self.log(f"{e['path']} pick {pk['label']} ({pk['name']}) confirmed {[e['dev'][k] for k in sorted(e['dev'])]}")
        return e

    # ---- meta-training (2c's generation, gen_table, rewrite and lineage, with this step's changes)
    def generation(self, L: str, g: int, name: str, program: str) -> dict:
        gdir = self.dir / "meta" / L / f"gen{g}"
        worlds = sorted(self.cfg["train_worlds"], key=lambda w: self.labels[L][w])
        futs = [self.meta_pool.submit(self.one_search3, gdir / w, w, f"meta:{name}", program,
                                      {"phase": "meta", "lineage": L, "generation": g, "program": name}) for w in worlds]
        entries = [f.result() for f in futs]
        means = [e["score"] for e in entries]
        states = [json.loads((self.dir / e["path"] / "state.json").read_text()) for e in entries]
        stats = {"best": [round(max([t["result"]["dev_exact"] for t in s["trials"] if t["result"]["status"] == "ok"], default=0.0), 3)
                          for s in states],
                 "invalid": sum(t["result"]["status"] == "invalid" for s in states for t in s["trials"]),
                 "failed": sum(t["result"]["status"] not in ("ok", "invalid") for s in states for t in s["trials"])}
        row = {"generation": g, "program": name, "program_sha": sha(program), "settings": entries[0]["settings"],
               "worlds": worlds, "search_means": means, "mean": sum(means) / len(means), "stats": stats, "searches": entries}
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

    def logs_for(self, L: str, row: dict) -> str:
        """Generation row's searches, in label order, as the rewrite reads them. Meta-training only."""
        searches = []
        for e in row["searches"]:
            if e.get("phase") != "meta" or e["world"] not in self.cfg["train_worlds"]:
                raise SystemExit("a rewrite may read only meta-training searches")
            st = json.loads((self.dir / e["path"] / "state.json").read_text())
            view = Q.start_view(self.starts[e["world"]], len(e["settings"]["trial_seeds"]))
            searches.append({"label": self.labels[L][e["world"]], "trials": st["trials"],
                             "analyses": st.get("analyses", {}), "base_dev": view["dev"]})
        return PG.format_logs(searches, S.fmt_trial)

    def task_example(self, n: int) -> str:
        w = self.cfg["train_worlds"][0]
        return Q.task_text3("scores", w, Q.start_view(self.starts[w], n), n)

    def rewrite(self, L: str, kind: str, g: int, parent: dict, latest: dict, hist: list) -> str:
        """kind: real (g{g+1} from the parent), facts or nologs (written once, from g0 and generation 0)."""
        pdir = self.dir / "programs" / L
        name = {"real": f"g{g + 1}", "facts": "facts", "nologs": "nologs"}[kind]
        out_prog = pdir / f"{name}.md"
        if out_prog.exists():
            return out_prog.read_text()
        n = len(latest["settings"]["trial_seeds"])
        logs = "" if kind == "nologs" else self.logs_for(L, latest)
        table = "" if kind == "nologs" else self.gen_table(hist, parent["name"])
        prompt = Q.l3_prompt(kind, parent["name"], parent["text"], self.task_example(n), table,
                             f"generation {g} (program {latest['program']})", logs, self.cfg["max_words"],
                             n_worlds=len(self.cfg["train_worlds"]))
        call = pdir / f"call_{name}.json"
        start = json.loads(call.read_text()).get("attempt", 1) - 1 if call.exists() else 0   # 2c: replay at the saved attempt
        val, _ = self.cached(S.SYSTEM_L2, prompt, S.L2_SCHEMA,
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
        if (pdir / "g0.md").exists() and (pdir / "g0.md").read_text() != self.g0:
            raise SystemExit(f"{pdir / 'g0.md'} differs from g0")
        (pdir / "g0.md").write_text(self.g0)
        (pdir / "labels.json").write_text(json.dumps(self.labels[L], indent=1))
        parent, cur = {"name": "g0", "text": self.g0}, {"name": "g0", "text": self.g0}
        parent_row, hist, out = None, [], {"g0": self.g0}
        for g in range(c["rewrites"] + 1):
            row = self.generation(L, g, cur["name"], cur["text"])
            if g == 0:
                row["kept"], parent_row = None, row
            else:
                rv = json.loads((pdir / f"revision_{cur['name']}.json").read_text())
                row |= {"parent": parent["name"], "parent_mean": parent_row["mean"], "kept": row["mean"] > parent_row["mean"],
                        "changes": rv["result"]["changes"], "header_fallback": rv["result"].get("header_fallback", False),
                        "header": PG.parse_program(cur["text"])[0], "parent_header": PG.parse_program(parent["text"])[0]}
                if row["kept"]:
                    parent, parent_row = cur, row
            hist.append(row)
            self.log(f"lineage {L} generation {g} ({cur['name']}): mean confirmed {row['mean']:.3f}"
                     + ("" if g == 0 else f" vs {row['parent_mean']:.3f} for {row['parent']}: {'kept' if row['kept'] else 'not kept'}"))
            if g == c["rewrites"]:
                break
            if g == 0:
                with ThreadPoolExecutor(3) as ex:
                    fs = {k: ex.submit(self.rewrite, L, k, 0, parent, row, hist) for k in ("real", "facts", "nologs")}
                    texts = {k: f.result() for k, f in fs.items()}
                out |= {"g1": texts["real"], "facts": texts["facts"], "nologs": texts["nologs"]}
                for arm in ("g1", "facts", "nologs"):
                    self.submit_arm(arm, L, out[arm])
                cur = {"name": "g1", "text": texts["real"]}
            else:
                cur = {"name": f"g{g + 1}", "text": self.rewrite(L, "real", g, parent, row, hist)}
        out["final"], out["final_name"] = parent["text"], parent["name"]
        if sha(out["final"]) == sha(out["g0"]):
            self.notes[L] = "final is g0: nothing was kept"
        elif sha(out["final"]) == sha(out["g1"]):
            self.notes[L] = "final is g1: searches shared"
        else:
            self.submit_arm("final", L, out["final"])
        summary = {"lineage": L, "final": parent["name"], "labels": self.labels[L],
                   "programs": {k: sha(v) for k, v in out.items() if k != "final_name"},
                   "generations": [{k: v for k, v in r.items() if k != "searches"} for r in hist]}
        (self.dir / "meta" / L / "lineage.json").write_text(json.dumps(summary, indent=1))
        return out

    # ---- meta-test on the held-out worlds
    def submit_arm(self, arm: str, L, program: str):
        n = self.cfg["g0_per_world"] if arm == "g0" else (self.cfg["check_per_world"] if arm == "check" else self.cfg["per_lineage_world"])
        for i in range(1, n + 1):
            for w in self.cfg["test_worlds"]:
                sdir = self.dir / "test" / arm / (L or "-") / w / f"s{i}"
                fu = self.test_pool.submit(self.one_search3, sdir, w, arm, program,
                                           {"phase": "test", "lineage": L, "search": i})
                with self.lock:
                    self.test_futs.append(((arm, L, w, i), fu))

    # ---- reviews, judge
    def judge3(self, entries: list) -> dict:
        f = self.dir / "judge.json"
        if f.exists():
            return json.loads(f.read_text())
        (self.dir / "judge").mkdir(exist_ok=True)
        out = {}
        for e in entries:
            if e.get("shared_with"):
                continue
            st = json.loads((self.dir / e["path"] / "state.json").read_text())
            analyses = [st.get("analyses", {}).get(str(r), "") for r in range(1, e["settings"]["R"] + 1)]
            val, _ = self.call_simple(RS.JUDGE_SYSTEM, Q.judge_prompt(e["world"], analyses, e["name"], e["hypothesis"]),
                                      RS.JUDGE_SCHEMA, RS.judge_validate, self.dir / "judge" / f"{e['path'].replace('/', '_')}.json")
            out[e["path"]] = val
        f.write_text(json.dumps(out, indent=1))
        return out

    def program_review(self) -> dict:
        """Descriptive: one blind call per distinct program (2c's program review, imported unchanged)."""
        f = self.dir / "program_review.json"
        if f.exists():
            return json.loads(f.read_text())
        progs = {}
        for L in self.cfg["lineages"]:
            for p in sorted((self.dir / "programs" / L).glob("*.md")):
                progs.setdefault(sha(p.read_text()), (p.read_text(), []))[1].append(f"{L}/{p.stem}")
        keys = sorted(progs)
        random.Random(2027).shuffle(keys)
        (self.dir / "program_review").mkdir(exist_ok=True)
        out = {}
        for n, h in enumerate(keys):
            letter = RV2C.letters(len(keys))[n]
            val, _ = self.call_simple(RV2C.PROGRAM_SYSTEM, RV2C.program_prompt(letter, progs[h][0]), RV2C.PROGRAM_SCHEMA,
                                      RV2C.validate_program, self.dir / "program_review" / f"{h}.json")
            out[h] = val | {"letter": letter, "programs": progs[h][1]}
        f.write_text(json.dumps(out, indent=1))
        return out

    def run(self):
        c = self.cfg
        if c["needs_launch"] and not (C.STEP3 / "LAUNCH_APPROVED.md").exists():
            raise SystemExit("the full run needs LAUNCH_APPROVED.md, written only on the user's approval of the dry run")
        pre = self.verify()
        self.log(f"Step 3 runner start ({self.preset}); preflight {pre}")
        t0 = time.time()
        worlds = list(dict.fromkeys(c["train_worlds"] + c["test_worlds"]))
        with ThreadPoolExecutor(len(worlds)) as ex:
            self.starts = dict(zip(worlds, ex.map(self.start3, worlds)))
        with ThreadPoolExecutor(c["meta_workers"]) as mp, ThreadPoolExecutor(c["test_workers"]) as tp:
            self.meta_pool, self.test_pool = mp, tp
            self.submit_arm("g0", None, self.g0)
            self.submit_arm("check", None, self.g0)
            with ThreadPoolExecutor(len(c["lineages"])) as lx:
                lfs = {L: lx.submit(self.lineage, L) for L in c["lineages"]}
                lins, errors = {}, []
                for L, fu in lfs.items():
                    try:
                        lins[L] = fu.result()
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"lineage {L}: {type(e).__name__}: {e}")
                        self.log(f"lineage {L} failed: {type(e).__name__}: {e}")
            entries = []
            for key, fu in list(self.test_futs):
                try:
                    entries.append(fu.result())
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{key}: {type(e).__name__}: {e}")
                    self.log(f"search {key} failed: {type(e).__name__}: {e}")
        if errors:
            raise SystemExit(f"{len(errors)} failures; rerun the same command to resume")
        for L, note in self.notes.items():
            if note.startswith("final is g1"):
                entries += [dict(e, arm="final", shared_with="g1") for e in entries if e["arm"] == "g1" and e["lineage"] == L]
        (self.dir / "summary.json").write_text(json.dumps({
            "entries": entries, "notes": self.notes,
            "starts": {w: {k: v for k, v in s.items() if k not in ("readouts", "code")} for w, s in self.starts.items()},
            "lineages": {L: json.loads((self.dir / "meta" / L / "lineage.json").read_text()) for L in c["lineages"]}}, indent=1))
        rev = self.review([e for e in entries if not e.get("shared_with")])
        jud = self.judge3(entries)
        prv = self.program_review()
        import analyze_step3 as AN
        res = AN.analyze(self.dir)
        post = self.verify()
        res.update(status="final", preset=self.preset, preflight=pre, postflight=post,
                   total_wall_h=round((time.time() - t0) / 3600, 2),
                   written_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        (self.dir / "results.json").write_text(json.dumps(res, indent=1))
        self.log(f"done: {res['summary_line']}; review {len(rev['verdicts'])} picks, judge {len(jud)} searches, "
                 f"{len(prv)} programs reviewed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    args = ap.parse_args()
    for f, what in ((C.STEP2_RESULTS, "Step 2"), (C.STEP3A_RESULTS, "Step 3a")):
        if not f.exists() or json.loads(f.read_text()).get("status") != "final":
            raise SystemExit(f"Step 3 needs {what}'s final results: {f}")
    Step3(C.STEP3 / PRESETS[args.preset]["campaign"], args.preset).run()


if __name__ == "__main__":
    main()
