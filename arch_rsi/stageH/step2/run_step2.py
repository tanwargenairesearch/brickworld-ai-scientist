"""Step 2 runner (PROTOCOL_STEP2.md). Resumable: everything finished is kept on disk and reused.

For each world (shift, mirror): the start (reference A) trains seeds 0 and 1 (with readouts) and 5001-5004. Then
12 searches per arm (told, readouts, scores, hidden), interleaved, up to 10 in flight: 4 rounds of 3 candidates, each
checked by Stage H1's check_h.py in the rule world (up to 2 repair calls), trained on seeds 0 and 1 with train_h.py and
scored with eval_h.py on the GPU; readout arms also get readouts.py for the seed-0 run. The pick is the best 2-seed
mean if it beats the start's, else the start; it is confirmed on seeds 5001-5004. Then the rule review (2b's reviewer,
unchanged) of every distinct pick, the diagnosis judge (descriptive), and analyze_step2.py.
usage (project root):
  nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/step2/run_step2.py --preset step2_dry   (1 search per arm and world)
  nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/step2/run_step2.py --preset step2       (needs LAUNCH_APPROVED.md)
"""
import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as C  # noqa: E402
import prompts as P  # noqa: E402

from arch_rsi import scientist as S  # noqa: E402
from arch_rsi.sandbox import sha  # noqa: E402

_spec = importlib.util.spec_from_file_location("review_2b", C.ROOT / "arch_rsi" / "stage2" / "2b" / "review_2b.py")
REV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REV)

PY = sys.executable
BASE = dict(R=4, K=3, trial_seeds=[0, 1], confirm_seeds=[5001, 5002, 5003, 5004], steps=131072, timeout=900,
            gpu_slots=4, check_slots=2, search_workers=10, worlds=list(C.WORLDS), arms=list(C.ARMS))
PRESETS = {"step2": dict(BASE, n_searches=12, campaign="campaigns/step2", needs_launch=True),
           "step2_dry": dict(BASE, n_searches=1, campaign="campaigns/step2_dry", needs_launch=False)}

JUDGE_SYSTEM = ("You are an independent reader judging whether a piece of text identifies a specific fact. Judge only "
                "against the fact given. Reply only with the JSON object requested.")
JUDGE_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["yes", "partly", "no"]},
                                                 "evidence": {"type": "string"}},
                "required": ["verdict", "evidence"], "additionalProperties": False}


def judge_prompt(world: str, analysis: str, name: str, hypothesis: str) -> str:
    return f"""A simulated brick world had this hidden rule:
{P.RULE_TEXT[world]}

Below is what an AI scientist wrote at the end of a search in that world: its last analysis, and the name and
hypothesis of the trial it ended with. Does the text state the rule?
- "yes": it states the rule's substance (what is changed and how).
- "partly": it identifies the right kind of change but not its specifics.
- "no": anything else.

## Last analysis
{analysis}

## Trial it ended with
name: {name}
hypothesis: {hypothesis}

Return a JSON object: "verdict" ("yes", "partly" or "no") and "evidence" (at most 80 words, quoting the text)."""


def judge_validate(o) -> dict:
    if not isinstance(o, dict) or o.get("verdict") not in ("yes", "partly", "no"):
        raise ValueError('"verdict" must be "yes", "partly" or "no"')
    return {"verdict": o["verdict"], "evidence": str(o.get("evidence", ""))[:1000]}


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hash_checks() -> dict:
    m2b = json.loads((C.ROOT / "arch_rsi" / "campaigns" / "stage2b" / "manifest.json").read_text())["hashes"]
    mh = json.loads((C.STAGEH / "manifest.json").read_text())["files"]
    m2 = json.loads((C.STEP2 / "manifest.json").read_text())["files"]
    bad = [p for d in (m2b, mh, m2) for p, h in d.items() if not (C.ROOT / p).exists() or sha_file(C.ROOT / p) != h]
    return {"stage2b": len(m2b), "stageH": len(mh), "step2": len(m2), "changed": bad, "ok": not bad}


class Step2:
    def __init__(self, cdir: Path, preset: str):
        self.cfg = PRESETS[preset]
        self.preset = preset
        self.dir = Path(cdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.gpu = threading.Semaphore(self.cfg["gpu_slots"])
        self.chk = threading.Semaphore(self.cfg["check_slots"])
        self.lock = threading.Lock()
        self.sha_locks = {}
        self.program = C.PROGRAM.read_text()
        self.ref_src = C.common.REF_A.read_text()

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.lock, (self.dir / "progress.log").open("a") as f:
            f.write(line + "\n")

    def verify(self):
        h = hash_checks()
        if not h["ok"]:
            (self.dir / "INTEGRITY_STOP").write_text(json.dumps(h))
            raise SystemExit(f"integrity stop: {h['changed']}")
        return h

    # ---- one training run (cached): train_h.py, eval_h.py, and readouts for a seed-0 run if asked
    def train_seed(self, rdir: Path, src_path: Path, world: str, seed: int, want_readouts: bool) -> dict:
        res_file = rdir / "result.json"
        if res_file.exists():
            res = json.loads(res_file.read_text())
            if not want_readouts or res.get("readouts") or res["status"] != "ok":
                return res
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "run.json").write_text(json.dumps({"candidate": str(src_path), "world": C.common.WORLD, "rule": world,
                                                   "seed": seed, "budget_steps": self.cfg["steps"], "device": "mps", "threads": 1}))
        with self.gpu:
            t0 = time.time()
            res = self._train_eval(rdir, world)
            if res["status"] == "ok" and want_readouts:
                res["readouts"] = self._readouts(rdir, world, seed)
            res.update(seed=seed, world=world, wall_s=round(time.time() - t0, 1))
        res_file.write_text(json.dumps(res))
        return res

    def _train_eval(self, rdir: Path, world: str) -> dict:
        if (rdir / "result.json").exists():
            old = json.loads((rdir / "result.json").read_text())
            if old["status"] == "ok":
                return {k: v for k, v in old.items() if k != "readouts"}
        try:
            p = subprocess.run([PY, "-I", "-B", "-W", "ignore", str(C.STAGEH / "train_h.py"), "--spec", str(rdir / "run.json"),
                                "--out", str(rdir / "run")], capture_output=True, text=True, cwd=C.ROOT, timeout=self.cfg["timeout"])
            lp = rdir / "run" / "train_log.json"
            tl = json.loads(lp.read_text()) if lp.exists() else {}
            status = tl.get("status", f"crashed (exit {p.returncode})")
            err = tl.get("error") or (None if status == "ok" else (p.stderr or "")[-400:])
        except subprocess.TimeoutExpired:
            tl, status, err = {}, "timeout", f"killed after {self.cfg['timeout']} s"
        dev = None
        if status == "ok":
            q = subprocess.run([PY, "-I", "-B", "-W", "ignore", str(C.STAGEH / "eval_h.py"), "--trial", str(rdir / "run"),
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
        q = subprocess.run([PY, "-I", "-B", "-W", "ignore", str(C.STEP2 / "readouts.py"), "--trial", str(rdir / "run"),
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
            p = subprocess.run([PY, "-I", "-B", "-W", "ignore", str(C.STAGEH / "check_h.py"), "--candidate", str(path),
                                "--rule", world, "--out", str(out)], capture_output=True, text=True, cwd=C.ROOT, timeout=1800)
        if p.returncode != 0 or not out.exists():
            return {"ok": False, "error": f"check process failed: {(p.stderr or '')[-300:]}"}
        return json.loads(out.read_text())

    def call(self, prompt: str, k: int, log_path: Path):
        return S.call_with_retries(S.SYSTEM_L1, prompt, S.l1_schema(k), lambda o: S.parse_proposals(o, k), log_path)

    def call_simple(self, system, prompt, schema, validate, log_path: Path):
        return S.call_with_retries(system, prompt, schema, validate, log_path)

    # ---- a candidate on several seeds (orchestrate_arch.Campaign.run_candidate's aggregation)
    def run_candidate(self, cdir: Path, src: str, world: str, seeds: list, check: dict, want_readouts: bool) -> dict:
        cdir.mkdir(parents=True, exist_ok=True)
        sp = cdir / "candidate.py"
        if not sp.exists():
            sp.write_text(src)
        with ThreadPoolExecutor(len(seeds)) as ex:
            per = list(ex.map(lambda s: self.train_seed(cdir / f"seed{s}", sp, world, s, want_readouts and s == seeds[0]), seeds))
        ok = all(r["status"] == "ok" for r in per)
        devs = [r["dev_exact"] if r["status"] == "ok" else 0.0 for r in per]
        bad = [r for r in per if r["status"] != "ok"]
        return {"status": "ok" if ok else bad[0]["status"], "error": None if ok else bad[0]["error"],
                "dev_exact": round(sum(devs) / len(devs), 4), "dev_per_seed": [round(d, 4) for d in devs],
                "train_exact": round(sum((r["train_exact"] or 0) for r in per) / len(per), 4),
                "params": check.get("params"), "flops_ratio": check.get("flops_ratio"),
                "train_steps": max((r.get("train_steps") or 0) for r in per),
                "wall_s": max(r["wall_s"] for r in per), "sha": sha(src), "readouts": per[0].get("readouts")}

    def confirm(self, world: str, src: str) -> dict:
        h = sha(src)
        with self.lock:
            lk = self.sha_locks.setdefault((world, h), threading.Lock())
        with lk:
            d = self.dir / "confirm" / world / h
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "candidate.py").exists():
                (d / "candidate.py").write_text(src)
            seeds = self.cfg["confirm_seeds"]
            with ThreadPoolExecutor(len(seeds)) as ex:
                res = list(ex.map(lambda s: self.train_seed(d / f"seed{s}", d / "candidate.py", world, s, False), seeds))
        return {s: r for s, r in zip(seeds, res)}

    def start(self, world: str) -> dict:
        f = self.dir / "start" / world / "start.json"
        if f.exists():
            return json.loads(f.read_text())
        r = self.run_candidate(self.dir / "start" / world, self.ref_src, world, self.cfg["trial_seeds"], {}, True)
        conf = self.confirm(world, self.ref_src)
        st = {"world": world, "code": self.ref_src, "dev": r["dev_exact"], "devs": r["dev_per_seed"], "status": r["status"],
              "readouts": r["readouts"], "confirm": {str(s): v["dev_exact"] if v["status"] == "ok" else 0.0 for s, v in conf.items()}}
        if st["status"] != "ok" or not st["readouts"]:
            raise SystemExit(f"the start failed in {world}: {r['status']} {r.get('error')}")
        f.write_text(json.dumps(st))
        self.log(f"start {world}: dev {st['devs']} confirm {st['confirm']}")
        return st

    # ---- one search
    def search(self, world: str, arm: str, i: int, start: dict) -> dict:
        sdir = self.dir / world / arm / f"s{i}"
        sdir.mkdir(parents=True, exist_ok=True)
        st_file = sdir / "state.json"
        st = json.loads(st_file.read_text()) if st_file.exists() else {"trials": [], "rounds_done": 0, "incumbent": None}
        R, K, seeds = self.cfg["R"], self.cfg["K"], self.cfg["trial_seeds"]
        want_ro = arm in C.READOUT_ARMS
        for r in range(st["rounds_done"] + 1, R + 1):
            self.verify()
            trials = st["trials"]
            seen = {sha(self.ref_src): "the starting architecture"} | {t["result"]["sha"]: t["label"] for t in trials
                                                                         if t.get("result", {}).get("sha")}
            inc = next((t for t in trials if t["label"] == st["incumbent"]), None)
            ros = {t["label"]: t["result"].get("readouts") for t in trials if t.get("result", {}).get("readouts")}
            prompt = P.build(arm, world, self.program, start, trials, inc, r, R, K, readouts=ros if want_ro else None)
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
                    self.log(f"{world}/{arm}/s{i} round {r} repair {j + 1} failed: {e}")
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
            self.log(f"{world}/{arm}/s{i} round {r}: " + ", ".join(f"{t['label']} {t['result']['status']} {t['result'].get('dev_exact')}"
                                                                    for t in new) + f" | best {st['incumbent']} {best:.3f}")
        return st

    def pick(self, st: dict, start: dict) -> dict:
        ok = [t for t in st["trials"] if t["result"]["status"] == "ok"]
        best = max(ok, key=lambda t: t["result"]["dev_exact"], default=None)
        if best is None or best["result"]["dev_exact"] <= start["dev"]:
            return {"label": "start", "name": "starting architecture", "src": self.ref_src, "sha": sha(self.ref_src),
                    "mean_selected": start["dev"], "hypothesis": ""}
        return {"label": best["label"], "name": best["name"], "src": best["code"], "sha": best["result"]["sha"],
                "mean_selected": best["result"]["dev_exact"], "hypothesis": best["hypothesis"]}

    def one_search(self, world: str, arm: str, i: int, start: dict) -> dict:
        ef = self.dir / world / arm / f"s{i}" / "entry.json"
        if ef.exists():
            return json.loads(ef.read_text())
        st = self.search(world, arm, i, start)
        pk = self.pick(st, start)
        conf = self.confirm(world, pk["src"])
        first = next((t["round"] for t in st["trials"] if t["result"]["status"] == "ok" and t["result"]["dev_exact"] >= C.FOUND), None)
        e = {"world": world, "arm": arm, "search": i, "label": pk["label"], "name": pk["name"], "sha": pk["sha"],
             "hypothesis": pk["hypothesis"], "mean_selected": pk["mean_selected"],
             "dev": {str(s): r["dev_exact"] for s, r in conf.items()}, "status": {str(s): r["status"] for s, r in conf.items()},
             "first_round_found": first, "last_analysis": st.get("analyses", {}).get(str(self.cfg["R"]), "")}
        ef.write_text(json.dumps(e, indent=1))
        self.log(f"{world}/{arm}/s{i} pick {pk['label']} ({pk['name']}) confirmed {[e['dev'][k] for k in sorted(e['dev'])]}")
        return e

    # ---- rule review and diagnosis judge
    def review(self, entries: list) -> dict:
        f = self.dir / "review.json"
        if f.exists():
            return json.loads(f.read_text())
        picks = {}
        for e in entries:
            if e["label"] != "start":
                picks[e["sha"]] = (self.dir / "confirm" / e["world"] / e["sha"] / "candidate.py").read_text()
        shas = sorted(picks)
        random.Random(2026).shuffle(shas)
        (self.dir / "review").mkdir(exist_ok=True)
        out = {}
        for n, h in enumerate(shas):
            letter = chr(ord("A") + n % 26) + (str(n // 26) if n >= 26 else "")
            val, _ = self.call_simple(REV.SYSTEM, REV.prompt(letter, picks[h]), REV.SCHEMA, REV.validate,
                                      self.dir / "review" / f"{h}.json")
            out[h] = val | {"letter": letter}
            self.log(f"review {h[:12]} {letter}: {val['verdict']}")
        res = {"model": S.MODEL, "effort": S.EFFORT, "verdicts": out}
        f.write_text(json.dumps(res, indent=1))
        return res

    def judge(self, entries: list) -> dict:
        f = self.dir / "judge.json"
        if f.exists():
            return json.loads(f.read_text())
        (self.dir / "judge").mkdir(exist_ok=True)
        out = {}
        for e in entries:
            key = f"{e['world']}/{e['arm']}/s{e['search']}"
            val, _ = self.call_simple(JUDGE_SYSTEM, judge_prompt(e["world"], e["last_analysis"], e["name"], e["hypothesis"]),
                                      JUDGE_SCHEMA, judge_validate, self.dir / "judge" / f"{key.replace('/', '_')}.json")
            out[key] = val
        f.write_text(json.dumps(out, indent=1))
        return out

    def run(self):
        if self.cfg["needs_launch"] and not (C.STEP2 / "LAUNCH_APPROVED.md").exists():
            raise SystemExit("the full run needs LAUNCH_APPROVED.md, written only on the user's approval of the dry run")
        pre = self.verify()
        self.log(f"Step 2 runner start ({self.preset}); preflight {pre}")
        t0 = time.time()
        starts = {w: self.start(w) for w in self.cfg["worlds"]}
        jobs = [(w, a, i) for i in range(self.cfg["n_searches"]) for w in self.cfg["worlds"] for a in self.cfg["arms"]]
        entries, errors = [], []
        with ThreadPoolExecutor(self.cfg["search_workers"]) as ex:
            futs = {ex.submit(self.one_search, w, a, i, starts[w]): (w, a, i) for w, a, i in jobs}
            for fu, job in futs.items():
                try:
                    entries.append(fu.result())
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{job}: {type(e).__name__}: {e}")
                    self.log(f"search {job} failed: {type(e).__name__}: {e}")
        if errors:
            raise SystemExit(f"{len(errors)} searches failed; rerun the same command to resume")
        (self.dir / "summary.json").write_text(json.dumps({"entries": entries, "starts": {w: {k: v for k, v in s.items()
                                               if k != "readouts"} for w, s in starts.items()}}, indent=1))
        rev = self.review(entries)
        jud = self.judge(entries)
        import analyze_step2 as AN
        res = AN.analyze(self.dir)
        post = self.verify()
        res.update(status="final", preset=self.preset, preflight=pre, postflight=post,
                   total_wall_h=round((time.time() - t0) / 3600, 2), written_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        (self.dir / "results.json").write_text(json.dumps(res, indent=1))
        self.log(f"done: {res['summary_line']}; review {len(rev['verdicts'])} picks, judge {len(jud)} searches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    args = ap.parse_args()
    Step2(C.STEP2 / PRESETS[args.preset]["campaign"], args.preset).run()


if __name__ == "__main__":
    main()
