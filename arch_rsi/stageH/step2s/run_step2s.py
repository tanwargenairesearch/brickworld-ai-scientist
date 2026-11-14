"""Step 2S runner (PROTOCOL_STEP2S.md). Resumable: everything finished is kept on disk and reused, and no finished
call is repeated (a saved reply is validated again and reused).

Step 2's runner (run_step2.Step2) is imported and reused for training, checks, repairs, confirmation, the pick, the
rule review and the diagnosis judge. One method is copied and changed: search(). Its only changes are the base prompt
(prompts_2s.base_prompt), no readouts in any arm, and, in the sandbox arms, the probe call and probes before each
proposal call (sandbox_round). test_run_step2s.py shows that for the scores arm it writes the same prompts and state as
Step 2's search(). The worlds come from Step 2's final results (s2s_common.qualify).
usage (project root):
  nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/step2s/run_step2s.py --preset step2s_dry   (1 search per arm and world)
  nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/step2s/run_step2s.py --preset step2s       (needs LAUNCH_APPROVED.md)
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2s_common as C  # noqa: E402

import prompts_2s as PR  # noqa: E402
import probes_2s as PB  # noqa: E402
import run_step2 as RS  # noqa: E402  (Step 2's)

from arch_rsi import scientist as S  # noqa: E402
from arch_rsi.sandbox import sha  # noqa: E402

BASE = dict(RS.BASE, arms=list(C.ARMS))
PRESETS = {"step2s": dict(BASE, campaign="campaigns/step2s", needs_launch=True, dry=False),
           "step2s_dry": dict(BASE, campaign="campaigns/step2s_dry", needs_launch=False, dry=True)}


def hash_checks() -> dict:
    """Step 2's checks (2b's 39 hashes, Stage H's 26 files, Step 2's manifest) plus Step 2S's own manifest."""
    h = RS.hash_checks()
    m = C.STEP2S / "manifest.json"
    own = json.loads(m.read_text())["files"] if m.exists() else {}
    bad = [p for p, x in own.items() if not (C.ROOT / p).exists() or RS.sha_file(C.ROOT / p) != x]
    h.update(step2s=len(own), changed=h["changed"] + bad + ([] if own else ["arch_rsi/stageH/step2s/manifest.json"]))
    h["ok"] = not h["changed"]
    return h


class Step2S(RS.Step2):
    def __init__(self, cdir: Path, preset: str, qual: dict = None):
        # Step2.__init__'s attributes, with this step's presets (test_run_step2s.py checks none is missing)
        self.cfg = PRESETS[preset]
        self.preset = preset
        self.dir = Path(cdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.gpu = threading.Semaphore(self.cfg["gpu_slots"])
        self.chk = threading.Semaphore(self.cfg["check_slots"])
        self.lock = threading.Lock()
        self.sha_locks = {}
        self.program = C.C2.PROGRAM.read_text()
        self.ref_src = C.common.REF_A.read_text()
        # this step's
        if qual is None:
            if not C.STEP2_RESULTS.exists():
                raise SystemExit("Step 2S waits for Step 2's full-run results: " + str(C.STEP2_RESULTS))
            qual = C.qualify(json.loads(C.STEP2_RESULTS.read_text()))
        self.qual = qual
        self.worlds = list(qual["worlds"])
        if not self.worlds:
            raise SystemExit("no world qualifies (PROTOCOL_STEP2S.md, \"Which worlds run\"): Step 2S doesn't run")
        self.n_per_arm = {a: 1 for a in C.ARMS} if self.cfg["dry"] else dict(qual["searches_per_world"])
        self.probe_lock = threading.Lock()
        self._designs, self._rules = None, {}

    def verify(self):
        h = hash_checks()
        if not h["ok"]:
            (self.dir / "INTEGRITY_STOP").write_text(json.dumps(h))
            raise SystemExit(f"integrity stop: {h['changed']}")
        return h

    # ---- calls: a finished call is never repeated
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

    def call_probes(self, prompt: str, log_path: Path):
        return self.cached(S.SYSTEM_L1, prompt, PB.SCHEMA, PB.parse, log_path)

    # ---- the sandbox
    def run_probes(self, world: str, probes: list) -> list:
        import torch
        from rules import Rule
        with self.probe_lock:                     # one probe at a time on the GPU; each is a few world steps
            if self._designs is None:
                self._designs = PB.load_designs()
            if world not in self._rules:
                self._rules[world] = Rule(world, PB.SPEC, torch.device("mps"))
            return [PB.run(self._rules[world], self._designs[p["design"] - 1], p["placements"]) for p in probes]

    def probe_history(self, sdir: Path, r: int) -> dict:
        hist = {}
        for q in range(1, r):
            f = sdir / f"round_{q}_probe_results.json"
            if f.exists():
                hist[q] = json.loads(f.read_text())
        return hist

    def sandbox_round(self, sdir: Path, world: str, base: str, r: int) -> str:
        """The probe call, the probes, and the proposal prompt with their results. Saved; a resume repeats neither."""
        hist = self.probe_history(sdir, r)
        probes, _ = self.call_probes(PR.probe_prompt(base, hist, r), sdir / f"round_{r}_probes.json")
        rf = sdir / f"round_{r}_probe_results.json"
        if rf.exists():
            res = json.loads(rf.read_text())
            if res["probes"] != probes:
                raise SystemExit(f"{rf}: saved probes differ from the saved call")
        else:
            res = {"probes": probes, "results": self.run_probes(world, probes)}
            rf.write_text(json.dumps(res))
        hist[r] = res
        return PR.proposal_prompt(base, hist, r)

    # ---- one search: run_step2.Step2.search, copied; changed lines are marked "2S"
    def search(self, world: str, arm: str, i: int, start: dict) -> dict:
        sdir = self.dir / world / arm / f"s{i}"
        sdir.mkdir(parents=True, exist_ok=True)
        st_file = sdir / "state.json"
        st = json.loads(st_file.read_text()) if st_file.exists() else {"trials": [], "rounds_done": 0, "incumbent": None}
        R, K, seeds = self.cfg["R"], self.cfg["K"], self.cfg["trial_seeds"]
        want_ro = False                                                                   # 2S: no arm sees readouts
        for r in range(st["rounds_done"] + 1, R + 1):
            self.verify()
            trials = st["trials"]
            seen = {sha(self.ref_src): "the starting architecture"} | {t["result"]["sha"]: t["label"] for t in trials
                                                                         if t.get("result", {}).get("sha")}
            inc = next((t for t in trials if t["label"] == st["incumbent"]), None)
            prompt = PR.base_prompt(arm, world, self.program, start, trials, inc, r, R, K)  # 2S
            if arm in C.SANDBOX_ARMS:                                                    # 2S
                prompt = self.sandbox_round(sdir, world, prompt, r)                      # 2S
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

    def jobs(self) -> list:
        n = max(self.n_per_arm.values())
        return [(w, a, i) for i in range(n) for w in self.worlds for a in C.ARMS if i < self.n_per_arm[a]]

    def run(self):
        if self.cfg["needs_launch"] and not (C.STEP2S / "LAUNCH_APPROVED.md").exists():
            raise SystemExit("the full run needs step2s/LAUNCH_APPROVED.md, written only on the user's approval of the dry run")
        pre = self.verify()
        self.log(f"Step 2S runner start ({self.preset}); worlds {self.worlds}; searches {self.n_per_arm}; preflight {pre}")
        t0 = time.time()
        starts = {w: self.start(w) for w in self.worlds}
        entries, errors = [], []
        with ThreadPoolExecutor(self.cfg["search_workers"]) as ex:
            futs = {ex.submit(self.one_search, w, a, i, starts[w]): (w, a, i) for w, a, i in self.jobs()}
            for fu, job in futs.items():
                try:
                    entries.append(fu.result())
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{job}: {type(e).__name__}: {e}")
                    self.log(f"search {job} failed: {type(e).__name__}: {e}")
        if errors:
            raise SystemExit(f"{len(errors)} searches failed; rerun the same command to resume")
        (self.dir / "summary.json").write_text(json.dumps({"entries": entries, "qualification": self.qual,
                                               "starts": {w: {k: v for k, v in s.items() if k != "readouts"}
                                                          for w, s in starts.items()}}, indent=1))
        rev = self.review(entries)
        jud = self.judge(entries)
        import analyze_step2s as AS
        step2 = json.loads(C.STEP2_RESULTS.read_text()) if C.STEP2_RESULTS.exists() else None
        res = AS.analyze(self.dir, step2)
        post = self.verify()
        res.update(status="final", preset=self.preset, qualification=self.qual, preflight=pre, postflight=post,
                   total_wall_h=round((time.time() - t0) / 3600, 2), written_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        (self.dir / "results.json").write_text(json.dumps(res, indent=1))
        self.log(f"done: {res['summary_line']}; review {len(rev['verdicts'])} picks, judge {len(jud)} searches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    args = ap.parse_args()
    Step2S(C.STEP2S / PRESETS[args.preset]["campaign"], args.preset).run()


if __name__ == "__main__":
    main()
