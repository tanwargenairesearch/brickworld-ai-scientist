"""Stage 2e (PROTOCOL_2E.md): do experiments guide a longer search? Forked pairs of searches on world W.

Reuses, imported and unchanged: Stage 2b's runner (GPU training and checks on W, the task text, the comparator),
Stage 1's two prompt builders, pick rule and confirmation, and the frozen call code. It adds:
- pairs: one shared round 1 written with the scores-hidden prompt, then two branches that continue from a copy of
  that state: "seen" (Stage 2b's g0 prompt) and "hidden" (the scores-hidden prompt). Selection uses the real scores
  in both;
- the current best and the pick rule on unrounded scores (in 2b a mean rounded to 4 decimals was compared with the
  unrounded bar, so an exact tie with the start counted as a new best);
- retest runs stored once per pair by code hash, so both branches reuse a round-1 candidate's retest;
- saved replies: a round whose model reply is on disk reuses it on resume, after checking the prompt is the same, and
  a trial folder is reused only if it holds the same code. Nothing is redrawn;
- the second-stage rule: after the first stage, more pairs run if the root mean square of the paired differences
  is above the threshold. It reads sizes only (analyze_2e.second_stage_rule);
- ceilings on elapsed hours and notional dollars: a new pair starts only while both are under their ceilings.
  Started pairs always finish. No stop reads a score.

usage (project root):
  caffeinate -i .venv/bin/python -I arch_rsi/stage2/2e/run_2e.py --preset 2e_dry   (tiny budgets)
  caffeinate -i .venv/bin/python -I arch_rsi/stage2/2e/run_2e.py --preset 2e       (needs LAUNCH_APPROVED.md)
Resumes from disk if rerun with the same preset. After a ceiling stop, CEILING_STOP.json must be removed by hand
(with new ceilings in LAUNCH_APPROVED.md) before more pairs can start. W's final split is never opened here.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
S2 = HERE.parent
A = S2.parent
ROOT = A.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

_spec = importlib.util.spec_from_file_location("run_2b", S2 / "2b" / "run_2b.py")
R2B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R2B)
O, S, S1 = R2B.O, R2B.S, R2B.S1
from arch_rsi.sandbox import sha  # noqa: E402

import analyze_2e as AZ  # noqa: E402
import review_2e as RV  # noqa: E402

W = "w"
BRANCHES = AZ.BRANCHES                      # ("seen", "hidden")
BASE = R2B.BASE
G0R = S2 / "programs" / "g0r.md"
CAMP_2B = A / "campaigns" / "stage2b"
PROTOCOL = HERE / "PROTOCOL_2E.md"
APPROVAL_FILE = HERE / "LAUNCH_APPROVED.md"
FILES_2E = ([HERE / p for p in ("run_2e.py", "analyze_2e.py", "review_2e.py", "PROTOCOL_2E.md", "ENDINGS_2E.md")]
            + [S2 / "2b" / p for p in ("analyze_2b.py", "review_2b.py")])

_base = {k: v for k, v in R2B.PRESETS["2b"].items() if k not in ("arms", "n_searches", "search_workers", "search_seed_base")}
PRESETS = {
    "2e": dict(_base, R=8, K=3, trial_seeds=[0, 1], retest_top=3, retest_seeds=[2, 3], base_seeds=[0, 1, 2, 3],
               confirm_seeds=list(range(2001, 2017)), comparator_seeds=list(range(2001, 2017)),
               pairs_stage1=32, pairs_max=48, rms_threshold=0.20, pair_workers=5,
               campaign="arch_rsi/campaigns/stage2e"),
}
PRESETS["2e_dry"] = dict(PRESETS["2e"], steps={W: 32768}, timeout={W: 600}, R=3, K=2, retest_top=2, retest_seeds=[2],
                         base_seeds=[0, 1, 2], confirm_seeds=[2001, 2002], comparator_seeds=[2001, 2002],
                         pairs_stage1=2, pairs_max=3, pair_workers=3, campaign="arch_rsi/campaigns/stage2e_dry")


# ---------------------------------------------------------------- pure helpers (tested in test_run_2e.py)
def better(score: float, best: float) -> bool:
    """A try becomes the current best only if its unrounded mean is strictly higher. An exact tie doesn't."""
    return score > best


def read_approval(text: str, protocol_hash: str) -> dict:
    """The ceilings from LAUNCH_APPROVED.md. It must name the hash of the protocol it approves."""
    m = re.search(r"^protocol_sha256:\s*([0-9a-f]{64})\s*$", text, re.M)
    if not m or m.group(1) != protocol_hash:
        raise SystemExit("LAUNCH_APPROVED.md doesn't name this PROTOCOL_2E.md's hash: the protocol changed after "
                         "approval, or the approval is for another protocol")

    def num(key):
        k = re.search(rf"^{key}:\s*([0-9]+(?:\.[0-9]+)?)\s*$", text, re.M)
        if not k:
            raise SystemExit(f"LAUNCH_APPROVED.md has no '{key}: <number>' line")
        return float(k.group(1))
    return {"hours": num("ceiling_hours"), "usd": num("ceiling_usd")}


def strkeys(e: dict) -> dict:
    """An entry with its per-seed dicts keyed by strings, as they are after a JSON round trip."""
    return {k: ({str(s): v for s, v in x.items()} if k in ("dev", "status", "wall_s") else x) for k, x in e.items()}


# ---------------------------------------------------------------- the campaign
class Stage2e(R2B.Stage2b):
    def __init__(self, cdir: Path, preset: str, presets: dict = None, ceilings: dict = None):
        cfg = dict((presets or PRESETS)[preset])
        O.PRESETS[preset] = cfg
        O.FROZEN = list(O.FROZEN) + [p for p in R2B.FILES_2B + FILES_2E if p not in O.FROZEN]
        O.Campaign.__init__(self, cdir, preset)
        self.run_locks = {}
        self.base = {}
        self.program = None
        self.ceil = dict(ceilings or {})
        self.now = time.time
        self.clock_lock = threading.Lock()
        cf = cdir / "clock.json"
        self.elapsed = json.loads(cf.read_text())["elapsed_s"] if cf.exists() else 0.0
        self.last = self.now()

    # ---- bookkeeping: elapsed time (only while the runner is alive) and notional cost
    def tick(self) -> float:
        with self.clock_lock:
            t = self.now()
            self.elapsed += t - self.last
            self.last = t
            (self.dir / "clock.json").write_text(json.dumps({"elapsed_s": round(self.elapsed, 1)}))
            return self.elapsed

    def log(self, msg: str):
        O.Campaign.log(self, msg)
        self.tick()

    def spent_usd(self) -> float:
        tot = 0.0
        for f in list((self.dir / "pairs").glob("*/*/round_*.json")) + list((self.dir / "review").glob("*.json")):
            if f.name.endswith(".failed.json"):
                continue
            try:
                tot += (json.loads(f.read_text()).get("usage") or {}).get("notional_cost_usd") or 0.0
            except json.JSONDecodeError:
                pass                                      # a log being written right now
        return tot

    def may_start(self, i: int) -> bool:
        """A new pair starts only while elapsed hours and notional dollars are both under their ceilings."""
        stop = self.dir / "CEILING_STOP.json"
        if stop.exists():
            return False
        hours, usd = self.tick() / 3600, self.spent_usd()
        over = [k for k, v in (("hours", hours), ("usd", usd)) if self.ceil.get(k) is not None and v >= self.ceil[k]]
        if not over:
            return True
        stop.write_text(json.dumps({"before_pair": i, "elapsed_hours": round(hours, 2), "notional_usd": round(usd, 2),
                                    "ceilings": self.ceil, "reached": over, "at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        self.log(f"CEILING_STOP before pair {i}: {round(hours, 2)} h, ${round(usd, 2)} (ceilings {self.ceil}). "
                 f"Started pairs finish; no new pair starts.")
        return False

    # ---- checks and calls that survive a resume
    def check(self, path: Path, world: str, mode: str = "both") -> dict:
        out = path.with_suffix(".check.json")
        meta = path.with_suffix(".check_for.json")
        key = {"sha": sha(path.read_text()), "world": world, "mode": mode}
        if out.exists() and meta.exists() and json.loads(meta.read_text()) == key:
            return json.loads(out.read_text())                       # resume: same file, world and mode
        res = R2B.Stage2b.check(self, path, world, mode)
        if out.exists():
            meta.write_text(json.dumps(key))
        return res

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

    @staticmethod
    def seed_score(tdir: Path, seed: int) -> float:
        """One training run's unrounded dev score; a run that didn't finish scores 0."""
        r = json.loads((tdir / f"seed{seed}" / "result.json").read_text())
        return r["dev_exact"] if r["status"] == "ok" else 0.0

    # ---- one round of one search
    def one_round(self, sdir: Path, st: dict, r: int, hidden: bool) -> dict:
        c = self.cfg
        R, K, seeds = c["R"], c["K"], c["trial_seeds"]
        base_src = BASE.read_text()
        base_dev = self.base[W]["shown_dev"]
        trials = st["trials"]
        seen = {sha(base_src): "the starting architecture"} | {t["result"]["sha"]: t["label"] for t in trials if t.get("result", {}).get("sha")}
        inc = next((t for t in trials if t["label"] == st["incumbent"]), None)
        task = self.task_for(W, hidden, "nofb" if hidden else "g0")
        if hidden:
            prompt = S1.l1_prompt_hidden(self.program, task, self.interface(), base_src, trials, r, R, K)
        else:
            prompt = S.l1_prompt(self.program, task, self.interface(), base_src, base_dev, trials, inc, r, R, K)
        val = self.saved_call(S.SYSTEM_L1, prompt, S.l1_schema(K), lambda o: S.parse_proposals(o, K), sdir / f"round_{r}.json")
        props, analysis = val["proposals"], val["analysis"]
        checks = [None] * K
        for j in range(3):
            todo = [i for i in range(K) if checks[i] is None or not checks[i]["ok"]]

            def run_check(i, j=j):
                p = sdir / "cands" / f"r{r}c{i + 1}_v{j}.py"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(props[i]["code"])
                return self.check(p, W, "both")
            with ThreadPoolExecutor(max(1, len(todo))) as ex:
                got = dict(zip(todo, ex.map(run_check, todo)))
            for i in todo:
                ck = got[i]
                h = sha(props[i]["code"])
                batch_dup = [k for k in range(K) if k < i and sha(props[k]["code"]) == h]
                if ck.get("ok") and (h in seen or batch_dup):
                    ck = {"ok": False, "error": f"identical to {seen.get(h) or f'proposal {batch_dup[0] + 1} in this round'}"}
                checks[i] = ck
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
                props[i] = {"name": f["name"] or props[i]["name"], "hypothesis": f["hypothesis"], "code": f["code"]}
        new, jobs = [], []
        for i in range(K):
            t = {"label": f"r{r}c{i + 1}", "round": r, "name": props[i]["name"], "hypothesis": props[i]["hypothesis"],
                 "code": props[i]["code"], "check": checks[i]}
            if not checks[i]["ok"]:
                t["result"] = {"status": "invalid", "error": checks[i]["error"], "dev_exact": None, "sha": sha(props[i]["code"])}
            else:
                jobs.append(t)
            new.append(t)

        def train(t):
            tdir = sdir / "trials" / t["label"]
            f = tdir / "candidate.py"
            if f.exists() and f.read_text() != t["code"]:
                raise SystemExit(f"{tdir} holds other code than this round's proposal; refusing to reuse its runs")
            return self.run_candidate(tdir, t["code"], W, seeds, t["check"])
        with ThreadPoolExecutor(max(1, len(jobs))) as ex:
            outs = list(ex.map(train, jobs))
        for t, o in zip(jobs, outs):
            t["result"] = o
            t["exact_mean"] = sum(self.seed_score(sdir / "trials" / t["label"], s) for s in seeds) / len(seeds)
        st["trials"] = trials + new
        st.setdefault("analyses", {})[str(r)] = analysis
        for t in new:
            if t["result"]["status"] == "ok" and better(t["exact_mean"], st["best"]):
                st["best"], st["incumbent"] = t["exact_mean"], t["label"]
        st["incumbent_by_round"][str(r)] = st["incumbent"]
        st["rounds_done"] = r
        (sdir / "state.json").write_text(json.dumps(st, indent=1))
        self.log(f"{sdir.relative_to(self.dir)} round {r}: " + ", ".join(
            f"{t['label']} {t['result']['status']} {t['result'].get('dev_exact')}" for t in new) + f" | current best {st['incumbent']} {st['best']:.4f}")
        return st

    # ---- a pair: the shared round, then the two branches
    def pair_dir(self, i: int) -> Path:
        return self.dir / "pairs" / f"p{i:02d}"

    def open_dir(self, sdir: Path):
        sdir.mkdir(parents=True, exist_ok=True)
        pf = sdir / "program.md"
        if pf.exists():
            if pf.read_text() != self.program:
                raise SystemExit(f"{sdir}: the program differs from the one this search started with")
        else:
            pf.write_text(self.program)
        f = sdir / "state.json"
        return json.loads(f.read_text()) if f.exists() else None

    def shared_round(self, pdir: Path) -> dict:
        sdir = pdir / "shared"
        st = self.open_dir(sdir) or {"trials": [], "rounds_done": 0, "incumbent": None, "best": self.base[W]["shown_dev"],
                                     "incumbent_by_round": {}}
        if st["rounds_done"] < 1:
            st = self.one_round(sdir, st, 1, hidden=True)
        return st

    def one_branch(self, pdir: Path, i: int, b: str, st1: dict) -> dict:
        sdir = pdir / b
        st = self.open_dir(sdir)
        fork = hashlib.sha256((pdir / "shared" / "state.json").read_bytes()).hexdigest()
        if st is None:
            st = dict(copy.deepcopy(st1), branch=b, forked_from=fork)
            (sdir / "state.json").write_text(json.dumps(st, indent=1))
        elif st.get("forked_from") != fork or st.get("branch") != b or st["trials"][:len(st1["trials"])] != st1["trials"]:
            raise SystemExit(f"{sdir}: this branch wasn't forked from the pair's shared round")
        for r in range(st["rounds_done"] + 1, self.cfg["R"] + 1):
            st = self.one_round(sdir, st, r, hidden=(b == "hidden"))
        pick = self.retest_and_pick_2e(pdir, b, st)
        seeds = self.cfg["confirm_seeds"]
        conf = self.confirm(pick["src"], seeds)
        rel = sdir.relative_to(self.dir)
        self.log(f"{rel} confirmed: {[round(r['dev_exact'], 3) if r['dev_exact'] is not None else r['status'] for r in conf.values()]}")
        return {"kind": "search", "pair": i, "arm": b, "search": i, "path": str(rel), "label": pick["label"], "name": pick["name"],
                "sha": pick["sha"], "mean_selected": pick["mean_selected"], "incumbent_2seed": pick["incumbent_2seed"],
                "seeds": seeds, "dev": {str(s): r["dev_exact"] for s, r in conf.items()},
                "status": {str(s): r["status"] for s, r in conf.items()}, "wall_s": {str(s): r["wall_s"] for s, r in conf.items()}}

    def one_pair(self, i: int) -> list:
        pdir = self.pair_dir(i)
        st1 = self.shared_round(pdir)
        with ThreadPoolExecutor(2) as ex:
            return list(ex.map(lambda b: self.one_branch(pdir, i, b, st1), BRANCHES))

    # ---- retest the top tries, then pick: Stage 1's rule on unrounded scores, with runs shared inside the pair
    def retest_and_pick_2e(self, pdir: Path, b: str, st: dict) -> dict:
        pick_file = pdir / b / "pick.json"
        if pick_file.exists():
            return json.loads(pick_file.read_text())
        c = self.cfg
        top = S1.top_for_retest(st["trials"], c["retest_top"])
        jobs = [(t, s) for t in top for s in c["retest_seeds"]]

        def run(job):
            t, s = job
            d = pdir / "retest" / t["result"]["sha"]
            d.mkdir(parents=True, exist_ok=True)
            with self.lock:
                f = d / "candidate.py"
                if not f.exists():
                    f.write_text(t["code"])
                elif f.read_text() != t["code"]:
                    raise SystemExit(f"{d} holds other code than the candidate with its hash")
            return self.locked_train(d / f"seed{s}", d / "candidate.py", W, s)
        with ThreadPoolExecutor(max(1, len(jobs))) as ex:
            outs = list(ex.map(run, jobs))
        res = {}
        for (t, s), o in zip(jobs, outs):
            res.setdefault(t["label"], {})[s] = o
        order = {t["label"]: k for k, t in enumerate(st["trials"])}
        rows = []
        for t in top:
            tdir = pdir / ("shared" if t["round"] == 1 else b) / "trials" / t["label"]
            first = [self.seed_score(tdir, s) for s in c["trial_seeds"]]
            more = [res[t["label"]][s]["dev_exact"] if res[t["label"]][s]["status"] == "ok" else 0.0 for s in c["retest_seeds"]]
            devs = first + more
            rows.append({"label": t["label"], "name": t["name"], "order": order[t["label"]],
                         "all_ok": all(res[t["label"]][s]["status"] == "ok" for s in c["retest_seeds"]),
                         "dev_trial_seeds": t["exact_mean"], "dev_all": devs, "mean": sum(devs) / len(devs), "sha": t["result"]["sha"]})
        start_mean = self.base[W]["mean"]
        label = S1.choose_pick(rows, start_mean)
        pick = {"label": label or "start", "start_mean": start_mean, "retested": rows, "incumbent_2seed": st["incumbent"]}
        if label:
            t = next(t for t in st["trials"] if t["label"] == label)
            pick |= {"name": t["name"], "src": t["code"], "sha": sha(t["code"]),
                     "mean_selected": next(r["mean"] for r in rows if r["label"] == label)}
        else:
            src = BASE.read_text()
            pick |= {"name": "the starting architecture", "src": src, "sha": sha(src), "mean_selected": start_mean}
        pick_file.write_text(json.dumps(pick, indent=1))
        self.log(f"{(pdir / b).relative_to(self.dir)} pick {pick['label']} ({pick['mean_selected']:.4f}); "
                 f"retested {[(r['label'], round(r['mean'], 4)) for r in rows]}; current best on the trial seeds {st['incumbent']}")
        return pick

    # ---- pairs in index order, the second-stage rule, and the run
    def run_pairs(self, idx: list) -> dict:
        """Run pairs in index order, at most pair_workers at once. A pair that already started always finishes;
        a new one starts only under the ceilings, so the finished pairs are always 1..N."""
        futs = {}
        sem = threading.Semaphore(self.cfg["pair_workers"])
        with ThreadPoolExecutor(self.cfg["pair_workers"]) as ex:
            for i in idx:
                sem.acquire()
                if not (self.pair_dir(i) / "shared").exists() and not self.may_start(i):
                    sem.release()
                    break
                f = ex.submit(self.one_pair, i)
                f.add_done_callback(lambda _: sem.release())
                futs[i] = f
            return {i: f.result() for i, f in futs.items()}

    def write_summary(self, entries: list):
        (self.dir / "confirm").mkdir(exist_ok=True)
        (self.dir / "confirm" / "summary.json").write_text(json.dumps(entries, indent=1))

    def second_stage(self, entries: list, comp: dict) -> dict:
        f = self.dir / "second_stage.json"
        if f.exists():
            return json.loads(f.read_text())
        verdicts = RV.review(self.dir, entries)
        pr = AZ.pair_rows(entries + [comp], verdicts)
        if pr["pending"]:
            raise SystemExit("the second-stage rule needs a verdict for every first-stage pick")
        out = AZ.second_stage_rule([r["d_builds"] / pr["denom"] for r in pr["rows"]], self.cfg["rms_threshold"])
        out["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write_text(json.dumps(out, indent=1))
        self.log(f"second-stage rule: root mean square of {out['n']} differences {out['rms']:.4f}, threshold "
                 f"{out['threshold']}: {'more pairs run' if out['extend'] else 'the run ends here'}")
        return out

    def run(self):
        c = self.cfg
        n1, n2 = c["pairs_stage1"], c["pairs_max"]
        self.log(f"start preset={self.preset} ceilings={self.ceil or 'none'}")
        self.base = {W: self.baselines_q()}
        self.program = G0R.read_text()
        (self.dir / "programs").mkdir(exist_ok=True)
        (self.dir / "programs" / "g0r.md").write_text(self.program)
        with ThreadPoolExecutor(1) as cx:
            comp_f = cx.submit(R2B.Stage2b.one_comparator, self, "start_g")
            done = self.run_pairs(list(range(1, n1 + 1)))
            comp = strkeys(comp_f.result())
        entries = [e for i in sorted(done) for e in done[i]]
        self.write_summary(entries + [comp])
        if len(done) == n1 and n2 > n1 and self.second_stage(entries, comp)["extend"]:
            more = self.run_pairs(list(range(n1 + 1, n2 + 1)))
            entries += [e for i in sorted(more) for e in more[i]]
            self.write_summary(entries + [comp])
        self.verify()
        self.log(f"STAGE2E_DONE: {len(entries) // 2} pairs finished. Run review_2e.py, then analyze_2e.py. "
                 f"W's final split was not opened.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    args = ap.parse_args()
    if (S2 / "FINAL_OPENED_W").exists():
        raise SystemExit("FINAL_OPENED_W exists: Stage 2e must not run after W's final split was opened")
    if "STAGE2B_DONE" not in (CAMP_2B / "progress.log").read_text():
        raise SystemExit("Stage 2b is still running (no STAGE2B_DONE): 2e would compete for the GPU")
    ceilings = None
    if args.preset == "2e":
        if not APPROVAL_FILE.exists():
            raise SystemExit(f"{APPROVAL_FILE.name} is missing: Stage 2e launches only on the user's go-ahead")
        ceilings = read_approval(APPROVAL_FILE.read_text(), hashlib.sha256(PROTOCOL.read_bytes()).hexdigest())
    Stage2e((ROOT / PRESETS[args.preset]["campaign"]).resolve(), args.preset, ceilings=ceilings).run()


if __name__ == "__main__":
    main()
