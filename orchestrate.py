"""Campaign runner. Every phase, arm, budget and seed rule is fixed in SETTINGS and protocol.md
before the first main-campaign trial; the manifest hashes both, plus all code and design sets.

usage: python orchestrate.py --campaign campaigns/main [--backend auto|api|cli] [--dry-run]
Resumable: finished trials (result.json) and proposer rounds (round_XX.json) are reused.
"""
import argparse
import hashlib
import itertools
import json
import os
import platform
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from brickworld.space import BASELINE, config_key, sample  # noqa: E402
import proposer  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")

SETTINGS = {
    "workers": 10,
    "threads_per_trial": 1,
    "small": {"budget_steps": 524288, "timeout_s": 600},
    "large": {"budget_steps": 524288, "width_mult": 2, "timeout_s": 3600},
    "eval_timeout_s": 600,
    "baseline_seeds": 5,
    "arms": {
        "random": {"kind": "random", "metric": "exact"},
        "sonnet": {"kind": "llm", "model": "claude-sonnet-5-5", "metric": "exact"},
        "opus": {"kind": "llm", "model": "claude-opus-5-5", "metric": "exact"},
        "proxy_sonnet": {"kind": "llm", "model": "claude-sonnet-5-5", "metric": "topview"},
    },
    "runs_per_arm": 5,
    "rounds": 8,
    "per_round": 4,
    "budgets_k": [4, 8, 16, 32],
    "fresh_seeds": 2,
    "transfer": {"top": 8, "others": 8, "small_extra_seeds": 2, "large_seeds": 2},
    "variety": {"recipe": {**BASELINE, "reward_mode": "iou_delta"},
                "conditions": ["fixed:64", "fixed:512", "fixed:4096", "stream"],
                "seeds": 3, "budget_steps": 2097152, "timeout_s": 2400,
                "ckpt_fracs": [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]},
}

FROZEN = ["brickworld/env.py", "brickworld/model.py", "brickworld/space.py", "brickworld/__init__.py",
          "train.py", "evaluate.py", "make_targets.py", "orchestrate.py", "proposer.py",
          "tests/test_world.py", "tests/test_proposer.py", "protocol.md",
          "targets/small_dev.pt", "targets/large_dev.pt", "targets/small_heldout_hashes.pt",
          "targets/large_heldout_hashes.pt", "holdout/small_final.pt", "holdout/large_final.pt"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_of(trial_id: str) -> int:
    return int(hashlib.sha256(trial_id.encode()).hexdigest()[:8], 16) % (2 ** 31)


class Campaign:
    def __init__(self, cdir: Path, backend: str, dry: bool):
        self.dir, self.dry = cdir, dry
        (cdir / "trials").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.q = queue.PriorityQueue()
        self.seq = itertools.count()
        self.manifest = self._manifest(backend)
        self.backend = self.manifest["proposer_backend"]
        self.stop = threading.Event()

    # ---------- integrity ----------
    def _manifest(self, backend):
        mf = self.dir / "manifest.json"
        hashes = {f: sha(ROOT / f) for f in FROZEN}
        if mf.exists():
            m = json.loads(mf.read_text())
            bad = [f for f in FROZEN if m["frozen"].get(f) != hashes[f]]
            if bad:
                raise SystemExit(f"frozen files changed since the campaign started: {bad}")
            return m
        if backend == "auto":
            backend = "api" if proposer.api_usable() else "cli"
        import torch
        m = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "frozen": hashes, "settings": SETTINGS,
             "analysis_code_at_launch": sha(ROOT / "analyze.py"),
             "proposer_backend": backend, "python": sys.version, "torch": torch.__version__,
             "machine": platform.platform(), "cpu": platform.processor()}
        mf.write_text(json.dumps(m, indent=1))
        return m

    def verify(self):
        for f in FROZEN:
            if sha(ROOT / f) != self.manifest["frozen"][f]:
                self.log(f"INTEGRITY FAILURE: {f} changed; stopping")
                self.stop.set()
                raise RuntimeError(f"frozen file changed: {f}")

    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with self.lock:
            with open(self.dir / "progress.log", "a") as f:
                f.write(line + "\n")
        print(line, flush=True)

    # ---------- worker pool ----------
    def submit(self, prio, fn, *args) -> Future:
        fut = Future()
        self.q.put((prio, next(self.seq), fn, args, fut))
        return fut

    def worker(self):
        while not self.stop.is_set():
            try:
                prio, _, fn, args, fut = self.q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                fut.set_result(fn(*args))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)

    def _run(self, cmd, timeout, logfile):
        with open(logfile, "w") as lf:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                return p.wait(timeout=timeout), False
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                return -9, True

    # ---------- one trial ----------
    def trial(self, tid: str, recipe: dict, scale="small", seed=None, extra=None, timeout=None) -> dict:
        tdir = self.dir / "trials" / tid
        res_f = tdir / "result.json"
        if res_f.exists():
            return json.loads(res_f.read_text())
        self.verify()
        tdir.mkdir(parents=True, exist_ok=True)
        sc = SETTINGS[scale]
        spec = {"recipe": recipe, "scale": scale, "seed": seed_of(tid) if seed is None else seed,
                "budget_steps": sc["budget_steps"], "train_targets": "stream", "device": "cpu",
                "threads": SETTINGS["threads_per_trial"], "width_mult": sc.get("width_mult", 1)}
        spec.update(extra or {})
        if self.dry:
            spec["budget_steps"] = 8192
        (tdir / "spec.json").write_text(json.dumps(spec))
        t0 = time.time()
        rc, to = self._run([PY, "train.py", "--spec", str(tdir / "spec.json"), "--out", str(tdir)],
                           timeout or sc["timeout_s"], tdir / "train.out")
        res = {"trial": tid, "scale": scale, "seed": spec["seed"], "recipe": recipe,
               "config_key": config_key(recipe), "train_wall_s": round(time.time() - t0, 1)}
        if to:
            res["status"] = "timeout"
        elif rc != 0 or not (tdir / "train_log.json").exists():
            res["status"] = "crash"
            res["error"] = (tdir / "train.out").read_text()[-800:]
        else:
            tl = json.loads((tdir / "train_log.json").read_text())
            res["train"] = {k: v for k, v in tl.items() if k != "curve"}
            res["status"] = "ok" if tl["status"] == "ok" else tl["status"]
            if res["status"] == "ok":
                res["dev"] = self._eval(tdir / "ckpt.pt", "dev", tdir / "eval_dev.json")
                if res["dev"] is None:
                    res["status"] = "eval_failed"
                if spec["train_targets"].startswith("fixed"):
                    res["trainset"] = self._eval(tdir / "ckpt.pt", "trainset", tdir / "eval_trainset.json",
                                                 designs=tdir / "train_designs.pt")
                for f in spec.get("ckpt_fracs", []):
                    ck = tdir / f"ckpt_{f:.3f}.pt"
                    res.setdefault("curve_dev", {})[f"{f:.3f}"] = self._eval(ck, "dev", tdir / f"eval_dev_{f:.3f}.json")
                    if spec["train_targets"].startswith("fixed"):
                        res.setdefault("curve_trainset", {})[f"{f:.3f}"] = self._eval(
                            ck, "trainset", tdir / f"eval_trainset_{f:.3f}.json", designs=tdir / "train_designs.pt")
        res_f.write_text(json.dumps(res))
        with self.lock:
            with open(self.dir / "journal.jsonl", "a") as f:
                f.write(json.dumps({k: v for k, v in res.items() if k != "recipe"}) + "\n")
        d = res.get("dev") or {}
        self.log(f"{tid}: {res['status']} exact={d.get('exact')} topview={d.get('topview')} "
                 f"train_s={res['train_wall_s']}")
        return res

    def _eval(self, ckpt, split, out, designs=None):
        cmd = [PY, "evaluate.py", "--ckpt", str(ckpt), "--split", split, "--out", str(out), "--threads", "1"]
        if designs:
            cmd += ["--designs", str(designs)]
        if split == "final":
            cmd += ["--campaign", str(self.dir)]
        rc, to = self._run(cmd, SETTINGS["eval_timeout_s"], str(out) + ".log")
        if rc != 0 or to or not Path(out).exists():
            return None
        r = json.loads(Path(out).read_text())
        return {k: r[k] for k in ("exact", "topview", "iou", "topview_minus_exact", "pressed_done", "n")}

    # ---------- searches ----------
    @staticmethod
    def shown(res, metric):
        """What a proposer is allowed to see about a trial."""
        tr = res.get("train") or {}
        d = res.get("dev") or {}
        return {"status": res["status"],
                "dev_success": round(d[metric], 4) if metric in d else None,
                "train_success": round(tr.get("train_exact_tail" if metric == "exact" else "train_hm_tail", 0), 4)
                if tr else None,
                "wall_s": res.get("train_wall_s"), "params": tr.get("params")}

    def llm_run(self, arm, run, noise_notes):
        a = SETTINGS["arms"][arm]
        base = self.baseline_futs[0].result()
        history = [{"label": "baseline", "round": 0, "config": BASELINE, "shown": self.shown(base, a["metric"])}]
        trials = []
        for rnd in range(1, SETTINGS["rounds"] + 1):
            if self.stop.is_set():
                return trials
            try:
                props = proposer.propose(self.backend, a["model"], history, rnd, SETTINGS["rounds"],
                                         SETTINGS["per_round"], noise_notes[a["metric"]],
                                         self.dir / "proposer" / arm / f"r{run}")
            except Exception as e:  # noqa: BLE001
                self.log(f"{arm} r{run}: proposer failed in round {rnd}: {e}")
                return trials
            futs = []
            for j, p in enumerate(props):
                i = (rnd - 1) * SETTINGS["per_round"] + j
                tid = f"{arm}-r{run}-t{i:02d}"
                futs.append((tid, p, self.submit(1, self.trial, tid, p["config"])))
            for tid, p, f in futs:
                res = f.result()
                trials.append(res)
                history.append({"label": tid.split("-")[-1], "round": rnd, "config": p["config"],
                                "hypothesis": p["hypothesis"], "shown": self.shown(res, a["metric"])})
        return trials

    def random_run(self, run):
        # Round by round, like the LLM arms, so every arm's trials share the machine with the same mix.
        rng = random.Random(f"random-run-{run}")
        k, out = SETTINGS["per_round"], []
        for rnd in range(SETTINGS["rounds"]):
            futs = [self.submit(1, self.trial, f"random-r{run}-t{i:02d}", sample(rng))
                    for i in range(rnd * k, (rnd + 1) * k)]
            out += [f.result() for f in futs]
        return out

    # ---------- phases ----------
    def go(self):
        workers = [threading.Thread(target=self.worker, daemon=True) for _ in range(SETTINGS["workers"])]
        for w in workers:
            w.start()
        self.log(f"campaign start; proposer backend={self.backend}; dry={self.dry}")
        # Scheduling (changes when a trial runs, not what it is): everything is FIFO at priority 1
        # except baseline seeds and large-scale transfer trials (priority 0). Variety needs nothing
        # from the proposers and starts at once; every arm submits round by round; transfer starts as
        # soon as the random arm is done, so the long large-scale trials do not all wait for the end.
        nb = SETTINGS["baseline_seeds"]
        self.baseline_futs = [self.submit(0, self.trial, f"baseline-s{s}", BASELINE) for s in range(nb)]
        variety_futs = self.variety_jobs()
        threads, results = {}, {}

        def run_and_store(key, fn, *args):
            results[key] = fn(*args)

        def start(arm, r, fn, *args):
            t = threading.Thread(target=run_and_store, args=((arm, r), fn, *args), daemon=True)
            t.start()
            threads[(arm, r)] = t

        for r in range(SETTINGS["runs_per_arm"]):
            start("random", r, self.random_run, r)
        base = [f.result() for f in self.baseline_futs]
        notes = {}
        for metric, label in (("exact", "exact"), ("topview", "topview")):
            v = [b["dev"][metric] for b in base if b.get("dev")]
            notes[metric] = (f"The baseline config was trained with {len(v)} different seeds: dev success mean "
                             f"{sum(v) / len(v):.3f}, min {min(v):.3f}, max {max(v):.3f}.") if v else ""
        self.log("baseline seeds done: " + notes["exact"])

        for arm, a in SETTINGS["arms"].items():
            if a["kind"] == "llm":
                for r in range(SETTINGS["runs_per_arm"]):
                    start(arm, r, self.llm_run, arm, r, notes)
        for r in range(SETTINGS["runs_per_arm"]):
            threads[("random", r)].join()
        self.log("random arm finished; starting transfer")
        transfer = self.transfer_jobs(results)
        for t in threads.values():
            t.join()
        self.log("all searches finished")

        fresh = self.fresh_seed_jobs(results)
        for f in fresh + transfer + variety_futs:
            f.result()
        self.log("fresh-seed, transfer and variety jobs finished")
        self.open_final()
        self.log("campaign complete")
        self.stop.set()

    def incumbents(self, results):
        """(arm, run, k) -> incumbent trial result, by the arm's own selection metric."""
        out = {}
        base0 = self.baseline_futs[0].result()
        arms = dict(SETTINGS["arms"])
        arms["random_topview"] = {"kind": "random", "metric": "topview", "source": "random"}
        for arm, a in arms.items():
            src = a.get("source", arm)
            for r in range(SETTINGS["runs_per_arm"]):
                trials = results.get((src, r), [])
                for k in SETTINGS["budgets_k"]:
                    pool = [base0] + [t for t in trials[:k] if t.get("dev")]
                    best = max(pool, key=lambda t: t["dev"][a["metric"]])  # first max wins ties
                    out[(arm, r, k)] = best
        return out

    def fresh_seed_jobs(self, results):
        inc = self.incumbents(results)
        (self.dir / "incumbents.json").write_text(json.dumps(
            {f"{a}|{r}|{k}": v["trial"] for (a, r, k), v in inc.items()}, indent=1))
        seen, futs = {}, []
        for v in inc.values():
            ck = v["config_key"]
            if ck in seen or v["trial"].startswith("baseline"):
                continue
            h = hashlib.sha256(ck.encode()).hexdigest()[:10]
            seen[ck] = h
            for s in range(SETTINGS["fresh_seeds"]):
                futs.append(self.submit(1, self.trial, f"fresh-{h}-s{s}", v["recipe"]))
        (self.dir / "fresh_map.json").write_text(json.dumps(seen, indent=1))
        return futs

    def transfer_jobs(self, results):
        pool = [t for r in range(SETTINGS["runs_per_arm"]) for t in results.get(("random", r), []) if t.get("dev")]
        pool.sort(key=lambda t: -t["dev"]["exact"])
        T = SETTINGS["transfer"]
        top, rest = pool[:T["top"]], pool[T["top"]:]
        others = random.Random("transfer-others").sample(rest, min(T["others"], len(rest)))
        chosen = [("baseline", BASELINE)] + [(t["trial"], t["recipe"]) for t in top + others]
        (self.dir / "transfer_set.json").write_text(json.dumps(
            {"top": [t["trial"] for t in top], "others": [t["trial"] for t in others]}, indent=1))
        futs = []
        for i, (src, rec) in enumerate(chosen):
            if src != "baseline":
                for s in range(T["small_extra_seeds"]):
                    futs.append(self.submit(1, self.trial, f"xfer-small-{i:02d}-s{s}", rec))
            for s in range(T["large_seeds"]):
                futs.append(self.submit(0, self.trial, f"xfer-large-{i:02d}-s{s}", rec, "large"))
        return futs

    def variety_jobs(self):
        V, futs = SETTINGS["variety"], []
        for cond in V["conditions"]:
            for s in range(V["seeds"]):
                tag = cond.replace(":", "")
                futs.append(self.submit(1, self.trial, f"variety-{tag}-s{s}", V["recipe"], "small", None,
                                        {"train_targets": cond, "budget_steps": V["budget_steps"],
                                         "ckpt_fracs": V["ckpt_fracs"]}, V["timeout_s"]))
        return futs

    def open_final(self):
        marker = self.dir / "FINAL_OPENED"
        if not marker.exists():
            marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") +
                              "\nAll search, fresh-seed, transfer and variety trials finished before this. "
                              "Every finished checkpoint is scored once on the final split; no selection uses it.\n")
        self.log("final split opened")
        futs = []
        for tdir in sorted((self.dir / "trials").iterdir()):
            out = tdir / "eval_final.json"
            if (tdir / "ckpt.pt").exists() and not out.exists():
                futs.append(self.submit(0, self._eval, tdir / "ckpt.pt", "final", out))
        for f in futs:
            f.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--backend", default="auto", choices=["auto", "api", "cli"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:  # tiny end-to-end check of every phase; real proposer calls, 8k-step trials
        SETTINGS.update({"runs_per_arm": 1, "rounds": 2, "budgets_k": [4, 8], "baseline_seeds": 2,
                         "fresh_seeds": 1})
        SETTINGS["transfer"] = {"top": 2, "others": 2, "small_extra_seeds": 1, "large_seeds": 1}
        SETTINGS["variety"] = {**SETTINGS["variety"], "seeds": 1, "conditions": ["fixed:64", "fixed:4096", "stream"]}
    Campaign(ROOT / args.campaign, args.backend, args.dry_run).go()


if __name__ == "__main__":
    main()
