"""Orchestrator for the architecture-search campaign with a self-revising research program.

usage (from the project root):
  .venv/bin/python -I arch_rsi/orchestrate_arch.py --campaign arch_rsi/campaigns/main --preset main
Phases (each resumes from what is on disk):
  0  starting architecture on both worlds, several seeds (noise figure shown to the scientist)
  1  meta-training on world P: for g = 0 .. G-1, S_P searches under program g, then the scientist
     rewrites program g into program g+1 from those searches' logs (Level 2)
  2  meta-testing on held-out world Q: S_Q searches each under program 0, program G, and a control
     that samples the main campaign's 144-option architecture menu at random
  3  confirmation: every Q search's selected architecture retrained with fresh seeds; the starting
     architecture retrained with the same seeds as the comparator
  4  FINAL_OPENED, then every confirmation run scored once on Q's hidden final designs
Every trial re-checks the hashes of the frozen files first and stops the campaign on any change.
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from arch_rsi import scientist as S  # noqa: E402
from arch_rsi.interface import INTERFACE_TEXT  # noqa: E402
from arch_rsi.menu import MENU  # noqa: E402
from arch_rsi.menu import source as menu_source  # noqa: E402
from arch_rsi.sandbox import sha  # noqa: E402
from arch_rsi.worlds import CAPS, SPECS  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
BASE_CFG = {"encoder": "conv", "width": 64, "depth": 2, "activation": "relu", "norm": "none"}

PRESETS = {
    "main": dict(P="large", Q="xl", steps={"large": 262144, "xl": 524288}, timeout={"large": 400, "xl": 1300},
                 R=4, K=3, S_P=3, G=3, S_Q=4, trial_seeds=[0, 1], base_seeds=[0, 1, 2, 3],
                 confirm_seeds=[101, 102, 103], comparator_seeds=[101, 102, 103], workers=10, max_words=1000),
    "dry": dict(P="large", Q="xl", steps={"large": 32768, "xl": 65536}, timeout={"large": 600, "xl": 600},
                R=2, K=2, S_P=1, G=1, S_Q=1, trial_seeds=[0], base_seeds=[0, 1],
                confirm_seeds=[101], comparator_seeds=[101], workers=8, max_words=1000),
}

FROZEN = [ROOT / p for p in ("train.py", "evaluate.py", "proposer.py", "brickworld/env.py", "brickworld/model.py",
                             "brickworld/space.py", "targets/large_dev.pt", "targets/large_heldout_hashes.pt")] + \
         [HERE / p for p in ("__init__.py", "worlds.py", "interface.py", "sandbox.py", "check.py", "train_arch.py",
                             "evaluate_arch.py", "scientist.py", "menu.py", "orchestrate_arch.py", "analyze_arch.py",
                             "candidates/baseline.py", "programs/g0.md", "protocol.md", "targets/xl_dev.pt",
                             "targets/xl_heldout_hashes.pt", "holdout/xl_final.pt")]


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Campaign:
    def __init__(self, cdir: Path, preset: str):
        self.dir = cdir
        self.cfg = PRESETS[preset]
        self.preset = preset
        cdir.mkdir(parents=True, exist_ok=True)
        self.slots = threading.Semaphore(self.cfg["workers"])
        self.check_slots = threading.Semaphore(4)  # pre-training checks run beside trainings; keep them few
        self.lock = threading.Lock()
        man = cdir / "manifest.json"
        if man.exists():
            m = json.loads(man.read_text())
            if m["preset"] != preset:
                raise SystemExit(f"campaign was started with preset {m['preset']}")
            self.hashes = m["hashes"]
        else:
            self.hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in FROZEN}
            man.write_text(json.dumps({"preset": preset, "config": self.cfg, "caps": CAPS, "hashes": self.hashes,
                                       "model": S.MODEL, "effort": S.EFFORT, "backend": "claude -p",
                                       "started": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=1))
        self.verify()

    # ---------- bookkeeping ----------
    def log(self, msg: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with self.lock:
            with open(self.dir / "progress.log", "a") as f:
                f.write(line + "\n")
        print(line, flush=True)

    def verify(self):
        for rel, h in self.hashes.items():
            if file_hash(ROOT / rel) != h:
                (self.dir / "INTEGRITY_STOP").write_text(f"{rel} changed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                raise SystemExit(f"INTEGRITY STOP: {rel} changed since launch")

    # ---------- trials ----------
    def train_one(self, rdir: Path, src_path: Path, world: str, seed: int, split: str = "dev") -> dict:
        """Train one seed and score it on dev. Cached on disk."""
        res_file = rdir / "result.json"
        if res_file.exists():
            return json.loads(res_file.read_text())
        rdir.mkdir(parents=True, exist_ok=True)
        spec = rdir / "run.json"
        spec.write_text(json.dumps({"candidate": str(src_path), "world": world, "seed": seed,
                                    "budget_steps": self.cfg["steps"][world], "threads": 1}))
        with self.slots:
            self.verify()
            t0 = time.time()
            try:
                p = subprocess.run([PY, "-I", str(HERE / "train_arch.py"), "--spec", str(spec), "--out", str(rdir / "run")],
                                   capture_output=True, text=True, timeout=self.cfg["timeout"][world])
                tl = json.loads((rdir / "run" / "train_log.json").read_text()) if (rdir / "run" / "train_log.json").exists() else {}
                status = tl.get("status", f"crashed (exit {p.returncode})")
                err = tl.get("error") or (None if status == "ok" else (p.stderr or "")[-400:])
            except subprocess.TimeoutExpired:
                tl, status, err = {}, "timeout", f"killed after {self.cfg['timeout'][world]} s"
            dev = None
            if status == "ok":
                q = subprocess.run([PY, "-I", str(HERE / "evaluate_arch.py"), "--trial", str(rdir / "run"),
                                    "--split", "dev", "--out", str(rdir / "dev.json")],
                                   capture_output=True, text=True, timeout=1800)
                if q.returncode == 0:
                    dev = json.loads((rdir / "dev.json").read_text())["exact"]
                else:
                    status, err = "eval_failed", (q.stderr or "")[-400:]
        res = {"status": status, "error": err, "dev_exact": dev, "train_exact": tl.get("train_exact_tail"),
               "params": tl.get("params"), "train_steps": tl.get("steps"), "flops_ratio": tl.get("flops_ratio"),
               "wall_s": round(time.time() - t0, 1), "seed": seed, "world": world}
        res_file.write_text(json.dumps(res))
        return res

    def run_candidate(self, cdir: Path, src: str, world: str, seeds: list, check: dict) -> dict:
        cdir.mkdir(parents=True, exist_ok=True)
        src_path = cdir / "candidate.py"
        if not src_path.exists():
            src_path.write_text(src)
        with ThreadPoolExecutor(len(seeds)) as ex:
            per = list(ex.map(lambda s: self.train_one(cdir / f"seed{s}", src_path, world, s), seeds))
        ok = all(r["status"] == "ok" for r in per)
        devs = [r["dev_exact"] if r["status"] == "ok" else 0.0 for r in per]
        bad = [r for r in per if r["status"] != "ok"]
        return {"status": "ok" if ok else bad[0]["status"], "error": None if ok else bad[0]["error"],
                "dev_exact": round(sum(devs) / len(devs), 4), "dev_per_seed": [round(d, 4) for d in devs] if len(seeds) > 1 else None,
                "train_exact": round(sum((r["train_exact"] or 0) for r in per) / len(per), 4),
                "params": check.get("params"), "flops_ratio": check.get("flops_ratio"),
                "train_steps": max((r.get("train_steps") or 0) for r in per),
                "wall_s": max(r["wall_s"] for r in per), "sha": sha(src)}

    def check(self, path: Path, world: str) -> dict:
        out = path.with_suffix(".check.json")
        with self.check_slots:
            p = subprocess.run([PY, "-I", str(HERE / "check.py"), "--candidate", str(path), "--world", world,
                                "--out", str(out)], capture_output=True, text=True, timeout=900)
        if p.returncode != 0 or not out.exists():
            return {"ok": False, "error": f"check process failed: {(p.stderr or '')[-300:]}"}
        return json.loads(out.read_text())

    # ---------- phase 0 ----------
    def baselines(self):
        out = {}
        src = (HERE / "candidates" / "baseline.py").read_text()
        chk = {}
        for w in (self.cfg["P"], self.cfg["Q"]):
            d = self.dir / "baseline" / w
            d.mkdir(parents=True, exist_ok=True)
            (d / "candidate.py").write_text(src)
            chk[w] = self.check(d / "candidate.py", w)
        jobs = [(w, s) for w in (self.cfg["P"], self.cfg["Q"]) for s in self.cfg["base_seeds"]]
        with ThreadPoolExecutor(len(jobs)) as ex:
            res = list(ex.map(lambda j: self.train_one(self.dir / "baseline" / j[0] / f"seed{j[1]}",
                                                       self.dir / "baseline" / j[0] / "candidate.py", j[0], j[1]), jobs))
        for (w, s), r in zip(jobs, res):
            out.setdefault(w, {})[s] = r
        summary = {}
        for w, rs in out.items():
            devs = [rs[s]["dev_exact"] for s in self.cfg["base_seeds"]]
            if any(d is None for d in devs):
                raise SystemExit(f"starting architecture failed on {w}: {rs}")
            shown = [rs[s]["dev_exact"] for s in self.cfg["trial_seeds"]]
            summary[w] = {"devs": devs, "mean": sum(devs) / len(devs), "min": min(devs), "max": max(devs),
                          "shown_dev": sum(shown) / len(shown), "check": chk[w]}
        (self.dir / "baseline" / "summary.json").write_text(json.dumps(summary, indent=1))
        self.log(f"phase 0 done: {json.dumps({w: [round(d, 3) for d in v['devs']] for w, v in summary.items()})}")
        return summary

    def noise_note(self, w: str) -> str:
        b = self.base[w]
        n = len(self.cfg["trial_seeds"])
        per = ", ".join(f"{d:.3f}" for d in b["devs"])
        return (f"every trial trains {n} seed{'s' if n > 1 else ''} and reports the mean dev success. The starting "
                f"architecture, trained with {len(b['devs'])} different seeds, scored {per} on dev (mean {b['mean']:.3f}): "
                f"training runs on this task can succeed or fail by seed luck alone, so treat single differences "
                f"of this size with care.")

    def task(self, w: str) -> str:
        n_dev = 512
        return S.task_text(SPECS[w], self.cfg["steps"][w], self.cfg["timeout"][w], n_dev, self.noise_note(w))

    def interface(self) -> str:
        return INTERFACE_TEXT.format(max_params=CAPS["max_params"], flops_ratio=CAPS["flops_ratio"],
                                     n_check=CAPS["n_check"], short_steps=CAPS["short_steps"],
                                     max_untrained=CAPS["max_untrained_exact"], max_clean=CAPS["max_untrained_clean"],
                                     max_short=CAPS["max_short_exact"])

    # ---------- one search ----------
    def search(self, sdir: Path, world: str, arm: str, program: str, search_seed: int) -> dict:
        sdir.mkdir(parents=True, exist_ok=True)
        st_file = sdir / "state.json"
        st = json.loads(st_file.read_text()) if st_file.exists() else {"trials": [], "rounds_done": 0, "incumbent": None}
        if program is not None and not (sdir / "program.md").exists():
            (sdir / "program.md").write_text(program)
        R, K, seeds = self.cfg["R"], self.cfg["K"], self.cfg["trial_seeds"]
        base_src = (HERE / "candidates" / "baseline.py").read_text()
        base_dev = self.base[world]["shown_dev"]
        if arm == "menu":
            pool = [c for c in MENU if c != BASE_CFG]
            order = random.Random(7919 * search_seed + 17).sample(pool, R * K)
        for r in range(st["rounds_done"] + 1, R + 1):
            trials = st["trials"]
            seen = {sha(base_src): "the starting architecture"} | {t["result"]["sha"]: t["label"] for t in trials if t.get("result", {}).get("sha")}
            inc = next((t for t in trials if t["label"] == st["incumbent"]), None)
            if arm == "menu":
                props = [{"name": f"menu {c['encoder']} w{c['width']} d{c['depth']} {c['activation']} {c['norm']}",
                          "hypothesis": f"random menu sample {json.dumps(c)}", "code": menu_source(c)}
                         for c in order[(r - 1) * K: r * K]]
                analysis = ""
            else:
                prompt = S.l1_prompt(program, self.task(world), self.interface(), base_src, base_dev, trials, inc, r, R, K)
                val, rec = S.call_with_retries(S.SYSTEM_L1, prompt, S.l1_schema(K), lambda o: S.parse_proposals(o, K),
                                               sdir / f"round_{r}.json")
                props, analysis = val["proposals"], val["analysis"]
            # pre-training checks, with up to 2 repair calls for the scientist
            checks = [None] * K
            for j in range(3):
                todo = [i for i in range(K) if checks[i] is None or not checks[i]["ok"]]

                def run_check(i, j=j):
                    p = sdir / "cands" / f"r{r}c{i + 1}_v{j}.py"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(props[i]["code"])
                    return self.check(p, world)
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
                if not failed or arm == "menu" or j == 2:
                    break
                fl = [{"name": props[i]["name"], "error": checks[i]["error"], "code": props[i]["code"]} for i in failed]
                try:
                    fix, _ = S.call_with_retries(S.SYSTEM_L1, S.repair_prompt(prompt, fl), S.l1_schema(len(failed)),
                                                 lambda o, n=len(failed): S.parse_proposals(o, n), sdir / f"round_{r}_repair{j + 1}.json")
                except RuntimeError as e:
                    self.log(f"{sdir.relative_to(self.dir)} round {r} repair {j + 1} failed: {e}")
                    break
                for i, f in zip(failed, fix["proposals"]):
                    props[i] = {"name": f["name"] or props[i]["name"], "hypothesis": f["hypothesis"], "code": f["code"]}
            # train the valid ones
            new = []
            jobs = []
            for i in range(K):
                t = {"label": f"r{r}c{i + 1}", "round": r, "name": props[i]["name"], "hypothesis": props[i]["hypothesis"],
                     "code": props[i]["code"], "check": checks[i]}
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

    # ---------- Level 2 ----------
    def revise(self, g: int, programs: dict, searches: list) -> str:
        out_prog = self.dir / "programs" / f"g{g + 1}.md"
        if out_prog.exists():
            return out_prog.read_text()
        P = self.cfg["P"]
        base_dev = self.base[P]["shown_dev"]
        logs = []
        for i, st in enumerate(searches):
            ok = [t for t in st["trials"] if t["result"]["status"] == "ok"]
            best = max([t["result"]["dev_exact"] for t in ok], default=None)
            parts = [f"### Search {i + 1}: best dev success {best} (starting architecture {base_dev:.3f}); "
                     f"selected: {st['incumbent'] or 'the starting architecture'}"]
            for r, a in sorted(st.get("analyses", {}).items()):
                parts.append(f"Your analysis before round {r}: {a}")
            parts.append("Trials:")
            parts += [S.fmt_trial(t, show_code=False) for t in st["trials"]]
            top = sorted(ok, key=lambda t: -t["result"]["dev_exact"])[:3]
            parts.append("Code of the three best trials:")
            parts += [f"{t['label']} ({t['name']}), dev {t['result']['dev_exact']}:\n```python\n{t['code'].strip()}\n```" for t in top]
            logs.append("\n".join(parts))
        table = []
        for h in range(g + 1):
            hs = self.gen_stats.get(h)
            if hs:
                table.append(f"- generation {h}: best dev success per search {hs['best']}, mean {hs['mean']:.3f}; "
                             f"invalid candidates {hs['invalid']}, failed trainings {hs['failed']}")
            rv = self.dir / "programs" / f"revision_g{h - 1}_to_g{h}.json"
            if h >= 1 and rv.exists():
                ch = json.loads(rv.read_text())["result"]["changes"]
                table.append(f"  changes you made when writing generation {h}: " + "; ".join(c.get("change", "")[:200] for c in ch))
        prompt = S.l2_prompt(g, programs[g], self.cfg["R"], self.cfg["K"], self.task(P), "\n".join(table) or "None.",
                             "\n\n".join(logs), self.cfg["max_words"])
        val, rec = S.call_with_retries(S.SYSTEM_L2, prompt, S.L2_SCHEMA, lambda o: S.parse_program(o, self.cfg["max_words"]),
                                       self.dir / "programs" / f"call_g{g}_to_g{g + 1}.json")
        prog = val["program"]
        (self.dir / "programs" / f"revision_g{g}_to_g{g + 1}.json").write_text(json.dumps({"result": val, "sha": sha(prog)}, indent=1))
        out_prog.write_text(prog)
        self.log(f"program g{g + 1} written ({len(prog.split())} words)")
        return prog

    def stats(self, searches: list) -> dict:
        best = [max([t["result"]["dev_exact"] for t in st["trials"] if t["result"]["status"] == "ok"], default=0.0) for st in searches]
        return {"best": [round(b, 3) for b in best], "mean": sum(best) / len(best),
                "invalid": sum(t["result"]["status"] == "invalid" for st in searches for t in st["trials"]),
                "failed": sum(t["result"]["status"] not in ("ok", "invalid") for st in searches for t in st["trials"])}

    # ---------- the whole campaign ----------
    def run(self):
        c = self.cfg
        self.log(f"start preset={self.preset}")
        self.base = self.baselines()
        (self.dir / "programs").mkdir(exist_ok=True)
        g0 = (HERE / "programs" / "g0.md").read_text()
        (self.dir / "programs" / "g0.md").write_text(g0)
        programs = {0: g0}
        self.gen_stats = {}
        # phase 1: meta-training on P
        for g in range(c["G"]):
            with ThreadPoolExecutor(c["S_P"]) as ex:
                searches = list(ex.map(lambda i: self.search(self.dir / "P" / f"g{g}" / f"s{i + 1}", c["P"], f"g{g}", programs[g], i + 1),
                                       range(c["S_P"])))
            self.gen_stats[g] = self.stats(searches)
            self.log(f"phase 1 generation {g} on {c['P']}: {self.gen_stats[g]}")
            programs[g + 1] = self.revise(g, programs, searches)
        (self.dir / "P" / "gen_stats.json").write_text(json.dumps(self.gen_stats, indent=1))
        # phase 2: meta-testing on Q
        arms = [("g0", programs[0]), (f"g{c['G']}", programs[c["G"]]), ("menu", None)]
        jobs = [(a, p, i + 1) for i in range(c["S_Q"]) for a, p in arms]
        with ThreadPoolExecutor(len(jobs)) as ex:
            qres = list(ex.map(lambda j: (j[0], j[2], self.search(self.dir / "Q" / j[0] / f"s{j[2]}", c["Q"], j[0], j[1], 100 + j[2])), jobs))
        self.log("phase 2 done")
        # phase 3: confirmation with fresh seeds
        Q = c["Q"]
        conf = []
        base_src = (HERE / "candidates" / "baseline.py").read_text()
        for arm, i, st in qres:
            inc = next((t for t in st["trials"] if t["label"] == st["incumbent"]), None)
            src = inc["code"] if inc else base_src
            conf.append({"kind": "search", "arm": arm, "search": i, "label": inc["label"] if inc else "start", "src": src,
                         "seeds": c["confirm_seeds"], "search_dev": inc["result"]["dev_exact"] if inc else self.base[Q]["shown_dev"]})
        conf.append({"kind": "comparator", "arm": "start", "search": 0, "label": "start", "src": base_src, "seeds": c["comparator_seeds"]})
        runs = {}
        for e in conf:
            for s in e["seeds"]:
                runs[(sha(e["src"]), s)] = e["src"]

        def do(key):
            h, s = key
            d = self.dir / "confirm" / h
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "candidate.py").exists():
                (d / "candidate.py").write_text(runs[key])
            return key, self.train_one(d / f"seed{s}", d / "candidate.py", Q, s)
        with ThreadPoolExecutor(len(runs)) as ex:
            cres = dict(ex.map(do, list(runs)))
        self.log(f"phase 3 done: {len(cres)} confirmation runs")
        # phase 4: open the final split once
        marker = self.dir / "FINAL_OPENED"
        if not marker.exists():
            marker.write_text(f"opened {time.strftime('%Y-%m-%d %H:%M:%S %Z')} after all searches and confirmation runs\n")
        finals = {}
        for (h, s), r in cres.items():
            rdir = self.dir / "confirm" / h / f"seed{s}"
            fpath = rdir / "final.json"
            if r["status"] == "ok" and not fpath.exists():
                subprocess.run([PY, "-I", str(HERE / "evaluate_arch.py"), "--trial", str(rdir / "run"), "--split", "final",
                                "--out", str(fpath), "--campaign", str(self.dir)], capture_output=True, text=True, timeout=1800, check=True)
            finals[f"{h}:{s}"] = json.loads(fpath.read_text())["exact"] if fpath.exists() else None
        summary = []
        for e in conf:
            h = sha(e["src"])
            summary.append({k: v for k, v in e.items() if k != "src"} | {
                "sha": h, "dev": {s: cres[(h, s)]["dev_exact"] for s in e["seeds"]},
                "final": {s: finals[f"{h}:{s}"] for s in e["seeds"]},
                "status": {s: cres[(h, s)]["status"] for s in e["seeds"]}})
        (self.dir / "confirm" / "summary.json").write_text(json.dumps(summary, indent=1))
        self.log("phase 4 done: final scores written; run analyze_arch.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    args = ap.parse_args()
    Campaign(Path(args.campaign).resolve(), args.preset).run()


if __name__ == "__main__":
    main()
