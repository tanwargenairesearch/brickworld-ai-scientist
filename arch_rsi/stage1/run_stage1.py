"""Stage 1 of PLAN.md: does the architecture search use its experiments? Pre-registered in protocol_stage1.md,
written before any Stage 1 result.

Reuses the frozen orchestrate_arch.Campaign (training, dev scoring, checks, Claude calls, hash checks) unchanged and
adds, for world Q only:
- arm "nofb": program g0 with every success score hidden from the scientist (it still sees status, check failures
  and errors); the code still selects with the real scores, so the only thing removed is adaptation to results;
- retest before selection, in every arm: after the 12 trials, the top 3 by 2-seed dev mean train 2 more seeds; the
  pick is the best 4-seed mean among those whose seeds all trained, or the starting architecture if none beats the
  start's own 4-seed mean;
- confirmation: each search's pick trains 8 fresh seeds, scored on dev (the final split stays locked);
- comparators, no search: the start, and the start with a separate critic and orthogonal init, 12 seeds each.

usage (project root):
  caffeinate -i .venv/bin/python -I arch_rsi/stage1/run_stage1.py --preset stage1      (the real run)
  caffeinate -i .venv/bin/python -I arch_rsi/stage1/run_stage1.py --preset stage1_dry  (tiny budgets)
Resumes from disk if rerun with the same preset.
"""
import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from arch_rsi import orchestrate_arch as O  # noqa: E402
from arch_rsi import scientist as S  # noqa: E402
from arch_rsi.menu import MENU  # noqa: E402
from arch_rsi.menu import source as menu_source  # noqa: E402
from arch_rsi.sandbox import sha  # noqa: E402

STAGE1_FILES = [HERE / p for p in ("run_stage1.py", "protocol_stage1.md", "analyze_stage1.py", "review_stage1.py",
                                   "candidates/start_critic_ortho.py")]

PRESETS = {
    "stage1": dict(O.PRESETS["main"], steps={"large": 131072, "xl": 131072}, timeout={"large": 400, "xl": 600},
                   arms=["g0", "nofb", "menu"], n_searches=10, search_workers=6, trial_seeds=[0, 1],
                   retest_top=3, retest_seeds=[2, 3], base_seeds=[0, 1, 2, 3],
                   confirm_seeds=list(range(1001, 1009)), comparator_seeds=list(range(1001, 1013)),
                   search_seed_base=500, campaign="arch_rsi/campaigns/stage1"),
}
PRESETS["stage1_dry"] = dict(PRESETS["stage1"], steps={"large": 16384, "xl": 16384}, R=2, K=2, n_searches=1,
                             search_workers=3, retest_top=2, retest_seeds=[2], base_seeds=[0, 1, 2],
                             confirm_seeds=[1001, 1002], comparator_seeds=[1001, 1002],
                             campaign="arch_rsi/campaigns/stage1_dry")

COMPARATORS = {"start": O.HERE / "candidates" / "baseline.py",
               "critic_ortho": HERE / "candidates" / "start_critic_ortho.py"}

FROZEN_SELECTION = "At the end of the search, the trial with the highest dev success is selected."

HIDDEN_NOTE = ("In this search you are not shown any success score: not for your trials and not for the starting "
               "architecture. You still see whether each trial trained without errors, and any check failure or error "
               "message. The selection at the end of the search still uses the real dev scores.")


# ---------------------------------------------------------------- pure helpers (tested in tests/test_stage1.py)
def selection_text(cfg) -> str:
    k, n_re = cfg["retest_top"], len(cfg["retest_seeds"])
    n_all = len(cfg["trial_seeds"]) + n_re
    return (f"At the end of the search, the {k} trials with the highest dev success each train {n_re} more "
            f"seed{'s' if n_re > 1 else ''}, and the one with the highest mean over its {n_all} seeds is selected "
            f"(the starting architecture if none beats the starting architecture's own {n_all}-seed mean).")


def fmt_trial_hidden(t: dict, show_code: bool = True) -> str:
    """S.fmt_trial without any outcome: no dev or training success, no training time."""
    r = t.get("result", {})
    head = {"trial": t["label"], "round": t["round"], "name": t.get("name"), "status": r.get("status"),
            "params": r.get("params"), "train_flops_vs_start": r.get("flops_ratio"), "train_steps": r.get("train_steps")}
    if r.get("error"):
        head["error"] = str(r["error"])[:400]
    s = json.dumps(head) + "\nhypothesis: " + str(t.get("hypothesis", ""))
    if show_code and t.get("code"):
        s += "\n```python\n" + t["code"].strip() + "\n```"
    return s


def l1_prompt_hidden(program, task, interface, base_code, trials, r, R, K) -> str:
    """S.l1_prompt with the scores, the current best and the start's score removed, and a note saying so."""
    hist = "No trials yet." if not trials else "\n\n".join(fmt_trial_hidden(t) for t in trials)
    left = (R - r + 1) * K
    return f"""## Your research program
{program}

{task}
{interface}

## Scores are hidden in this search
{HIDDEN_NOTE}

## Starting architecture
```python
{base_code.strip()}
```

## Results so far in this search
{hist}

## Budget
This is round {r} of {R}. Each round trains {K} candidates in parallel; {left} trials remain including this round.

## What to return
A JSON object with:
- "analysis": at most 150 words on what the results so far suggest.
- "proposals": exactly {K} objects, each with "name" (a few words), "hypothesis" (at most 80 words: what changes, why it should help, what result would refute it) and "code" (the complete candidate file)."""


def choose_pick(rows: list, start_mean: float):
    """rows: [{"label", "order", "all_ok", "mean"}] for the retested trials. Returns the label of the pick, or None
    for the starting architecture. Only trials whose seeds all trained are eligible; a trial must beat the start's
    mean strictly; ties between trials go to the earlier one."""
    best, pick = start_mean, None
    for row in sorted(rows, key=lambda x: x["order"]):
        if row["all_ok"] and row["mean"] > best:
            best, pick = row["mean"], row["label"]
    return pick


def top_for_retest(trials: list, k: int) -> list:
    ok = [(i, t) for i, t in enumerate(trials) if t["result"]["status"] == "ok"]
    ok.sort(key=lambda it: (-it[1]["result"]["dev_exact"], it[0]))
    return [t for _, t in ok[:k]]


def search_order(arms: list, n: int) -> list:
    """Interleaved, so every arm meets the same machine load: (g0 1, nofb 1, menu 1, g0 2, ...)."""
    return [(a, i) for i in range(1, n + 1) for a in arms]


# ---------------------------------------------------------------- the campaign
class Stage1(O.Campaign):
    def __init__(self, cdir: Path, preset: str):
        O.PRESETS[preset] = PRESETS[preset]
        O.FROZEN = list(O.FROZEN) + [p for p in STAGE1_FILES if p not in O.FROZEN]
        super().__init__(cdir, preset)
        self.run_locks = {}

    # phase 0, Q only: the start on the base seeds (the scientist's noise note and the selection's start mean)
    def baselines_q(self) -> dict:
        Q, c = self.cfg["Q"], self.cfg
        d = self.dir / "baseline" / Q
        d.mkdir(parents=True, exist_ok=True)
        src = (O.HERE / "candidates" / "baseline.py").read_text()
        (d / "candidate.py").write_text(src)
        chk = self.check(d / "candidate.py", Q)
        with ThreadPoolExecutor(len(c["base_seeds"])) as ex:
            res = list(ex.map(lambda s: self.train_one(d / f"seed{s}", d / "candidate.py", Q, s), c["base_seeds"]))
        devs = [r["dev_exact"] for r in res]
        if any(x is None for x in devs):
            raise SystemExit(f"starting architecture failed on {Q}: {res}")
        shown = [devs[c["base_seeds"].index(s)] for s in c["trial_seeds"]]
        summ = {"devs": devs, "mean": sum(devs) / len(devs), "min": min(devs), "max": max(devs),
                "shown_dev": sum(shown) / len(shown), "check": chk, "seeds": c["base_seeds"]}
        (self.dir / "baseline" / "summary.json").write_text(json.dumps({Q: summ}, indent=1))
        self.log(f"phase 0 done: {Q} {[round(x, 3) for x in devs]}")
        return summ

    def task_for(self, w: str, hidden: bool) -> str:
        n = len(self.cfg["trial_seeds"])
        noise = (f"every trial trains {n} seeds. Training runs on this task can succeed or fail by seed luck alone."
                 if hidden else self.noise_note(w))
        t = S.task_text(O.SPECS[w], self.cfg["steps"][w], self.cfg["timeout"][w], 512, noise)
        if FROZEN_SELECTION not in t:
            raise SystemExit("task text changed: selection sentence not found")
        return t.replace(FROZEN_SELECTION, selection_text(self.cfg))

    def locked_train(self, rdir: Path, src_path: Path, world: str, seed: int) -> dict:
        """train_one with a lock per run directory, for runs that several searches may share."""
        with self.lock:
            lk = self.run_locks.setdefault(str(rdir), threading.Lock())
        with lk:
            return self.train_one(rdir, src_path, world, seed)

    # one search: the frozen search loop, plus the hidden-score prompt for "nofb"
    def search(self, sdir: Path, world: str, arm: str, program: str, search_seed: int) -> dict:
        sdir.mkdir(parents=True, exist_ok=True)
        st_file = sdir / "state.json"
        st = json.loads(st_file.read_text()) if st_file.exists() else {"trials": [], "rounds_done": 0, "incumbent": None}
        if program is not None and not (sdir / "program.md").exists():
            (sdir / "program.md").write_text(program)
        R, K, seeds = self.cfg["R"], self.cfg["K"], self.cfg["trial_seeds"]
        hidden = arm == "nofb"
        base_src = (O.HERE / "candidates" / "baseline.py").read_text()
        base_dev = self.base[world]["shown_dev"]
        if arm == "menu":
            pool = [c for c in MENU if c != O.BASE_CFG]
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
                task = self.task_for(world, hidden)
                if hidden:
                    prompt = l1_prompt_hidden(program, task, self.interface(), base_src, trials, r, R, K)
                else:
                    prompt = S.l1_prompt(program, task, self.interface(), base_src, base_dev, trials, inc, r, R, K)
                val, rec = S.call_with_retries(S.SYSTEM_L1, prompt, S.l1_schema(K), lambda o: S.parse_proposals(o, K),
                                               sdir / f"round_{r}.json")
                props, analysis = val["proposals"], val["analysis"]
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
            new, jobs = [], []
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

    # retest the top trials, then pick
    def retest_and_pick(self, sdir: Path, st: dict, world: str) -> dict:
        pick_file = sdir / "pick.json"
        if pick_file.exists():
            return json.loads(pick_file.read_text())
        c = self.cfg
        top = top_for_retest(st["trials"], c["retest_top"])
        jobs = [(t, s) for t in top for s in c["retest_seeds"]]

        def run(job):
            t, s = job
            d = sdir / "trials" / t["label"]
            return self.train_one(d / f"seed{s}", d / "candidate.py", world, s)
        with ThreadPoolExecutor(max(1, len(jobs))) as ex:
            outs = list(ex.map(run, jobs))
        res = {}
        for (t, s), o in zip(jobs, outs):
            res.setdefault(t["label"], {})[s] = o
        order = {t["label"]: i for i, t in enumerate(st["trials"])}
        rows = []
        for t in top:
            first = t["result"]["dev_per_seed"] or [t["result"]["dev_exact"]]
            more = [res[t["label"]][s]["dev_exact"] if res[t["label"]][s]["status"] == "ok" else 0.0 for s in c["retest_seeds"]]
            all_ok = all(res[t["label"]][s]["status"] == "ok" for s in c["retest_seeds"])
            devs = list(first) + more
            rows.append({"label": t["label"], "name": t["name"], "order": order[t["label"]], "all_ok": all_ok,
                         "dev_2seed": t["result"]["dev_exact"], "dev_all": [round(x, 4) for x in devs],
                         "mean": sum(devs) / len(devs), "sha": t["result"]["sha"]})
        start_mean = self.base[world]["mean"]
        label = choose_pick(rows, start_mean)
        pick = {"label": label or "start", "start_mean": start_mean, "retested": rows,
                "incumbent_2seed": st["incumbent"]}
        if label:
            t = next(t for t in st["trials"] if t["label"] == label)
            pick |= {"name": t["name"], "src": t["code"], "sha": sha(t["code"]),
                     "mean_selected": next(r["mean"] for r in rows if r["label"] == label)}
        else:
            src = (O.HERE / "candidates" / "baseline.py").read_text()
            pick |= {"name": "the starting architecture", "src": src, "sha": sha(src), "mean_selected": start_mean}
        pick_file.write_text(json.dumps(pick, indent=1))
        self.log(f"{sdir.relative_to(self.dir)} pick {pick['label']} ({pick['mean_selected']:.3f}); "
                 f"retested {[(r['label'], round(r['mean'], 3)) for r in rows]}; 2-seed incumbent {st['incumbent']}")
        return pick

    def confirm(self, src: str, seeds: list) -> dict:
        h = sha(src)
        d = self.dir / "confirm" / h
        d.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if not (d / "candidate.py").exists():
                (d / "candidate.py").write_text(src)
        with ThreadPoolExecutor(len(seeds)) as ex:
            res = list(ex.map(lambda s: self.locked_train(d / f"seed{s}", d / "candidate.py", self.cfg["Q"], s), seeds))
        return {s: r for s, r in zip(seeds, res)}

    def one_search(self, arm: str, i: int, program: str) -> dict:
        Q = self.cfg["Q"]
        sdir = self.dir / "Q" / arm / f"s{i}"
        st = self.search(sdir, Q, arm, program, self.cfg["search_seed_base"] + i)
        pick = self.retest_and_pick(sdir, st, Q)
        conf = self.confirm(pick["src"], self.cfg["confirm_seeds"])
        self.log(f"Q/{arm}/s{i} confirmed: {[round(r['dev_exact'], 3) if r['dev_exact'] is not None else r['status'] for r in conf.values()]}")
        return {"kind": "search", "arm": arm, "search": i, "label": pick["label"], "name": pick["name"],
                "sha": pick["sha"], "mean_selected": pick["mean_selected"], "incumbent_2seed": pick["incumbent_2seed"],
                "seeds": self.cfg["confirm_seeds"], "dev": {s: r["dev_exact"] for s, r in conf.items()},
                "status": {s: r["status"] for s, r in conf.items()}, "wall_s": {s: r["wall_s"] for s, r in conf.items()}}

    def one_comparator(self, name: str) -> dict:
        src = COMPARATORS[name].read_text()
        p = self.dir / "comparators" / name / "candidate.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
        chk = self.check(p, self.cfg["Q"])
        if not chk.get("ok"):
            raise SystemExit(f"comparator {name} failed the checks: {chk}")
        conf = self.confirm(src, self.cfg["comparator_seeds"])
        self.log(f"comparator {name}: {[round(r['dev_exact'], 3) if r['dev_exact'] is not None else r['status'] for r in conf.values()]}")
        return {"kind": "comparator", "arm": name, "search": 0, "label": name, "name": name, "sha": sha(src),
                "seeds": self.cfg["comparator_seeds"], "dev": {s: r["dev_exact"] for s, r in conf.items()},
                "status": {s: r["status"] for s, r in conf.items()}, "wall_s": {s: r["wall_s"] for s, r in conf.items()}}

    def run(self):
        c = self.cfg
        self.log(f"start preset={self.preset}")
        self.base = {c["Q"]: self.baselines_q()}
        g0 = (O.HERE / "programs" / "g0.md").read_text()
        (self.dir / "programs").mkdir(exist_ok=True)
        (self.dir / "programs" / "g0.md").write_text(g0)
        programs = {"g0": g0, "nofb": g0, "menu": None}
        order = search_order(c["arms"], c["n_searches"])
        with ThreadPoolExecutor(2) as cx, ThreadPoolExecutor(c["search_workers"]) as sx:
            comp_f = [cx.submit(self.one_comparator, n) for n in COMPARATORS]
            search_f = [sx.submit(self.one_search, a, i, programs[a]) for a, i in order]
            entries = [f.result() for f in search_f] + [f.result() for f in comp_f]
        (self.dir / "confirm" / "summary.json").write_text(json.dumps(entries, indent=1))
        self.verify()
        self.log("STAGE1_DONE: run analyze_stage1.py, then review_stage1.py. The final split was not opened.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    args = ap.parse_args()
    Stage1((ROOT / PRESETS[args.preset]["campaign"]).resolve(), args.preset).run()


if __name__ == "__main__":
    main()
