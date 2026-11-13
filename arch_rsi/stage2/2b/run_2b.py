"""Stage 2b (PROTOCOL_2B.md): does the search use its experiments when it controls the recipe? World W.

Reuses Stage 1's campaign (run_stage1.Stage1, itself the frozen orchestrate_arch.Campaign) for the search loop,
the hidden-score prompt, retest-and-pick, confirmation and Claude calls, and changes only:
- training: stage2/train_2b.py on the GPU (RECIPE block, fixed-FLOP budget), scored by the frozen evaluator on W;
- checks: stage2/check_2b.py (Stage 1's checks on W, plus the RECIPE block and the per-arm rules);
- arms: g0, nofb, netonly, reconly (Claude, program g0r) and random (menu network x random recipe);
- comparators: reference A, the start with Stage G's recipe, the start with the frozen recipe (seeds 1001-1012).

usage (project root):
  caffeinate -i .venv/bin/python -I arch_rsi/stage2/2b/run_2b.py --preset 2b       (the real run)
  caffeinate -i .venv/bin/python -I arch_rsi/stage2/2b/run_2b.py --preset 2b_dry   (tiny budgets)
Resumes from disk if rerun with the same preset.
"""
import argparse
import importlib.util
import json
import random
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

from arch_rsi import orchestrate_arch as O  # noqa: E402
from arch_rsi import scientist as S  # noqa: E402
from arch_rsi.menu import MENU  # noqa: E402
from arch_rsi.menu import source as menu_source  # noqa: E402
from arch_rsi.sandbox import sha  # noqa: E402
from arch_rsi.stage2 import w2b, w_world  # noqa: E402

w_world.register()
spec_ = importlib.util.spec_from_file_location("run_stage1", A / "stage1" / "run_stage1.py")
S1 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(S1)

PY = str(ROOT / ".venv" / "bin" / "python")
W = "w"
CLAUDE_ARMS = ("g0", "nofb", "netonly", "reconly")
CHECK_MODE = {"g0": "both", "nofb": "both", "netonly": "fixed", "reconly": "recipe_only", "random": "both"}
REF_A = A / "campaigns" / "stage1" / "confirm" / "1c9c0e6b521fc0bb" / "candidate.py"
BASE = A / "candidates" / "baseline.py"
FILES_2B = ([HERE / p for p in ("run_2b.py", "PROTOCOL_2B.md")]
            + [S2 / p for p in ("w2b.py", "w_world.py", "train_2b.py", "check_2b.py", "programs/g0r.md",
                                "targets/w_dev.pt", "targets/w_heldout_hashes.pt", "targets/w_check.pt", "holdout/w_final.pt")]
            + [A / "stage1" / "run_stage1.py", A / "stage1" / "analyze_stage1.py", A / "check.py", REF_A])

PRESETS = {
    "2b": dict(O.PRESETS["main"], Q=W, steps={W: 131072}, timeout={W: 900}, R=4, K=3,
               arms=["g0", "nofb", "netonly", "reconly", "random"], n_searches=16, search_workers=10, workers=4,
               trial_seeds=[0, 1], retest_top=3, retest_seeds=[2, 3], base_seeds=[0, 1, 2, 3],
               confirm_seeds=list(range(1001, 1009)), comparator_seeds=list(range(1001, 1013)),
               search_seed_base=500, campaign="arch_rsi/campaigns/stage2b"),
}
PRESETS["2b_dry"] = dict(PRESETS["2b"], steps={W: 32768}, timeout={W: 600}, R=2, K=2, n_searches=1, search_workers=5,
                         retest_top=2, retest_seeds=[2], base_seeds=[0, 1, 2], confirm_seeds=[1001, 1002],
                         comparator_seeds=[1001, 1002], campaign="arch_rsi/campaigns/stage2b_dry")

TASK_2B = """## Task
An agent must build a target design in a brick world.
- The world is a {X} x {Y} base, up to {Z} layers high. Bricks are one layer high with footprints 1x1, 1x2, 2x1 or 2x2.
- A placed brick drops until it rests on the highest filled cell under its footprint (or the ground), so overhangs with empty cells under them are possible and placement order matters.
- Each step the agent picks a brick type and an anchor cell, or DONE. An episode ends at DONE or after {max_steps} placements. Bricks cannot be removed.
- Target designs are made by {k_min} to {k_max} random brick drops. Training uses a stream of fresh random designs; the held-out designs never appear in training.

## Training and evaluation
- Each trial trains a PPO agent from scratch on a GPU with a fixed training compute: the floating-point operations the starting architecture uses in {steps:,} environment steps with the default recipe. Compute counts one forward pass to collect each sample (a third of a forward-and-backward pass) and one forward-and-backward pass per sample per PPO epoch. So a trial trains for {steps:,} x (4 + 1/3) / ((ppo_epochs + 1/3) x r) steps, rounded down to whole updates of n_envs x rollout_len steps, where r is the network's training FLOPs per sample relative to the starting architecture (r below 1 counts as 1). More epochs or a bigger network means fewer samples; fewer epochs means more. The budget must allow at least 4 updates.
{recipe_text}
- Fixed for every trial: Adam with the learning rate decaying linearly to 0, value coefficient 0.5, gradient-norm clip 1.0, normalised advantages. Reward per placement: (target cells filled minus cells filled outside the target) / the target's cell count, minus 0.01; plus 1 when the episode ends with an exact build.
- A trial whose training runs longer than {timeout:,} seconds is killed and counts as a failure. Some operations cost much more time than their FLOP count suggests (3D convolutions run several times slower than 2D ones of equal FLOPs on this GPU).
- After training, the agent runs greedily on {n_dev} held-out dev designs. Success means the finished build equals the target design exactly, cell for cell.
- Selection metric: dev success (higher is better). At the end of the search, the trial with the highest dev success is selected. It is then retrained with new seeds and tested once on a separate hidden set of designs, so a lucky seed does not help.
- Noise: {noise}
"""

BOUNDS_TEXT = ("lr 1e-4 to 3e-3; n_envs 64 to 1024 (integer, parallel worlds per update); rollout_len 8 to 64 (integer); "
               "n_minibatches 4 to 64 (integer, must divide n_envs x rollout_len, at least 64 samples per minibatch); "
               "ppo_epochs 1 to 8 (integer); ent_coef 0 to 0.05; gamma 0.9 to 0.995; gae_lambda 0.8 to 0.99; clip_eps 0.1 to 0.3")
DEFAULTS_TEXT = ", ".join(f"{k} {v}" for k, v in w2b.STAGE_G.items())
RECIPE_TEXT = {
    "both": ("- In this search you choose the network and the training recipe. To set the recipe, add one top-level constant "
             "to your file, `RECIPE = {{...}}`, a literal dict with any of these keys and bounds: " + BOUNDS_TEXT +
             ". Keys you leave out take the default recipe: " + DEFAULTS_TEXT + ". A file without RECIPE trains with the default."),
    "fixed": ("- In this search you change only the network. The training recipe is fixed at: " + DEFAULTS_TEXT +
              ". Do not add a RECIPE constant to your file."),
    "recipe_only": ("- In this search the network is fixed: every trial uses the starting architecture's code unchanged. You change "
                    "only the training recipe, by returning the starting architecture's file with one top-level constant added, "
                    "`RECIPE = {{...}}`, a literal dict with any of these keys and bounds: " + BOUNDS_TEXT +
                    ". Keys you leave out take the default recipe: " + DEFAULTS_TEXT + ". Only the RECIPE constant of your "
                    "file is used; the network is always the starting architecture."),
}


def random_proposals(search_seed: int, n: int, base_flops: int, budget_steps: int) -> list:
    """The random arm's n (menu network, recipe) draws, deterministic in the search seed (PROTOCOL_2B.md)."""
    from arch_rsi.check import train_flops
    from arch_rsi.interface import arch_spec
    rng = random.Random(7919 * search_seed + 17)
    pool = [c for c in MENU if c != O.BASE_CFG]
    nets = rng.sample(pool, n)
    aspec = arch_spec(O.SPECS[W])
    out = []
    for c in nets:
        src = menu_source(c)
        r = train_flops(src, aspec) / base_flops
        for _ in range(1000):
            rec = w2b.random_recipe(rng)
            if not w2b.validate(rec) and w2b.budget(budget_steps, r, rec)["ok"]:
                break
        else:
            raise SystemExit(f"no valid random recipe for {c}")
        out.append({"name": f"menu {c['encoder']} w{c['width']} d{c['depth']} {c['activation']} {c['norm']} + random recipe",
                    "hypothesis": f"random menu sample {json.dumps(c)} with random recipe {json.dumps(rec)}",
                    "code": w2b.with_recipe(src, rec)})
    return out


class Stage2b(S1.Stage1):
    def __init__(self, cdir: Path, preset: str):
        O.PRESETS[preset] = PRESETS[preset]
        O.FROZEN = list(O.FROZEN) + [p for p in FILES_2B if p not in O.FROZEN]
        O.Campaign.__init__(self, cdir, preset)
        self.run_locks = {}

    # ---- training and checks: the 2b trainer and check on W
    def train_one(self, rdir: Path, src_path: Path, world: str, seed: int, split: str = "dev") -> dict:
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
                p = subprocess.run([PY, "-I", str(S2 / "train_2b.py"), "--spec", str(spec), "--out", str(rdir / "run")],
                                   capture_output=True, text=True, timeout=self.cfg["timeout"][world])
                tl = json.loads((rdir / "run" / "train_log.json").read_text()) if (rdir / "run" / "train_log.json").exists() else {}
                status = tl.get("status", f"crashed (exit {p.returncode})")
                err = tl.get("error") or (None if status == "ok" else (p.stderr or "")[-400:])
            except subprocess.TimeoutExpired:
                tl, status, err = {}, "timeout", f"killed after {self.cfg['timeout'][world]} s"
            train_wall = round(time.time() - t0, 1)
            dev = None
            if status == "ok":
                q = subprocess.run([PY, "-I", str(S2 / "w2b.py"), "eval", "--trial", str(rdir / "run"),
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
        with self.check_slots:
            p = subprocess.run([PY, "-I", str(S2 / "check_2b.py"), "--candidate", str(path), "--mode", mode,
                                "--budget", str(self.cfg["steps"][world]), "--out", str(out)], capture_output=True, text=True, timeout=1200)
        if p.returncode != 0 or not out.exists():
            return {"ok": False, "error": f"check process failed: {(p.stderr or '')[-300:]}"}
        return json.loads(out.read_text())

    def task_for(self, w: str, hidden: bool, arm: str = "g0") -> str:
        n = len(self.cfg["trial_seeds"])
        noise = (f"every trial trains {n} seeds. Training runs on this task can succeed or fail by seed luck alone."
                 if hidden else self.noise_note(w))
        sp = O.SPECS[w]
        t = TASK_2B.format(X=sp.X, Y=sp.Y, Z=sp.Z, max_steps=sp.max_steps, k_min=sp.k_min, k_max=sp.k_max,
                           steps=self.cfg["steps"][w], timeout=self.cfg["timeout"][w], n_dev=512, noise=noise,
                           recipe_text=RECIPE_TEXT[CHECK_MODE[arm]].replace("{{", "{").replace("}}", "}"))
        if S1.FROZEN_SELECTION not in t:
            raise SystemExit("task text changed: selection sentence not found")
        return t.replace(S1.FROZEN_SELECTION, S1.selection_text(self.cfg))

    # ---- one search: Stage 1's loop with the 2b arms
    def search(self, sdir: Path, world: str, arm: str, program: str, search_seed: int) -> dict:
        sdir.mkdir(parents=True, exist_ok=True)
        st_file = sdir / "state.json"
        st = json.loads(st_file.read_text()) if st_file.exists() else {"trials": [], "rounds_done": 0, "incumbent": None}
        if program is not None and not (sdir / "program.md").exists():
            (sdir / "program.md").write_text(program)
        R, K, seeds = self.cfg["R"], self.cfg["K"], self.cfg["trial_seeds"]
        hidden, mode = arm == "nofb", CHECK_MODE[arm]
        base_src = BASE.read_text()
        base_dev = self.base[world]["shown_dev"]
        if arm == "random":
            from arch_rsi.check import train_flops
            from arch_rsi.interface import arch_spec
            drawn = random_proposals(search_seed, R * K, train_flops(base_src, arch_spec(O.SPECS[W])), self.cfg["steps"][W])
        for r in range(st["rounds_done"] + 1, R + 1):
            trials = st["trials"]
            seen = {sha(base_src): "the starting architecture"} | {t["result"]["sha"]: t["label"] for t in trials if t.get("result", {}).get("sha")}
            inc = next((t for t in trials if t["label"] == st["incumbent"]), None)
            prompt = None
            if arm == "random":
                props, analysis = drawn[(r - 1) * K: r * K], ""
            else:
                task = self.task_for(world, hidden, arm)
                if hidden:
                    prompt = S1.l1_prompt_hidden(program, task, self.interface(), base_src, trials, r, R, K)
                else:
                    prompt = S.l1_prompt(program, task, self.interface(), base_src, base_dev, trials, inc, r, R, K)
                val, _ = S.call_with_retries(S.SYSTEM_L1, prompt, S.l1_schema(K), lambda o: S.parse_proposals(o, K),
                                             sdir / f"round_{r}.json")
                props, analysis = val["proposals"], val["analysis"]

            def normalise(p):
                if arm != "reconly":
                    return p
                try:
                    rec = w2b.parse_recipe(p["code"]) or {}
                except (ValueError, SyntaxError):
                    rec = {}
                return dict(p, code=w2b.with_recipe(base_src, rec) if rec else base_src)
            props = [normalise(p) for p in props]
            checks = [None] * K
            for j in range(3):
                todo = [i for i in range(K) if checks[i] is None or not checks[i]["ok"]]

                def run_check(i, j=j):
                    p = sdir / "cands" / f"r{r}c{i + 1}_v{j}.py"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(props[i]["code"])
                    return self.check(p, world, mode)
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
                if not failed or arm == "random" or j == 2:
                    break
                fl = [{"name": props[i]["name"], "error": checks[i]["error"], "code": props[i]["code"]} for i in failed]
                try:
                    fix, _ = S.call_with_retries(S.SYSTEM_L1, S.repair_prompt(prompt, fl), S.l1_schema(len(failed)),
                                                 lambda o, n=len(failed): S.parse_proposals(o, n), sdir / f"round_{r}_repair{j + 1}.json")
                except RuntimeError as e:
                    self.log(f"{sdir.relative_to(self.dir)} round {r} repair {j + 1} failed: {e}")
                    break
                for i, f in zip(failed, fix["proposals"]):
                    props[i] = normalise({"name": f["name"] or props[i]["name"], "hypothesis": f["hypothesis"], "code": f["code"]})
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

    def one_search(self, arm: str, i: int, program: str) -> dict:
        sdir = self.dir / "W" / arm / f"s{i}"
        st = self.search(sdir, W, arm, program, self.cfg["search_seed_base"] + i)
        pick = self.retest_and_pick(sdir, st, W)
        conf = self.confirm(pick["src"], self.cfg["confirm_seeds"])
        self.log(f"W/{arm}/s{i} confirmed: {[round(r['dev_exact'], 3) if r['dev_exact'] is not None else r['status'] for r in conf.values()]}")
        return {"kind": "search", "arm": arm, "search": i, "label": pick["label"], "name": pick["name"],
                "sha": pick["sha"], "mean_selected": pick["mean_selected"], "incumbent_2seed": pick["incumbent_2seed"],
                "seeds": self.cfg["confirm_seeds"], "dev": {s: r["dev_exact"] for s, r in conf.items()},
                "status": {s: r["status"] for s, r in conf.items()}, "wall_s": {s: r["wall_s"] for s, r in conf.items()}}

    def comparators(self) -> dict:
        base = BASE.read_text()
        return {"refA": REF_A.read_text(), "start_g": base, "start_frozen": w2b.with_recipe(base, w2b.FROZEN_AS_BLOCK)}

    def one_comparator(self, name: str) -> dict:
        src = self.comparators()[name]
        p = self.dir / "comparators" / name / "candidate.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
        chk = self.check(p, W)
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
        self.base = {W: self.baselines_q()}
        g0r = (S2 / "programs" / "g0r.md").read_text()
        (self.dir / "programs").mkdir(exist_ok=True)
        (self.dir / "programs" / "g0r.md").write_text(g0r)
        programs = {a: g0r for a in CLAUDE_ARMS} | {"random": None}
        order = S1.search_order(c["arms"], c["n_searches"])
        with ThreadPoolExecutor(3) as cx, ThreadPoolExecutor(c["search_workers"]) as sx:
            comp_f = [cx.submit(self.one_comparator, n) for n in self.comparators()]
            search_f = [sx.submit(self.one_search, a, i, programs[a]) for a, i in order]
            entries = [f.result() for f in search_f] + [f.result() for f in comp_f]
        (self.dir / "confirm" / "summary.json").write_text(json.dumps(entries, indent=1))
        self.verify()
        self.log("STAGE2B_DONE: run review_2b.py, then analyze_2b.py. W's final split was not opened.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    args = ap.parse_args()
    Stage2b((ROOT / PRESETS[args.preset]["campaign"]).resolve(), args.preset).run()


if __name__ == "__main__":
    main()
