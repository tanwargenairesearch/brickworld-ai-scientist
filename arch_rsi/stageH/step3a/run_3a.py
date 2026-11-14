"""Step 3a runner (PROTOCOL_STEP3A.md). Resumable: finished runs (runs/b*/CELL_sSEED/result.json), checks and the
passed tests are kept.

Phases:
  guard      Step 2's full run must be over (its results.json exists) and no Step 2 or Step 2S runner may be alive:
             they share the GPU, and Step 2's trials have a 900 s limit
  preflight  Stage 2b's 39 frozen hashes, Stage H's 26 files, Step 2's and Step 2S's manifests and every file in
             step3a/manifest.json must be unchanged, or nothing runs
  tests      test_rules_3a.py and test_identity_3a.py on the GPU; a failure stops the run
  checks     check_3a.py for each fix in its rule world, on the GPU
  main       150 runs at 131,072 steps (25 cells x seeds 4000-4005), four at once on the GPU; each is train_3a.py
             (900 s limit) then eval_3a.py on hplain_dev. A failed or timed-out run is rerun once on the same seed; a run
             cut short by the runner itself stopping is redone, not counted as a failure.
  fallback   only rules with damage >= 25 points that fail pass rule 1: their three cells and plain_A at 262,144 steps
  analysis   analyze_3a.py -> runs/results.json
  postflight the same hash checks again, recorded in results.json
usage (project root): nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/step3a/run_3a.py
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common  # noqa: E402

import analyze_3a as A  # noqa: E402
import rules3a as R3  # noqa: E402
from make_fixes_3a import FIXES  # noqa: E402

PY = [sys.executable, "-I", "-B", "-W", "ignore"]
RUNS = A.RUNS
PAR, TRAIN_TIMEOUT, EVAL_TIMEOUT = 4, 900, 600
LOG = RUNS / "progress.log"
STAGEH = HERE.parent
STEP2_RESULTS = STAGEH / "step2" / "campaigns" / "step2" / "results.json"   # only its existence is checked
MANIFESTS = {"stageH_files": STAGEH / "manifest.json", "step2_files": STAGEH / "step2" / "manifest.json",
             "step2s_files": STAGEH / "step2s" / "manifest.json", "step3a_files": HERE / "manifest.json"}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gpu_free() -> dict:
    """Step 2's full run is over, and no Step 2 or Step 2S runner process is alive."""
    p = subprocess.run(["pgrep", "-fl", r"python.* arch_rsi/stageH/step2s?/run_step2s?\.py"], capture_output=True,
                       text=True)   # the runners themselves, not shells that merely mention them
    alive = [ln for ln in p.stdout.splitlines() if ln.strip()]
    return {"step2_results_exist": STEP2_RESULTS.exists(), "step2_runners_alive": alive,
            "ok": STEP2_RESULTS.exists() and not alive}


def hash_checks() -> dict:
    m2b = json.loads((common.ROOT / "arch_rsi" / "campaigns" / "stage2b" / "manifest.json").read_text())["hashes"]
    out = {"stage2b_frozen": len(m2b), "stage2b_changed": [p for p, h in m2b.items() if sha_file(common.ROOT / p) != h]}
    ok = not out["stage2b_changed"]
    for k, mp in MANIFESTS.items():
        if not mp.exists():
            out[k], out[f"{k}_changed"], ok = 0, [f"{mp.relative_to(common.ROOT)} is missing"], False
            continue
        files = json.loads(mp.read_text())["files"]
        bad = [p for p, h in files.items() if not (common.ROOT / p).exists() or sha_file(common.ROOT / p) != h]
        out[k], out[f"{k}_changed"] = len(files), bad
        ok = ok and not bad
    out["ok"] = ok
    return out


def candidate(cand: str) -> Path:
    return common.REF_A if cand == "A" else FIXES[cand]


def run_tests():
    d = RUNS / "tests"
    d.mkdir(parents=True, exist_ok=True)
    for name, args in (("test_rules_3a", ["--device", "mps"]), ("test_identity_3a", [])):
        mark = d / f"{name}.passed"
        if mark.exists():
            continue
        t0 = time.time()
        p = subprocess.run(PY + [str(HERE / f"{name}.py")] + args, capture_output=True, text=True, cwd=common.ROOT,
                           timeout=3600)
        (d / f"{name}.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr[-4000:])
        log(f"{name}: exit {p.returncode} ({time.time() - t0:.0f} s); last line: {p.stdout.strip().splitlines()[-1:] }")
        if p.returncode != 0:
            raise SystemExit(f"{name} failed; see {d / (name + '.log')}. Nothing else was run.")
        mark.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def run_checks():
    (RUNS / "checks").mkdir(parents=True, exist_ok=True)
    for r in A.RULE_NAMES:
        out = RUNS / "checks" / f"{r}.json"
        if out.exists():
            continue
        t0 = time.time()
        p = subprocess.run(PY + [str(HERE / "check_3a.py"), "--candidate", str(FIXES[r]), "--rule", r,
                                 "--out", str(out)], capture_output=True, text=True, cwd=common.ROOT, timeout=1800)
        if not out.exists():
            out.write_text(json.dumps({"ok": False, "rule": r, "error": f"check crashed: {(p.stderr or '')[-400:]}"}))
        c = json.loads(out.read_text())
        log(f"check {R3.IDS[r]} {r}: ok={c['ok']} untrained={c.get('check_untrained')} "
            f"short={c.get('check_short_training')} error={c.get('error')} ({time.time() - t0:.1f} s)")


def attempt(d: Path, cell, rule, cand, seed, budget) -> dict:
    """One training + evaluation attempt in directory d -> attempt record."""
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    spec = {"candidate": str(candidate(cand)), "world": common.WORLD, "rule": rule, "seed": seed,
            "budget_steps": budget, "device": "mps", "threads": 1}
    (d / "run.json").write_text(json.dumps(spec))
    t0 = time.time()
    rec = {"status": "ok", "error": None}
    try:
        p = subprocess.run(PY + [str(HERE / "train_3a.py"), "--spec", str(d / "run.json"), "--out", str(d / "run")],
                           capture_output=True, text=True, cwd=common.ROOT, timeout=TRAIN_TIMEOUT)
        lp = d / "run" / "train_log.json"
        tl = json.loads(lp.read_text()) if lp.exists() else {"status": "crashed", "error": (p.stderr or "")[-600:]}
    except subprocess.TimeoutExpired:
        tl = {"status": "timeout", "error": f"training exceeded {TRAIN_TIMEOUT} s"}
    rec.update(train_status=tl["status"], train_wall_s=tl.get("wall_s"), steps=tl.get("steps"),
               n_updates=tl.get("n_updates"), flops_ratio=tl.get("flops_ratio"), train_exact_tail=tl.get("train_exact_tail"),
               designs_rejected_heldout=tl.get("designs_rejected_heldout"))
    if tl["status"] != "ok":
        rec.update(status=tl["status"], error=tl.get("error"))
    else:
        try:
            p = subprocess.run(PY + [str(HERE / "eval_3a.py"), "--trial", str(d / "run"), "--rule", rule,
                                     "--designs", f"{R3.DESIGNS[rule]}_dev", "--out", str(d / "dev.json")],
                               capture_output=True, text=True, cwd=common.ROOT, timeout=EVAL_TIMEOUT)
            if p.returncode != 0 or not (d / "dev.json").exists():
                rec.update(status="eval_failed", error=(p.stderr or "")[-600:])
        except subprocess.TimeoutExpired:
            rec.update(status="eval_failed", error=f"evaluation exceeded {EVAL_TIMEOUT} s")
    if rec["status"] == "ok":
        ev = json.loads((d / "dev.json").read_text())
        rec.update(dev_exact=ev["exact"], iou=ev["iou"], pressed_done=ev["pressed_done"], exact_hidden=ev["exact_hidden"],
                   n_hidden=ev["n_hidden"], wrong_cells_yx=ev["wrong_cells_yx"], split=ev["split"])
    rec["wall_s"] = round(time.time() - t0, 2)
    (d / "attempt.json").write_text(json.dumps(rec))
    return rec


def run_one(cell, rule, cand, seed, budget) -> dict:
    base = RUNS / f"b{budget}" / f"{cell}_s{seed}"
    rp = base / "result.json"
    if rp.exists():
        return json.loads(rp.read_text())
    tries = []
    for n in (1, 2):
        d = base / f"try{n}"
        if (d / "attempt.json").exists():
            rec = json.loads((d / "attempt.json").read_text())   # a finished attempt from before a restart
        else:
            rec = attempt(d, cell, rule, cand, seed, budget)
        tries.append(rec)
        if rec["status"] == "ok":
            break
    rec = dict(tries[-1])
    rec.update(cell=cell, rule=rule, cand=cand, seed=seed, budget=budget, attempts=len(tries), dir=f"try{len(tries)}",
               earlier_attempts=[{"status": t["status"], "error": t["error"]} for t in tries[:-1]])
    rp.write_text(json.dumps(rec))
    return rec


def run_batch(jobs, label):
    done, t0 = 0, time.time()
    log(f"{label}: {len(jobs)} runs, {PAR} at once")
    with ThreadPoolExecutor(PAR) as ex:
        futs = {ex.submit(run_one, *j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            done += 1
            try:
                r = f.result()
                log(f"{label} {done}/{len(jobs)} {j[0]} s{j[3]}: {r['status']}"
                    + (f" exact {r['dev_exact']:.4f}" if r["status"] == "ok" else f" ({str(r['error'])[:120]})")
                    + f", {r['attempts']} attempt(s), {r['wall_s']} s")
            except Exception as e:  # noqa: BLE001
                log(f"{label} {done}/{len(jobs)} {j[0]} s{j[3]}: runner error {type(e).__name__}: {e}")
    log(f"{label}: finished in {(time.time() - t0) / 60:.1f} min")


def main_jobs() -> list:
    return [(c, r, k, s, A.BUDGET) for c, r, k in A.all_cells() for s in A.SEEDS]


def fallback_jobs(fb) -> list:
    keep = {v for r in fb for v in A.cells_for(r).values()} - {"plain_A"}
    cells = [("plain_A", "plain", "A")] + [x for x in A.all_cells() if x[0] in keep]
    return [(c, r, k, s, A.FALLBACK) for c, r, k in cells for s in A.SEEDS]


def main():
    RUNS.mkdir(exist_ok=True)
    log("Step 3a runner start")
    g = gpu_free()
    log(f"guard: {g}")
    if not g["ok"]:
        raise SystemExit("guard failed: Step 2's full run isn't over or a Step 2/2S runner is alive; nothing was run")
    pre = hash_checks()
    log(f"preflight: {pre}")
    if not pre["ok"]:
        raise SystemExit("preflight failed: a frozen or hashed file changed; nothing was run")
    t_start = time.time()
    run_tests()
    run_checks()
    run_batch(main_jobs(), "main")
    ev = A.evaluate(A.load_results(RUNS), A.load_checks(RUNS))
    fb = [r for r in A.RULE_NAMES if ev["rules"][r]["fallback_trigger"]]
    log(f"fallback: {fb or 'none'}")
    if fb:
        run_batch(fallback_jobs(fb), "fallback")
    res = A.analyze(RUNS, tuple(fb))
    post = hash_checks()
    res.update(status="final", guard=g, preflight=pre, postflight=post,
               total_wall_min=round((time.time() - t_start) / 60, 1),
               written_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    (RUNS / "results.json").write_text(json.dumps(res, indent=1))
    log(f"postflight: {post}")
    e = res["evaluation"]
    for r in A.RULE_NAMES:
        d = e["rules"][r]
        log(f"{d['id']} {r}: means {d['means']} repair {d['repair']} p_holm {d['p_holm']:.4g} interaction "
            f"{d['interaction']} legal {d['legal']} complete {d['complete']} -> {'PASS' if d['passes'] else 'no'}")
    log(f"families: { {f: v['passing'] for f, v in e['families'].items()} }")
    log(f"decision: {e['decision']} (usable families: {e['usable_families']}); results in {RUNS / 'results.json'}")


if __name__ == "__main__":
    main()
