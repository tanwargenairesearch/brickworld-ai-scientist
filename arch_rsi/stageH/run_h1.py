"""Stage H1 runner (PROTOCOL_H1.md). Resumable: finished runs (runs/b*/CELL_sSEED/result.json) and checks are kept.

Phases:
  preflight  Stage 2b's 39 frozen hashes and every file in stageH/manifest.json must be unchanged, or nothing runs
  checks     check_h.py for each fix in its rule world (shift/F1, mirror/F2, torus/F3, late/F4), on the GPU
  main       78 runs at 131,072 steps (13 cells x seeds 4000-4005), four at once on the GPU; each is train_h.py
             (900 s limit) then eval_h.py on the rule's dev designs. A failed or timed-out run is rerun once on the
             same seed; a run cut short by the runner itself stopping is redone, not counted as a failure.
  fallback   only rules with damage >= 25 points that fail pass rule 1: their three cells and plain_A at 262,144 steps
  analysis   analyze_h1.py -> runs/results.json
  postflight the same hash checks again, recorded in results.json
usage (project root): nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/run_h1.py
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
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import analyze_h1 as A  # noqa: E402

PY = [sys.executable, "-I", "-B", "-W", "ignore"]
RUNS = A.RUNS
PAR, TRAIN_TIMEOUT, EVAL_TIMEOUT = 4, 900, 600
LOG = RUNS / "progress.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hash_checks() -> dict:
    m2b = json.loads((common.ROOT / "arch_rsi" / "campaigns" / "stage2b" / "manifest.json").read_text())["hashes"]
    bad2b = [p for p, h in m2b.items() if sha_file(common.ROOT / p) != h]
    mh = json.loads((HERE / "manifest.json").read_text())["files"]
    badh = [p for p, h in mh.items() if not (common.ROOT / p).exists() or sha_file(common.ROOT / p) != h]
    return {"stage2b_frozen": len(m2b), "stage2b_changed": bad2b, "stageH_files": len(mh), "stageH_changed": badh,
            "ok": not bad2b and not badh}


def candidate(cand: str) -> Path:
    return common.REF_A if cand == "A" else common.FIXES[cand]


def run_checks():
    (RUNS / "checks").mkdir(parents=True, exist_ok=True)
    for r in A.RULE_NAMES:
        out = RUNS / "checks" / f"{r}.json"
        if out.exists():
            continue
        t0 = time.time()
        p = subprocess.run(PY + [str(HERE / "check_h.py"), "--candidate", str(common.FIXES[r]), "--rule", r,
                                 "--out", str(out)], capture_output=True, text=True, cwd=common.ROOT, timeout=1800)
        if not out.exists():
            out.write_text(json.dumps({"ok": False, "rule": r, "error": f"check crashed: {(p.stderr or '')[-400:]}"}))
        c = json.loads(out.read_text())
        log(f"check {r}: ok={c['ok']} untrained={c.get('check_untrained')} short={c.get('check_short_training')} "
            f"error={c.get('error')} ({time.time() - t0:.1f} s)")


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
        p = subprocess.run(PY + [str(HERE / "train_h.py"), "--spec", str(d / "run.json"), "--out", str(d / "run")],
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
            p = subprocess.run(PY + [str(HERE / "eval_h.py"), "--trial", str(d / "run"), "--rule", rule,
                                     "--designs", f"{common.DESIGNS[rule]}_dev", "--out", str(d / "dev.json")],
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


def main():
    RUNS.mkdir(exist_ok=True)
    log("Stage H1 runner start")
    pre = hash_checks()
    log(f"preflight: {pre}")
    if not pre["ok"]:
        raise SystemExit("preflight failed: a frozen or Stage H file changed; nothing was run")
    t_start = time.time()
    run_checks()
    main_jobs = [(c, r, k, s, A.BUDGET) for c, r, k in A.all_cells() for s in A.SEEDS]
    run_batch(main_jobs, "main")
    ev = A.evaluate(A.load_results(RUNS), A.load_checks(RUNS))
    fb = [r for r in A.RULE_NAMES if ev["rules"][r]["fallback_trigger"]]
    log(f"fallback: {fb or 'none'}")
    if fb:
        cells = [("plain_A", "plain", "A")] + [x for x in A.all_cells() if x[0] in
                                                 {v for r in fb for v in A.cells_for(r).values()} - {"plain_A"}]
        run_batch([(c, r, k, s, A.FALLBACK) for c, r, k in cells for s in A.SEEDS], "fallback")
    res = A.analyze(RUNS, tuple(fb))
    post = hash_checks()
    res.update(status="final", preflight=pre, postflight=post, total_wall_min=round((time.time() - t_start) / 60, 1),
               written_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    (RUNS / "results.json").write_text(json.dumps(res, indent=1))
    log(f"postflight: {post}")
    e = res["evaluation"]
    for r in A.RULE_NAMES:
        d = e["rules"][r]
        log(f"{r}: means {d['means']} repair {d['repair']} p_holm {d['p_holm']:.4g} interaction {d['interaction']} "
            f"legal {d['legal']} complete {d['complete']} -> {'PASS' if d['passes'] else 'no'}")
    log(f"decision: {e['decision']} (passing: {e['passing']}); results in {RUNS / 'results.json'}")


if __name__ == "__main__":
    main()
