"""Stage 2a: calibrate world W (PROTOCOL_2A.md). No Claude calls; W's dev designs only.

usage (project root): caffeinate -i .venv/bin/python -I arch_rsi/stage2/2a/run_2a.py [--dry]
Resumable: a run with a result.json is not re-run. Writes runs/ (or dry/): decisions.json, summary.json, run.log.
"""
import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
S2 = HERE.parent
A = S2.parent
ROOT = A.parent
PY = str(ROOT / ".venv" / "bin" / "python")
sys.path.insert(0, str(ROOT))

from arch_rsi.orchestrate_arch import FROZEN, file_hash  # noqa: E402

spec_ = importlib.util.spec_from_file_location("analyze_stage1", A / "stage1" / "analyze_stage1.py")
AS1 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(AS1)

WW = S2 / "w_world.py"
CONF = A / "campaigns" / "stage1" / "confirm"
NETS = {"start": A / "candidates" / "baseline.py",
        "refA": CONF / "1c9c0e6b521fc0bb" / "candidate.py",   # no-feedback search 9's pick, retest 95.4%
        "refB": CONF / "722ae3f704b47af7" / "candidate.py"}   # no-feedback search 5's pick, retest 94.1%
RECIPES = {"g": {"n_envs": 256, "lr": 0.001, "n_minibatches": 32}, "f": {}}  # Stage G's recipe; frozen recipe
W_FILES = [S2 / "targets" / f for f in ("w_dev.pt", "w_heldout_hashes.pt", "w_check.pt")] + [S2 / "holdout" / "w_final.pt"]
EASY_S, EASY_R, HARD = 0.50, 0.90, 0.20
B2_RUNS, B2_COMP = 80 * 38, 24  # Stage 2b: 5 arms x 16 searches x 38 runs, plus 24 comparator runs

CFG = {"full": dict(out="runs", budgets=[262144, 524288, 1048576], fallback=131072,
                    lad_start_g=list(range(2000, 2006)), lad_start_f=[2000, 2001], lad_ref=[2000, 2001],
                    noise_start_g=list(range(2000, 2010)) + list(range(3000, 3010)),
                    noise_ref=list(range(2000, 2006)), noise_start_f=list(range(2000, 2006)), workers=4, timeout=3600),
       "dry": dict(out="dry", budgets=[8192, 16384], fallback=4096,
                   lad_start_g=[2000, 2001], lad_start_f=[2000], lad_ref=[2000],
                   noise_start_g=[2000, 2001, 3000, 3001], noise_ref=[2000, 2001], noise_start_f=[2000, 2001],
                   workers=4, timeout=900, force_noise=True)}
C: dict = {}
OUT = HERE
MANIFEST: dict = {}
LOCK = threading.Lock()


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z {msg}"
    with LOCK:
        print(line, flush=True)
        with open(OUT / "run.log", "a") as f:
            f.write(line + "\n")


def hidden_mask():
    v = torch.load(S2 / "targets" / "w_dev.pt").bool()
    above = (v.flip(1).cumsum(1).flip(1) - v.int()) > 0
    return ((~v) & above).flatten(1).any(1).numpy()


H = hidden_mask()


def hashes():
    files = list(FROZEN) + [A / "gpu" / "train_gpu.py", WW, S2 / "make_targets_w.py", HERE / "PROTOCOL_2A.md",
                            Path(__file__).resolve()] + W_FILES + list(NETS.values())
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def verify():
    now = hashes()
    bad = [k for k, v in MANIFEST["hashes"].items() if now.get(k) != v]
    if bad:
        (OUT / "STOPPED").write_text(f"INTEGRITY: changed {bad}")
        log(f"INTEGRITY: changed {bad}")
        raise SystemExit(f"INTEGRITY: changed {bad}")


def run(cell):
    net, rk, seed, budget = cell
    d = OUT / f"b{budget}" / f"{net}_{rk}_s{seed}"
    rf = d / "result.json"
    if rf.exists():
        return json.loads(rf.read_text())
    verify()
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"candidate": str(NETS[net]), "world": "w", "seed": seed, "budget_steps": budget,
                                            "threads": 1, "device": "mps", "recipe": RECIPES[rk]}))
    t0, rc, err = time.time(), None, None
    try:
        p = subprocess.run([PY, "-I", str(WW), "train", "--spec", str(d / "run.json"), "--out", str(d / "run")],
                           capture_output=True, text=True, timeout=C["timeout"])
        rc, err = p.returncode, p.stderr[-500:]
    except subprocess.TimeoutExpired:
        rc = "timeout"
    lf = d / "run" / "train_log.json"
    tl = json.loads(lf.read_text()) if (lf.exists() and rc != "timeout") else {}
    status = "timeout" if rc == "timeout" else tl.get("status", f"crashed rc={rc}")
    res = {"net": net, "recipe": rk, "seed": seed, "budget": budget, "status": status,
           "error": None if status == "ok" else (tl.get("error") or err), "steps": tl.get("steps"),
           "flops_ratio": tl.get("flops_ratio"), "train_wall_s": tl.get("wall_s"), "dev_exact": None, "hidden": None,
           "learned": False}
    if status == "ok":
        subprocess.run([PY, "-I", str(WW), "eval", "--trial", str(d / "run"), "--split", "dev", "--out", str(d / "dev.json")],
                       capture_output=True, text=True, timeout=C["timeout"])
        if (d / "dev.json").exists():
            pe = np.array(json.loads((d / "dev.json").read_text())["per_design"]["exact"], bool)
            res |= {"dev_exact": float(pe.mean()), "hidden": float(pe[H].mean())}
            res["learned"] = res["hidden"] >= AS1.LEARNED
        else:
            res |= {"status": "eval_failed", "error": "no dev.json"}
    res["wall_s"] = round(time.time() - t0, 1)
    rf.write_text(json.dumps(res))
    log(f"b{budget} {net}_{rk}_s{seed}: {res['status']} steps {res['steps']} wall {res['wall_s']}s"
        + (f" dev {res['dev_exact']:.4f}" if res["dev_exact"] is not None else f" ERROR {str(res['error'])[:200]}"))
    return res


def score(r):
    return r["dev_exact"] if (r["status"] == "ok" and r["dev_exact"] is not None) else 0.0


def run_all(cells):
    with ThreadPoolExecutor(C["workers"]) as ex:
        outs = list(ex.map(run, cells))
    return dict(zip(cells, outs))


def group(res, net, rk, seeds, budget):
    rr = [res[(net, rk, s, budget)] for s in seeds]
    sc = [score(r) for r in rr]
    return {"per_seed": [round(v, 4) for v in sc], "mean": round(AS1.mean(sc), 4),
            "sd": round(statistics.stdev(sc), 4) if len(sc) > 1 else None,
            "learned": sum(r["learned"] for r in rr), "n": len(rr), "failed": sum(r["status"] != "ok" for r in rr),
            "median_wall_s": statistics.median(r["wall_s"] for r in rr)}


def ladder_cells(b):
    return ([("start", "g", s, b) for s in C["lad_start_g"]] + [("start", "f", s, b) for s in C["lad_start_f"]]
            + [(n, "g", s, b) for n in ("refA", "refB") for s in C["lad_ref"]])


def read_budget(res, b):
    g = {"start_g": group(res, "start", "g", C["lad_start_g"], b), "start_f": group(res, "start", "f", C["lad_start_f"], b),
         "refA": group(res, "refA", "g", C["lad_ref"], b), "refB": group(res, "refB", "g", C["lad_ref"], b)}
    S, R = g["start_g"]["mean"], max(g["refA"]["mean"], g["refB"]["mean"])
    reading = "too easy" if (S > EASY_S or R >= EASY_R) else ("too hard" if max(S, R) < HARD else "passes")
    return {"budget": b, "S": S, "R": R, "reading": reading, "cells": g}


def main():
    global C, OUT, MANIFEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    C = CFG["dry" if a.dry else "full"]
    OUT = HERE / C["out"]
    OUT.mkdir(parents=True, exist_ok=True)
    mf = OUT / "manifest.json"
    if mf.exists():
        MANIFEST = json.loads(mf.read_text())
    else:
        MANIFEST = {"started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "mode": "dry" if a.dry else "full",
                    "config": C, "recipes": RECIPES, "nets": {k: str(v.relative_to(ROOT)) for k, v in NETS.items()},
                    "hashes": hashes()}
        mf.write_text(json.dumps(MANIFEST, indent=1))
    verify()
    log(f"stage 2a {'DRY ' if a.dry else ''}start; protocol sha {MANIFEST['hashes']['arch_rsi/stage2/2a/PROTOCOL_2A.md'][:16]}")

    res, ladder = {}, []
    for b in C["budgets"]:
        res |= run_all(ladder_cells(b))
        rd = read_budget(res, b)
        ladder.append(rd)
        log(f"ladder b{b}: start (Stage G recipe) {rd['S']:.4f}, better reference {rd['R']:.4f} -> {rd['reading']}")
        if rd["reading"] == "too easy":
            break
    passing = [r for r in ladder if r["reading"] == "passes"]
    status = "chosen"
    if not passing and ladder[0]["reading"] == "too easy":
        b = C["fallback"]
        res |= run_all(ladder_cells(b))
        rd = read_budget(res, b)
        ladder.append(rd)
        log(f"ladder fallback b{b}: start {rd['S']:.4f}, better reference {rd['R']:.4f} -> {rd['reading']}")
        passing = [rd] if rd["reading"] == "passes" else []
        status = "chosen" if passing else "W too easy"
    elif not passing:
        status = "W too hard" if all(r["reading"] == "too hard" for r in ladder) else "no budget passes"
    if not passing and C.get("force_noise"):  # dry run only: exercise the noise step on the last ladder budget
        passing, status = [ladder[-1]], "dry: forced"
    decisions = {"ladder": ladder, "status": status, "chosen_budget": max(r["budget"] for r in passing) if passing else None}
    (OUT / "decisions.json").write_text(json.dumps(decisions, indent=1))
    if not passing:
        log(f"no budget passes: {status}. Back to the user; STAGE2A_DONE")
        return

    B = decisions["chosen_budget"]
    log(f"chosen budget {B}; noise step")
    t0 = time.time()
    cells = ([("start", "g", s, B) for s in C["noise_start_g"]] + [(n, "g", s, B) for n in ("refA", "refB") for s in C["noise_ref"]]
             + [("start", "f", s, B) for s in C["noise_start_f"]])
    res |= run_all(cells)
    sg = group(res, "start", "g", C["noise_start_g"], B)
    half = len(C["noise_start_g"]) // 2
    r1, r2 = sg["per_seed"][:half], sg["per_seed"][half:]
    noise = {"budget": B, "start_g": sg, "start_g_range1": {"seeds": C["noise_start_g"][:half], "mean": round(AS1.mean(r1), 4),
                                                            "sd": round(statistics.stdev(r1), 4)},
             "start_g_range2": {"seeds": C["noise_start_g"][half:], "mean": round(AS1.mean(r2), 4), "sd": round(statistics.stdev(r2), 4)},
             "ranges_perm_p": round(AS1.perm_p(r1, r2), 4),
             "refA": group(res, "refA", "g", C["noise_ref"], B), "refB": group(res, "refB", "g", C["noise_ref"], B),
             "start_f": group(res, "start", "f", C["noise_start_f"], B), "headroom": round(1 - sg["mean"], 4)}
    ref_walls = [res[(n, "g", s, B)]["wall_s"] for n in ("refA", "refB") for s in C["noise_ref"]]
    start_walls = [res[("start", "g", s, B)]["wall_s"] for s in C["noise_start_g"]]
    t_ref, t_start = AS1.mean(ref_walls) / C["workers"], AS1.mean(start_walls) / C["workers"]
    proj_h = (B2_RUNS * t_ref + B2_COMP * t_start) / 3600
    noise["projection_2b"] = {"runs": B2_RUNS + B2_COMP, "eff_s_per_run_ref": round(t_ref, 1), "eff_s_per_run_start": round(t_start, 1),
                              "hours": round(proj_h, 1), "over_48h": proj_h > 48}
    noise["noise_step_wall_s"] = round(time.time() - t0, 1)
    summary = {"decisions": decisions, "noise": noise}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"noise: start (Stage G recipe) {sg['mean']:.4f} (SD {sg['sd']}), ranges {noise['start_g_range1']['mean']:.4f} vs "
        f"{noise['start_g_range2']['mean']:.4f} (perm p {noise['ranges_perm_p']}); refA {noise['refA']['mean']:.4f}, "
        f"refB {noise['refB']['mean']:.4f}; start (frozen recipe) {noise['start_f']['mean']:.4f}")
    log(f"projected Stage 2b machine time: {proj_h:.1f} h{' (OVER 48 h: back to the user)' if proj_h > 48 else ''}")
    log("STAGE2A_DONE")


if __name__ == "__main__":
    main()
