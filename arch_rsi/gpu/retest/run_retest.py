"""Retest Stage 1's picks under Stage G's recipe (PROTOCOL_R.md). No Claude calls; dev designs only.

usage (project root): caffeinate -i .venv/bin/python -I arch_rsi/gpu/retest/run_retest.py [--dry]
Resumable: a run with a result.json is not re-run. Writes runs/results.json (or dry/results.json).
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

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
A = ROOT / "arch_rsi"
PY = str(ROOT / ".venv" / "bin" / "python")
sys.path.insert(0, str(ROOT))

from arch_rsi.orchestrate_arch import FROZEN, file_hash  # noqa: E402
from arch_rsi.sandbox import sha  # noqa: E402

spec_ = importlib.util.spec_from_file_location("analyze_stage1", A / "stage1" / "analyze_stage1.py")
AS1 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(AS1)

S1 = A / "campaigns" / "stage1"
RES1 = json.loads((S1 / "results.json").read_text())
CONF = S1 / "confirm"
TRAIN_GPU, EVAL = A / "gpu" / "train_gpu.py", A / "evaluate_arch.py"
START_SHA = sha((S1 / "comparators" / "start" / "candidate.py").read_text())
CRITIC_SHA = sha((S1 / "comparators" / "critic_ortho" / "candidate.py").read_text())
RECIPE = {"n_envs": 256, "lr": 0.001, "n_minibatches": 32}
ARMS = ["g0", "nofb", "menu"]
DEV = A / "targets" / "xl_dev.pt"
CFG = {"full": dict(out="runs", budget=262144, pick_seeds=list(range(1001, 1009)), comp_seeds=list(range(1001, 1013)),
                    searches=list(range(1, 11)), workers=4, timeout=1800),
       "dry": dict(out="dry", budget=8192, pick_seeds=[1001], comp_seeds=[1001, 1002], searches=[1], workers=4, timeout=600)}
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
    v = torch.load(DEV).bool()
    above = (v.flip(1).cumsum(1).flip(1) - v.int()) > 0
    return ((~v) & above).flatten(1).any(1).numpy()


H = hidden_mask()


def searches(arm):
    return [s for s in RES1["per_arm"][arm]["searches"] if s["search"] in C["searches"]]


def candidate_path(h):
    p = CONF / h / "candidate.py"
    assert sha(p.read_text()) == h, h
    return p


def jobs_by_sha():
    js = {START_SHA: C["comp_seeds"], CRITIC_SHA: C["comp_seeds"]}
    for arm in ARMS:
        for s in searches(arm):
            js.setdefault(s["sha"], C["pick_seeds"])
    return js


def hashes(shas):
    h = {str(p.relative_to(ROOT)): file_hash(p) for p in FROZEN}
    for p in [TRAIN_GPU, HERE / "PROTOCOL_R.md", Path(__file__).resolve(), A / "stage1" / "analyze_stage1.py"]:
        h[str(p.relative_to(ROOT))] = file_hash(p)
    for s in shas:
        p = candidate_path(s)
        h[str(p.relative_to(ROOT))] = file_hash(p)
    return h


def verify():
    now = hashes(MANIFEST["shas"])
    bad = [k for k, v in MANIFEST["hashes"].items() if now.get(k) != v]
    if bad:
        (OUT / "STOPPED").write_text(f"INTEGRITY: changed {bad}")
        log(f"INTEGRITY: changed {bad}")
        raise SystemExit(f"INTEGRITY: changed {bad}")


def run(h, seed):
    d = OUT / h / f"seed{seed}"
    rf = d / "result.json"
    if rf.exists():
        return json.loads(rf.read_text())
    verify()
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"candidate": str(candidate_path(h)), "world": "xl", "seed": seed,
                                            "budget_steps": C["budget"], "threads": 1, "device": "mps", "recipe": RECIPE}))
    t0, rc, err = time.time(), None, None
    try:
        p = subprocess.run([PY, "-I", str(TRAIN_GPU), "--spec", str(d / "run.json"), "--out", str(d / "run")],
                           capture_output=True, text=True, timeout=C["timeout"])
        rc, err = p.returncode, p.stderr[-500:]
    except subprocess.TimeoutExpired:
        rc = "timeout"
    lf = d / "run" / "train_log.json"
    tl = json.loads(lf.read_text()) if (lf.exists() and rc != "timeout") else {}
    status = "timeout" if rc == "timeout" else tl.get("status", f"crashed rc={rc}")
    res = {"sha": h, "seed": seed, "status": status, "error": None if status == "ok" else (tl.get("error") or err),
           "steps": tl.get("steps"), "flops_ratio": tl.get("flops_ratio"), "wall_s": round(time.time() - t0, 1),
           "dev_exact": None, "hidden": None, "learned": False}
    if status == "ok":
        subprocess.run([PY, "-I", str(EVAL), "--trial", str(d / "run"), "--split", "dev", "--out", str(d / "dev.json")],
                       capture_output=True, text=True, timeout=1800)
        if (d / "dev.json").exists():
            pe = np.array(json.loads((d / "dev.json").read_text())["per_design"]["exact"], bool)
            res |= {"dev_exact": float(pe.mean()), "hidden": float(pe[H].mean())}
            res["learned"] = res["hidden"] >= AS1.LEARNED
        else:
            res |= {"status": "eval_failed", "error": "no dev.json"}
    rf.write_text(json.dumps(res))
    log(f"{h}/seed{seed}: {res['status']} steps {res['steps']} wall {res['wall_s']}s"
        + (f" dev {res['dev_exact']:.4f}" if res["dev_exact"] is not None else f" ERROR {str(res['error'])[:200]}"))
    return res


def score(r):
    return r["dev_exact"] if (r["status"] == "ok" and r["dev_exact"] is not None) else 0.0


def spearman(x, y):
    def ranks(v):
        v = np.asarray(v, float)
        order = v.argsort()
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for val in np.unique(v):  # average ranks for ties
            m = v == val
            r[m] = r[m].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse(res):
    def seeds_of(h, seeds):
        return [res[(h, s)] for s in seeds]

    groups, detail, old_new = {}, {}, []
    for arm in ARMS:
        rows = []
        for s in searches(arm):
            rr = seeds_of(s["sha"], C["pick_seeds"])
            sc = [score(r) for r in rr]
            rows.append({"search": s["search"], "pick": s["pick"], "name": s["name"], "sha": s["sha"],
                         "score": round(AS1.mean(sc), 4), "per_seed": [round(v, 4) for v in sc],
                         "stage1_score": s["primary_score"], "seeds_learned": sum(r["learned"] for r in rr),
                         "failed": sum(r["status"] != "ok" for r in rr)})
        groups[arm] = [r["score"] for r in rows]
        detail[arm] = {"mean": round(AS1.mean(groups[arm]), 4), "stage1_mean": round(AS1.mean([r["stage1_score"] for r in rows]), 4),
                       "sd_between_searches": round(statistics.stdev(groups[arm]), 4) if len(rows) > 1 else None,
                       "seeds_learned": sum(r["seeds_learned"] for r in rows), "seeds": len(rows) * len(C["pick_seeds"]),
                       "searches": rows}
    for name, h in (("start", START_SHA), ("critic_ortho", CRITIC_SHA)):
        rr = seeds_of(h, C["comp_seeds"])
        groups[name] = [score(r) for r in rr]
        detail[name] = {"mean": round(AS1.mean(groups[name]), 4), "per_seed": [round(v, 4) for v in groups[name]],
                        "stage1_mean": RES1["per_arm"][name]["mean"], "seeds_learned": sum(r["learned"] for r in rr),
                        "seeds": len(rr), "failed": sum(r["status"] != "ok" for r in rr)}
    prim = AS1.family(groups, [("g0", "start"), ("nofb", "start")])
    sec = AS1.family(groups, [("g0", "nofb"), ("menu", "start"), ("critic_ortho", "start"), ("g0", "menu"), ("nofb", "menu")])
    passes = {k: v["diff"] > 0 and v["p_holm"] < 0.05 for k, v in prim.items()}
    reading = "lead survives" if all(passes.values()) else ("lead gone" if not any(passes.values()) else "mixed")
    leads = {arm: {"stage1": round(detail[arm]["stage1_mean"] - detail["start"]["stage1_mean"], 4),
                   "retest": round(detail[arm]["mean"] - detail["start"]["mean"], 4)} for arm in ARMS}
    distinct = {}
    for arm in ARMS:
        for r in detail[arm]["searches"]:
            if r["sha"] != START_SHA:
                distinct[r["sha"]] = (r["stage1_score"], r["score"])
    rho = spearman([v[0] for v in distinct.values()], [v[1] for v in distinct.values()]) if len(distinct) > 2 else None
    return {"recipe": RECIPE, "budget": C["budget"], "world": "xl", "split": "dev", "per_arm": detail,
            "primary_family": {"contrasts": prim, "passes": passes, "reading": reading},
            "secondary_family": sec, "leads_over_start": leads,
            "rank_agreement": {"spearman": None if rho is None else round(rho, 4), "n_distinct_picks": len(distinct)}}


def main():
    global C, OUT, MANIFEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    C = CFG["dry" if a.dry else "full"]
    OUT = HERE / C["out"]
    OUT.mkdir(parents=True, exist_ok=True)
    js = jobs_by_sha()
    mf = OUT / "manifest.json"
    if mf.exists():
        MANIFEST = json.loads(mf.read_text())
    else:
        MANIFEST = {"started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "mode": "dry" if a.dry else "full",
                    "config": C, "recipe": RECIPE, "shas": sorted(js), "hashes": hashes(sorted(js))}
        mf.write_text(json.dumps(MANIFEST, indent=1))
    verify()
    jobs = [(h, s) for h, seeds in js.items() for s in seeds]
    log(f"retest {'DRY ' if a.dry else ''}start: {len(js)} networks, {len(jobs)} runs; protocol sha "
        f"{MANIFEST['hashes']['arch_rsi/gpu/retest/PROTOCOL_R.md'][:16]}")
    with ThreadPoolExecutor(C["workers"]) as ex:
        outs = list(ex.map(lambda j: run(*j), jobs))
    res = {(j[0], j[1]): o for j, o in zip(jobs, outs)}
    out = analyse(res)
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    p = out["primary_family"]
    log("primary: " + "; ".join(f"{k} {v['diff']:+.4f} [{v['ci95_boot'][0]:+.4f}, {v['ci95_boot'][1]:+.4f}] p_holm {v['p_holm']}"
                                for k, v in p["contrasts"].items()) + f" -> {p['reading']}")
    log("means: " + ", ".join(f"{k} {v['mean']:.4f}" for k, v in out["per_arm"].items()))
    log(f"rank agreement (Spearman, {out['rank_agreement']['n_distinct_picks']} picks): {out['rank_agreement']['spearman']}")
    log("RETEST_DONE")


if __name__ == "__main__":
    main()
