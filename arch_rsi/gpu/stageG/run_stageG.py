"""Stage G: calibrate the GPU recipe for Stage 2 (route B of ../GPU_PLAN.md). Pilot, disclosed; no Claude calls; dev only.
The rules are in PROTOCOL_G.md, whose hash manifest.json records before the first run.

usage (project root): caffeinate -i .venv/bin/python -I arch_rsi/gpu/stageG/run_stageG.py [--dry]
Resumable: a run with a result.json is not re-run, and a step with a decision in decisions.json is not re-decided.
"""
import argparse
import hashlib
import itertools
import json
import math
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
from arch_rsi.worlds import RECIPE  # noqa: E402

TRAIN_GPU, TRAIN_CPU, EVAL = A / "gpu" / "train_gpu.py", A / "train_arch.py", A / "evaluate_arch.py"
T0 = A / "pilot" / "stage0" / "campaign" / "Q" / "g0" / "s1" / "trials"
CANDS = {"start": A / "candidates" / "baseline.py", "r1c1": T0 / "r1c1" / "candidate.py", "r4c3": T0 / "r4c3" / "candidate.py"}
NETS = ["start", "r1c1", "r4c3"]
REFS = ["r1c1", "r4c3"]
ROLLOUT = RECIPE["rollout_len"]
STAGE1_RESULTS = A / "campaigns" / "stage1" / "results.json"
DEV = {"large": ROOT / "targets" / "large_dev.pt", "xl": A / "targets" / "xl_dev.pt"}
WORLD_NAME = {"xl": "Q", "large": "P"}

CFG = {
    "full": dict(out="runs", g1_envs=[256, 1024], g1_conc=[1, 2, 4], g1_budget=262144, g1_seed=900, cpu_runs=10,
                 cpu_budget=65536, g2_lr=[0.001, 0.002], g2_mb=[8, 32], g2_seeds=[400, 401, 402, 403], g2_updates=128,
                 cal_updates=[64, 128, 256], start_seeds=list(range(300, 306)), ref_seeds=[300, 301], ceiling=0.90,
                 tie=0.05, g5_seeds=list(range(306, 310)), g6_seeds=list(range(1001, 1013)), g6_budget=131072,
                 timeout=3600, power_sims=1500),
    "dry": dict(out="dry", g1_envs=[256], g1_conc=[1, 2], g1_budget=16384, g1_seed=900, cpu_runs=2, cpu_budget=4096,
                g2_lr=[0.001, 0.002], g2_mb=[8, 32], g2_seeds=[400], g2_updates=4, cal_updates=[2, 4],
                start_seeds=[300, 301], ref_seeds=[300], ceiling=0.90, tie=0.05, g5_seeds=[302],
                g6_seeds=[1001, 1002], g6_budget=4096, timeout=900, power_sims=40),
}
C: dict = {}
OUT = HERE
DEC: dict = {}
MANIFEST: dict = {}
LOCK = threading.Lock()


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z {msg}"
    with LOCK:
        print(line, flush=True)
        with open(OUT / "run.log", "a") as f:
            f.write(line + "\n")


def hidden_mask(p):
    v = torch.load(p).bool()
    above = (v.flip(1).cumsum(1).flip(1) - v.int()) > 0
    return ((~v) & above).flatten(1).any(1).numpy()


H = {w: hidden_mask(p) for w, p in DEV.items()}


def hashes():
    h = {str(p.relative_to(ROOT)): file_hash(p) for p in FROZEN}
    for p in [TRAIN_GPU, HERE / "PROTOCOL_G.md", Path(__file__).resolve()] + [CANDS[n] for n in REFS]:
        h[str(p.relative_to(ROOT))] = file_hash(p)
    return h


def verify():
    now = hashes()
    bad = [k for k, v in MANIFEST["hashes"].items() if now.get(k) != v]
    if bad:
        (OUT / "STOPPED").write_text(f"INTEGRITY: changed {bad}")
        log(f"INTEGRITY: changed {bad}")
        raise SystemExit(f"INTEGRITY: changed {bad}")


def save_dec():
    (OUT / "decisions.json").write_text(json.dumps(DEC, indent=1))


def weights_sha(p):
    sd = torch.load(p, map_location="cpu")["state_dict"]
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def run(d, net, world, seed, budget, recipe=None, trainer="gpu", evaluate=True):
    rf = d / "result.json"
    if rf.exists():
        return json.loads(rf.read_text())
    verify()
    d.mkdir(parents=True, exist_ok=True)
    spec = {"candidate": str(CANDS[net]), "world": world, "seed": seed, "budget_steps": budget, "threads": 1}
    if trainer == "gpu":
        spec |= {"device": "mps", "recipe": recipe or {}}
    (d / "run.json").write_text(json.dumps(spec))
    script = TRAIN_GPU if trainer == "gpu" else TRAIN_CPU
    t0, rc, err = time.time(), None, None
    try:
        p = subprocess.run([PY, "-I", str(script), "--spec", str(d / "run.json"), "--out", str(d / "run")],
                           capture_output=True, text=True, timeout=C["timeout"])
        rc, err = p.returncode, p.stderr[-500:]
    except subprocess.TimeoutExpired:
        rc = "timeout"
    t1 = time.time()
    lf = d / "run" / "train_log.json"
    tl = json.loads(lf.read_text()) if (lf.exists() and rc != "timeout") else {}
    status = "timeout" if rc == "timeout" else tl.get("status", f"crashed rc={rc}")
    res = {"net": net, "world": world, "seed": seed, "budget": budget, "trainer": trainer,
           "recipe": tl.get("recipe", recipe), "status": status,
           "error": None if status == "ok" else (tl.get("error") or err),
           "steps": tl.get("steps"), "n_updates": tl.get("n_updates"), "flops_ratio": tl.get("flops_ratio"),
           "loop_t0": tl.get("loop_t0"), "loop_t1": tl.get("loop_t1"), "loop_sps": tl.get("loop_sps"),
           "sps": tl.get("sps"), "proc_t0": t0, "proc_t1": t1, "train_wall_s": round(t1 - t0, 1),
           "dev_exact": None, "hidden": None, "learned": None}
    if status == "ok" and evaluate:
        e = subprocess.run([PY, "-I", str(EVAL), "--trial", str(d / "run"), "--split", "dev", "--out", str(d / "dev.json")],
                           capture_output=True, text=True, timeout=1800)
        if (d / "dev.json").exists():
            pe = np.array(json.loads((d / "dev.json").read_text())["per_design"]["exact"], bool)
            res |= {"dev_exact": float(pe.mean()), "hidden": float(pe[H[world]].mean())}
            res["learned"] = res["hidden"] >= 0.10
            res["weights_sha"] = weights_sha(d / "run" / "ckpt.pt")
        else:
            res |= {"status": "eval_failed", "error": e.stderr[-500:]}
    rf.write_text(json.dumps(res))
    tag = d.relative_to(OUT)
    log(f"{tag}: {res['status']} steps {res['steps']} wall {res['train_wall_s']}s loop_sps {res['loop_sps']}"
        + (f" dev {res['dev_exact']:.4f} hidden {res['hidden']:.3f}" if res["dev_exact"] is not None else "")
        + (f" ERROR {str(res['error'])[:200]}" if res["status"] != "ok" else ""))
    return res


def pool(jobs, k):
    with ThreadPoolExecutor(max(1, k)) as ex:
        return list(ex.map(lambda j: run(**j), jobs))


def sc(r):
    return r["dev_exact"] if (r["status"] == "ok" and r["dev_exact"] is not None) else 0.0


def mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


# ---------------------------------------------------------------- G1
def g1():
    if "G1" in DEC:
        return
    log("G1 throughput: start")
    rows = []
    for N in C["g1_envs"]:
        for k in C["g1_conc"]:
            for net in NETS:
                jobs = [dict(d=OUT / "g1" / f"{net}_N{N}_k{k}_r{i}", net=net, world="xl", seed=C["g1_seed"] + i,
                             budget=C["g1_budget"], recipe={"n_envs": N}, evaluate=False) for i in range(k)]
                rs = pool(jobs, k)
                ok = all(r["status"] == "ok" for r in rs)
                total = sum(r["steps"] for r in rs) / (max(r["loop_t1"] for r in rs) - min(r["loop_t0"] for r in rs)) if ok else 0.0
                rows.append({"n_envs": N, "conc": k, "net": net, "total_sps": round(total, 1),
                             "per_run_loop_sps": [r["loop_sps"] for r in rs], "statuses": [r["status"] for r in rs]})
                log(f"G1 {net} N={N} k={k}: total {total:.0f} steps/s; per run {[r['loop_sps'] for r in rs]}")
    jobs = [dict(d=OUT / "g1" / f"cpu_start_r{i}", net="start", world="xl", seed=C["g1_seed"] + i, budget=C["cpu_budget"],
                 trainer="cpu", evaluate=False) for i in range(C["cpu_runs"])]
    rs = pool(jobs, C["cpu_runs"])
    ok = all(r["status"] == "ok" for r in rs)
    cpu_total = sum(r["steps"] for r in rs) / (max(r["proc_t1"] for r in rs) - min(r["proc_t0"] for r in rs)) if ok else 0.0
    log(f"G1 CPU, {C['cpu_runs']} runs at once, frozen recipe, timed from process start: total {cpu_total:.0f} steps/s")
    scores = {}
    for N in C["g1_envs"]:
        for k in C["g1_conc"]:
            v = [r["total_sps"] for r in rows if r["n_envs"] == N and r["conc"] == k]
            scores[(N, k)] = math.exp(mean([math.log(x) for x in v])) if all(x > 0 for x in v) else 0.0
    best = max(scores.values())
    if best <= 0:
        log("G1: no setting finished on all three networks; stopping")
        raise SystemExit("G1 failed")
    eligible = sorted(s for s, v in scores.items() if v >= 0.9 * best)
    N, k = eligible[0]
    DEC["G1"] = {"n_envs": N, "conc": k, "best_score": round(best, 1), "chosen_score": round(scores[(N, k)], 1),
                 "scores": [{"n_envs": a, "conc": b, "geomean_total_sps": round(v, 1)} for (a, b), v in sorted(scores.items())],
                 "rows": rows, "cpu_10_runs_total_sps": round(cpu_total, 1),
                 "cpu_per_run_sps": [r["sps"] for r in rs]}
    save_dec()
    log(f"G1 decision: {N} worlds per update, {k} GPU runs at once (score {scores[(N, k)]:.0f}; best {best:.0f})")


# ---------------------------------------------------------------- G2
def g2():
    if "G2" in DEC:
        return
    N, k = DEC["G1"]["n_envs"], DEC["G1"]["conc"]
    budget = C["g2_updates"] * N * ROLLOUT
    log(f"G2 recipe grid: start; {N} worlds; {budget} steps; {k} at once")
    cells = [(lr, mb) for lr in C["g2_lr"] for mb in C["g2_mb"]]
    jobs = [dict(d=OUT / "g2" / f"lr{lr}_mb{mb}_s{s}", net="start", world="xl", seed=s, budget=budget,
                 recipe={"n_envs": N, "lr": lr, "n_minibatches": mb}) for s in C["g2_seeds"] for (lr, mb) in cells]
    rs = pool(jobs, k)
    table = []
    for lr, mb in cells:
        rr = [r for r, j in zip(rs, jobs) if j["recipe"]["lr"] == lr and j["recipe"]["n_minibatches"] == mb]
        table.append({"lr": lr, "n_minibatches": mb, "mean": round(mean([sc(r) for r in rr]), 4),
                      "per_seed": [round(sc(r), 4) for r in rr], "learned": sum(bool(r["learned"]) for r in rr),
                      "failed": sum(r["status"] != "ok" for r in rr)})
    best = max(t["mean"] for t in table)
    tied = [t for t in table if t["mean"] >= best - C["tie"]]
    pick = min(tied, key=lambda t: (t["n_minibatches"], t["lr"]))
    DEC["G2"] = {"lr": pick["lr"], "n_minibatches": pick["n_minibatches"], "budget": budget, "best_mean": best,
                 "tied": [(t["lr"], t["n_minibatches"]) for t in tied], "table": table}
    save_dec()
    log(f"G2 decision: lr {pick['lr']}, {pick['n_minibatches']} minibatches (mean {pick['mean']:.3f}; best {best:.3f}; tied {DEC['G2']['tied']})")


def gpu_recipe():
    return {"n_envs": DEC["G1"]["n_envs"], "lr": DEC["G2"]["lr"], "n_minibatches": DEC["G2"]["n_minibatches"]}


def cal_job(world, net, budget, seed):
    return dict(d=OUT / "cal" / world / f"{net}_b{budget}_s{seed}", net=net, world=world, seed=seed, budget=budget,
                recipe=gpu_recipe())


# ---------------------------------------------------------------- G3 / G4
def calibrate(world, step):
    if step in DEC:
        return
    N, k = DEC["G1"]["n_envs"], DEC["G1"]["conc"]
    log(f"{step} budget and ceiling on {WORLD_NAME[world]} ({world}): recipe {gpu_recipe()}")
    rows = []
    for u in C["cal_updates"]:
        b = u * N * ROLLOUT
        jobs = [cal_job(world, "start", b, s) for s in C["start_seeds"]] + \
               [cal_job(world, n, b, s) for n in REFS for s in C["ref_seeds"]]
        rs = pool(jobs, k)
        by = {n: [r for r in rs if r["net"] == n] for n in NETS}
        row = {"updates": u, "budget": b, "start_mean": round(mean([sc(r) for r in by["start"]]), 4),
               "start_per_seed": [round(sc(r), 4) for r in by["start"]],
               "start_learned": sum(bool(r["learned"]) for r in by["start"]),
               "refs": {n: {"mean": round(mean([sc(r) for r in by[n]]), 4), "per_seed": [round(sc(r), 4) for r in by[n]]} for n in REFS},
               "failed": sum(r["status"] != "ok" for r in rs),
               "mean_wall_s": {n: round(mean([r["train_wall_s"] for r in by[n]]), 1) for n in NETS}}
        row["better_ref"] = max(row["refs"][n]["mean"] for n in REFS)
        rows.append(row)
        log(f"{step} {b} steps: start {row['start_mean']:.3f} ({row['start_learned']} learned), "
            + ", ".join(f"{n} {row['refs'][n]['mean']:.3f}" for n in REFS) + f"; walls {row['mean_wall_s']}")
        if row["better_ref"] >= C["ceiling"]:
            log(f"{step}: better reference design reached {row['better_ref']:.3f} >= {C['ceiling']}; larger budgets not run")
            break
    passing = [r for r in rows if r["better_ref"] < C["ceiling"]]
    if passing:
        L = max(passing, key=lambda r: r["budget"])
        tied = [r for r in passing if r["better_ref"] >= L["better_ref"] - C["tie"]]
        pick = min(tied, key=lambda r: (abs(r["start_mean"] - 0.5), -r["budget"]))
        flag = None
    else:
        pick, flag = rows[0], "ceiling gate failed at every budget run; chose the smallest"
    DEC[step] = {"world": world, "budget": pick["budget"], "updates": pick["updates"], "flag": flag, "rows": rows}
    save_dec()
    log(f"{step} decision: {pick['budget']} steps ({pick['updates']} updates); start {pick['start_mean']:.3f}, "
        f"better reference {pick['better_ref']:.3f}" + (f"; FLAG {flag}" if flag else ""))


# ---------------------------------------------------------------- G5
def perm_p_mc(a, b, M):
    x = np.r_[a, b]
    n = len(a)
    d = (x * M).sum(1) / n - (x * ~M).sum(1) / n
    return (np.abs(d) >= abs(a.mean() - b.mean()) - 1e-12).mean()


def power_table(hi, hi_sd, lo, lo_sd, sims):
    rng = np.random.default_rng(3)

    def arm(n, p, sd_p, m):
        ps = np.clip(rng.normal(p, sd_p, n), 0.02, 0.98)
        learn = rng.random((n, m)) < ps[:, None]
        s = np.where(learn, rng.normal(hi, hi_sd, (n, m)), rng.normal(lo, lo_sd, (n, m)))
        return np.clip(s, 0, 1).mean(1)

    gap = hi - lo
    out = []
    for sd_p in (0.15, 0.25):
        for m in (8, 12, 16):
            for n in (10, 12, 16, 20):
                M = np.argsort(rng.random((4000, 2 * n)), 1) < n
                row = {"sd_p": sd_p, "seeds": m, "searches": n}
                for d in (0.0, 0.10, 0.15, 0.20):
                    dp = d / gap  # difference in learning chance that gives a d-point difference in mean score
                    hits = sum(perm_p_mc(arm(n, .5 + dp / 2, sd_p, m), arm(n, .5 - dp / 2, sd_p, m), M) < .05 for _ in range(sims))
                    row["false_pos" if d == 0 else f"+{int(round(d * 100))}"] = round(hits / sims, 3)
                out.append(row)
    return out


def g5():
    if "G5" in DEC:
        return
    k = DEC["G1"]["conc"]
    log("G5 noise: start, extra seeds at each chosen budget")
    noise = {}
    for world, step in (("xl", "G3"), ("large", "G4")):
        b = DEC[step]["budget"]
        rs = pool([cal_job(world, "start", b, s) for s in C["start_seeds"] + C["g5_seeds"]], k)
        x = np.array([sc(r) for r in rs])
        noise[world] = {"budget": b, "seeds": C["start_seeds"] + C["g5_seeds"], "exact": [round(v, 4) for v in x],
                        "hidden": [round(r["hidden"] or 0.0, 4) for r in rs], "mean": round(float(x.mean()), 4),
                        "sd": round(float(x.std(ddof=1)), 4) if len(x) > 1 else None,
                        "learned": sum(bool(r["learned"]) for r in rs), "failed": sum(r["status"] != "ok" for r in rs),
                        "mean_wall_s": round(mean([r["train_wall_s"] for r in rs]), 1)}
        log(f"G5 {WORLD_NAME[world]} at {b}: mean {noise[world]['mean']:.3f}, SD {noise[world]['sd']}, "
            f"{noise[world]['learned']}/{len(rs)} learned")
    bq = DEC["G3"]["budget"]
    runs = [json.loads(p.read_text()) for p in (OUT / "cal" / "xl").glob(f"*_b{bq}_s*/result.json")]
    runs = [r for r in runs if r["status"] == "ok"]
    hi_s = [r["dev_exact"] for r in runs if r["learned"]]
    lo_s = [r["dev_exact"] for r in runs if not r["learned"]]
    levels = {"source": "Stage G runs at the chosen Q budget", "hi": mean(hi_s), "hi_sd": float(np.std(hi_s)) if len(hi_s) > 1 else 0.04,
              "lo": mean(lo_s), "lo_sd": float(np.std(lo_s)) if len(lo_s) > 1 else 0.10, "n_hi": len(hi_s), "n_lo": len(lo_s)}
    if len(hi_s) < 2 or len(lo_s) < 2 or levels["hi"] - levels["lo"] < 0.25:
        levels = {"source": "Stage 0's levels (too few Stage G seeds in one hump)", "hi": 0.88, "hi_sd": 0.04, "lo": 0.30,
                  "lo_sd": 0.10, "n_hi": len(hi_s), "n_lo": len(lo_s)}
    log(f"G5 power model levels: {levels}")
    t = time.time()
    table = power_table(levels["hi"], levels["hi_sd"], levels["lo"], levels["lo_sd"], C["power_sims"])
    log(f"G5 power table done in {time.time() - t:.0f}s")
    DEC["G5"] = {"noise": noise, "levels": levels, "power": table, "sims_per_cell": C["power_sims"]}
    save_dec()


# ---------------------------------------------------------------- G6
def exact_perm(a, b):
    x = np.array(list(a) + list(b), float)
    n, tot = len(a), x.sum()
    obs = abs(np.mean(a) - np.mean(b))
    hit = total = 0
    combos = itertools.combinations(range(len(x)), n)
    while True:
        chunk = np.fromiter(itertools.chain.from_iterable(itertools.islice(combos, 200_000)), dtype=np.int16)
        if chunk.size == 0:
            break
        idx = chunk.reshape(-1, n)
        sa = x[idx].sum(1)
        diff = sa / n - (tot - sa) / (len(x) - n)
        hit += int((np.abs(diff) >= obs - 1e-12).sum())
        total += len(idx)
    return hit / total, total


def g6():
    if "G6" in DEC:
        return
    k = DEC["G1"]["conc"]
    log("G6 device checks")
    seeds = C["g6_seeds"]
    frozen = [dict(d=OUT / "g6" / f"start_frozen_s{s}", net="start", world="xl", seed=s, budget=C["g6_budget"], recipe={})
              for s in seeds]
    g2c = DEC["G2"]
    g2_src = dict(d=OUT / "g2" / f"lr{g2c['lr']}_mb{g2c['n_minibatches']}_s{C['g2_seeds'][0]}", net="start", world="xl",
                  seed=C["g2_seeds"][0], budget=g2c["budget"],
                  recipe={"n_envs": DEC["G1"]["n_envs"], "lr": g2c["lr"], "n_minibatches": g2c["n_minibatches"]})
    repeats = [dict(frozen[0], d=OUT / "g6" / f"repeat_start_frozen_s{seeds[0]}"),
               dict(g2_src, d=OUT / "g6" / f"repeat_g2_lr{g2c['lr']}_mb{g2c['n_minibatches']}_s{C['g2_seeds'][0]}")]
    rs = pool(frozen + repeats, k)
    gpu = [sc(r) for r in rs[:len(seeds)]]
    s1 = json.loads(STAGE1_RESULTS.read_text())["per_arm"]["start"]["per_seed"][:len(seeds)]
    p, n_splits = exact_perm(gpu, s1)
    first = {"frozen": rs[0], "g2": run(**g2_src)}
    det = []
    for name, orig, rep in (("frozen recipe, seed %d" % seeds[0], first["frozen"], rs[len(seeds)]),
                            ("GPU recipe (G2's pick), seed %d" % C["g2_seeds"][0], first["g2"], rs[len(seeds) + 1])):
        det.append({"run": name, "dev": [orig["dev_exact"], rep["dev_exact"]],
                    "weights_sha": [orig.get("weights_sha"), rep.get("weights_sha")],
                    "same_score": orig["dev_exact"] == rep["dev_exact"],
                    "same_weights": orig.get("weights_sha") is not None and orig.get("weights_sha") == rep.get("weights_sha")})
    DEC["G6"] = {"a": {"gpu_per_seed": [round(x, 4) for x in gpu], "gpu_mean": round(mean(gpu), 4),
                       "cpu_stage1_per_seed": s1, "cpu_mean": round(mean(s1), 4), "diff": round(mean(gpu) - mean(s1), 4),
                       "gpu_learned": sum(bool(r["learned"]) for r in rs[:len(seeds)]),
                       "exact_perm_p_two_sided": round(p, 4), "splits": n_splits, "stop": p < 0.05},
                 "b": det}
    save_dec()
    log(f"G6a: GPU mean {mean(gpu):.3f} vs Stage 1 CPU {mean(s1):.3f}, exact permutation p = {p:.4f}"
        + ("  -> STOP: the device alone shifts scores; report before Stage 2" if p < 0.05 else ""))
    for d in det:
        log(f"G6b {d['run']}: same score {d['same_score']}, same weights {d['same_weights']} ({d['dev']}, {d['weights_sha']})")


def main():
    global C, OUT, DEC, MANIFEST
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
                    "config": C, "hashes": hashes(), "torch": torch.__version__}
        mf.write_text(json.dumps(MANIFEST, indent=1))
    verify()
    DEC = json.loads((OUT / "decisions.json").read_text()) if (OUT / "decisions.json").exists() else {}
    log(f"Stage G {'DRY ' if a.dry else ''}start; protocol sha {MANIFEST['hashes']['arch_rsi/gpu/stageG/PROTOCOL_G.md'][:16]}")
    g1()
    g2()
    calibrate("xl", "G3")
    calibrate("large", "G4")
    g5()
    g6()
    (OUT / "summary.json").write_text(json.dumps({"manifest": MANIFEST, "decisions": DEC}, indent=1))
    log("STAGEG_DONE")


if __name__ == "__main__":
    main()
