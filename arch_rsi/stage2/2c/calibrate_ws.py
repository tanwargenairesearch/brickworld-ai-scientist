"""Stage 2c, before anything else: calibrate world W_s (PROTOCOL_2C.md, decision 1) with Stage 2a's ladder rules.

Imports Stage 2a's runner (stage2/2a/run_2a.py, not edited) and uses its rules and helpers unchanged: the budgets,
the 12 ladder runs per budget, the reading (too easy if the start under Stage G's recipe averages over 50% or the
better reference network reaches 90%; too hard if both are under 20%), the one fallback budget, and the noise step.
One addition, approved by the user on 10 November before any W_s run (PROTOCOL_2C.md, Amendment 2): if the fallback
is also too easy, the ladder keeps halving (65,536, then 32,768, then 16,384 steps) with the same 12 runs and the same
reading, and stops at the first budget that passes (the largest passing one) or at the first that is too hard.
Only what must change for W_s is changed:
- the world: W_s (8x8x6, 6-16 bricks) and its 512 dev designs; "learned" uses W_s's dev designs with hidden cells;
- the trainer: Stage 2c's own (ws_world.py train, i.e. the frozen stage2/train_2b.py), so the budget is calibrated on
  the code 2c will run. With no RECIPE block it trains with Stage G's recipe. The frozen recipe is the RECIPE block
  2b's start_frozen comparator uses;
- the machine-time projection: Stage 2c's runs instead of 2b's.
The 3,600 s limit per calibration run is 2a's. Stage 2c itself will use 900 s per run on W_s, as on W.
No model calls; W_s's designs only (W_s has no final split).

If a budget is chosen, writes stage2/2c/ws_budget.json (the budget and 2c's 900 s limit), which run_2c.py needs.
If none passes, it stops, writes no budget, and the result goes back to the user, as in 2a.

usage (project root): caffeinate -i .venv/bin/python -I arch_rsi/stage2/2c/calibrate_ws.py [--dry]
Resumable: a run with a result.json is not re-run. Writes stage2/2c/calib/ (or calib_dry/).
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
S2 = HERE.parent
A = S2.parent
ROOT = A.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import ws_world  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_2a", S2 / "2a" / "run_2a.py")
R2A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R2A)
from arch_rsi.orchestrate_arch import FROZEN, file_hash  # noqa: E402
from arch_rsi.stage2 import w2b  # noqa: E402

PY = R2A.PY
WS_PY = HERE / "ws_world.py"
TIMEOUT_2C = 900
BUDGET_PATH = HERE / "ws_budget.json"
# Stage 2c's runs (PROTOCOL_2C.md): meta-training on W_s, 2 lineages x 4 generations x 6 searches x (30 + 3), plus
# the start on 5 seeds; meta-test on W, 4 arms x 16 searches x (30 + 8), timed with 2a's measured W figure.
C2_WS_RUNS, C2_W_RUNS = 2 * 4 * 6 * 33 + 5, 4 * 16 * 38
CFG = {"full": dict(R2A.CFG["full"], out="calib", down=[65536, 32768, 16384]),
       "dry": dict(R2A.CFG["dry"], out="calib_dry", down=[2048])}
C: dict = {}
OUT = HERE
MANIFEST: dict = {}
LOCK = threading.Lock()
CANDS: dict = {}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z {msg}"
    with LOCK:
        print(line, flush=True)
        with open(OUT / "run.log", "a") as f:
            f.write(line + "\n")


def hidden_mask(dev: torch.Tensor) -> np.ndarray:
    """run_2a.hidden_mask, for a given design set: designs with an empty cell under a filled one."""
    v = dev.bool()
    above = (v.flip(1).cumsum(1).flip(1) - v.int()) > 0
    return ((~v) & above).flatten(1).any(1).numpy()


H = hidden_mask(torch.load(ws_world.TARGETS / "ws_dev.pt"))


def candidates(out: Path) -> dict:
    """(network, recipe) -> candidate file. Stage G's recipe is the default (no RECIPE block); the frozen recipe is
    2b's start_frozen file: the start plus w2b.FROZEN_AS_BLOCK."""
    sf = out / "start_frozen.py"
    src = w2b.with_recipe(R2A.NETS["start"].read_text(), w2b.FROZEN_AS_BLOCK)
    if sf.exists() and sf.read_text() != src:
        raise SystemExit(f"{sf} differs from the start plus the frozen recipe")
    sf.write_text(src)
    return {("start", "g"): R2A.NETS["start"], ("start", "f"): sf, ("refA", "g"): R2A.NETS["refA"], ("refB", "g"): R2A.NETS["refB"]}


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def hashes() -> dict:
    files = (list(FROZEN) + [S2 / p for p in ("train_2b.py", "w2b.py", "w_world.py")]
             + [HERE / p for p in ("ws_world.py", "calibrate_ws.py", "PROTOCOL_2C.md", "make_targets_ws.py",
                                   "targets/ws_dev.pt", "targets/ws_check.pt", "targets/ws_heldout_hashes.pt")]
             + [S2 / "2a" / "run_2a.py", S2 / "2a" / "PROTOCOL_2A.md"] + list(CANDS.values()))
    return {rel(p): file_hash(Path(p)) for p in dict.fromkeys(files)}


def verify():
    now = hashes()
    bad = [k for k, v in MANIFEST["hashes"].items() if now.get(k) != v]
    if bad:
        (OUT / "STOPPED").write_text(f"INTEGRITY: changed {bad}")
        log(f"INTEGRITY: changed {bad}")
        raise SystemExit(f"INTEGRITY: changed {bad}")


def run(cell):
    """run_2a.run on W_s through Stage 2c's trainer and evaluator."""
    net, rk, seed, budget = cell
    d = OUT / f"b{budget}" / f"{net}_{rk}_s{seed}"
    rf = d / "result.json"
    if rf.exists():
        return json.loads(rf.read_text())
    verify()
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"candidate": str(CANDS[(net, rk)]), "world": "ws", "seed": seed,
                                            "budget_steps": budget, "threads": 1, "device": "mps"}))
    t0, rc, err = time.time(), None, None
    try:
        p = subprocess.run([PY, "-I", str(WS_PY), "train", "--spec", str(d / "run.json"), "--out", str(d / "run")],
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
        subprocess.run([PY, "-I", str(WS_PY), "eval", "--trial", str(d / "run"), "--split", "dev", "--out", str(d / "dev.json")],
                       capture_output=True, text=True, timeout=C["timeout"])
        if (d / "dev.json").exists():
            pe = np.array(json.loads((d / "dev.json").read_text())["per_design"]["exact"], bool)
            res |= {"dev_exact": float(pe.mean()), "hidden": float(pe[H].mean())}
            res["learned"] = res["hidden"] >= R2A.AS1.LEARNED
        else:
            res |= {"status": "eval_failed", "error": "no dev.json"}
    res["wall_s"] = round(time.time() - t0, 1)
    rf.write_text(json.dumps(res))
    log(f"b{budget} {net}_{rk}_s{seed}: {res['status']} steps {res['steps']} wall {res['wall_s']}s"
        + (f" dev {res['dev_exact']:.4f}" if res["dev_exact"] is not None else f" ERROR {str(res['error'])[:200]}"))
    return res


def ladder() -> tuple:
    """run_2a.main's ladder and fallback, step for step, then Amendment 2's downward steps. Returns (results, decisions)."""
    res, lad = {}, []
    for b in C["budgets"]:
        res |= R2A.run_all(R2A.ladder_cells(b))
        rd = R2A.read_budget(res, b)
        lad.append(rd)
        log(f"ladder b{b}: start (Stage G recipe) {rd['S']:.4f}, better reference {rd['R']:.4f} -> {rd['reading']}")
        if rd["reading"] == "too easy":
            break
    passing = [r for r in lad if r["reading"] == "passes"]
    status = "chosen"
    if not passing and lad[0]["reading"] == "too easy":
        b = C["fallback"]
        res |= R2A.run_all(R2A.ladder_cells(b))
        rd = R2A.read_budget(res, b)
        lad.append(rd)
        log(f"ladder fallback b{b}: start {rd['S']:.4f}, better reference {rd['R']:.4f} -> {rd['reading']}")
        passing = [rd] if rd["reading"] == "passes" else []
        status = "chosen" if passing else "W_s too easy"
        if rd["reading"] == "too easy":                      # Amendment 2: keep halving while it stays too easy
            for b in C["down"]:
                res |= R2A.run_all(R2A.ladder_cells(b))
                rd = R2A.read_budget(res, b)
                lad.append(rd)
                log(f"ladder down b{b}: start {rd['S']:.4f}, better reference {rd['R']:.4f} -> {rd['reading']}")
                if rd["reading"] == "passes":
                    passing, status = [rd], "chosen"
                    break
                if rd["reading"] == "too hard":
                    status = "no budget passes: too easy at the next budget up, too hard here"
                    break
    elif not passing:
        status = "W_s too hard" if all(r["reading"] == "too hard" for r in lad) else "no budget passes"
    if not passing and C.get("force_noise"):  # dry run only, as in 2a: exercise the noise step on the last budget
        passing, status = [lad[-1]], "dry: forced"
    return res, {"ladder": lad, "status": status, "chosen_budget": max(r["budget"] for r in passing) if passing else None}


def noise(res: dict, B: int) -> dict:
    """run_2a.main's noise step at the chosen budget, with the projection for Stage 2c."""
    t0 = time.time()
    cells = ([("start", "g", s, B) for s in C["noise_start_g"]] + [(n, "g", s, B) for n in ("refA", "refB") for s in C["noise_ref"]]
             + [("start", "f", s, B) for s in C["noise_start_f"]])
    res |= R2A.run_all(cells)
    sg = R2A.group(res, "start", "g", C["noise_start_g"], B)
    half = len(C["noise_start_g"]) // 2
    r1, r2 = sg["per_seed"][:half], sg["per_seed"][half:]
    out = {"budget": B, "start_g": sg,
           "start_g_range1": {"seeds": C["noise_start_g"][:half], "mean": round(R2A.AS1.mean(r1), 4), "sd": round(R2A.statistics.stdev(r1), 4)},
           "start_g_range2": {"seeds": C["noise_start_g"][half:], "mean": round(R2A.AS1.mean(r2), 4), "sd": round(R2A.statistics.stdev(r2), 4)},
           "ranges_perm_p": round(R2A.AS1.perm_p(r1, r2), 4),
           "refA": R2A.group(res, "refA", "g", C["noise_ref"], B), "refB": R2A.group(res, "refB", "g", C["noise_ref"], B),
           "start_f": R2A.group(res, "start", "f", C["noise_start_f"], B), "headroom": round(1 - sg["mean"], 4)}
    walls = [res[(n, "g", s, B)]["wall_s"] for n in ("refA", "refB") for s in C["noise_ref"]]
    t_ws = R2A.AS1.mean(walls) / C["workers"]
    t_w = json.loads((S2 / "2a" / "runs" / "summary.json").read_text())["noise"]["projection_2b"]["eff_s_per_run_ref"]
    h_ws, h_w = C2_WS_RUNS * t_ws / 3600, C2_W_RUNS * t_w / 3600
    out["projection_2c"] = {"ws_runs": C2_WS_RUNS, "eff_s_per_run_ws": round(t_ws, 1), "ws_hours": round(h_ws, 1),
                            "w_runs": C2_W_RUNS, "eff_s_per_run_w_from_2a": t_w, "w_hours": round(h_w, 1),
                            "hours": round(h_ws + h_w, 1), "over_48h": h_ws + h_w > 48}
    slowest = max(r["wall_s"] for k, r in res.items() if k[3] == B)
    out["slowest_run_s"] = slowest
    out["slowest_over_half_2c_limit"] = slowest > TIMEOUT_2C / 2
    out["noise_step_wall_s"] = round(time.time() - t0, 1)
    return out


def main(argv=None):
    global C, OUT, MANIFEST, CANDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    C = CFG["dry" if a.dry else "full"]
    R2A.C, R2A.run = C, run                      # 2a's helpers read their config and run function from their module
    OUT = HERE / C["out"]
    OUT.mkdir(parents=True, exist_ok=True)
    CANDS = candidates(OUT)
    mf = OUT / "manifest.json"
    if mf.exists():
        MANIFEST = json.loads(mf.read_text())
    else:
        MANIFEST = {"started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "mode": "dry" if a.dry else "full",
                    "config": C, "world": "ws", "cands": {f"{n}_{r}": rel(p) for (n, r), p in CANDS.items()},
                    "hashes": hashes()}
        mf.write_text(json.dumps(MANIFEST, indent=1))
    verify()
    log(f"W_s calibration {'DRY ' if a.dry else ''}start; 2c protocol sha "
        f"{MANIFEST['hashes']['arch_rsi/stage2/2c/PROTOCOL_2C.md'][:16]}, 2a protocol sha "
        f"{MANIFEST['hashes']['arch_rsi/stage2/2a/PROTOCOL_2A.md'][:16]}")
    res, decisions = ladder()
    (OUT / "decisions.json").write_text(json.dumps(decisions, indent=1))
    if decisions["chosen_budget"] is None:
        log(f"no budget passes: {decisions['status']}. No ws_budget.json written; back to the user. CALIBRATION_DONE")
        return decisions
    B = decisions["chosen_budget"]
    log(f"chosen budget {B}; noise step")
    nz = noise(res, B)
    (OUT / "summary.json").write_text(json.dumps({"decisions": decisions, "noise": nz}, indent=1))
    p = nz["projection_2c"]
    log(f"noise: start (Stage G recipe) {nz['start_g']['mean']:.4f} (SD {nz['start_g']['sd']}), learned "
        f"{nz['start_g']['learned']} of {nz['start_g']['n']}; refA {nz['refA']['mean']:.4f}, refB {nz['refB']['mean']:.4f}; "
        f"start (frozen recipe) {nz['start_f']['mean']:.4f}; slowest run {nz['slowest_run_s']} s")
    log(f"projected Stage 2c machine time: {p['hours']} h (W_s {p['ws_hours']} h, W {p['w_hours']} h)"
        + (" (OVER 48 h: back to the user)" if p["over_48h"] else ""))
    if not a.dry:
        bf = BUDGET_PATH
        if bf.exists() and json.loads(bf.read_text())["steps"] != B:
            raise SystemExit(f"{bf} exists with another budget; refusing to overwrite")
        bf.write_text(json.dumps({"steps": B, "timeout": TIMEOUT_2C, "chosen_by": "calibrate_ws.py",
                                  "decisions": "arch_rsi/stage2/2c/calib/decisions.json",
                                  "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}, indent=1))
        log(f"wrote {bf.name}: {B} steps, {TIMEOUT_2C} s per run")
    log("CALIBRATION_DONE")
    return decisions


if __name__ == "__main__":
    if "STAGE2B_DONE" not in (A / "campaigns" / "stage2b" / "progress.log").read_text():
        raise SystemExit("Stage 2b is still running (no STAGE2B_DONE): calibration would compete for the GPU")
    main()
