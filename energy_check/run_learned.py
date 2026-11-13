"""Fit, tune and score the learned brick energy (brick_cram.py) with its Metal search kernel (metal_search.py).

usage (project root; needs an Apple GPU):
  .venv/bin/python -I -B energy_check/run_learned.py tune     # every setting of GRID on the tuning designs
  .venv/bin/python -I -B energy_check/run_learned.py freeze   # pick one setting by RULE, write learned_settings.json
  .venv/bin/python -I -B energy_check/run_learned.py dev      # score each dev set once with learned_settings.json
  .venv/bin/python -I -B energy_check/run_learned.py pl       # the pseudolikelihood diagnostic, on training designs
  .venv/bin/python -I -B energy_check/run_learned.py speed    # kernel throughput against the NumPy reference

Protocol.
- Training designs: solved designs of the smallest world only (5x5x3), from the frozen generator (seed TRAIN_SEED),
  with every dev, final and tuning design of that world rejected by hash. The fit is `brick_cram.count_fit`.
- The same weights are then used in all four worlds: the energy is shared by every cell and has no grid size.
- Tuning designs: the 256 per world of run_check.py. No setting is chosen on a dev design.
- RULE: the setting that builds the most tuning designs exactly, summed over the worlds; ties go to the fewest
  training designs, then to the fewest search moves (mean over stopped designs, summed over the worlds).
- `dev` refuses to run without learned_settings.json, and refuses to run twice: it writes
  results/learned_dev.started before it scores anything and refuses when that file or results/learned_dev.json exists.
  The guard is a file check and no more: nothing but the protocol stops someone deleting those files.
- The final splits are never read.

Every build is made by stepping the frozen `BrickWorld` and scored by the frozen checker `evaluate.check`.
The stopping test is the energy's alone: the world's step limit is not part of it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import brick_cram as BC  # noqa: E402
import brick_energy as BE  # noqa: E402
import metal_search as MS  # noqa: E402
import run_check as RC  # noqa: E402
from brickworld.env import design_hash, has_hidden_cells, zobrist_codes  # noqa: E402
from evaluate import check  # noqa: E402

RESULTS = HERE / "results"
SETTINGS = HERE / "learned_settings.json"
TRAIN_SEED = 20261114
TRAIN_WORLD = "small"
GRID = [{"n_train": n, "temp": temp, "chains": 4, "max_moves": 20000, "seed": 0}
        for n in (1000, 4000, 16000) for temp in (0.4, 0.6, 0.8)]
RULE = ("most tuning designs built exactly, summed over the worlds; ties: fewest training designs, "
        "then fewest mean search moves")


def training_set(n: int):
    """(labels, tokens) of n solved designs of the training world, none of them a dev, final or tuning design."""
    spec, _, hashes = RC.worlds()[TRAIN_WORLD]
    tune = design_hash(RC.tuning_designs(spec, hashes), zobrist_codes(spec))
    target, bricks = BC.solved_designs(spec, n, TRAIN_SEED, torch.cat([torch.load(hashes), tune]))
    return BC.labels_from_bricks(target, bricks, torch.Generator().manual_seed(TRAIN_SEED)), BC.tokens(target)


def fitted_tables(n: int):
    lab, tok = training_set(n)
    K, B, seconds = BC.count_fit(lab, tok)
    tabs = BC.search_tables(K, B, lab, tok)
    fit = {"n_train": n, "filled_cells": int((tok >= 0).sum()), "fit_seconds": seconds,
           "parameters": int(K.numel() - 4 + B[:, 1:].numel()), "nonzero_parameters": int((K != 0).sum() + (B != 0).sum()),
           **BC.compare_with_rules(K > 0, B > 0)}
    return tabs, fit


def solve(spec, designs: torch.Tensor, tabs: dict, s: dict, per_design: bool = False) -> dict:
    """Search every design on the GPU, build each answer in the frozen world and score it with the frozen checker."""
    dims, n, chains = (spec.Z, spec.Y, spec.X), len(designs), s["chains"]
    tok = np.repeat(BC.tokens(designs).reshape(n, -1).numpy().astype(np.int32), chains, 0)
    y, found, gpu_seconds = MS.search_gpu(tabs, tok, dims, s["max_moves"], s["temp"], s["seed"], chains=chains)
    f = found.reshape(n, chains)
    stopped_np = (f >= 0).any(1)
    left = BC.violations(y, tok, tabs, BC.neighbours(*dims)).sum(1).reshape(n, chains)
    pick = np.where(stopped_np, np.where(f >= 0, f, 1 << 30).argmin(1), left.argmin(1))   # first chain to stop
    labels = torch.from_numpy(y.reshape(n, chains, *designs.shape[1:])[np.arange(n), pick]).long()
    board = BC.decode(labels)
    bricks = board.flatten(1).sum(1)
    built, illegal = BE.build(BE.bricks_to_actions(board, spec.done_action), designs, spec)
    exact = torch.from_numpy(check(built.numpy(), designs.numpy())["exact"])
    stopped = torch.from_numpy(stopped_np)
    moves_all = torch.from_numpy(np.where(stopped_np, f[np.arange(n), pick], s["max_moves"]))
    moves = moves_all[stopped].float()
    res = {
        "world": spec.name, "grid": [spec.X, spec.Y, spec.Z], "n": n,
        "exact": int(exact.sum()), "exact_rate": float(exact.float().mean()),
        "stopped": int(stopped.sum()),
        "stopped_but_not_exact": int((stopped & ~exact).sum()),
        "stopped_over_step_limit": int((stopped & (bricks > spec.max_steps)).sum()),
        "not_stopped": int((~stopped).sum()),
        "illegal_placements": illegal,
        "moves_mean": float(moves.mean()) if len(moves) else None,
        "moves_median": float(moves.quantile(0.5)) if len(moves) else None,      # the mean of the two middle values
        "moves_p90": float(moves.quantile(0.9)) if len(moves) else None,
        "moves_max": int(moves.max()) if len(moves) else None,
        "bricks_mean": float(bricks[stopped].float().mean()) if len(moves) else None,
        "bricks_max": int(bricks[stopped].max()) if len(moves) else None,
        "step_limit": spec.max_steps,
        "share_with_hidden_cell": float(has_hidden_cells(designs).float().mean()),
        "gpu_search_seconds": round(gpu_seconds, 3),
    }
    if per_design:
        res["per_design"] = {"exact": exact.int().tolist(), "stopped": stopped.int().tolist(),
                             "moves": moves_all.tolist(), "bricks": bricks.tolist()}
    return res


def warm_up():
    """Compile the kernel and make one launch, so that no timed search pays for either."""
    tabs = BC.rule_tables()
    MS.search_gpu(tabs, np.zeros((4, 8), np.int32), (2, 2, 2), 8, 0.5, 0)


def cmd_tune(args):
    W = RC.worlds()
    tune = {name: RC.tuning_designs(spec, hashes) for name, (spec, _, hashes) in W.items()}
    rows, fits = [], {}
    for s in GRID:
        if s["n_train"] not in fits:
            fits[s["n_train"]] = fitted_tables(s["n_train"])
            print(json.dumps(fits[s["n_train"]][1]), flush=True)
        for name, (spec, _, _) in W.items():
            r = solve(spec, tune[name], fits[s["n_train"]][0], s)
            rows.append({"setting": s, **r})
            print(json.dumps({"n_train": s["n_train"], "temp": s["temp"], **{k: r[k] for k in (
                "world", "exact", "n", "not_stopped", "stopped_but_not_exact", "stopped_over_step_limit",
                "moves_median", "moves_max", "bricks_max", "step_limit", "gpu_search_seconds")}}), flush=True)
    RC.dump({"train_seed": TRAIN_SEED, "train_world": TRAIN_WORLD, "tune_seed": RC.TUNE_SEED, "n_tune": RC.N_TUNE,
             "rule": RULE, "fits": [f for _, f in fits.values()], "rows": rows}, RESULTS / "learned_tune.json")


def cmd_freeze(args):
    rec = json.loads((RESULTS / "learned_tune.json").read_text())
    n_worlds = len(RC.worlds())
    table = []
    for s in GRID:
        rows = [r for r in rec["rows"] if r["setting"] == s]
        assert len(rows) == n_worlds
        table.append((-sum(r["exact"] for r in rows), s["n_train"], sum(r["moves_mean"] or float("inf") for r in rows), s))
    table.sort(key=lambda it: it[:3])
    best = table[0]
    RC.dump({"setting": best[3], "rule": rec["rule"], "tune_exact_total": -best[0],
             "tune_n_total": RC.N_TUNE * n_worlds, "tune_moves_mean_sum": best[2]}, SETTINGS)
    for neg, n, mv, s in table:
        print(f"n_train {n:<6} temp {s['temp']:<4} exact {-neg}/{RC.N_TUNE * n_worlds}  mean moves (sum over worlds) {mv:.1f}")
    print("frozen:", json.dumps(best[3]))


def cmd_dev(args):
    if not SETTINGS.exists():
        raise SystemExit("refused: no learned_settings.json (run tune, then freeze)")
    out, started = RESULTS / "learned_dev.json", RESULTS / "learned_dev.started"
    if out.exists() or started.exists():
        raise SystemExit("refused: results/learned_dev.json or .started exists; each dev set is scored once")
    s = json.loads(SETTINGS.read_text())["setting"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    started.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")      # a run that began counts as the one run
    tabs, fit = fitted_tables(s["n_train"])
    print(json.dumps(fit), flush=True)
    rows = []
    for name, (spec, dev, _) in RC.worlds().items():
        r = solve(spec, torch.load(dev), tabs, s, per_design=True)
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "per_design"}), flush=True)
    RC.dump({"setting": s, "fit": fit, "rows": rows}, out)


def cmd_pl(args):
    """The paper's objective on the same training designs: does a cost threshold separate forbidden terms?"""
    lab, tok = training_set(1000)
    okK, okB = BC.true_rules()
    rows = []
    for nonneg in (False, True):
        for lr in (30.0, 100.0, 300.0):
            K, B, seconds, loss = BC.fit(lab, tok, lr=lr, updates=300, device="mps", nonneg=nonneg)
            tabs = BC.search_tables(K, B, lab, tok)
            cmp = BC.compare_with_rules(torch.from_numpy(tabs["bad8"][:4] != 0), torch.from_numpy(tabs["badB"] != 0))
            rows.append({"lr": lr, "updates": 300, "nonnegative_costs": nonneg, "loss": loss, "gpu_fit_seconds": seconds,
                         "tau_pair": tabs["tau_pair"], "forbidden_pair_cost_median": float(K[~okK].median()),
                         "forbidden_pair_cost_min": float(K[~okK].min()), **cmp})
            print(json.dumps(rows[-1]), flush=True)
    RC.dump({"n_train": 1000, "rows": rows}, RESULTS / "learned_pl_diagnostic.json")


def cmd_speed(args):
    """Kernel throughput with every term flagged (no chain stops, every move does the full work), against the NumPy
    reference. The reference's time includes its set-up (first labels and flags) and its thread count is NumPy's own:
    the key 'numpy_reference_4_threads' names torch's setting, which does not govern NumPy."""
    dims = (6, 10, 10)                                             # the largest world
    rows = [MS.moves_per_second(dims, chains) for chains in (2048, 16384, 131072)]
    for r in rows:
        print(json.dumps(r), flush=True)
    tabs = BC.rule_tables()
    tabs = {**tabs, "bad8": np.ones_like(tabs["bad8"]), "badB": np.ones_like(tabs["badB"])}
    rng = np.random.default_rng(0)
    tok = np.where(rng.random((2048, 600)) < 0.5, 0, -1).astype(np.int32)
    tok[:, 0] = 0
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        BC.search_reference(tabs, tok, dims, 64, 0.5, seed=0)
        times.append(time.perf_counter() - t0)
    ref = {"chains": 2048, "cells": 600, "moves": 2048 * 64, "seconds_median": float(np.median(times))}
    ref["moves_per_second"] = ref["moves"] / ref["seconds_median"]
    print(json.dumps(ref), flush=True)
    RC.dump({"dims_zyx": dims, "metal_kernel": rows, "numpy_reference_4_threads": ref}, RESULTS / "kernel_speed.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "freeze", "dev", "pl", "speed"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.cmd != "freeze":
        warm_up()
    {"tune": cmd_tune, "freeze": cmd_freeze, "dev": cmd_dev, "pl": cmd_pl, "speed": cmd_speed}[args.cmd](args)
