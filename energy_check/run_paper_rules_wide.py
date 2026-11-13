"""A wider test of the paper's fit with the paper's stopping rules, on the GPU. Post hoc, like run_paper_rules.py:
it changes no scored setting, and no dev or final design is read.

Why. run_paper_rules.py tried two pseudolikelihood fits and three temperatures on the CPU, and no rule worked. Two
fits are a thin basis for saying the paper's fit does not work here: the paper tunes its learning rate, its number
of updates and its L1 strength per task, over wide ranges (20 to 4,000 updates).

What it can show.
- Whether any of 36 pseudolikelihood fits, at any of 4 temperatures, with any of the 5 rules, gives a solver that
  makes no wrong stop. If one does, the fit of this folder could be the paper's own.
- If none does: how close the best setting of each rule comes, on 16,384 fresh designs.
What it cannot show. Anything about the paper's other training methods (it also reports contrastive divergence and
others), other label designs, an annealed temperature, or the paper's own code.

Fixed before the first run (this file was committed before any of its commands was run):
- Fits (`fit`): `brick_cram.fit` on the frozen 4,000 training designs, from zero, momentum 0.5, cosine schedule;
  learning rate 30, 100 or 300; 30, 100, 300 or 1,000 updates; L1 strength 0, 1e-4 or 1e-3. 36 fits.
- Rules: all five of paper_rules.py. Temperatures 0.1, 0.3, 1 and 3 (units of hbar); 0 for `fixed`.
- Search: the Metal kernel of metal_paper_search.py, 4 chains, 20,000 moves, seed 0. A design counts as stopped
  only if the stop also holds when its labels are judged again in float64.
- Tuning (`tune`): the first 128 tuning designs of each world.
- Choice (`freeze`), per rule: the most tuning designs stopped and built exactly, summed over the worlds; ties go
  to the fewest wrong stops, then the lower temperature, learning rate, update count and L1 strength.
- Fresh designs (`fresh`): 4,096 per world (seed FRESH_SEED), none a dev, final, tuning, training or screen design,
  nor a fresh design of run_paper_rules.py. Each rule's chosen setting once, with the control (the counted fit,
  rule terms1, temperature 0.6, through the same kernel). `fresh` refuses to run twice.
- A rule WORKS as in run_paper_rules.py: no wrong stop and at least 99% of the fresh designs built exactly.

usage (project root; needs an Apple GPU, about 20 minutes in all):
  .venv/bin/python -I -B energy_check/run_paper_rules_wide.py fit | tune | freeze | fresh
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
import metal_paper_search as MP  # noqa: E402
import paper_rules as PR  # noqa: E402
import run_check as RC  # noqa: E402
import run_learned as RL  # noqa: E402
import run_paper_rules as RP  # noqa: E402
from brickworld.env import StreamSource, design_hash, zobrist_codes  # noqa: E402
from evaluate import check  # noqa: E402

RESULTS = RL.RESULTS
LRS, UPDATES, L1S = (30.0, 100.0, 300.0), (30, 100, 300, 1000), (0.0, 1e-4, 1e-3)
TEMPS = (0.1, 0.3, 1.0, 3.0)
CHAINS, CAP, N_TUNE, N_FRESH, FRESH_SEED, SEARCH_SEED = 4, 20000, 128, 4096, 20261120, 0
CONTROL = {"fit": "count", "rule": "terms1", "temperature": 0.6}


def fit_name(lr, updates, l1):
    return f"lr{lr:g}_u{updates}_l1_{l1:g}"


def fit_grid():
    for lr in LRS:
        for updates in UPDATES:
            for l1 in L1S:
                yield fit_name(lr, updates, l1), {"lr": lr, "updates": updates, "l1": l1}


def load_fits():
    lab, tok = RP.training()
    z = np.load(RESULTS / "paper_rules_wide_fits.npz")
    info = json.loads((RESULTS / "paper_rules_wide_fits.json").read_text())["fits"]
    out = {name: (torch.from_numpy(z[f"K_{name}"]), torch.from_numpy(z[f"B_{name}"]), info[name]["statistics"], lab, tok)
           for name, _ in fit_grid()}
    K, B, _ = BC.count_fit(lab, tok)
    out["count"] = (K, B, PR.statistics(K, B, lab, tok), lab, tok)
    return out


def score(spec, designs, fit, rule, temperature) -> dict:
    """One GPU search over `designs`, built in the frozen world and scored by the frozen checker."""
    K, B, st, _, _ = fit
    dims, n = (spec.Z, spec.Y, spec.X), len(designs)
    tok = RP.flat_tok(designs)
    out = MP.search(K, B, st, tok, dims, rule, temperature, CHAINS, CAP, SEARCH_SEED, tabs=RP.tables(fit, rule))
    labels = torch.from_numpy(out["labels"].reshape(n, *designs.shape[1:])).long()
    board = BC.decode(labels)
    bricks = board.flatten(1).sum(1).numpy()
    built, illegal = BE.build(BE.bricks_to_actions(board, spec.done_action), designs, spec)
    exact = check(built.numpy(), designs.numpy())["exact"]
    valid = ~BC.violations(out["labels"], tok, BC.rule_tables(), BC.neighbours(*dims)).any(1)   # diagnosis only
    stopped = out["stopped"] & out["confirmed"]
    over = stopped & ~exact & valid & (bricks > spec.max_steps)
    mv = out["moves"][stopped]
    return {"world": spec.name, "n": n, "exact": int(exact.sum()), "stopped": int(stopped.sum()),
            "stopped_and_exact": int((stopped & exact).sum()), "wrong_stops": int((stopped & ~exact & ~over).sum()),
            "stopped_valid_over_step_limit": int(over.sum()), "not_stopped": int((~stopped).sum()),
            "stops_not_confirmed_in_float64": int((out["stopped"] & ~out["confirmed"]).sum()),
            "exact_but_not_stopped": int((~stopped & exact).sum()), "illegal_placements": int(illegal),
            "moves_median": float(np.median(mv)) if len(mv) else None, "moves_max": int(mv.max()) if len(mv) else None,
            "gpu_search_seconds": round(out["gpu_seconds"], 3)}


def configs():
    for name, _ in fit_grid():
        for rule in PR.RULES:
            for temp in ((0.0,) if rule == "fixed" else TEMPS):
                yield {"fit": name, "rule": rule, "temperature": temp}


def cmd_fit(args):
    out = RESULTS / "paper_rules_wide_fits.npz"
    if out.exists():
        raise SystemExit("refused: results/paper_rules_wide_fits.npz exists")
    lab, tok = RP.training()
    arrays, info = {}, {}
    for name, f in fit_grid():
        K, B, seconds, _ = BC.fit(lab, tok, lr=f["lr"], updates=f["updates"], device="mps", l1=f["l1"])
        info[name] = {**f, "loss": float(BC.pl_loss(K, B, lab, tok)), "gpu_fit_seconds": seconds,
                      "nonzero_costs": int((K != 0).sum() + (B != 0).sum()), "statistics": PR.statistics(K, B, lab, tok)}
        arrays[f"K_{name}"], arrays[f"B_{name}"] = K.numpy(), B.numpy()
        print(name, json.dumps({k: info[name][k] for k in ("loss", "gpu_fit_seconds", "nonzero_costs")}), flush=True)
    np.savez(out, **arrays)
    RC.dump({"n_train": len(lab), "fits": info}, RESULTS / "paper_rules_wide_fits.json")


def cmd_tune(args):
    fits, rows = load_fits(), []
    W = RC.worlds()
    designs = {name: RC.tuning_designs(spec, hashes)[:N_TUNE] for name, (spec, _, hashes) in W.items()}
    t0 = time.perf_counter()
    for i, cfg in enumerate(list(configs()) + [CONTROL]):
        st = fits[cfg["fit"]][2]
        usable = all(np.isfinite(st[k]) for k in ("hbar", "ebar", "tau", "delta"))
        for name, (spec, _, _) in W.items():
            if not usable:
                rows.append({**cfg, "world": spec.name, "skipped": "a statistic of the fit is not finite"})
            else:
                rows.append({**cfg, **score(spec, designs[name], fits[cfg["fit"]], cfg["rule"], cfg["temperature"])})
        mine = rows[-len(W):]
        if "skipped" in mine[0]:
            print(i, json.dumps(cfg), "skipped", flush=True)
        else:
            print(i, json.dumps(cfg), "stopped and exact", sum(r["stopped_and_exact"] for r in mine), "wrong stops",
                  sum(r["wrong_stops"] for r in mine), "not stopped", sum(r["not_stopped"] for r in mine),
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    RC.dump({"n_tune_per_world": N_TUNE, "chains": CHAINS, "cap": CAP, "rows": rows}, RESULTS / "paper_rules_wide_tune.json")


def cmd_freeze(args):
    rows = json.loads((RESULTS / "paper_rules_wide_tune.json").read_text())["rows"]
    grid = dict(fit_grid())
    by = {}
    for r in rows:
        if "skipped" not in r and r["fit"] != "count":
            by.setdefault((r["fit"], r["rule"], r["temperature"]), []).append(r)
    chosen = {}
    for rule in PR.RULES:
        best = None
        for (fit, rl, temp), mine in by.items():
            if rl != rule or len(mine) != len(RC.worlds()):
                continue
            f = grid[fit]
            key = (-sum(r["stopped_and_exact"] for r in mine), sum(r["wrong_stops"] for r in mine), temp, f["lr"],
                   f["updates"], f["l1"])
            if best is None or key < best[0]:
                best = (key, {"fit": fit, "rule": rule, "temperature": temp})
        if best is not None:
            chosen[rule] = {**best[1], "tune_stopped_and_exact": -best[0][0], "tune_wrong_stops": best[0][1]}
        print(rule, json.dumps(chosen.get(rule)))
    zero = sorted({(k[0], k[1], k[2]) for k, mine in by.items() if sum(r["wrong_stops"] for r in mine) == 0
                   and sum(r["stopped_and_exact"] for r in mine) > 0})
    print("settings with no wrong stop on the tuning designs and at least one correct stop:", len(zero))
    RC.dump({"chosen": chosen, "settings_tried": len(by), "settings_with_no_wrong_stop_and_a_correct_stop": zero,
             "rule": "most tuning designs stopped and built exactly; ties: fewest wrong stops, lower temperature, "
                     "learning rate, update count, L1 strength"}, RESULTS / "paper_rules_wide_settings.json")


def cmd_fresh(args):
    out, started = RESULTS / "paper_rules_wide_fresh.json", RESULTS / "paper_rules_wide_fresh.started"
    if out.exists() or started.exists():
        raise SystemExit("refused: results/paper_rules_wide_fresh.json or .started exists; the check is run once")
    chosen = json.loads((RESULTS / "paper_rules_wide_settings.json").read_text())["chosen"]
    started.write_text("started\n")
    fits, rows = load_fits(), []
    W = RC.worlds()
    designs = {}
    for name, (spec, _, _) in W.items():
        codes = zobrist_codes(spec)
        held = torch.cat([RP.held_hashes(name), design_hash(RP.screen_designs(name)[0], codes)])
        first = StreamSource(spec, RP.FRESH_SEED, held, chunk=8192)(RP.N_FRESH).clone()   # run_paper_rules.py's fresh designs
        held = torch.cat([held, design_hash(first, codes)])
        designs[name] = StreamSource(spec, FRESH_SEED, held, chunk=8192)(N_FRESH).clone()
        assert not bool(torch.isin(design_hash(designs[name], codes), held).any())
    runs = [{k: c[k] for k in ("fit", "rule", "temperature")} for c in chosen.values()] + [CONTROL]
    for cfg in runs:
        for name, (spec, _, _) in W.items():
            rows.append({**cfg, **score(spec, designs[name], fits[cfg["fit"]], cfg["rule"], cfg["temperature"])})
            print(json.dumps(rows[-1]), flush=True)
    summary = []
    for cfg in runs:
        mine = [r for r in rows if all(r[k] == cfg[k] for k in cfg)]
        tot = {k: sum(r[k] for r in mine) for k in ("n", "exact", "stopped_and_exact", "wrong_stops",
                                                    "stopped_valid_over_step_limit", "not_stopped",
                                                    "stops_not_confirmed_in_float64")}
        tot["works"] = bool(tot["wrong_stops"] == 0 and tot["exact"] >= 0.99 * tot["n"])
        summary.append({**cfg, **tot})
        print(json.dumps(summary[-1]), flush=True)
    RC.dump({"post_hoc": True, "fresh_seed": FRESH_SEED, "n_fresh_per_world": N_FRESH, "chains": CHAINS, "cap": CAP,
             "works_means": RP.WORKS, "summary": summary, "rows": rows}, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "tune", "freeze", "fresh"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    {"fit": cmd_fit, "tune": cmd_tune, "freeze": cmd_freeze, "fresh": cmd_fresh}[args.cmd](args)
