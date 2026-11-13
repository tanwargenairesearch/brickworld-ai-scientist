"""Does the paper's own fit, with any of the paper's stopping rules, give a solver for the brick-assembly designs?

Added after the dev results were seen (post hoc). It changes no scored setting, and no dev or final design is read.

Why. The scored energy of this folder is fitted by counting, not by the paper's pseudolikelihood. The diagnostics
(`run_learned.py pl`, `pl_l1`) showed that pseudolikelihood costs fail one stopping rule: the per-term threshold.
The paper uses four rules, one per task (paper_rules.py). This tests all of them with the paper's fit.

What it can show.
- For each rule: whether the paper's fit plus that rule builds these designs without stopping on a wrong board.
  The paper reports that its rules never stop on a wrong board.
- If one does, the fit of this folder could be the paper's own, with numbers from fresh designs.
- If none does, the departure from the paper's fit stands on a tested footing, and the failure of each rule is
  described: it never stops, or it stops on wrong boards.
What it cannot show.
- Anything about other label designs, other learning rates or update counts, the paper's annealed temperature
  schedule, or the paper's own code. Two fits and three temperatures are tried, no more.
- Anything about the networks, the dev sets or the final splits.

Fixed before the first run (this file was committed before any of its commands was run):
- Fits (`fit`): pseudolikelihood by `brick_cram.fit` on the frozen 4,000 training designs, from zero, momentum 0.5,
  cosine schedule, 300 updates, learning rate 100 (the lower loss of 30 and 100 in results/learned_pl_diagnostic.json).
  Fit A has no L1 term (the paper's maze and N-Queens recipe). Fit B has L1 strength 1e-4 (its graph-coloring recipe).
- Control: the counted fit with rule terms1 at temperature 0.6, the scored setting, run through the same NumPy search.
- Screen (`screen`, no search): 256 solved designs per world from the frozen generator (seed SCREEN_SEED), none a dev,
  final, tuning or training design. For each fit and rule: the share of these true solutions that pass the rule, and
  the share that still pass after one random cell is given another label.
- Tuning (`tune`): the first 64 tuning designs of each world. Every fit x rule, with temperatures 0.3, 1 and 3 (in
  units of hbar) for the rules that sample, and temperature 0 for `fixed`. 4 chains, 20,000 moves.
- Choice (`freeze`), per rule: the fit and temperature with the most tuning designs stopped and built exactly, summed
  over the worlds; ties go to the fewest wrong stops, then the lower temperature, then fit A.
- Fresh designs (`fresh`): 256 per world (seed FRESH_SEED), none a dev, final, tuning, training or screen design.
  Each rule's chosen setting is run once, with the control. `fresh` refuses to run twice.
- A rule WORKS here if, on the fresh designs, it makes no wrong stop and at least 99% are built exactly.
- A wrong stop: the search stopped and the build is not exact. A tiling that the world's rules allow but that needs
  more bricks than the step limit is counted apart, as in the rest of this folder.
- Every build is made in the frozen `BrickWorld` and scored by the frozen checker.

usage (project root; `fit` needs an Apple GPU for about half a minute, the rest runs on the CPU):
  .venv/bin/python -I -B energy_check/run_paper_rules.py fit | screen | tune | freeze | fresh
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
import paper_rules as PR  # noqa: E402
import run_check as RC  # noqa: E402
import run_learned as RL  # noqa: E402
from brickworld.env import StreamSource, design_hash, zobrist_codes  # noqa: E402
from evaluate import check  # noqa: E402

RESULTS = RL.RESULTS
FITS = {"A": {"lr": 100.0, "updates": 300, "l1": 0.0}, "B": {"lr": 100.0, "updates": 300, "l1": 1e-4}}
TEMPS = (0.3, 1.0, 3.0)
CHAINS, CAP, N_TUNE, N_SCREEN, N_FRESH = 4, 20000, 64, 256, 256
SCREEN_SEED, FRESH_SEED, SEARCH_SEED = 20261118, 20261119, 0
CONTROL = {"fit": "count", "rule": "terms1", "temperature": 0.6}
WORKS = "no wrong stop and at least 99% of the fresh designs built exactly"


def training():
    s = json.loads(RL.SETTINGS.read_text())["setting"]
    return RL.training_set(s["n_train"])


def load_fits():
    """name -> (K, B, statistics, lab, tok): the two pseudolikelihood fits of `fit`, and the counted fit."""
    lab, tok = training()
    z = np.load(RESULTS / "paper_rules_fits.npz")
    stats = json.loads((RESULTS / "paper_rules_fits.json").read_text())["fits"]
    out = {name: (torch.from_numpy(z[f"K_{name}"]), torch.from_numpy(z[f"B_{name}"]), stats[name]["statistics"], lab, tok)
           for name in FITS}
    K, B, _ = BC.count_fit(lab, tok)
    out["count"] = (K, B, PR.statistics(K, B, lab, tok), lab, tok)
    return out


def tables(fit, rule):
    K, B, st, lab, tok = fit
    return PR.term_tables(K, B, st, lab, tok, rule) if rule in ("terms", "terms1") else None


def held_hashes(name):
    """Hashes of the dev, final and tuning designs of a world and, in the training world, of the training designs."""
    spec, _, hashes = RC.worlds()[name]
    codes = zobrist_codes(spec)
    held = [torch.load(hashes), design_hash(RC.tuning_designs(spec, hashes), codes)]
    if name == RL.TRAIN_WORLD:
        s = json.loads(RL.SETTINGS.read_text())["setting"]
        tune = design_hash(RC.tuning_designs(spec, hashes), codes)
        target, _ = BC.solved_designs(spec, s["n_train"], RL.TRAIN_SEED, torch.cat([torch.load(hashes), tune]))
        held.append(design_hash(target, codes))
    return torch.cat(held)


def screen_designs(name):
    """(target, labels) of N_SCREEN solved designs of a world, with the generator's own bricks."""
    spec = RC.worlds()[name][0]
    target, bricks = BC.solved_designs(spec, N_SCREEN, SCREEN_SEED, held_hashes(name))
    return target, BC.labels_from_bricks(target, bricks, torch.Generator().manual_seed(SCREEN_SEED))


def flat_tok(designs):
    return BC.tokens(designs).reshape(len(designs), -1).numpy().astype(np.int32)


def score(spec, designs, fit, rule, temperature) -> dict:
    """One search over `designs`, built in the frozen world and scored by the frozen checker."""
    K, B, st, _, _ = fit
    dims, n = (spec.Z, spec.Y, spec.X), len(designs)
    tok = flat_tok(designs)
    t0 = time.perf_counter()
    out = PR.search(K, B, st, tok, dims, rule, temperature, CHAINS, CAP, SEARCH_SEED, tabs=tables(fit, rule))
    seconds = time.perf_counter() - t0
    labels = torch.from_numpy(out["labels"].reshape(n, *designs.shape[1:])).long()
    board = BC.decode(labels)
    bricks = board.flatten(1).sum(1).numpy()
    built, illegal = BE.build(BE.bricks_to_actions(board, spec.done_action), designs, spec)
    exact = check(built.numpy(), designs.numpy())["exact"]
    valid = ~BC.violations(out["labels"], tok, BC.rule_tables(), BC.neighbours(*dims)).any(1)   # diagnosis only
    stopped = out["stopped"]
    over = stopped & ~exact & valid & (bricks > spec.max_steps)
    mv = out["moves"][stopped]
    return {"world": spec.name, "n": n, "exact": int(exact.sum()), "stopped": int(stopped.sum()),
            "stopped_and_exact": int((stopped & exact).sum()), "wrong_stops": int((stopped & ~exact & ~over).sum()),
            "stopped_valid_over_step_limit": int(over.sum()), "not_stopped": int((~stopped).sum()),
            "exact_but_not_stopped": int((~stopped & exact).sum()), "illegal_placements": int(illegal),
            "moves_median": float(np.median(mv)) if len(mv) else None, "moves_max": int(mv.max()) if len(mv) else None,
            "cpu_seconds": round(seconds, 1)}


def configs():
    for name in FITS:
        for rule in PR.RULES:
            for temp in ((0.0,) if rule == "fixed" else TEMPS):
                yield {"fit": name, "rule": rule, "temperature": temp}


def cmd_fit(args):
    out = RESULTS / "paper_rules_fits.npz"
    if out.exists():
        raise SystemExit("refused: results/paper_rules_fits.npz exists")
    lab, tok = training()
    arrays, info = {}, {}
    for name, f in FITS.items():
        K, B, seconds, _ = BC.fit(lab, tok, lr=f["lr"], updates=f["updates"], device="mps", l1=f["l1"])
        st = PR.statistics(K, B, lab, tok)
        info[name] = {**f, "loss": float(BC.pl_loss(K, B, lab, tok)), "gpu_fit_seconds": seconds,
                      "nonzero_costs": int((K != 0).sum() + (B != 0).sum()), "statistics": st}
        arrays[f"K_{name}"], arrays[f"B_{name}"] = K.numpy(), B.numpy()
        print(name, json.dumps(info[name]), flush=True)
    np.savez(out, **arrays)
    RC.dump({"n_train": len(lab), "fits": info}, RESULTS / "paper_rules_fits.json")


def cmd_screen(args):
    fits, rows = load_fits(), []
    rng = np.random.default_rng(SCREEN_SEED)
    for name, (spec, _, _) in RC.worlds().items():
        target, lab = screen_designs(name)
        tok, y = flat_tok(target), lab.reshape(len(lab), -1).numpy().astype(np.int32)
        y1 = y.copy()                                  # one filled cell per design given another label
        for i in range(len(y)):
            c = rng.choice(np.flatnonzero(tok[i] >= 0))
            y1[i, c] = 1 + (y1[i, c] - 1 + rng.integers(1, PR.C - 1)) % (PR.C - 1)
        for fname, fit in fits.items():
            K, B, st, _, _ = fit
            for rule in PR.RULES:
                tabs = tables(fit, rule)
                dims = (spec.Z, spec.Y, spec.X)
                rows.append({"fit": fname, "rule": rule, "world": spec.name, "n": len(y),
                             "true_solutions_passing": int(PR.passes_rule(K, B, st, tok, dims, y, rule, tabs).sum()),
                             "one_wrong_label_passing": int(PR.passes_rule(K, B, st, tok, dims, y1, rule, tabs).sum())})
                print(json.dumps(rows[-1]), flush=True)
    RC.dump({"screen_seed": SCREEN_SEED, "rows": rows}, RESULTS / "paper_rules_screen.json")


def cmd_tune(args):
    fits, rows = load_fits(), []
    W = RC.worlds()
    designs = {name: RC.tuning_designs(spec, hashes)[:N_TUNE] for name, (spec, _, hashes) in W.items()}
    for cfg in list(configs()) + [CONTROL]:
        for name, (spec, _, _) in W.items():
            if not np.isfinite(fits[cfg["fit"]][2]["hbar"]):
                rows.append({**cfg, "world": spec.name, "skipped": "no positive pair cost, so no temperature unit"})
            else:
                rows.append({**cfg, **score(spec, designs[name], fits[cfg["fit"]], cfg["rule"], cfg["temperature"])})
            print(json.dumps(rows[-1]), flush=True)
    RC.dump({"n_tune_per_world": N_TUNE, "chains": CHAINS, "cap": CAP, "rows": rows}, RESULTS / "paper_rules_tune.json")


def cmd_freeze(args):
    rows = json.loads((RESULTS / "paper_rules_tune.json").read_text())["rows"]
    chosen = {}
    for rule in PR.RULES:
        best = None
        for cfg in configs():
            if cfg["rule"] != rule:
                continue
            mine = [r for r in rows if all(r[k] == cfg[k] for k in cfg) and "skipped" not in r]
            if len(mine) != len(RC.worlds()):
                continue
            key = (-sum(r["stopped_and_exact"] for r in mine), sum(r["wrong_stops"] for r in mine), cfg["temperature"],
                   cfg["fit"])
            if best is None or key < best[0]:
                best = (key, cfg)
        if best is not None:
            chosen[rule] = {**best[1], "tune_stopped_and_exact": -best[0][0], "tune_wrong_stops": best[0][1]}
        print(rule, json.dumps(chosen.get(rule)))
    RC.dump({"chosen": chosen, "rule": "most tuning designs stopped and built exactly; ties: fewest wrong stops, "
             "lower temperature, fit A"}, RESULTS / "paper_rules_settings.json")


def cmd_fresh(args):
    out, started = RESULTS / "paper_rules_fresh.json", RESULTS / "paper_rules_fresh.started"
    if out.exists() or started.exists():
        raise SystemExit("refused: results/paper_rules_fresh.json or .started exists; the check is run once")
    chosen = json.loads((RESULTS / "paper_rules_settings.json").read_text())["chosen"]
    started.write_text("started\n")
    fits, rows = load_fits(), []
    W = RC.worlds()
    designs = {}
    for name, (spec, _, _) in W.items():
        codes = zobrist_codes(spec)
        held = torch.cat([held_hashes(name), design_hash(screen_designs(name)[0], codes)])
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
                                                    "stopped_valid_over_step_limit", "not_stopped")}
        tot["works"] = bool(tot["wrong_stops"] == 0 and tot["exact"] >= 0.99 * tot["n"])
        summary.append({**cfg, **tot})
        print(json.dumps(summary[-1]), flush=True)
    RC.dump({"post_hoc": True, "fresh_seed": FRESH_SEED, "n_fresh_per_world": N_FRESH, "chains": CHAINS, "cap": CAP,
             "works_means": WORKS, "summary": summary, "rows": rows}, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "screen", "tune", "freeze", "fresh"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    {"fit": cmd_fit, "screen": cmd_screen, "tune": cmd_tune, "freeze": cmd_freeze, "fresh": cmd_fresh}[args.cmd](args)
