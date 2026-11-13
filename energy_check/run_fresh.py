"""A robustness check of the learned brick energy on fresh designs. Added after the dev results were seen (post hoc):
it selects nothing and changes no setting.

What it can show.
- Whether the dev score depended on the one search seed used: the frozen fit, search seeds 0, 1 and 2.
- Whether it depended on the one sample of training designs: two refits on other training samples, search seed 0.
- A miss rate from 8 times as many designs per world as a dev set holds.
What it cannot show: anything about the networks, which are not evaluated here, or about the final splits.

Fixed before the first run:
- N_FRESH designs per world from the frozen generator (seed FRESH_SEED), with every dev, final and tuning design
  rejected by hash, and, in the training world, every training design of the three fits.
- The settings are learned_settings.json's, unchanged, apart from the search seed where stated.
- Every run is reported. A run is one fit and one search seed over all four worlds.

usage (project root; needs an Apple GPU):
  .venv/bin/python -I -B energy_check/run_fresh.py          # writes results/learned_fresh.json; refuses to run twice
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import brick_cram as BC  # noqa: E402
import run_check as RC  # noqa: E402
import run_learned as RL  # noqa: E402
from brickworld.env import StreamSource, design_hash, zobrist_codes  # noqa: E402

FRESH_SEED = 20261116                    # equals the second refit seed by accident: the two seed different generators
                                         # (the design stream here, the solved designs of the training world there), and
                                         # every training design is rejected from the fresh designs by hash
N_FRESH = 4096
SEARCH_SEEDS = (0, 1, 2)
REFIT_SEEDS = (RL.TRAIN_SEED + 1, RL.TRAIN_SEED + 2)
OUT = RL.RESULTS / "learned_fresh.json"


def training(n: int, seed: int):
    """(designs, labels, tokens) of n solved designs of the training world: run_learned.training_set with a seed."""
    spec, _, hashes = RC.worlds()[RL.TRAIN_WORLD]
    tune = design_hash(RC.tuning_designs(spec, hashes), zobrist_codes(spec))
    target, bricks = BC.solved_designs(spec, n, seed, torch.cat([torch.load(hashes), tune]))
    return target, BC.labels_from_bricks(target, bricks, torch.Generator().manual_seed(seed)), BC.tokens(target)


def main():
    started = OUT.with_suffix(".started")
    if OUT.exists() or started.exists():
        raise SystemExit("refused: results/learned_fresh.json or .started exists; the check is run once")
    started.write_text("started\n")
    torch.set_num_threads(4)
    s = json.loads(RL.SETTINGS.read_text())["setting"]
    W = RC.worlds()

    # the three fits: the frozen one (which must equal run_learned's) and two on other training samples
    fits, train_hashes = {}, []
    spec_t = W[RL.TRAIN_WORLD][0]
    for seed in (RL.TRAIN_SEED,) + REFIT_SEEDS:
        target, lab, tok = training(s["n_train"], seed)
        K, B, _ = BC.count_fit(lab, tok)
        tabs = BC.search_tables(K, B, lab, tok)
        fits[seed] = (tabs, BC.compare_with_rules(K > 0, B > 0))
        train_hashes.append(design_hash(target, zobrist_codes(spec_t)))
    frozen, _ = RL.fitted_tables(s["n_train"])
    for k in ("W8", "B", "bad8", "badB"):
        assert np.array_equal(fits[RL.TRAIN_SEED][0][k], frozen[k]), f"the frozen fit was not reproduced ({k})"

    # the fresh designs
    fresh = {}
    for name, (spec, _, hashes) in W.items():
        codes = zobrist_codes(spec)
        held = [torch.load(hashes), design_hash(RC.tuning_designs(spec, hashes), codes)]
        if name == RL.TRAIN_WORLD:
            held += train_hashes
        held = torch.cat(held)
        designs = StreamSource(spec, FRESH_SEED, held, chunk=8192)(N_FRESH).clone()
        assert not bool(torch.isin(design_hash(designs, codes), held).any())
        fresh[name] = designs

    RL.warm_up()
    runs = [{"fit_seed": RL.TRAIN_SEED, "search_seed": k, "fit": "frozen"} for k in SEARCH_SEEDS]
    runs += [{"fit_seed": f, "search_seed": 0, "fit": "refit"} for f in REFIT_SEEDS]
    rows = []
    for run in runs:
        tabs, cmp = fits[run["fit_seed"]]
        for name, (spec, _, _) in W.items():
            r = RL.solve(spec, fresh[name], tabs, {**s, "seed": run["search_seed"]})
            rows.append({**run, "allowed_pair_terms_flagged": cmp["allowed_pair_terms_flagged"],
                         "forbidden_pair_terms_flagged": cmp["forbidden_pair_terms_flagged"], **r})
            print(json.dumps({**run, **{k: r[k] for k in (
                "world", "exact", "n", "not_stopped", "stopped_but_not_exact", "stopped_over_step_limit",
                "illegal_placements", "moves_median", "moves_max", "gpu_search_seconds")}}), flush=True)
    RC.dump({"post_hoc": True, "fresh_seed": FRESH_SEED, "n_fresh": N_FRESH, "setting": s, "rows": rows}, OUT)
    for run in runs:
        mine = [r for r in rows if r["fit_seed"] == run["fit_seed"] and r["search_seed"] == run["search_seed"]]
        print(f"{run['fit']:6} fit seed {run['fit_seed']} search seed {run['search_seed']}: "
              f"{sum(r['exact'] for r in mine):,} of {sum(r['n'] for r in mine):,} built exactly; "
              f"not stopped {sum(r['not_stopped'] for r in mine)}, over the step limit "
              f"{sum(r['stopped_over_step_limit'] for r in mine)}, stopped on a wrong board "
              f"{sum(r['stopped_but_not_exact'] - r['stopped_over_step_limit'] for r in mine)}")


if __name__ == "__main__":
    main()
