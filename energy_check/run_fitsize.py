"""Does a larger training sample remove the designs the counted fit makes impossible? A second check added after the
dev results were seen (post hoc): it selects nothing and changes no setting. The scored setting stays 4,000 designs.

Why. The review of this folder found that every design the search failed to stop on has no labelling of energy 0 under
the fit: an allowed term that the training designs never showed is flagged, and some design needs it. The tuning rule
broke the tie between 4,000 and 16,000 training designs in favour of 4,000 (both built all 1,024 tuning designs).

What it can show.
- Whether such impossible designs become rarer when the fit counts 16,000 solved designs in place of 4,000: three
  training samples, each at both sizes (the 16,000 designs of a sample begin with its 4,000; the anchors of the
  labels are drawn afresh for each fit, as run_learned.py draws them).
- What is left when they are gone: tilings that need more bricks than the world's step limit allows.
What it cannot show: anything about the networks, about the dev sets (not rescored) or about the final splits.

Fixed before the first run:
- N_FRESH designs per world from the frozen generator (seed FRESH_SEED, not used before), with every dev, final and
  tuning design rejected by hash and, in the training world, every training design of the six fits.
- The settings are learned_settings.json's apart from the number of training designs; search seed 0.
- `impossible` is counted by `brick_cram.unsolvable` (arc consistency), with no search.
- Every run is reported. A run is one fit over all four worlds.

usage (project root; needs an Apple GPU):
  .venv/bin/python -I -B energy_check/run_fitsize.py        # writes results/learned_fitsize.json; refuses to run twice
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
import run_fresh as RF  # noqa: E402
import run_learned as RL  # noqa: E402
from brickworld.env import StreamSource, design_hash, zobrist_codes  # noqa: E402

FRESH_SEED = 20261117
N_FRESH = 4096
FIT_SEEDS = (RL.TRAIN_SEED, RL.TRAIN_SEED + 1, RL.TRAIN_SEED + 2)
SIZES = (4000, 16000)
OUT = RL.RESULTS / "learned_fitsize.json"


def main():
    started = OUT.with_suffix(".started")
    if OUT.exists() or started.exists():
        raise SystemExit("refused: results/learned_fitsize.json or .started exists; the check is run once")
    started.write_text("started\n")
    torch.set_num_threads(4)
    s = json.loads(RL.SETTINGS.read_text())["setting"]
    assert s["n_train"] == SIZES[0] and s["seed"] == 0
    W = RC.worlds()
    spec_t = W[RL.TRAIN_WORLD][0]

    fits, train_hashes = {}, []
    for seed in FIT_SEEDS:
        for n in SIZES:                           # each fit is made as run_learned makes it: its own labelling
            target, lab, tok = RF.training(n, seed)
            hashes = design_hash(target, zobrist_codes(spec_t))
            K, B, _ = BC.count_fit(lab, tok)
            fits[(seed, n)] = (BC.search_tables(K, B, lab, tok), BC.compare_with_rules(K > 0, B > 0),
                               int(torch.unique(hashes).numel()))
        train_hashes.append(hashes)               # the larger sample, which begins with the smaller one
    frozen, _ = RL.fitted_tables(s["n_train"])
    for k in ("W8", "B", "bad8", "badB"):
        assert np.array_equal(fits[(RL.TRAIN_SEED, s["n_train"])][0][k], frozen[k]), f"the frozen fit was not reproduced ({k})"

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
    rows = []
    for (seed, n), (tabs, cmp, distinct) in fits.items():
        for name, (spec, _, _) in W.items():
            d = fresh[name]
            r = RL.solve(spec, d, tabs, {**s, "n_train": n}, per_design=True)
            imp = BC.unsolvable(tabs, BC.tokens(d).reshape(len(d), -1).numpy().astype(np.int32), (spec.Z, spec.Y, spec.X))
            stopped = np.array(r.pop("per_design")["stopped"], bool)
            rows.append({"fit_seed": seed, "n_train": n, "distinct_training_designs": distinct,
                         "allowed_pair_terms_flagged": cmp["allowed_pair_terms_flagged"],
                         "forbidden_pair_terms_flagged": cmp["forbidden_pair_terms_flagged"],
                         "forbidden_pair_terms": cmp["forbidden_pair_terms"],
                         "impossible": int(imp.sum()), "impossible_and_stopped": int((imp & stopped).sum()),
                         "not_stopped_and_not_proved_impossible": int((~imp & ~stopped).sum()), **r})
            print(json.dumps({k: rows[-1][k] for k in (
                "fit_seed", "n_train", "world", "exact", "n", "impossible", "not_stopped",
                "not_stopped_and_not_proved_impossible", "stopped_over_step_limit", "stopped_but_not_exact",
                "illegal_placements", "moves_median", "moves_max", "gpu_search_seconds")}), flush=True)
    RC.dump({"post_hoc": True, "fresh_seed": FRESH_SEED, "n_fresh": N_FRESH, "setting": s, "rows": rows}, OUT)
    for (seed, n) in fits:
        mine = [r for r in rows if r["fit_seed"] == seed and r["n_train"] == n]
        print(f"fit seed {seed}, {n:>6,} training designs ({mine[0]['distinct_training_designs']:,} distinct, "
              f"{mine[0]['allowed_pair_terms_flagged']} allowed pair terms flagged): "
              f"{sum(r['exact'] for r in mine):,} of {sum(r['n'] for r in mine):,} built exactly; impossible "
              f"{sum(r['impossible'] for r in mine)}, not stopped {sum(r['not_stopped'] for r in mine)}, over the step "
              f"limit {sum(r['stopped_over_step_limit'] for r in mine)}, stopped on a wrong board "
              f"{sum(r['stopped_but_not_exact'] - r['stopped_over_step_limit'] for r in mine)}")


if __name__ == "__main__":
    main()
