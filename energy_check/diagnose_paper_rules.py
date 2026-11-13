"""What do the wrong stops of the paper's fit look like? A diagnosis on tuning designs, added after the results of
run_paper_rules_wide.py were seen. It selects nothing.

For each rule's chosen setting (results/paper_rules_wide_settings.json) and for the control, on the first 128 tuning
designs of each world: every stop that does not build the design is sorted by which of the world's rules its labels
break. `brick_cram.true_rules` is used for that sorting only.
  pair rule:  two neighbouring cells carry labels that cannot belong together (a brick with a missing or wrong piece)
  input rule: a brick's anchor cell rests on nothing, or a 1x1 brick does

usage (project root; needs an Apple GPU for a few seconds):
  .venv/bin/python -I -B energy_check/diagnose_paper_rules.py     # writes results/paper_rules_wide_diagnosis.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import brick_cram as BC  # noqa: E402
import brick_energy as BE  # noqa: E402
import metal_paper_search as MP  # noqa: E402
import run_check as RC  # noqa: E402
import run_paper_rules as RP  # noqa: E402
import run_paper_rules_wide as RW  # noqa: E402
from evaluate import check  # noqa: E402


def main():
    torch.set_num_threads(4)
    chosen = json.loads((RW.RESULTS / "paper_rules_wide_settings.json").read_text())["chosen"]
    fits = RW.load_fits()
    rules = BC.rule_tables()
    pair_only = {**rules, "badB": np.zeros_like(rules["badB"])}
    input_only = {**rules, "bad8": np.zeros_like(rules["bad8"])}
    rows = []
    for cfg in [{k: c[k] for k in ("fit", "rule", "temperature")} for c in chosen.values()] + [RW.CONTROL]:
        fit = fits[cfg["fit"]]
        K, B, st = fit[:3]
        tot = {"n": 0, "stopped_and_exact": 0, "wrong_stops": 0, "break_a_pair_rule_only": 0,
               "break_an_input_rule_only": 0, "break_both": 0, "break_neither": 0}
        share = []
        for name, (spec, _, hashes) in RC.worlds().items():
            d = RC.tuning_designs(spec, hashes)[:RW.N_TUNE]
            tok, dims = RP.flat_tok(d), (spec.Z, spec.Y, spec.X)
            out = MP.search(K, B, st, tok, dims, cfg["rule"], cfg["temperature"], RW.CHAINS, RW.CAP, RW.SEARCH_SEED,
                            tabs=RP.tables(fit, cfg["rule"]))
            labels = torch.from_numpy(out["labels"].reshape(len(d), *d.shape[1:])).long()
            built, _ = BE.build(BE.bricks_to_actions(BC.decode(labels), spec.done_action), d, spec)
            exact = check(built.numpy(), d.numpy())["exact"]
            stopped = out["stopped"] & out["confirmed"]
            nbr = BC.neighbours(*dims)
            vp = BC.violations(out["labels"], tok, pair_only, nbr)
            vi = BC.violations(out["labels"], tok, input_only, nbr)
            wrong = stopped & ~exact
            p, i = vp.any(1), vi.any(1)
            tot["n"] += len(d)
            tot["stopped_and_exact"] += int((stopped & exact).sum())
            tot["wrong_stops"] += int(wrong.sum())
            tot["break_a_pair_rule_only"] += int((wrong & p & ~i).sum())
            tot["break_an_input_rule_only"] += int((wrong & ~p & i).sum())
            tot["break_both"] += int((wrong & p & i).sum())
            tot["break_neither"] += int((wrong & ~p & ~i).sum())
            share += ((vp | vi).sum(1)[wrong] / (tok >= 0).sum(1)[wrong]).tolist()
        tot["median_share_of_cells_breaking_a_rule_at_a_wrong_stop"] = float(np.median(share)) if share else None
        rows.append({**cfg, **tot})
        print(json.dumps(rows[-1]), flush=True)
    RC.dump({"designs": "the first 128 tuning designs of each world", "rows": rows},
            RW.RESULTS / "paper_rules_wide_diagnosis.json")


if __name__ == "__main__":
    main()
