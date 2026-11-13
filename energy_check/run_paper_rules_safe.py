"""One more post hoc check of the paper's fit, added after the results of run_paper_rules_wide.py were seen.

Why. The wider test chose, per rule, the setting that finished the most tuning designs, and every chosen setting
made wrong stops. But its tuning table also holds 16 settings that made NO wrong stop on the 512 tuning designs
and stopped correctly on some of them: all use the graph-coloring rule (`terms`) with a lightly trained fit
(learning rate 30). The choice rule fixed before that run did not pick them, so none was scored on fresh designs.

What it can show.
- Whether the best of those settings also makes no wrong stop on 16,384 new fresh designs, and how many of them
  it finishes. No wrong stop: the paper's fit with the paper's graph-coloring rule gives a stop that can be
  trusted here, for a share of the designs. Any wrong stop: it does not.
- How its stop test relates to the counted fit's: which cost terms each of them flags.
What it cannot show. That the choice was free of hindsight: the setting is picked on the tuning designs after the
wider test was read, which is why the fresh designs here are new. It says nothing about other label designs,
other training methods, or the paper's own code.

Fixed before the first run (this file was committed before it was run):
- The setting: among the settings of results/paper_rules_wide_settings.json with no wrong stop and a correct stop
  on the tuning designs, the one with the most tuning designs stopped and built exactly; ties go to the lower
  temperature, learning rate, update count and L1 strength.
- Fresh designs: 4,096 per world, seed FRESH_SEED, none a dev, final, tuning, training or screen design, nor a
  fresh design of run_paper_rules.py or run_paper_rules_wide.py. Scored once; the script refuses a second run.
- Search and scoring: those of run_paper_rules_wide.py (Metal kernel, 4 chains, 20,000 moves, seed 0; a stop
  counts only if it holds in float64).

usage (project root; needs an Apple GPU, under a minute):
  .venv/bin/python -I -B energy_check/run_paper_rules_safe.py     # writes results/paper_rules_safe.json
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
import run_check as RC  # noqa: E402
import run_paper_rules as RP  # noqa: E402
import run_paper_rules_wide as RW  # noqa: E402
from arch_rsi.stage2.designs import StreamSource, design_hash, zobrist_codes  # noqa: E402

FRESH_SEED = 20261121
N_FRESH = 4096


def pick() -> dict:
    """The setting fixed in the docstring, from the wider test's tuning table."""
    safe = {tuple(s) for s in json.loads((RW.RESULTS / "paper_rules_wide_settings.json").read_text())[
        "settings_with_no_wrong_stop_and_a_correct_stop"]}
    rows = json.loads((RW.RESULTS / "paper_rules_wide_tune.json").read_text())["rows"]
    info = json.loads((RW.RESULTS / "paper_rules_wide_fits.json").read_text())["fits"]
    tot = {}
    for r in rows:
        key = (r["fit"], r["rule"], r["temperature"])
        if key in safe:
            t = tot.setdefault(key, {"tune_stopped_and_exact": 0, "tune_wrong_stops": 0, "tune_not_stopped": 0})
            t["tune_stopped_and_exact"] += r["stopped_and_exact"]
            t["tune_wrong_stops"] += r["wrong_stops"]
            t["tune_not_stopped"] += r["not_stopped"]
    assert len(tot) == len(safe) and all(t["tune_wrong_stops"] == 0 for t in tot.values())
    key = min(tot, key=lambda k: (-tot[k]["tune_stopped_and_exact"], k[2], info[k[0]]["lr"], info[k[0]]["updates"],
                                  info[k[0]]["l1"]))
    return {"fit": key[0], "rule": key[1], "temperature": key[2], **tot[key], "candidates": len(tot)}


def flagged_terms(fits, cfg) -> dict:
    """Which cost terms the setting's stop test flags, next to the counted fit's and to the world's true rules.
    Pair terms are counted once (4 offsets x 26 x 26), input terms over 2 x 26."""
    mine = RP.tables(fits[cfg["fit"]], cfg["rule"])
    count = RP.tables(fits["count"], "terms1")
    rules = BC.rule_tables()
    out = {}
    for name, key, sl in (("pair", "bad8", slice(0, 4)), ("input", "badB", slice(None))):
        a, c, r = (np.asarray(t[key])[sl] != 0 for t in (mine, count, rules))
        out[name] = {"flagged_by_this_fit": int(a.sum()), "flagged_by_the_counted_fit": int(c.sum()),
                     "forbidden_by_the_true_rules": int(r.sum()),
                     "flagged_by_this_fit_but_not_by_the_counted_fit": int((a & ~c).sum()),
                     "forbidden_by_the_true_rules_but_not_flagged_by_this_fit": int((r & ~a).sum()),
                     "forbidden_by_the_true_rules_but_not_flagged_by_the_counted_fit": int((r & ~c).sum())}
    return out


def main():
    torch.set_num_threads(4)
    out, started = RW.RESULTS / "paper_rules_safe.json", RW.RESULTS / "paper_rules_safe.started"
    if out.exists() or started.exists():
        raise SystemExit("refused: results/paper_rules_safe.json or .started exists; the check is run once")
    cfg = pick()
    print("setting", json.dumps(cfg), flush=True)
    started.write_text("started\n")
    fits = RW.load_fits()
    terms = flagged_terms(fits, cfg)
    print("terms", json.dumps(terms), flush=True)
    rows = []
    for name, (spec, _, _) in RC.worlds().items():
        codes = zobrist_codes(spec)
        held = torch.cat([RP.held_hashes(name), design_hash(RP.screen_designs(name)[0], codes)])
        for seed, n in ((RP.FRESH_SEED, RP.N_FRESH), (RW.FRESH_SEED, RW.N_FRESH)):   # the earlier fresh designs
            held = torch.cat([held, design_hash(StreamSource(spec, seed, held, chunk=8192)(n).clone(), codes)])
        designs = StreamSource(spec, FRESH_SEED, held, chunk=8192)(N_FRESH).clone()
        assert not bool(torch.isin(design_hash(designs, codes), held).any())
        for c in ({k: cfg[k] for k in ("fit", "rule", "temperature")}, RW.CONTROL):
            rows.append({**c, **RW.score(spec, designs, fits[c["fit"]], c["rule"], c["temperature"])})
            print(json.dumps(rows[-1]), flush=True)
    summary = []
    for c in ({k: cfg[k] for k in ("fit", "rule", "temperature")}, RW.CONTROL):
        mine = [r for r in rows if all(r[k] == c[k] for k in c)]
        tot = {k: sum(r[k] for r in mine) for k in ("n", "exact", "stopped_and_exact", "wrong_stops",
                                                    "stopped_valid_over_step_limit", "not_stopped",
                                                    "exact_but_not_stopped", "stops_not_confirmed_in_float64")}
        summary.append({**c, **tot})
        print(json.dumps(summary[-1]), flush=True)
    RC.dump({"post_hoc": True, "setting": cfg, "fresh_seed": FRESH_SEED, "n_fresh_per_world": N_FRESH,
             "chains": RW.CHAINS, "cap": RW.CAP, "flagged_terms": terms, "summary": summary, "rows": rows}, out)


if __name__ == "__main__":
    main()
