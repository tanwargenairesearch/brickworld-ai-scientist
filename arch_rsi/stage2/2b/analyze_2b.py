"""Pre-registered analysis of Stage 2b (PROTOCOL_2B.md). Written before any Stage 2b result.

usage (project root): .venv/bin/python -I arch_rsi/stage2/2b/analyze_2b.py --campaign arch_rsi/campaigns/stage2b
Reads W's dev results only (the final split stays locked). Writes results.json in the campaign directory.

Unit: one search, scored as the mean over its pick's 8 confirmation seeds (a failed seed scores 0). Comparators are
scored per seed. Primary score: a pick the rule review (review_2b.py) judges to break a rule is scored as the start
with Stage G's recipe on the same seeds; the as-computed score is reported beside it. Without review.json the primary
family is "pending review". Tests: Stage 1's exact permutation test, Holm within each family, bootstrap intervals.
Primary family: g0 - nofb (the gate), g0 - refA. Gate: g0 - nofb positive with Holm-adjusted p < 0.05.
"""
import argparse
import ast
import importlib.util
import json
import statistics
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
S2 = HERE.parent
A = S2.parent
ROOT = A.parent
spec_ = importlib.util.spec_from_file_location("analyze_stage1", A / "stage1" / "analyze_stage1.py")
AS1 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(AS1)

PRIMARY = [("g0", "nofb"), ("g0", "refA")]
SECONDARY = [("g0", "netonly"), ("g0", "reconly"), ("g0", "random"), ("g0", "start_g"), ("nofb", "refA")]
BY_CONSTRUCTION = ("random", "reconly")  # menu networks and the start's network: compliant by construction
mean = AS1.mean


def hidden_flags() -> np.ndarray:
    v = torch.load(S2 / "targets" / "w_dev.pt").bool()
    above = (v.flip(1).cumsum(1).flip(1) - v.int()) > 0
    return ((~v) & above).flatten(1).any(1).numpy()


def recipe_of(src: str):
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RECIPE" for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def analyse(cdir: Path, entries: list, review, H) -> dict:
    sk = str
    comp = {e["arm"]: e for e in entries if e["kind"] == "comparator"}
    start_by_seed = {sk(s): (comp["start_g"]["dev"].get(sk(s)) or 0.0) for s in comp["start_g"]["seeds"]}

    def seed_scores(e):
        return [(e["dev"].get(sk(s)) if e["status"].get(sk(s)) == "ok" else 0.0) or 0.0 for s in e["seeds"]]

    as_computed, primary, detail = {}, {}, {}
    for e in entries:
        sc = seed_scores(e)
        det = [AS1.seed_detail(cdir, e["sha"], s, H) if e["status"].get(sk(s)) == "ok" else {"hidden": 0.0, "learned": False}
               for s in e["seeds"]]
        timeouts = sum(e["status"].get(sk(s)) == "timeout" for s in e["seeds"])
        if e["kind"] == "comparator":
            as_computed[e["arm"]] = primary[e["arm"]] = sc
            detail[e["arm"]] = {"per_seed": [round(x, 4) for x in sc], "mean": round(mean(sc), 4),
                                "sd": round(statistics.stdev(sc), 4) if len(sc) > 1 else None,
                                "seeds_learned": sum(d["learned"] for d in det), "seeds": len(sc), "timeouts": timeouts}
            continue
        verdict = (review or {}).get(e["sha"], {}).get("verdict", "pending" if review is None else "compliant")
        if e["label"] == "start" or e["arm"] in BY_CONSTRUCTION:
            verdict = "compliant (by construction)"
        x = mean(sc)
        y = mean([start_by_seed[sk(s)] for s in e["seeds"]]) if verdict == "violation" else x
        as_computed.setdefault(e["arm"], []).append(x)
        primary.setdefault(e["arm"], []).append(y)
        src = (cdir / "confirm" / e["sha"] / "candidate.py").read_text()
        d = detail.setdefault(e["arm"], {"searches": []})
        d["searches"].append({"search": e["search"], "pick": e["label"], "name": e["name"], "sha": e["sha"],
                              "verdict": verdict, "score": round(x, 4), "primary_score": round(y, 4),
                              "selection_mean": round(e["mean_selected"], 4), "per_seed": [round(v, 4) for v in sc],
                              "seeds_learned": sum(q["learned"] for q in det), "timeouts": timeouts,
                              "recipe": recipe_of(src)})
    for a, d in detail.items():
        if "searches" not in d:
            continue
        ss = d["searches"]
        d |= {"mean": round(mean(as_computed[a]), 4), "primary_mean": round(mean(primary[a]), 4),
              "sd_between_searches": round(statistics.stdev(as_computed[a]), 4) if len(ss) > 1 else None,
              "share_seeds_learned": round(sum(s["seeds_learned"] for s in ss) / sum(len(s["per_seed"]) for s in ss), 4),
              "selection_optimism": round(mean([s["selection_mean"] - s["score"] for s in ss]), 4),
              "picked_start": sum(s["pick"] == "start" for s in ss),
              "picks_with_recipe": sum(s["recipe"] is not None for s in ss),
              "violations": sum(s["verdict"] == "violation" for s in ss)}
    pending = review is None
    prim = AS1.family(primary, PRIMARY)
    g = prim.get("g0 - nofb")
    gate = None if (pending or g is None) else (g["diff"] > 0 and g["p_holm"] < 0.05)
    return {"per_arm": detail,
            "primary_family": {"status": "pending review" if pending else "final", "contrasts": prim, "gate_passed": gate},
            "as_computed_primary_family": AS1.family(as_computed, PRIMARY),
            "secondary_family": AS1.family(as_computed if pending else primary, SECONDARY)}


def search_curves(cdir: Path, arms: list) -> dict:
    out = {}
    for a in arms:
        rounds, invalid, failed, timeouts, trials = {}, 0, 0, 0, 0
        for st_file in sorted((cdir / "W" / a).glob("s*/state.json")):
            st = json.loads(st_file.read_text())
            best = 0.0
            for r in sorted({t["round"] for t in st["trials"]}):
                ok = [t["result"]["dev_exact"] for t in st["trials"] if t["round"] == r and t["result"]["status"] == "ok"]
                rb = max(ok, default=0.0)
                best = max(best, rb)
                rounds.setdefault(r, {"round_best": [], "best_so_far": []})
                rounds[r]["round_best"].append(rb)
                rounds[r]["best_so_far"].append(best)
            trials += len(st["trials"])
            invalid += sum(t["result"]["status"] == "invalid" for t in st["trials"])
            failed += sum(t["result"]["status"] not in ("ok", "invalid") for t in st["trials"])
            timeouts += sum(t["result"]["status"] == "timeout" for t in st["trials"])
        out[a] = {"per_round": {r: {k: round(mean(v), 4) for k, v in d.items()} for r, d in sorted(rounds.items())},
                  "trials": trials, "invalid": invalid, "failed_training": failed, "timeouts": timeouts}
    return out


def costs(cdir: Path, arms: list) -> dict:
    out = {}
    for a in arms:
        wall = calls = tok_out = 0
        usd = 0.0
        for sdir in sorted((cdir / "W" / a).glob("s*")):
            for f in sdir.glob("trials/*/seed*/result.json"):
                wall += json.loads(f.read_text()).get("wall_s") or 0
            for f in sdir.glob("round_*.json"):
                if f.name.endswith(".failed.json"):
                    continue
                u = json.loads(f.read_text()).get("usage", {})
                calls += 1
                usd += u.get("notional_cost_usd") or 0.0
                tok_out += u.get("output_tokens") or 0
        out[a] = {"search_training_wall_hours": round(wall / 3600, 2), "claude_calls": calls,
                  "claude_notional_usd": round(usd, 2), "output_tokens": tok_out}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    cdir = Path(args.campaign).resolve()
    if (S2 / "FINAL_OPENED_W").exists():
        raise SystemExit("FINAL_OPENED_W exists: Stage 2b must not open W's final split")
    man = json.loads((cdir / "manifest.json").read_text())
    entries = json.loads((cdir / "confirm" / "summary.json").read_text())
    rv = cdir / "review.json"
    review = json.loads(rv.read_text())["verdicts"] if rv.exists() else None
    H = hidden_flags()
    res = {"campaign": str(cdir.relative_to(ROOT)), "preset": man["preset"], "split": "dev (W final locked)",
           "learned_threshold": AS1.LEARNED, "dev_designs_with_hidden_cells": int(H.sum()),
           "baseline": json.loads((cdir / "baseline" / "summary.json").read_text())}
    res |= analyse(cdir, entries, review, H)
    arms = man["config"]["arms"]
    res["search_curves"] = search_curves(cdir, arms)
    res["costs"] = costs(cdir, arms)
    (cdir / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("primary_family", "secondary_family")}, indent=1))


if __name__ == "__main__":
    main()
