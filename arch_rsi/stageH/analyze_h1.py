"""Stage H1 analysis (PROTOCOL_H1.md, "Pass rules" and "Also recorded"). Reads runs/ and writes runs/results.json.

Pass rules, for each rule r in shift, mirror, torus, late (A = reference A, F = the rule's fix):
  1 repair     mean(r_F) - mean(r_A) >= 25 points, and Stage 1's exact two-sided permutation test (perm_p, imported
               unchanged) gives Holm-adjusted p < 0.05 over the four rules, with a positive difference. A rule judged on
               fallback runs (262,144 steps) has its p doubled (capped at 1) before Holm. An incomplete cell gives p = 1.
  2 specific   (mean(r_F) - mean(r_A)) - (mean(plain_F_r) - mean(plain_A)) >= 20 points (point estimate; a four-group
               bootstrap 95% interval is reported)
  3 legal      check_h.py passed for the fix in its rule world
  4 complete   all 24 runs of the rule's four cells finished "ok" (each failed run having been rerun once)
Fallback trigger (judged on the 131,072-step runs): damage mean(plain_A) - mean(r_A) >= 25 points and rule 1 fails.
Decision: 3 or 4 rules pass -> Step 2 and Step 3 possible; 2 -> Step 2 only; 0 or 1 -> stop.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/analyze_h1.py [--fallback shift,late]
"""
import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

from arch_rsi.stage1.analyze_stage1 import boot_diff, holm, perm_p  # noqa: E402,F401  (unchanged)

RULE_NAMES = ("shift", "mirror", "torus", "late")
SEEDS = tuple(range(4000, 4006))
BUDGET, FALLBACK = 131072, 262144
REPAIR_MIN, SPECIFIC_MIN, DAMAGE_MIN, ALPHA = 0.25, 0.20, 0.25, 0.05
EPS = 1e-9
RUNS = HERE / "runs"
REF_2C = [0.498, 0.709, 0.7188, 0.7266, 0.7617, 0.7266]  # 2c calibration, reference A on ws_dev at 131,072 steps


def cells_for(r: str) -> dict:
    return {"pA": "plain_A", "pF": f"plain_F_{r}", "rA": f"{r}_A", "rF": f"{r}_F"}


def all_cells() -> list:
    """(cell, rule, candidate) for the 13 cells; candidate "A" is reference A, otherwise the fix for that rule."""
    out = [("plain_A", "plain", "A")]
    for r in RULE_NAMES:
        out += [(f"plain_F_{r}", "plain", r), (f"{r}_A", r, "A"), (f"{r}_F", r, r)]
    return out


def group(results: dict, budget: int, cell: str) -> tuple:
    """(scores of the runs that finished ok, in seed order; whether all six did)."""
    rs = [results.get((budget, cell, s)) for s in SEEDS]
    vals = [r["dev_exact"] for r in rs if r and r.get("status") == "ok"]
    return vals, len(vals) == len(SEEDS)


def boot_interaction(rF, rA, pF, pA, draws: int = 10_000, seed: int = 0) -> list:
    """95% interval of (mean rF - mean rA) - (mean pF - mean pA), resampling each group with replacement
    (boot_diff's method and percentile indices, extended to four groups)."""
    rng = random.Random(seed)

    def m(x):
        return mean([rng.choice(x) for _ in x])

    vals = sorted((m(rF) - m(rA)) - (m(pF) - m(pA)) for _ in range(draws))
    return [round(vals[int(0.025 * draws)], 4), round(vals[int(0.975 * draws) - 1], 4)]


def evaluate(results: dict, checks: dict, fallback_rules=()) -> dict:
    """The pass rules and decision. results: {(budget, cell, seed): result}; checks: {rule: check_h output}."""
    per, raw_p = {}, {}
    for r in RULE_NAMES:
        b = FALLBACK if r in fallback_rules else BUDGET
        c = cells_for(r)
        g = {k: group(results, b, v) for k, v in c.items()}
        vals = {k: v[0] for k, v in g.items()}
        complete = all(v[1] for v in g.values())
        have = all(len(v) >= 1 for v in vals.values())
        mu = {k: (mean(v) if v else None) for k, v in vals.items()}
        repair = mu["rF"] - mu["rA"] if have else None
        p = perm_p(vals["rF"], vals["rA"]) if have and len(vals["rF"]) and len(vals["rA"]) else 1.0
        if not complete:
            p = 1.0
        p_used = min(1.0, 2 * p) if r in fallback_rules else p
        raw_p[r] = p_used
        plain_gain = mu["pF"] - mu["pA"] if have else None
        inter = repair - plain_gain if have else None
        damage = mu["pA"] - mu["rA"] if have else None
        per[r] = {
            "budget": b, "fallback": r in fallback_rules, "cells": c,
            "means": {k: (round(v, 4) if v is not None else None) for k, v in mu.items()},
            "per_seed": vals, "complete": complete,
            "repair": None if repair is None else round(repair, 4),
            "repair_ci95_boot": boot_diff(vals["rF"], vals["rA"]) if have else None,
            "p_raw": p, "p_used": p_used,
            "plain_gain": None if plain_gain is None else round(plain_gain, 4),
            "interaction": None if inter is None else round(inter, 4),
            "interaction_ci95_boot": boot_interaction(vals["rF"], vals["rA"], vals["pF"], vals["pA"]) if have else None,
            "damage": None if damage is None else round(damage, 4),
            "recovery_fraction": (round(repair / damage, 4) if have and damage and damage > 0 else None),
            "legal": bool(checks.get(r, {}).get("ok")), "check": checks.get(r),
        }
    adj = holm(raw_p)
    n_pass = 0
    for r in RULE_NAMES:
        d = per[r]
        d["p_holm"] = adj[r]
        d["pass_repair"] = (d["repair"] is not None and d["repair"] >= REPAIR_MIN - EPS and d["repair"] > 0
                            and d["p_holm"] < ALPHA)
        d["pass_specific"] = d["interaction"] is not None and d["interaction"] >= SPECIFIC_MIN - EPS
        d["pass_legal"] = d["legal"]
        d["pass_complete"] = d["complete"]
        d["passes"] = all((d["pass_repair"], d["pass_specific"], d["pass_legal"], d["pass_complete"]))
        d["fallback_trigger"] = (not d["fallback"] and d["damage"] is not None and d["damage"] >= DAMAGE_MIN - EPS
                                 and not d["pass_repair"])
        n_pass += d["passes"]
    decision = "step2_and_step3_possible" if n_pass >= 3 else "step2_only" if n_pass == 2 else "stop"
    return {"rules": per, "n_pass": n_pass, "passing": [r for r in RULE_NAMES if per[r]["passes"]],
            "decision": decision, "fallback_rules": list(fallback_rules)}


# ---- descriptives -----------------------------------------------------------------------------------------------

def load_results(runs: Path = RUNS) -> dict:
    out = {}
    for p in sorted(runs.glob("b*/*/result.json")):
        r = json.loads(p.read_text())
        out[(r["budget"], r["cell"], r["seed"])] = r
    return out


def load_checks(runs: Path = RUNS) -> dict:
    return {r: json.loads((runs / "checks" / f"{r}.json").read_text())
            for r in RULE_NAMES if (runs / "checks" / f"{r}.json").exists()}


def edge_share(yx: list) -> dict:
    """Share of wrong cells in the wrapped edge rows and columns (rows 0 and 7, columns 0 and 7) vs their 28/64 share."""
    Y, X = len(yx), len(yx[0])
    tot = sum(map(sum, yx))
    edge = sum(yx[y][x] for y in range(Y) for x in range(X) if y in (0, Y - 1) or x in (0, X - 1))
    return {"wrong_total": tot, "edge_share": round(edge / tot, 4) if tot else None,
            "board_edge_share": round((2 * X + 2 * Y - 4) / (X * Y), 4)}


def describe(results: dict, runs: Path, budgets) -> dict:
    out = {}
    for b in budgets:
        for cell, rule, cand in all_cells():
            rs = [results.get((b, cell, s)) for s in SEEDS]
            rs = [r for r in rs if r]
            if not rs:
                continue
            ok = [r for r in rs if r["status"] == "ok"]
            sc = [r["dev_exact"] for r in ok]
            yx = None
            for r in ok:
                m = r["wrong_cells_yx"]
                yx = m if yx is None else [[a + c for a, c in zip(u, v)] for u, v in zip(yx, m)]
            curves = []
            for r in ok:
                lp = runs / f"b{b}" / f"{cell}_s{r['seed']}" / r["dir"] / "run" / "train_log.json"
                if lp.exists():
                    curves.append(json.loads(lp.read_text()).get("curve", []))
            curve = []
            if curves and all(len(c) == len(curves[0]) for c in curves):
                for i, pt in enumerate(curves[0]):
                    curve.append({"steps": pt["steps"], "train_exact": round(mean(c[i]["train_exact"] for c in curves), 4)})
            hid = [r["exact_hidden"] for r in ok if r.get("exact_hidden") is not None]
            out[f"b{b}/{cell}"] = {
                "rule": rule, "candidate": cand, "n_ok": len(ok), "n": len(rs),
                "statuses": [r["status"] for r in rs], "attempts": [r["attempts"] for r in rs],
                "mean": round(mean(sc), 4) if sc else None, "sd": round(stdev(sc), 4) if len(sc) > 1 else None,
                "per_seed": dict(zip([r["seed"] for r in ok], sc)),
                "exact_hidden_mean": round(mean(hid), 4) if hid else None,
                "train_wall_s_median": sorted(r["train_wall_s"] for r in ok)[len(ok) // 2] if ok else None,
                "wrong_cells_yx": yx, "edge": edge_share(yx) if yx else None,
                "learning_curve": curve,
            }
    return out


def h1_identity(results: dict, runs: Path) -> dict:
    """shift_F against plain_A on the same seeds: identical inputs, so identical scores are expected."""
    rows = []
    for s in SEEDS:
        a, f = results.get((BUDGET, "plain_A", s)), results.get((BUDGET, "shift_F", s))
        if not (a and f and a["status"] == f["status"] == "ok"):
            rows.append({"seed": s, "comparable": False})
            continue
        da = json.loads((runs / f"b{BUDGET}" / f"plain_A_s{s}" / a["dir"] / "dev.json").read_text())["per_design"]
        df = json.loads((runs / f"b{BUDGET}" / f"shift_F_s{s}" / f["dir"] / "dev.json").read_text())["per_design"]
        rows.append({"seed": s, "comparable": True, "plain_A": a["dev_exact"], "shift_F": f["dev_exact"],
                     "same_score": a["dev_exact"] == f["dev_exact"], "same_per_design": da == df})
    return {"rows": rows, "all_identical": all(r.get("same_per_design") for r in rows)}


def analyze(runs: Path = RUNS, fallback_rules=()) -> dict:
    results, checks = load_results(runs), load_checks(runs)
    ev = evaluate(results, checks, fallback_rules)
    main_ev = evaluate(results, checks, ()) if fallback_rules else ev
    budgets = [BUDGET] + ([FALLBACK] if fallback_rules else [])
    pa, _ = group(results, BUDGET, "plain_A")
    return {
        "evaluation": ev,
        "main_budget_evaluation": main_ev,
        "fallback_triggered": [r for r in RULE_NAMES if main_ev["rules"][r]["fallback_trigger"]],
        "cells": describe(results, runs, budgets),
        "h1_identity": h1_identity(results, runs),
        "plain_A_vs_2c": {"plain_A_mean": round(mean(pa), 4) if pa else None, "ref_2c_mean": round(mean(REF_2C), 4),
                          "ref_2c_per_seed": REF_2C, "perm_p": perm_p(pa, REF_2C) if pa else None,
                          "note": "2c scored ws_dev on the CPU; plain_A scores hplain_dev on the GPU. Descriptive only."},
        "n_runs": len(results),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fallback", default="")
    ap.add_argument("--out", default=str(RUNS / "results.json"))
    args = ap.parse_args()
    fb = tuple(x for x in args.fallback.split(",") if x)
    res = analyze(RUNS, fb)
    Path(args.out).write_text(json.dumps(res, indent=1))
    ev = res["evaluation"]
    for r in RULE_NAMES:
        d = ev["rules"][r]
        print(r, d["means"], "repair", d["repair"], "p_holm", round(d["p_holm"], 4), "interaction", d["interaction"],
              "legal", d["legal"], "complete", d["complete"], "PASS" if d["passes"] else "no")
    print("decision:", ev["decision"], "| passing:", ev["passing"])
