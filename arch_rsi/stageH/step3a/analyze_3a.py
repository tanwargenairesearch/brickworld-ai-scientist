"""Step 3a analysis (PROTOCOL_STEP3A.md, "Pass rules", "What happens next" and "Also recorded"). Reads runs/ and
writes runs/results.json.

Stage H1's pass rules and thresholds, unchanged, for each new rule r (A = reference A, F = the rule's fix):
  1 repair     mean(r_F) - mean(r_A) >= 25 points and Stage 1's exact two-sided permutation test (perm_p, imported
               unchanged) gives Holm-adjusted p < 0.05 over the eight new rules, with a positive difference. A rule judged
               on fallback runs (262,144 steps) has its p doubled (capped at 1) before Holm. An incomplete cell gives p = 1.
  2 specific   (mean(r_F) - mean(r_A)) - (mean(plain_F_r) - mean(plain_A)) >= 20 points (point estimate; Stage H1's
               four-group bootstrap 95% interval, imported unchanged, is reported)
  3 legal      check_3a.py passed for the fix in its rule world
  4 complete   all 24 runs of the rule's four cells finished "ok" (each failed run having been rerun once)
Fallback trigger (judged on the 131,072-step runs): damage mean(plain_A) - mean(r_A) >= 25 points and rule 1 fails.
Families: usable for Step 3 with at least two passing members, counting H1 and H2, which passed in Step 1.
Decision: 0 or 1 new rule passes -> "stop"; otherwise 3 or 4 usable families -> "step3_can_hold_out_a_family", and
fewer -> "options_back_to_user" (the protocol's "2 usable families" case, read as "fewer than 3").
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3a/analyze_3a.py [--fallback handshift,vmirror]
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common  # noqa: E402,F401

import rules3a as R3  # noqa: E402
from analyze_h1 import boot_interaction  # noqa: E402  (Stage H1's, unchanged)
from arch_rsi.stage1.analyze_stage1 import boot_diff, holm, perm_p  # noqa: E402  (unchanged)

RULE_NAMES = R3.NEW
SEEDS = tuple(range(4000, 4006))
BUDGET, FALLBACK = 131072, 262144
REPAIR_MIN, SPECIFIC_MIN, DAMAGE_MIN, ALPHA = 0.25, 0.20, 0.25, 0.05
EPS = 1e-9
RUNS = HERE / "runs"
H1_RUNS = HERE.parent / "runs"  # Step 1's runs, read only: the plain cell's reproduction check


def cells_for(r: str) -> dict:
    return {"pA": "plain_A", "pF": f"plain_F_{r}", "rA": f"{r}_A", "rF": f"{r}_F"}


def all_cells() -> list:
    """(cell, rule, candidate) for the 25 cells; candidate "A" is reference A, otherwise the fix for that rule."""
    out = [("plain_A", "plain", "A")]
    for r in RULE_NAMES:
        out += [(f"plain_F_{r}", "plain", r), (f"{r}_A", r, "A"), (f"{r}_F", r, r)]
    return out


def group(results: dict, budget: int, cell: str) -> tuple:
    """(scores of the runs that finished ok, in seed order; whether all six did)."""
    rs = [results.get((budget, cell, s)) for s in SEEDS]
    vals = [r["dev_exact"] for r in rs if r and r.get("status") == "ok"]
    return vals, len(vals) == len(SEEDS)


def families(passing_new) -> dict:
    out = {}
    for f in R3.FAMILIES:
        members = [r for r in R3.FAMILY if R3.FAMILY[r] == f]
        passing = [r for r in members if r in R3.PASSED_BEFORE or r in passing_new]
        out[f] = {"members": [R3.IDS[r] for r in members], "passing": [R3.IDS[r] for r in passing],
                  "usable": len(passing) >= 2}
    return out


def evaluate(results: dict, checks: dict, fallback_rules=()) -> dict:
    """The pass rules, the families and the decision. results: {(budget, cell, seed): result}; checks: {rule: check}."""
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
        p = perm_p(vals["rF"], vals["rA"]) if have else 1.0
        if not complete:
            p = 1.0
        p_used = min(1.0, 2 * p) if r in fallback_rules else p
        raw_p[r] = p_used
        plain_gain = mu["pF"] - mu["pA"] if have else None
        inter = repair - plain_gain if have else None
        damage = mu["pA"] - mu["rA"] if have else None
        per[r] = {
            "id": R3.IDS[r], "family": R3.FAMILY[r], "budget": b, "fallback": r in fallback_rules, "cells": c,
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
    passing = [r for r in RULE_NAMES if per[r]["passes"]]
    fam = families(passing)
    usable = [f for f in R3.FAMILIES if fam[f]["usable"]]
    if len(passing) <= 1:
        decision = "stop"
    elif len(usable) >= 3:
        decision = "step3_can_hold_out_a_family"
    else:
        decision = "options_back_to_user"
    return {"rules": per, "n_pass": len(passing), "passing": passing, "passing_ids": [R3.IDS[r] for r in passing],
            "families": fam, "usable_families": usable, "decision": decision, "fallback_rules": list(fallback_rules)}


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


def describe(results: dict, runs: Path, budgets) -> dict:
    out = {}
    for b in budgets:
        for cell, rule, cand in all_cells():
            rs = [r for r in (results.get((b, cell, s)) for s in SEEDS) if r]
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
                "wrong_cells_yx": yx,
                "wrong_by_row": [sum(row) for row in yx] if yx else None,
                "wrong_by_col": [sum(col) for col in zip(*yx)] if yx else None,
                "learning_curve": curve,
            }
    return out


def per_design(runs: Path, budget: int, cell: str, res: dict):
    p = runs / f"b{budget}" / f"{cell}_s{res['seed']}" / res["dir"] / "dev.json"
    return json.loads(p.read_text())["per_design"] if p.exists() else None


def identity(results: dict, runs: Path, h1_runs: Path = H1_RUNS) -> dict:
    """Descriptive: the rerun plain cell against Step 1's on the same seeds, and each booklet rule's fixed cell against
    the plain cell (the fix gives the network exactly the plain input, so identical results are expected)."""
    h1 = load_results(h1_runs)
    rows = []
    for s in SEEDS:
        a, o = results.get((BUDGET, "plain_A", s)), h1.get((BUDGET, "plain_A", s))
        ok = bool(a and o and a["status"] == o["status"] == "ok")
        rows.append({"seed": s, "comparable": ok, **({"here": a["dev_exact"], "step1": o["dev_exact"],
                     "same_per_design": per_design(runs, BUDGET, "plain_A", a) == per_design(h1_runs, BUDGET, "plain_A", o)}
                     if ok else {})})
    book = {}
    for r in R3.BOOKLET:
        brow = []
        for s in SEEDS:
            a, f = results.get((BUDGET, "plain_A", s)), results.get((BUDGET, f"{r}_F", s))
            ok = bool(a and f and a["status"] == f["status"] == "ok")
            brow.append({"seed": s, "comparable": ok, **({"plain_A": a["dev_exact"], "fixed": f["dev_exact"],
                         "same_per_design": per_design(runs, BUDGET, "plain_A", a) == per_design(runs, BUDGET, f"{r}_F", f)}
                         if ok else {})})
        book[r] = {"rows": brow, "all_identical": all(x.get("same_per_design") for x in brow)}
    return {"plain_reproduces_step1": {"rows": rows, "all_identical": all(x.get("same_per_design") for x in rows)},
            "booklet_fix_matches_plain": book}


def analyze(runs: Path = RUNS, fallback_rules=(), h1_runs: Path = H1_RUNS) -> dict:
    results, checks = load_results(runs), load_checks(runs)
    ev = evaluate(results, checks, fallback_rules)
    main_ev = evaluate(results, checks, ()) if fallback_rules else ev
    budgets = [BUDGET] + ([FALLBACK] if fallback_rules else [])
    return {
        "evaluation": ev,
        "main_budget_evaluation": main_ev,
        "fallback_triggered": [r for r in RULE_NAMES if main_ev["rules"][r]["fallback_trigger"]],
        "cells": describe(results, runs, budgets),
        "identity": identity(results, runs, h1_runs),
        "n_runs": len(results),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fallback", default="")
    ap.add_argument("--out", default=str(RUNS / "results.json"))
    args = ap.parse_args()
    res = analyze(RUNS, tuple(x for x in args.fallback.split(",") if x))
    Path(args.out).write_text(json.dumps(res, indent=1))
    ev = res["evaluation"]
    for r in RULE_NAMES:
        d = ev["rules"][r]
        print(d["id"], r, d["means"], "repair", d["repair"], "p_holm", round(d["p_holm"], 4), "interaction",
              d["interaction"], "legal", d["legal"], "complete", d["complete"], "PASS" if d["passes"] else "no")
    print("families:", {f: v["passing"] for f, v in ev["families"].items()}, "| decision:", ev["decision"])
