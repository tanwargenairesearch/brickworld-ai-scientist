"""Step 2S analysis (PROTOCOL_STEP2S.md, "Analysis, fixed now"). analyze(cdir, step2_results) reads summary.json,
review.json, judge.json and the saved probes from a campaign directory; run_step2s.py writes the result to results.json.

Unit: one search, scored as in Step 2 (analyze_step2.rows, imported: mean of the pick's 4 confirmation seeds; a failed
seed scores 0; a pick the rule review judges a violation scores as the start). Primary, one test: P1S = sandbox -
scores only, pooled over the worlds that ran, with Step 2's permutation test shuffling arm labels within each world
(analyze_step2.contrast, 1,000,000 seeded shuffles); it passes if positive with p < 0.05. Precondition for reading P1S
as discovery: at least 75% (9 of 12) of the told-with-sandbox searches score >= 38.3%. Everything else is reported
with no decision: found per arm and world (Fisher), first round found, the judge's verdicts, probe use, dig.bench's
prediction (told_sandbox - sandbox in probes per search, permutation p), the drift check against Step 2's scores-only
arm, and calls and cost (probe results files are not counted as calls).
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2s/analyze_step2s.py --campaign DIR
"""
import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2s_common as C  # noqa: E402

import analyze_step2 as AN  # noqa: E402  (Step 2's)

ALPHA, PRE_SHARE = 0.05, 0.75
ENDINGS = {
    "pass": "Given a sandbox to experiment in, the AI scientist found fixes for hidden rules it wasn't told more often "
            "than with scores alone.",
    "pass_no_claim": "Sandbox searches scored higher than scores alone, but even told the rule it couldn't reliably "
                     "write the fix, so I make no claim about discovery.",
    "fail": "Experimenting in a sandbox didn't measurably help it find hidden rules here.",
}


def caveat(n_worlds: int) -> str:
    return f"{'One rule' if n_worlds == 1 else 'Two rules'} of one kind, toy scale, one model, whose weights never changed."


def probe_use(cdir: Path, rs: list, R: int = 4) -> dict:
    """Probes per search, placements per probe, searches with no probes; from each search's saved probe results."""
    per = {}
    for r in rs:
        sdir = cdir / r["world"] / r["arm"] / f"s{r['search']}"
        files = [sdir / f"round_{q}_probe_results.json" for q in range(1, R + 1)]
        got = [json.loads(f.read_text()) for f in files if f.exists()]
        per[(r["world"], r["arm"], r["search"])] = {"rounds": len(got), "probes": sum(len(g["probes"]) for g in got),
                                                    "placements": [len(p["placements"]) for g in got for p in g["probes"]]}
    out = {}
    for a in C.SANDBOX_ARMS:
        mine = [v for (w, arm, s), v in per.items() if arm == a]
        pl = [x for v in mine for x in v["placements"]]
        out[a] = {"searches": len(mine), "probes_per_search": [v["probes"] for v in mine],
                  "mean_probes_per_search": round(mean(v["probes"] for v in mine), 3) if mine else None,
                  "mean_placements_per_probe": round(mean(pl), 3) if pl else None,
                  "searches_with_no_probes": sum(v["probes"] == 0 for v in mine)}
    return {"per_arm": out, "per_search": {"/".join(map(str, k)): v for k, v in per.items()}}


def call_costs(cdir: Path) -> dict:
    """analyze_step2.call_costs, changed only to skip saved probe results (round_*_probe_results.json), which are not
    calls. Probe calls (round_*_probes.json) count with their arm, like proposal and repair calls."""
    out = {}
    for p in cdir.rglob("*.json"):
        rel = p.relative_to(cdir).parts
        if p.name.startswith("round_") and not p.name.endswith((".failed.json", "_probe_results.json")):
            key = rel[1]
        elif rel[0] in ("review", "judge"):
            key = rel[0]
        else:
            continue
        rec = json.loads(p.read_text())
        u = rec.get("usage") or {}
        c = out.setdefault(key, {"calls": 0, "cost_usd": 0.0})
        c["calls"] += 1
        c["cost_usd"] = round(c["cost_usd"] + (u.get("notional_cost_usd") or 0.0), 4)
    return out


def groups_of(per_search: dict, arm: str, worlds: list) -> dict:
    return {w: [v["probes"] for k, v in per_search.items() if k.split("/")[0] == w and k.split("/")[1] == arm] for w in worlds}


def analyze(cdir: Path, step2_results: dict = None, n_perm=AN.N_PERM) -> dict:
    cdir = Path(cdir)
    rs = AN.rows(cdir)
    worlds = sorted({r["world"] for r in rs})
    p1s = AN.contrast(rs, "sandbox", "scores", n_perm)
    p1s["passes"] = p1s["diff"] is not None and p1s["diff"] > 0 and p1s["p"] < ALPHA
    told = [r for r in rs if r["arm"] == "told_sandbox"]
    need = math.ceil(PRE_SHARE * len(told))
    pre = {"found": sum(r["found"] for r in told), "n": len(told), "needed": need,
           "holds": len(told) > 0 and sum(r["found"] for r in told) >= need}
    jud = json.loads((cdir / "judge.json").read_text()) if (cdir / "judge.json").exists() else {}
    per_arm = {}
    for a in C.ARMS:
        g = AN.groups(rs, a)
        vs = [v["verdict"] for k, v in jud.items() if k.split("/")[1] == a]
        per_arm[a] = {"mean": round(mean(v for w in g for v in g[w]), 4) if any(g.values()) else None,
                      "by_world": {w: round(mean(v), 4) if v else None for w, v in g.items()},
                      "found": sum(r["found"] for r in rs if r["arm"] == a), "n": sum(len(v) for v in g.values()),
                      "found_by_world": {w: sum(r["found"] for r in rs if r["arm"] == a and r["world"] == w) for w in worlds},
                      "first_round_found": [r["first_round_found"] for r in rs if r["arm"] == a],
                      "picks_start": sum(r["label"] == "start" for r in rs if r["arm"] == a),
                      "violations": sum(r["violation"] for r in rs if r["arm"] == a),
                      "judge": {x: vs.count(x) for x in ("yes", "partly", "no")}}
    pu = probe_use(cdir, rs)
    ga, gb = groups_of(pu["per_search"], "told_sandbox", worlds), groups_of(pu["per_search"], "sandbox", worlds)
    if all(ga[w] and gb[w] for w in worlds):
        d = mean(v for w in worlds for v in ga[w]) - mean(v for w in worlds for v in gb[w])
        pred = {"diff_told_minus_sandbox": round(d, 4), "p": AN.perm_p(ga, gb, n_perm),
                "as_predicted": d < 0, "note": "dig.bench: a player who knows the rules hardly uses creative mode"}
    else:
        pred = {"diff_told_minus_sandbox": None, "p": None, "note": "an arm has no searches"}
    drift = None
    if step2_results is not None:
        g2 = {w: [r["score"] for r in step2_results["rows"] if r["arm"] == "scores" and r["world"] == w] for w in worlds}
        g1 = AN.groups(rs, "scores")
        if all(g1.get(w) and g2[w] for w in worlds):
            d = mean(v for w in worlds for v in g1[w]) - mean(v for w in worlds for v in g2[w])
            drift = {"diff_step2s_minus_step2": round(d, 4), "p": AN.perm_p(g1, g2, n_perm), "ci95_boot": AN.boot(g1, g2),
                     "found": [sum(v >= C.FOUND for w in worlds for v in g1[w]), sum(v >= C.FOUND for w in worlds for v in g2[w])],
                     "n": [sum(map(len, g1.values())), sum(map(len, g2.values()))]}
    ending = ("pass" if pre["holds"] else "pass_no_claim") if p1s["passes"] else "fail"
    fd = lambda x: "n/a" if x is None else f"{x:+.3f}"  # noqa: E731
    line = (f"P1S {fd(p1s['diff'])} (p {p1s['p']:.4g}, {'passes' if p1s['passes'] else 'fails'}); told with sandbox found "
            f"{pre['found']}/{pre['n']} (precondition {'holds' if pre['holds'] else 'fails'})")
    return {"rows": rs, "worlds": worlds, "primary": {"P1S": p1s}, "precondition": pre, "per_arm": per_arm,
            "probes": pu, "digbench_prediction": pred, "drift": drift, "ending": ending,
            "ending_text": ENDINGS[ending] + " " + caveat(len(worlds)), "discovery_claim_allowed": pre["holds"],
            "all_reviewed": all(r["reviewed"] for r in rs), "calls": call_costs(cdir), "summary_line": line}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    s2 = json.loads(C.STEP2_RESULTS.read_text()) if C.STEP2_RESULTS.exists() else None
    res = analyze(Path(args.campaign), s2)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1)[:4000])
