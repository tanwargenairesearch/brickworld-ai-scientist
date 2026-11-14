"""Step 2 analysis (PROTOCOL_STEP2.md, "Analysis, fixed now"). analyze(cdir) reads summary.json, review.json and
judge.json from a campaign directory and returns the results (run_step2.py writes them to results.json).

Unit: one search, scored by the mean of its pick's 4 confirmation seeds (a failed seed scores 0; a pick the rule review
judges a violation scores as the start). Contrasts pool both worlds: the difference in mean score between two arms,
with a two-sided permutation test that shuffles arm labels within each world (1,000,000 seeded random shuffles) and a
bootstrap 95% interval (resampling within world and arm). Primary, Holm over 2 (Stage 1's holm, unchanged):
P1 readouts - hidden, P2 readouts - scores; each passes if positive with Holm-adjusted p < 0.05. Check: at least 75%
of the told arm's searches score >= 38.3%. Secondary, Holm over 2: told - readouts, scores - hidden.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2/analyze_step2.py --campaign DIR
"""
import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as C  # noqa: E402

import numpy as np  # noqa: E402

from arch_rsi.stage1.analyze_stage1 import holm  # noqa: E402

N_PERM, N_BOOT, ALPHA, CHECK_SHARE = 1_000_000, 10_000, 0.05, 0.75
PRIMARY = {"P1": ("readouts", "hidden"), "P2": ("readouts", "scores")}
SECONDARY = {"S1": ("told", "readouts"), "S2": ("scores", "hidden")}
ENDINGS = {
    "both": "With readouts of its own runs, the AI scientist found fixes for hidden rules it wasn't told, more often than with scores only or with none.",
    "p1": "With readouts of its own runs, the AI scientist found fixes for hidden rules it wasn't told, more often than with no scores. Readouts beyond the score weren't shown to matter.",
    "none": "It didn't measurably find hidden rules from evidence here.",
}
CAVEAT = "Two rules of one kind, toy scale, one model, whose weights never changed."


def rows(cdir: Path) -> list:
    summ = json.loads((cdir / "summary.json").read_text())
    rev = json.loads((cdir / "review.json").read_text())["verdicts"] if (cdir / "review.json").exists() else {}
    out = []
    for e in summ["entries"]:
        devs = [e["dev"][s] if e["status"][s] == "ok" else 0.0 for s in sorted(e["dev"])]
        sc = mean(devs)
        viol = e["label"] != "start" and rev.get(e["sha"], {}).get("verdict") == "violation"
        if viol:
            sc = mean(summ["starts"][e["world"]]["confirm"].values())
        out.append({"world": e["world"], "arm": e["arm"], "search": e["search"], "label": e["label"], "name": e["name"],
                    "score": sc, "found": sc >= C.FOUND, "violation": viol, "first_round_found": e.get("first_round_found"),
                    "reviewed": e["label"] == "start" or e["sha"] in rev})
    return out


def groups(rs, arm):
    return {w: [r["score"] for r in rs if r["arm"] == arm and r["world"] == w] for w in sorted({r["world"] for r in rs})}


def perm_p(ga: dict, gb: dict, n=N_PERM, seed=0) -> float:
    """Two-sided permutation p for pooled mean(a) - mean(b), shuffling arm labels within each world."""
    rng = np.random.default_rng(seed)
    NA = sum(len(v) for v in ga.values())
    NB = sum(len(v) for v in gb.values())
    tot = sum(sum(ga[w]) + sum(gb[w]) for w in ga)
    obs = sum(map(sum, ga.values())) / NA - sum(map(sum, gb.values())) / NB
    hits, done = 0, 0
    while done < n:
        m = min(100_000, n - done)
        sa = np.zeros(m)
        for w in ga:
            vals = np.array(ga[w] + gb[w], dtype=float)
            idx = rng.random((m, len(vals))).argsort(axis=1)[:, :len(ga[w])]
            sa += vals[idx].sum(axis=1)
        d = sa / NA - (tot - sa) / NB
        hits += int((np.abs(d) >= abs(obs) - 1e-12).sum())
        done += m
    return (hits + 1) / (n + 1)


def boot(ga: dict, gb: dict, draws=N_BOOT, seed=0) -> list:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        a = np.concatenate([rng.choice(ga[w], len(ga[w])) for w in ga])
        b = np.concatenate([rng.choice(gb[w], len(gb[w])) for w in gb])
        vals.append(a.mean() - b.mean())
    vals.sort()
    return [round(float(vals[int(0.025 * draws)]), 4), round(float(vals[int(0.975 * draws) - 1]), 4)]


def fisher(a, na, b, nb) -> float:
    k, N = a + b, na + nb

    def p(x):
        return math.comb(na, x) * math.comb(nb, k - x) / math.comb(N, k)
    obs = p(a)
    return sum(p(x) for x in range(max(0, k - nb), min(k, na) + 1) if p(x) <= obs * (1 + 1e-9))


def contrast(rs, a, b, n_perm) -> dict:
    ga, gb = groups(rs, a), groups(rs, b)
    if not all(ga.get(w) and gb.get(w) for w in set(ga) | set(gb)):
        return {"arms": [a, b], "diff": None, "p": 1.0, "ci95_boot": None, "per_world_found": {}, "note": "an arm has no searches"}
    d = mean(v for w in ga for v in ga[w]) - mean(v for w in gb for v in gb[w])
    fw = {}
    for w in ga:
        fa = sum(r["found"] for r in rs if r["arm"] == a and r["world"] == w)
        fb = sum(r["found"] for r in rs if r["arm"] == b and r["world"] == w)
        fw[w] = {"found": [fa, fb], "n": [len(ga[w]), len(gb[w])], "fisher_p": round(fisher(fa, len(ga[w]), fb, len(gb[w])), 4)}
    return {"arms": [a, b], "diff": round(d, 4), "p": perm_p(ga, gb, n_perm), "ci95_boot": boot(ga, gb), "per_world_found": fw}


def call_costs(cdir: Path) -> dict:
    out = {}
    for p in cdir.rglob("*.json"):
        rel = p.relative_to(cdir).parts
        if p.name.startswith("round_") and not p.name.endswith(".failed.json"):
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


def analyze(cdir: Path, n_perm=N_PERM) -> dict:
    cdir = Path(cdir)
    rs = rows(cdir)
    prim = {k: contrast(rs, a, b, n_perm) for k, (a, b) in PRIMARY.items()}
    adj = holm({k: v["p"] for k, v in prim.items()})
    for k, v in prim.items():
        v["p_holm"] = adj[k]
        v["passes"] = v["diff"] is not None and v["diff"] > 0 and adj[k] < ALPHA
    sec = {k: contrast(rs, a, b, n_perm) for k, (a, b) in SECONDARY.items()}
    adj2 = holm({k: v["p"] for k, v in sec.items()})
    for k, v in sec.items():
        v["p_holm"] = adj2[k]
    told = [r for r in rs if r["arm"] == "told"]
    need = math.ceil(CHECK_SHARE * len(told))
    check = {"found": sum(r["found"] for r in told), "n": len(told), "needed": need,
             "passes": len(told) > 0 and sum(r["found"] for r in told) >= need}
    ending = "both" if prim["P1"]["passes"] and prim["P2"]["passes"] else "p1" if prim["P1"]["passes"] else "none"
    per_arm = {}
    for a in C.ARMS:
        g = groups(rs, a)
        per_arm[a] = {"mean": round(mean(v for w in g for v in g[w]), 4) if any(g.values()) else None,
                      "by_world": {w: round(mean(v), 4) if v else None for w, v in g.items()},
                      "found": sum(r["found"] for r in rs if r["arm"] == a), "n": sum(len(v) for v in g.values()),
                      "first_round_found": [r["first_round_found"] for r in rs if r["arm"] == a],
                      "picks_start": sum(r["label"] == "start" for r in rs if r["arm"] == a),
                      "violations": sum(r["violation"] for r in rs if r["arm"] == a)}
    jud = json.loads((cdir / "judge.json").read_text()) if (cdir / "judge.json").exists() else {}
    for a in C.ARMS:
        vs = [v["verdict"] for k, v in jud.items() if k.split("/")[1] == a]
        per_arm[a]["judge"] = {x: vs.count(x) for x in ("yes", "partly", "no")}
    fd = lambda x: "n/a" if x is None else f"{x:+.3f}"  # noqa: E731
    line = (f"P1 {fd(prim['P1']['diff'])} (Holm p {prim['P1']['p_holm']:.4g}), P2 {fd(prim['P2']['diff'])} "
            f"(Holm p {prim['P2']['p_holm']:.4g}); told found {check['found']}/{check['n']} (check {'passes' if check['passes'] else 'fails'})")
    return {"rows": rs, "primary": prim, "secondary": sec, "check": check, "per_arm": per_arm, "ending": ending,
            "ending_text": ENDINGS[ending] + " " + CAVEAT, "discovery_claim_allowed": check["passes"],
            "all_reviewed": all(r["reviewed"] for r in rs), "calls": call_costs(cdir), "summary_line": line}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    res = analyze(Path(args.campaign))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1)[:4000])
