"""Pre-registered analysis of Stage 2e (PROTOCOL_2E.md). Written before any Stage 2e result.

usage (project root): .venv/bin/python -I arch_rsi/stage2/2e/analyze_2e.py --campaign arch_rsi/campaigns/stage2e
Reads W's dev results only (the final split stays locked). Writes results.json in the campaign directory.

Unit: one pair of searches that share round 1. A branch scores the number of exact builds over its pick's
confirmation seeds (512 dev designs per seed; a failed seed scores 0). A pick the rule review (review_2e.py) judges
to break a rule is scored as the start on the same seeds; the as-computed score is reported beside it.
d = seen - hidden, per pair. One primary test: the exact two-sided sign-flip test on the mean of d, over all 2^N
sign patterns, computed on the integer build counts. Gate: mean d > 0 and p < 0.05. No second primary test.
Interval: bootstrap 95% over pairs (10,000 draws, seed 0). Readings: pass / worse / bounded (upper end below +10
points) / open. Without a verdict for every pick the status is "pending review" and no gate is declared.

run_2e.py imports the pure functions here for the second-stage rule, which reads only the sizes of the differences.
"""
import argparse
import importlib.util
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S2 = HERE.parent
A = S2.parent
ROOT = A.parent
_spec = importlib.util.spec_from_file_location("analyze_2b", S2 / "2b" / "analyze_2b.py")
AN2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN2)          # frozen, unchanged: hidden_flags, recipe_of, and Stage 1's AS1.seed_detail
AS1 = AN2.AS1

N_DESIGNS = 512
BRANCHES = ("seen", "hidden")
ALPHA = 0.05
BOUND = 0.10            # reading "bounded": the interval's upper end is below +10 points
mean = AS1.mean


# ---------------------------------------------------------------- pure helpers (tested in test_analyze_2e.py)
def builds(x: float) -> int:
    """A dev score as a whole number of exact builds out of 512."""
    k = round(x * N_DESIGNS)
    if abs(k / N_DESIGNS - x) > 1e-9:
        raise ValueError(f"{x} is not a whole number of builds out of {N_DESIGNS}")
    return k


def seed_builds(e: dict) -> list:
    return [builds(e["dev"][str(s)]) if e["status"][str(s)] == "ok" else 0 for s in e["seeds"]]


def verdict_of(e: dict, verdicts) -> str:
    if e["label"] == "start":
        return "compliant (by construction)"
    return (verdicts or {}).get(e["sha"], {}).get("verdict", "pending")


def pair_rows(entries: list, verdicts) -> dict:
    """Per pair: each branch's as-computed and primary build counts, and d = seen - hidden (integers)."""
    comp = next(e for e in entries if e["kind"] == "comparator" and e["arm"] == "start_g")
    start = dict(zip((str(s) for s in comp["seeds"]), seed_builds(comp)))
    by = {}
    for e in entries:
        if e["kind"] == "search":
            by.setdefault(e["pair"], {})[e["arm"]] = e
    pairs = sorted(i for i, v in by.items() if all(b in v for b in BRANCHES))
    if pairs != list(range(1, len(pairs) + 1)):
        raise ValueError(f"finished pairs are not 1..N: {pairs}")
    rows, n_seeds = [], None
    for i in pairs:
        row = {"pair": i}
        for b in BRANCHES:
            e = by[i][b]
            n_seeds = n_seeds or len(e["seeds"])
            if len(e["seeds"]) != n_seeds or any(str(s) not in start for s in e["seeds"]):
                raise ValueError(f"pair {i} {b}: confirmation seeds differ from the start's")
            v = verdict_of(e, verdicts)
            as_c = sum(seed_builds(e))
            prim = sum(start[str(s)] for s in e["seeds"]) if v == "violation" else as_c
            row[b] = {"pick": e["label"], "name": e["name"], "sha": e["sha"], "verdict": v, "builds": as_c,
                      "primary_builds": prim, "per_seed_builds": seed_builds(e), "selection_mean": e["mean_selected"]}
        row["d_builds"] = row["seen"]["primary_builds"] - row["hidden"]["primary_builds"]
        row["d_builds_as_computed"] = row["seen"]["builds"] - row["hidden"]["builds"]
        row["same_pick"] = row["seen"]["sha"] == row["hidden"]["sha"]
        rows.append(row)
    pending = sum(r[b]["verdict"] == "pending" for r in rows for b in BRANCHES)
    return {"rows": rows, "denom": N_DESIGNS * (n_seeds or 1), "pending": pending, "start_builds": start}


def pick_round(label: str) -> int:
    """The round a pick was proposed in: 0 for the start, else the r in "r<r>c<c>"."""
    return 0 if label == "start" else int(label[1:].split("c")[0])


def signflip_p(d: list) -> float:
    """Exact two-sided sign-flip p-value for the sum (so the mean) of integer differences d: the share of the
    2^N sign patterns whose |sum| is at least the observed |sum|. A zero difference is the same under either sign."""
    a = [abs(int(x)) for x in d]
    n, total = len(a), sum(a)
    if not 0 < n <= 62:
        raise ValueError("the sign-flip count needs 1 to 62 pairs")
    if any(int(x) != x for x in d):
        raise ValueError("differences must be integers (build counts)")
    if total == 0:
        return 1.0
    dist = np.zeros(2 * total + 1, dtype=np.int64)      # dist[total + s] = number of sign patterns with sum s
    dist[total] = 1
    for x in a:
        if x == 0:
            dist = dist * 2
            continue
        new = np.zeros_like(dist)
        new[x:] += dist[:-x]
        new[:-x] += dist[x:]
        dist = new
    obs = abs(sum(int(x) for x in d))
    far = np.abs(np.arange(-total, total + 1)) >= obs
    return int(dist[far].sum()) / 2 ** n


def boot_interval(d: list, draws: int = 10_000, seed: int = 0) -> list:
    """Bootstrap 95% interval for the mean of d, resampling pairs (the percentile rule Stage 1's boot_diff uses)."""
    rng = random.Random(seed)
    vals = sorted(mean([rng.choice(d) for _ in d]) for _ in range(draws))
    return [vals[int(0.025 * draws)], vals[int(0.975 * draws) - 1]]


def reading(mean_d: float, p: float, upper: float) -> str:
    if p < ALPHA and mean_d > 0:
        return "pass"
    if p < ALPHA and mean_d < 0:
        return "worse"
    return "bounded" if upper < BOUND else "open"


def rms(d: list) -> float:
    return math.sqrt(sum(x * x for x in d) / len(d))


def second_stage_rule(d: list, threshold: float) -> dict:
    """PROTOCOL_2E.md: more pairs run if the root mean square of the first stage's differences is above the
    threshold. It reads sizes only, so swapping any pair's two labels can't change it."""
    r = rms([abs(x) for x in d])
    return {"rms": r, "threshold": threshold, "extend": r > threshold, "n": len(d)}


def test_block(d_builds: list, denom: int) -> dict:
    d = [x / denom for x in d_builds]
    m, p, ci = mean(d), signflip_p(d_builds), boot_interval(d)
    return {"n_pairs": len(d), "mean_d": round(m, 6), "p": p, "ci95_boot": [round(ci[0], 6), round(ci[1], 6)],
            "sd_d": round(statistics.stdev(d), 6) if len(d) > 1 else None, "rms_d": round(rms(d), 6),
            "n_zero": sum(x == 0 for x in d_builds), "n_positive": sum(x > 0 for x in d_builds),
            "n_negative": sum(x < 0 for x in d_builds), "gate_passed": bool(p < ALPHA and m > 0),
            "reading": reading(m, p, ci[1])}


# ---------------------------------------------------------------- descriptive, read from the campaign's files
def running_best(cdir: Path, pairs: list, floor: float) -> dict:
    """Per round: the mean over pairs of each branch's best unrounded trial score so far (never below the start's
    score on the trial seeds), and of their difference. Selection scores on the trial seeds, not confirmed scores."""
    out = {}
    for i in pairs:
        cur = {}
        for b in BRANCHES:
            st = json.loads((cdir / "pairs" / f"p{i:02d}" / b / "state.json").read_text())
            best, by_round = floor, {}
            for r in sorted({t["round"] for t in st["trials"]}):
                ok = [t["exact_mean"] for t in st["trials"] if t["round"] == r and t["result"]["status"] == "ok"]
                best = max([best] + ok)
                by_round[r] = best
            cur[b] = by_round
        for r in cur["seen"]:
            o = out.setdefault(r, {"seen": [], "hidden": [], "d": []})
            o["seen"].append(cur["seen"][r])
            o["hidden"].append(cur["hidden"][r])
            o["d"].append(cur["seen"][r] - cur["hidden"][r])
    return {str(r): {k: round(mean(v), 4) for k, v in o.items()} for r, o in sorted(out.items())}


def counts_and_costs(cdir: Path, pairs: list) -> dict:
    out = {k: {"trials": 0, "invalid": 0, "failed_training": 0, "timeouts": 0, "calls": 0, "notional_usd": 0.0,
               "training_wall_hours": 0.0} for k in ("shared",) + BRANCHES}
    for i in pairs:
        pdir = cdir / "pairs" / f"p{i:02d}"
        for k in out:
            st = json.loads((pdir / k / "state.json").read_text())
            mine = [t for t in st["trials"] if (t["round"] == 1) == (k == "shared")]
            out[k]["trials"] += len(mine)
            out[k]["invalid"] += sum(t["result"]["status"] == "invalid" for t in mine)
            out[k]["failed_training"] += sum(t["result"]["status"] not in ("ok", "invalid") for t in mine)
            out[k]["timeouts"] += sum(t["result"]["status"] == "timeout" for t in mine)
            for f in (pdir / k).glob("round_*.json"):
                if not f.name.endswith(".failed.json"):
                    out[k]["calls"] += 1
                    out[k]["notional_usd"] += (json.loads(f.read_text()).get("usage") or {}).get("notional_cost_usd") or 0.0
            for f in (pdir / k).glob("trials/*/seed*/result.json"):
                out[k]["training_wall_hours"] += (json.loads(f.read_text()).get("wall_s") or 0) / 3600
    for k in out:
        out[k]["notional_usd"] = round(out[k]["notional_usd"], 2)
        out[k]["training_wall_hours"] = round(out[k]["training_wall_hours"], 2)
    rv = [json.loads(f.read_text()) for f in (cdir / "review").glob("*.json") if not f.name.endswith(".failed.json")]
    out["review"] = {"calls": len(rv), "notional_usd": round(sum((r.get("usage") or {}).get("notional_cost_usd") or 0.0 for r in rv), 2)}
    return out


def analyse(cdir: Path, entries: list, verdicts, H) -> dict:
    pr = pair_rows(entries, verdicts)
    rows, denom = pr["rows"], pr["denom"]
    n_seeds = denom // N_DESIGNS
    comp = next(e for e in entries if e["kind"] == "comparator" and e["arm"] == "start_g")
    start_scores = [k / N_DESIGNS for k in seed_builds(comp)]
    res = {"n_pairs": len(rows), "confirmation_seeds": n_seeds, "denominator": denom, "pending_verdicts": pr["pending"],
           "status": "pending review" if pr["pending"] else "final"}
    for r in rows:
        r["d"] = round(r["d_builds"] / denom, 6)
        for b in BRANCHES:
            r[b]["score"] = round(r[b]["builds"] / denom, 6)
            r[b]["primary_score"] = round(r[b]["primary_builds"] / denom, 6)
            src = (cdir / "confirm" / r[b]["sha"] / "candidate.py").read_text()
            seeds = next(e for e in entries if e["kind"] == "search" and e["pair"] == r["pair"] and e["arm"] == b)["seeds"]
            r[b]["seeds_learned"] = sum(AS1.seed_detail(cdir, r[b]["sha"], s, H)["learned"] for s in seeds)
            r[b]["recipe"] = AN2.recipe_of(src)
    res["pairs"] = rows
    as_computed = test_block([r["d_builds_as_computed"] for r in rows], denom)
    res["as_computed"] = as_computed
    if pr["pending"]:
        res["primary"] = {"status": "pending review", "gate_passed": None, "reading": None}
    else:
        res["primary"] = test_block([r["d_builds"] for r in rows], denom) | {"status": "final"}
    per = {}
    last = max(pick_round(r[b]["pick"]) for r in rows for b in BRANCHES)
    for b in BRANCHES:
        sc = [r[b]["primary_score"] for r in rows]
        per[b] = {"mean": round(mean(sc), 6), "sd_between_pairs": round(statistics.stdev(sc), 6) if len(sc) > 1 else None,
                  "min": min(sc), "max": max(sc), "minus_start": round(mean(sc) - mean(start_scores), 6),
                  "share_seeds_learned": round(sum(r[b]["seeds_learned"] for r in rows) / (len(rows) * n_seeds), 4),
                  "selection_optimism": round(mean([r[b]["selection_mean"] - r[b]["score"] for r in rows]), 6),
                  "picked_start": sum(r[b]["pick"] == "start" for r in rows),
                  "picks_with_recipe": sum(r[b]["recipe"] is not None for r in rows),
                  "violations": sum(r[b]["verdict"] == "violation" for r in rows),
                  "picks_by_round": {str(k): sum(pick_round(r[b]["pick"]) == k for r in rows) for k in range(last + 1)}}
    res["per_branch"] = per
    res["start"] = {"per_seed": start_scores, "mean": round(mean(start_scores), 6),
                    "sd": round(statistics.stdev(start_scores), 6) if len(start_scores) > 1 else None}
    res["same_pick_pairs"] = sum(r["same_pick"] for r in rows)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    cdir = Path(args.campaign).resolve()
    if (S2 / "FINAL_OPENED_W").exists():
        raise SystemExit("FINAL_OPENED_W exists: Stage 2e must not open W's final split")
    man = json.loads((cdir / "manifest.json").read_text())
    entries = json.loads((cdir / "confirm" / "summary.json").read_text())
    rv = cdir / "review.json"
    verdicts = json.loads(rv.read_text())["verdicts"] if rv.exists() else None
    base = json.loads((cdir / "baseline" / "summary.json").read_text())
    res = {"campaign": str(cdir.relative_to(ROOT)) if cdir.is_relative_to(ROOT) else str(cdir), "preset": man["preset"],
           "split": "dev (W final locked)",
           "learned_threshold": AS1.LEARNED, "baseline": base}
    res |= analyse(cdir, entries, verdicts, AN2.hidden_flags())
    pairs = [r["pair"] for r in res["pairs"]]
    for name in ("second_stage", "CEILING_STOP"):
        f = cdir / f"{name}.json"
        res[name.lower()] = json.loads(f.read_text()) if f.exists() else None
    res["running_best_by_round"] = running_best(cdir, pairs, base["w"]["shown_dev"])
    res["counts_and_costs"] = counts_and_costs(cdir, pairs)
    (cdir / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("status", "n_pairs", "primary", "as_computed", "second_stage")}, indent=1))


if __name__ == "__main__":
    main()
