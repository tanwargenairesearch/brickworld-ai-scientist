"""Pre-registered analysis of the architecture-search campaign. Frozen before launch.

usage: .venv/bin/python -I arch_rsi/analyze_arch.py --campaign arch_rsi/campaigns/main
Writes results.json in the campaign directory and prints it.

Unit of analysis: one search. Its score is the mean, over the fresh confirmation seeds, of the selected
architecture's exact-build success on Q's 1,024 hidden final designs; a confirmation seed whose training
fails (timeout, crash, divergence) scores 0. Intervals: 95% percentile bootstrap over searches (10,000
draws, seed 0), resampling each arm independently; comparator seeds are resampled too. Contrasts also get
an exact two-sided permutation p-value over searches.
"""
import argparse
import itertools
import json
import random
import statistics
from pathlib import Path


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def boot_ci(groups_fn, draws=10_000, seed=0):
    rng = random.Random(seed)
    vals = sorted(groups_fn(rng) for _ in range(draws))
    return [round(vals[int(0.025 * draws)], 4), round(vals[int(0.975 * draws) - 1], 4)]


def resample(rng, xs):
    return [rng.choice(xs) for _ in xs]


def perm_p(a, b):
    """Exact two-sided permutation p-value for a difference in means (all splits)."""
    pooled, na = a + b, len(a)
    obs = abs(mean(a) - mean(b))
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), na):
        s = set(idx)
        x = [pooled[i] for i in idx]
        y = [pooled[i] for i in range(len(pooled)) if i not in s]
        total += 1
        hits += abs(mean(x) - mean(y)) >= obs - 1e-12
    return round(hits / total, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    cdir = Path(args.campaign)
    man = json.loads((cdir / "manifest.json").read_text())
    cfg = man["config"]
    G = cfg["G"]
    summ = json.loads((cdir / "confirm" / "summary.json").read_text())

    def score(e):
        return mean([(e["final"][str(s)] if str(s) in e["final"] else e["final"].get(s)) or 0.0 for s in e["seeds"]])

    arms = {}
    for e in summ:
        if e["kind"] == "search":
            arms.setdefault(e["arm"], []).append(score(e))
    comp = {e["arm"]: [(e["final"][str(s)] if str(s) in e["final"] else e["final"].get(s)) or 0.0 for s in e["seeds"]]
            for e in summ if e["kind"] == "comparator"}
    gG = f"g{G}"
    res = {"per_arm": {}, "comparators": {}, "contrasts": {}}
    for a, xs in arms.items():
        res["per_arm"][a] = {"n_searches": len(xs), "scores": [round(x, 4) for x in xs], "mean": round(mean(xs), 4),
                             "ci95": boot_ci(lambda r, xs=xs: mean(resample(r, xs)))}
    for a, xs in comp.items():
        res["comparators"][a] = {"seeds": len(xs), "scores": [round(x, 4) for x in xs], "mean": round(mean(xs), 4)}

    def contrast(name, a, b, b_is_comp=False):
        xa = arms.get(a)
        xb = comp.get(b) if b_is_comp else arms.get(b)
        if not xa or not xb:
            return
        res["contrasts"][name] = {
            "diff": round(mean(xa) - mean(xb), 4),
            "ci95": boot_ci(lambda r: mean(resample(r, xa)) - mean(resample(r, xb))),
            "perm_p": None if b_is_comp else perm_p(xa, xb)}
    contrast(f"C1_compounding_{gG}_minus_g0", gG, "g0")
    contrast("C2_code_g0_minus_menu", "g0", "menu")
    contrast(f"C3_{gG}_minus_start", gG, "start", b_is_comp=True)
    contrast(f"S1_{gG}_minus_menu", gG, "menu")
    contrast("S2_g0_minus_start", "g0", "start", b_is_comp=True)
    contrast("S3_menu_minus_start", "menu", "start", b_is_comp=True)

    # secondary: search curves on Q, validity, compute, optimism of the selected dev score
    sec = {}
    for arm in arms:
        curves, invalid, failed, params, flops, steps = [], 0, 0, [], [], []
        for sd in sorted((cdir / "Q" / arm).glob("s*")):
            st = json.loads((sd / "state.json").read_text())
            base_dev = json.loads((cdir / "baseline" / "summary.json").read_text())[cfg["Q"]]["shown_dev"]
            best, curve = base_dev, []
            for t in st["trials"]:
                r = t["result"]
                invalid += r["status"] == "invalid"
                failed += r["status"] not in ("ok", "invalid")
                if r["status"] == "ok":
                    best = max(best, r["dev_exact"])
                curve.append(round(best, 4))
            curves.append(curve)
            inc = next((t for t in st["trials"] if t["label"] == st["incumbent"]), None)
            if inc:
                params.append(inc["result"]["params"])
                flops.append(inc["result"]["flops_ratio"])
                steps.append(inc["result"].get("train_steps"))
        ksteps = [cfg["K"] * r for r in range(1, cfg["R"] + 1)]
        sec[arm] = {"best_dev_after_k_trials": {k: round(mean([c[k - 1] for c in curves if len(c) >= k]), 4) for k in ksteps},
                    "invalid_candidates": invalid, "failed_trainings": failed,
                    "selected_params": params, "selected_flops_ratio": flops, "selected_train_steps": steps}
    for e in summ:
        if e["kind"] == "search":
            sec[e["arm"]].setdefault("search_dev_minus_final", []).append(round(e["search_dev"] - score(e), 4))
    res["secondary_Q"] = sec
    gp = cdir / "P" / "gen_stats.json"
    if gp.exists():
        res["P_in_sample_by_generation"] = json.loads(gp.read_text())
    res["baseline"] = json.loads((cdir / "baseline" / "summary.json").read_text())
    calls, cost, lat = 0, 0.0, []
    for p in list(cdir.rglob("round_*.json")) + list(cdir.rglob("call_g*.json")):
        if p.name.endswith(".failed.json"):
            continue
        rec = json.loads(p.read_text())
        calls += 1
        cost += (rec.get("usage") or {}).get("notional_cost_usd") or 0.0
        lat.append(rec.get("latency_s") or 0)
    res["claude_calls"] = {"n": calls, "notional_cost_usd": round(cost, 2),
                           "median_latency_s": statistics.median(lat) if lat else None}
    (cdir / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
