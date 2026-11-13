"""Pre-registered analysis of Stage 1 (protocol_stage1.md). Written and hashed before any Stage 1 result.

usage (project root): .venv/bin/python -I arch_rsi/stage1/analyze_stage1.py --campaign arch_rsi/campaigns/stage1
Reads dev results only (the final split stays locked). Writes results.json in the campaign directory.

Unit: one search. Its score is the mean, over the 8 confirmation seeds, of its pick's exact-build success on Q's
512 dev designs; a confirmation seed whose training failed scores 0. Comparators are scored per seed.
Primary score: compliant picks only. A search whose pick the independent review (review_stage1.py) judges to break
a rule is scored as if it had picked the starting architecture (the start's scores on the same seeds). The
as-computed score is reported beside it. Without review.json the primary analysis is marked pending.
Tests: exact two-sided permutation test on the difference in means (all splits), Holm-corrected within each
family. Primary family: g0 - nofb, g0 - start. Gate: both differences positive with Holm-adjusted p < 0.05.
Bootstrap 95% intervals (10,000 draws, seed 0, each group resampled independently) are descriptive only.
"""
import argparse
import itertools
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEV_DESIGNS = ROOT / "arch_rsi" / "targets" / "xl_dev.pt"
LEARNED = 0.10  # a seed "learned the hidden-cell skill" if it builds at least 10% of the dev designs with hidden cells

PRIMARY = [("g0", "nofb"), ("g0", "start")]
SECONDARY = [("g0", "menu"), ("nofb", "menu"), ("nofb", "start"), ("menu", "start"), ("g0", "critic_ortho"),
             ("critic_ortho", "start")]


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def perm_p(a: list, b: list) -> float:
    """Exact two-sided permutation p-value for mean(a) - mean(b), over all splits of the pooled values."""
    pooled, na, nb = a + b, len(a), len(b)
    total = sum(pooled)
    obs = abs(mean(a) - mean(b))
    hits = n = 0
    for idx in itertools.combinations(range(len(pooled)), na):
        sa = sum(pooled[i] for i in idx)
        n += 1
        hits += abs(sa / na - (total - sa) / nb) >= obs - 1e-12
    return hits / n


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def boot_diff(a: list, b: list, draws: int = 10_000, seed: int = 0) -> list:
    rng = random.Random(seed)
    vals = sorted(mean([rng.choice(a) for _ in a]) - mean([rng.choice(b) for _ in b]) for _ in range(draws))
    return [round(vals[int(0.025 * draws)], 4), round(vals[int(0.975 * draws) - 1], 4)]


def hidden_flags() -> np.ndarray:
    v = torch.load(DEV_DESIGNS).bool()
    above = (v.flip(1).cumsum(1).flip(1) - v.int()) > 0
    return ((~v) & above).flatten(1).any(1).numpy()


def seed_detail(cdir: Path, h: str, s, H: np.ndarray) -> dict:
    f = cdir / "confirm" / h / f"seed{s}" / "dev.json"
    if not f.exists():
        return {"hidden": 0.0, "learned": False}
    p = np.array(json.loads(f.read_text())["per_design"]["exact"], bool)
    hid = float(p[H].mean())
    return {"hidden": hid, "learned": hid >= LEARNED}


def family(groups: dict, pairs: list) -> dict:
    out, ps = {}, {}
    for a, b in pairs:
        if a in groups and b in groups and groups[a] and groups[b]:
            name = f"{a} - {b}"
            ps[name] = perm_p(groups[a], groups[b])
            out[name] = {"diff": round(mean(groups[a]) - mean(groups[b]), 4), "p": round(ps[name], 4),
                         "ci95_boot": boot_diff(groups[a], groups[b]), "n": [len(groups[a]), len(groups[b])]}
    for k, p in holm(ps).items():
        out[k]["p_holm"] = round(p, 4)
    return out


def analyse(cdir: Path, entries: list, review: dict | None, H: np.ndarray) -> dict:
    def s_key(s):
        return str(s)

    comp = {e["arm"]: e for e in entries if e["kind"] == "comparator"}
    start_by_seed = {s_key(s): (comp["start"]["dev"].get(s_key(s)) or 0.0) for s in comp["start"]["seeds"]} if "start" in comp else {}

    def seed_scores(e):
        return [(e["dev"].get(s_key(s)) if e["status"].get(s_key(s)) == "ok" else 0.0) or 0.0 for s in e["seeds"]]

    as_computed, primary, detail = {}, {}, {}
    for e in entries:
        sc = seed_scores(e)
        det = [seed_detail(cdir, e["sha"], s, H) if e["status"].get(s_key(s)) == "ok" else {"hidden": 0.0, "learned": False}
               for s in e["seeds"]]
        if e["kind"] == "comparator":
            as_computed[e["arm"]] = sc
            primary[e["arm"]] = sc
            detail[e["arm"]] = {"per_seed": [round(x, 4) for x in sc], "hidden_per_seed": [round(d["hidden"], 4) for d in det],
                                "seeds_learned": sum(d["learned"] for d in det), "seeds": len(sc)}
            continue
        verdict = (review or {}).get(e["sha"], {}).get("verdict", "pending" if review is None else "compliant")
        if e["label"] == "start" or e["arm"] == "menu":
            verdict = "compliant (start or menu file)"
        x = mean(sc)
        y = mean([start_by_seed[s_key(s)] for s in e["seeds"]]) if verdict == "violation" else x
        as_computed.setdefault(e["arm"], []).append(x)
        primary.setdefault(e["arm"], []).append(y)
        d = detail.setdefault(e["arm"], {"searches": []})
        d["searches"].append({"search": e["search"], "pick": e["label"], "name": e["name"], "sha": e["sha"],
                              "verdict": verdict, "score": round(x, 4), "primary_score": round(y, 4),
                              "selection_mean": round(e["mean_selected"], 4), "per_seed": [round(v, 4) for v in sc],
                              "hidden_mean": round(mean([q["hidden"] for q in det]), 4),
                              "seeds_learned": sum(q["learned"] for q in det)})
    for a, d in detail.items():
        if "searches" in d:
            ss = d["searches"]
            d |= {"mean": round(mean(as_computed[a]), 4), "primary_mean": round(mean(primary[a]), 4),
                  "sd_between_searches": round(statistics.stdev(as_computed[a]), 4) if len(ss) > 1 else None,
                  "hidden_mean": round(mean([s["hidden_mean"] for s in ss]), 4),
                  "share_seeds_learned": round(sum(s["seeds_learned"] for s in ss) / sum(len(s["per_seed"]) for s in ss), 4),
                  "selection_optimism": round(mean([s["selection_mean"] - s["score"] for s in ss]), 4),
                  "picked_start": sum(s["pick"] == "start" for s in ss),
                  "violations": sum(s["verdict"] == "violation" for s in ss)}
        else:
            d |= {"mean": round(mean(as_computed[a]), 4)}
    pending = review is None
    prim = family(primary, PRIMARY)
    gate = None if pending else all(v["diff"] > 0 and v["p_holm"] < 0.05 for v in prim.values()) and len(prim) == 2
    return {"per_arm": detail,
            "primary_family": {"status": "pending review" if pending else "final", "contrasts": prim, "gate_passed": gate},
            "as_computed_primary_family": family(as_computed, PRIMARY),
            "secondary_family": family(primary if not pending else as_computed, SECONDARY)}


def search_curves(cdir: Path, arms: list) -> dict:
    """Descriptive: per round, the mean over searches of the round's best 2-seed dev score, and of the best so far.
    Best-so-far curves rise even under pure noise; compare arms, not slopes."""
    out = {}
    for a in arms:
        rounds, invalid, failed, trials = {}, 0, 0, 0
        for st_file in sorted((cdir / "Q" / a).glob("s*/state.json")):
            st = json.loads(st_file.read_text())
            best = 0.0
            for r in sorted({t["round"] for t in st["trials"]}):
                rt = [t for t in st["trials"] if t["round"] == r]
                ok = [t["result"]["dev_exact"] for t in rt if t["result"]["status"] == "ok"]
                rb = max(ok, default=0.0)
                best = max(best, rb)
                rounds.setdefault(r, {"round_best": [], "best_so_far": []})
                rounds[r]["round_best"].append(rb)
                rounds[r]["best_so_far"].append(best)
            trials += len(st["trials"])
            invalid += sum(t["result"]["status"] == "invalid" for t in st["trials"])
            failed += sum(t["result"]["status"] not in ("ok", "invalid") for t in st["trials"])
        out[a] = {"per_round": {r: {k: round(mean(v), 4) for k, v in d.items()} for r, d in sorted(rounds.items())},
                  "trials": trials, "invalid": invalid, "failed_training": failed}
    return out


def costs(cdir: Path, arms: list) -> dict:
    out = {}
    for a in arms:
        wall = calls = 0
        usd = 0.0
        tok_out = 0
        for sdir in sorted((cdir / "Q" / a).glob("s*")):
            for f in sdir.glob("trials/*/seed*/result.json"):
                wall += json.loads(f.read_text()).get("wall_s") or 0
            for f in sdir.glob("round_*.json"):
                if f.name.endswith(".failed.json"):
                    continue
                u = json.loads(f.read_text()).get("usage", {})
                calls += 1
                usd += u.get("notional_cost_usd") or 0.0
                tok_out += u.get("output_tokens") or 0
        # input tokens are not reported: the frozen logger drops the CLI's cache-creation count, where the prompt lands
        out[a] = {"search_training_wall_hours": round(wall / 3600, 2), "claude_calls": calls,
                  "claude_notional_usd": round(usd, 2), "output_tokens": tok_out}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    cdir = Path(args.campaign).resolve()
    if (cdir / "FINAL_OPENED").exists():
        raise SystemExit("FINAL_OPENED exists: Stage 1 must not open the final split")
    man = json.loads((cdir / "manifest.json").read_text())
    entries = json.loads((cdir / "confirm" / "summary.json").read_text())
    rv_file = cdir / "review.json"
    review = json.loads(rv_file.read_text())["verdicts"] if rv_file.exists() else None
    H = hidden_flags()
    res = {"campaign": str(cdir.relative_to(ROOT)), "preset": man["preset"], "split": "dev (final locked)",
           "learned_threshold": LEARNED, "dev_designs_with_hidden_cells": int(H.sum()),
           "baseline": json.loads((cdir / "baseline" / "summary.json").read_text())}
    res |= analyse(cdir, entries, review, H)
    arms = man["config"]["arms"]
    res["search_curves"] = search_curves(cdir, arms)
    res["costs"] = costs(cdir, arms)
    (cdir / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
