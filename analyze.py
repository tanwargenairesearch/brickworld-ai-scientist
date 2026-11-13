"""Pre-registered analyses (protocol.md) on a finished campaign. Writes results.json, report.md
and figures/. Reads only files the campaign produced; changes nothing in it.

usage: python analyze.py --campaign campaigns/main
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
B = 10_000
RNG = np.random.default_rng(20261107)


def load(cdir):
    res, fin = {}, {}
    for t in sorted((cdir / "trials").iterdir()):
        if (t / "result.json").exists():
            res[t.name] = json.loads((t / "result.json").read_text())
        if (t / "eval_final.json").exists():
            fin[t.name] = json.loads((t / "eval_final.json").read_text())
    return res, fin


def boot_mean(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    if len(x) == 1:
        return [float(x[0]), float(x[0])]
    m = x[RNG.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    d = a[RNG.integers(0, len(a), (B, len(a)))].mean(1) - b[RNG.integers(0, len(b), (B, len(b)))].mean(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def summ(x):
    x = [v for v in x if v is not None]
    return {"mean": float(np.mean(x)) if x else None, "ci95": boot_mean(x), "n": len(x),
            "values": [round(float(v), 4) for v in x]}


def fexact(fin, tid):
    return fin[tid]["exact"] if tid in fin else None


def fscore(res, fin, tid):
    """Final exact of a retrain or transfer trial. A trial that timed out or crashed scores 0
    (the recipe did not finish within its budget), as pre-registered; a missing trial is None."""
    if tid not in res:
        return None
    if res[tid]["status"] in ("timeout", "crash"):
        return 0.0
    return fexact(fin, tid)


def recipe_score(res, fin, tid, fresh_map, n_baseline):
    """Final exact of a trial's recipe retrained with fresh seeds (baseline: its own seeds)."""
    if tid.startswith("baseline"):
        ids = [f"baseline-s{s}" for s in range(n_baseline)]
    else:
        h = fresh_map.get(res[tid]["config_key"])
        ids = [k for k in res if h and k.startswith(f"fresh-{h}-s")]
    v = [x for x in (fscore(res, fin, k) for k in ids) if x is not None]
    return float(np.mean(v)) if v else None


def exploit_recipe(r):
    return r["reward_mode"] == "heightmap_delta" or r["obs_mode"] == "heightmaps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    cdir = ROOT / args.campaign
    man = json.loads((cdir / "manifest.json").read_text())
    S = man["settings"]
    res, fin = load(cdir)
    inc = {tuple(k.split("|")): v for k, v in json.loads((cdir / "incumbents.json").read_text()).items()}
    fresh_map = json.loads((cdir / "fresh_map.json").read_text())
    nb = S["baseline_seeds"]
    out = {"manifest_created": man["created"], "proposer_backend": man["proposer_backend"]}

    # integrity
    bad = [f for f, h in man["frozen"].items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    out["integrity"] = {"frozen_files_unchanged": not bad, "changed": bad,
                        "final_opened": (cdir / "FINAL_OPENED").read_text().splitlines()[0]
                        if (cdir / "FINAL_OPENED").exists() else None}
    status = {}
    for tid, r in res.items():
        arm = tid.split("-")[0]
        status.setdefault(arm, {}).setdefault(r["status"], 0)
        status[arm][r["status"]] += 1
    out["trial_status"] = status

    # noise floor
    bdev = [res[f"baseline-s{s}"]["dev"]["exact"] for s in range(nb) if res.get(f"baseline-s{s}", {}).get("dev")]
    bfin = [fexact(fin, f"baseline-s{s}") for s in range(nb)]
    out["baseline"] = {"dev_exact": summ(bdev), "final_exact": summ(bfin),
                       "dev_topview": summ([res[f"baseline-s{s}"]["dev"]["topview"] for s in range(nb)
                                            if res.get(f"baseline-s{s}", {}).get("dev")])}

    # 3. toy r
    arms = list(S["arms"]) + ["random_topview"]
    ks = S["budgets_k"]
    toy = {}
    for arm in arms:
        runs = sorted({int(r) for (a, r, k) in inc if a == arm})
        per_k = {k: [] for k in ks}
        ck_k = {k: [] for k in ks}
        dev_k = {k: [] for k in ks}
        gap_k = {k: [] for k in ks}
        expl_k = {k: [] for k in ks}
        slopes = []
        for r in runs:
            ys = []
            for k in ks:
                tid = inc[(arm, str(r), str(k))]
                sc = recipe_score(res, fin, tid, fresh_map, nb)
                per_k[k].append(sc)
                ck_k[k].append(fexact(fin, tid))
                dev_k[k].append(res[tid]["dev"]["exact"])
                gap_k[k].append(res[tid]["dev"]["topview"] - res[tid]["dev"]["exact"])
                expl_k[k].append(1.0 if exploit_recipe(res[tid]["recipe"]) else 0.0)
                ys.append(sc)
            if all(y is not None for y in ys):
                slopes.append(100 * np.polyfit(np.log2(ks), ys, 1)[0])
        toy[arm] = {
            "runs": len(runs),
            "fresh_final_exact_by_k": {k: summ(v) for k, v in per_k.items()},
            "checkpoint_final_exact_by_k": {k: summ(v) for k, v in ck_k.items()},
            "dev_exact_by_k_selection_biased": {k: summ(v) for k, v in dev_k.items()},
            "dev_topview_minus_exact_by_k": {k: summ(v) for k, v in gap_k.items()},
            "exploit_found_share_by_k": {k: summ(v) for k, v in expl_k.items()},
            "toy_r_points_per_doubling": summ(slopes),
            "_slopes": slopes, "_level32": per_k[ks[-1]],
        }
    comp = {}
    for llm in ("sonnet", "opus"):
        if llm in toy and "random" in toy:
            comp[f"{llm}_minus_random"] = {
                "slope_diff_ci95": boot_diff(toy[llm]["_slopes"], toy["random"]["_slopes"]),
                "slope_diff_mean": (float(np.mean(toy[llm]["_slopes"]) - np.mean(toy["random"]["_slopes"]))
                                    if toy[llm]["_slopes"] and toy["random"]["_slopes"] else None),
                "level_k32_diff_ci95": boot_diff([v for v in toy[llm]["_level32"] if v is not None],
                                                 [v for v in toy["random"]["_level32"] if v is not None]),
            }
    # random arm, pooled expectation (secondary): best-of-k by dev exact among all random trials
    pool = [t for t in res if t.startswith("random-") and res[t].get("dev") and t in fin]
    if pool and "baseline-s0" in res:
        pooled = {}
        for k in ks:
            vals = []
            for _ in range(2000):
                pick = list(RNG.choice(pool, size=min(k, len(pool)), replace=False)) + ["baseline-s0"]
                best = max(pick, key=lambda t: res[t]["dev"]["exact"])
                vals.append(fexact(fin, best))
            pooled[k] = float(np.mean(vals))
        comp["random_pooled_checkpoint_final_exact_by_k"] = pooled
    out["toy_r"] = {a: {k: v for k, v in d.items() if not k.startswith("_")} for a, d in toy.items()}
    out["toy_r_comparisons"] = comp

    # exploit drift across rounds (all proxy trials, not just incumbents)
    drift = {}
    for arm in ("proxy_sonnet", "sonnet", "opus"):
        rows = [(int(t.split("-t")[1]) // S["per_round"] + 1, exploit_recipe(res[t]["recipe"]))
                for t in res if t.startswith(arm + "-")]
        drift[arm] = {rd: float(np.mean([e for r_, e in rows if r_ == rd]))
                      for rd in sorted({r_ for r_, _ in rows})}
    out["exploit_recipe_share_by_round"] = drift

    # 1. transfer
    tset = json.loads((cdir / "transfer_set.json").read_text())
    chosen = ["baseline"] + tset["top"] + tset["others"]
    rows = []
    for i, src in enumerate(chosen):
        if src == "baseline":
            small = [fexact(fin, f"baseline-s{s}") for s in range(nb)]
        else:
            small = [fexact(fin, src)] + [fscore(res, fin, f"xfer-small-{i:02d}-s{s}")
                                          for s in range(S["transfer"]["small_extra_seeds"])]
        large = [fscore(res, fin, f"xfer-large-{i:02d}-s{s}") for s in range(S["transfer"]["large_seeds"])]
        failed_large = sum(res.get(f"xfer-large-{i:02d}-s{s}", {}).get("status") in ("timeout", "crash")
                           for s in range(S["transfer"]["large_seeds"]))
        small, large = [v for v in small if v is not None], [v for v in large if v is not None]
        rows.append({"i": i, "source": src, "group": "baseline" if i == 0 else ("top" if src in tset["top"] else "other"),
                     "small": float(np.mean(small)) if small else None,
                     "large": float(np.mean(large)) if large else None,
                     "n_small": len(small), "n_large": len(large), "large_timeouts_or_crashes": failed_large,
                     "recipe": res[src]["recipe"] if src in res else None})
    ok = [r for r in rows if r["small"] is not None and r["large"] is not None]
    xs, ys = np.array([r["small"] for r in ok]), np.array([r["large"] for r in ok])
    rho = float(spearmanr(xs, ys).statistic) if len(ok) > 2 else None
    bs = []
    for _ in range(B):
        idx = RNG.integers(0, len(ok), len(ok))
        if len(set(xs[idx])) > 1 and len(set(ys[idx])) > 1:
            bs.append(spearmanr(xs[idx], ys[idx]).statistic)
    base_row = rows[0]
    top_rows = [r for r in ok if r["group"] == "top"]
    out["transfer"] = {
        "spearman_rho": rho, "rho_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None,
        "n_configs": len(ok),
        "top8_beat_baseline_small": sum(r["small"] > base_row["small"] for r in top_rows) if base_row["small"] is not None else None,
        "top8_beat_baseline_large": sum(r["large"] > base_row["large"] for r in top_rows) if base_row["large"] is not None else None,
        "small_winner": max(ok, key=lambda r: r["small"])["source"] if ok else None,
        "large_winner": max(ok, key=lambda r: r["large"])["source"] if ok else None,
        "rows": rows,
    }

    # 2. variety
    V = S["variety"]
    var = {}
    for cond in V["conditions"]:
        tag = cond.replace(":", "")
        tids = [f"variety-{tag}-s{s}" for s in range(V["seeds"])]
        curve = {}
        for f in V["ckpt_fracs"] + [1.0]:
            vals = []
            for t in tids:
                r = res.get(t, {})
                if f == 1.0:
                    vals.append((r.get("dev") or {}).get("exact"))
                else:
                    vals.append(((r.get("curve_dev") or {}).get(f"{f:.3f}") or {}).get("exact"))
            curve[f"{f:.3f}"] = summ(vals)
        train_end = [((res.get(t, {}).get("trainset")) or {}).get("exact") for t in tids]
        var[cond] = {"dev_exact_curve": curve, "final_exact_end": summ([fexact(fin, t) for t in tids]),
                     "trainset_exact_end": summ(train_end)}
    out["variety"] = var

    # proposer usage
    usage = {}
    for f in (cdir / "proposer").rglob("round_*.json"):
        if f.name.endswith("_failed.json"):
            continue
        d = json.loads(f.read_text())
        arm = f.parts[-3]
        u = usage.setdefault(arm, {"calls": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0,
                                   "notional_cost_usd": 0.0, "latency_s": []})
        u["calls"] += 1
        u["retries"] += d["attempt"] - 1
        u["input_tokens"] += (d["usage"].get("input_tokens") or 0) + (d["usage"].get("cache_creation_input_tokens") or 0)
        u["output_tokens"] += d["usage"].get("output_tokens") or 0
        u["notional_cost_usd"] += d["usage"].get("notional_cost_usd") or 0.0
        u["latency_s"].append(d["latency_s"])
    for u in usage.values():
        u["median_latency_s"] = float(np.median(u.pop("latency_s"))) if u["calls"] else None
    out["proposer_usage"] = usage
    failed = [str(p.relative_to(cdir)) for p in (cdir / "proposer").rglob("*_failed.json")]
    out["proposer_failures_logged"] = failed

    (cdir / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("integrity", "trial_status", "baseline", "toy_r_comparisons")},
                     indent=1, default=float))
    figures(cdir, out, ks)


def figures(cdir, out, ks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = cdir / "figures"
    fd.mkdir(exist_ok=True)
    colors = {"random": "#6b7280", "sonnet": "#2563eb", "opus": "#7c3aed", "proxy_sonnet": "#dc2626",
              "random_topview": "#f59e0b"}
    names = {"random": "Random search", "sonnet": "Claude Sonnet 5.5", "opus": "Claude Opus 5.5",
             "proxy_sonnet": "Sonnet 5.5, flawed checker", "random_topview": "Random, flawed checker"}

    def band(ax, arm, key, label=None):
        d = out["toy_r"][arm][key]
        m = [d[k]["mean"] for k in ks]
        lo = [d[k]["ci95"][0] if d[k]["ci95"] else np.nan for k in ks]
        hi = [d[k]["ci95"][1] if d[k]["ci95"] else np.nan for k in ks]
        ax.plot(ks, [100 * v if v is not None else np.nan for v in m], "-o", color=colors[arm], lw=2,
                label=label or names[arm])
        ax.fill_between(ks, 100 * np.array(lo, float), 100 * np.array(hi, float), color=colors[arm], alpha=0.15)

    b = out["baseline"]["final_exact"]["mean"]
    for fname, arms in (("toy_r.png", ["random", "sonnet", "opus"]),
                        ("exploit.png", ["sonnet", "proxy_sonnet", "random", "random_topview"])):
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
        for a in arms:
            if a in out["toy_r"]:
                band(ax, a, "fresh_final_exact_by_k")
        if b is not None:
            ax.axhline(100 * b, color="#9ca3af", ls="--", lw=1, label="Baseline recipe")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("Search budget (training runs)")
        ax.set_ylabel("Exact success on hidden designs (%)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fd / fname)
        plt.close(fig)

    rows = [r for r in out["transfer"]["rows"] if r["small"] is not None and r["large"] is not None]
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=200)
    for g, c in (("baseline", "#111827"), ("top", "#2563eb"), ("other", "#9ca3af")):
        rr = [r for r in rows if r["group"] == g]
        ax.scatter([100 * r["small"] for r in rr], [100 * r["large"] for r in rr], color=c, s=36,
                   label={"baseline": "Baseline", "top": "Top 8 at small scale", "other": "Other random configs"}[g])
    ax.set_xlabel("Small world: exact success (%)")
    ax.set_ylabel("Large world: exact success (%)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "transfer.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
    for cond, d in out["variety"].items():
        xs = [float(f) for f in d["dev_exact_curve"]]
        ys = [100 * v["mean"] if v["mean"] is not None else np.nan for v in d["dev_exact_curve"].values()]
        ax.plot(xs, ys, "-o", lw=2, ms=3, label=cond.replace("fixed:", "Fixed ").replace("stream", "Fresh generator") +
                (" designs" if cond.startswith("fixed") else ""))
    ax.set_xlabel("Share of the 2.1M-step training budget")
    ax.set_ylabel("Exact success on held-out dev designs (%)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "variety.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
