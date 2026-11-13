"""The physics check in PLAN.md. Run after validate.py passes:
  .venv/bin/python run_check.py
Writes records.jsonl (one row per exact build) and results.json.
"""
import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from brickphys import classify, classify_prefixes, order_class, replay

HERE = Path(__file__).parent
SPECS = {"small": {"X": 5, "Y": 5, "Z": 3}, "large": {"X": 6, "Y": 6, "Z": 4}}
GROUPS = {
    "small": {"baseline": "baseline", "random_k32": "random", "sonnet_k32": "claude", "opus_k32": "claude"},
    "large": {"baseline": "baseline", "best_transfer": "best_transfer"},
}
SEEDS = 3
BOOT = 2000


def has_hidden(occ):
    occ = np.array(occ, dtype=bool)
    above = np.flip(np.logical_or.accumulate(np.flip(occ, 0), 0), 0)  # filled at or above
    return bool((above & ~occ).any())


def job(args):
    scale, agent, di, bricks = args
    pre = classify_prefixes(bricks)
    out = {"scale": scale, "agent": agent, "design": di, "n_bricks": len(bricks),
           "final_class": pre[-1], "order_class": order_class(pre), "mujoco": []}
    for k in range(SEEDS):
        seed = int.from_bytes(hashlib.sha256(f"{scale}/{agent}/{di}/{k}".encode()).digest()[:4], "little")
        stands, errs, _ = replay(bricks, SPECS[scale], seed=seed)
        out["mujoco"].append(bool(stands))
    return out


def design_job(args):
    scale, di, bricks = args
    pre = classify_prefixes(bricks)
    return {"scale": scale, "design": di, "final_class": pre[-1], "order_class": order_class(pre)}


def boot_share(rows, key_fn, designs, rng):
    """Share of rows where key_fn is true, with a bootstrap over designs."""
    by_d = {}
    for r in rows:
        by_d.setdefault(r["design"], []).append(1.0 if key_fn(r) else 0.0)
    if not by_d:
        return None
    ds = list(by_d)
    sums = np.array([sum(by_d[d]) for d in ds])
    cnts = np.array([len(by_d[d]) for d in ds])
    point = sums.sum() / cnts.sum()
    idx = rng.integers(0, len(ds), size=(BOOT, len(ds)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    return {"share": float(point), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
            "n_builds": int(cnts.sum()), "n_designs": len(ds)}


def boot_paired(rows_a, rows_b, key_fn, rng):
    """Mean over designs built exactly by both groups of (share in b - share in a)."""
    def per_design(rows):
        out = {}
        for r in rows:
            out.setdefault(r["design"], []).append(1.0 if key_fn(r) else 0.0)
        return {d: np.mean(v) for d, v in out.items()}
    a, b = per_design(rows_a), per_design(rows_b)
    common = sorted(set(a) & set(b))
    diff = np.array([b[d] - a[d] for d in common])
    idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    bs = diff[idx].mean(1)
    return {"diff_pp": float(diff.mean() * 100), "lo_pp": float(np.quantile(bs, 0.025) * 100),
            "hi_pp": float(np.quantile(bs, 0.975) * 100), "n_designs": len(common)}


def main():
    ro = json.loads((HERE / "rollouts.json").read_text())
    agent_group = {}
    for scale, groups in ro["agents"].items():
        for g, trials in groups.items():
            for t in trials:
                agent_group[(scale, t)] = GROUPS[scale][g]
    jobs = []
    for key, eps in ro["episodes"].items():
        scale, agent = key.split("/")
        for di, e in enumerate(eps):
            if e["exact"]:
                jobs.append((scale, agent, di, e["bricks"]))
    djobs = [(s, di, d["bricks"]) for s in ro["designs"] for di, d in enumerate(ro["designs"][s])]
    hidden = {s: [has_hidden(d["occ"]) for d in ro["designs"][s]] for s in ro["designs"]}

    t0 = time.time()
    with Pool(10) as pool:
        probe = pool.map(job, jobs[:100])
        per = (time.time() - t0) / 100
        print(f"projected time for {len(jobs)} builds: {per * len(jobs) / 60:.1f} min", flush=True)
        if per * len(jobs) > 3600:
            raise SystemExit("projected over 1 hour: switch to the subset rule in PLAN.md")
        rows = probe + pool.map(job, jobs[100:], chunksize=16)
        drows = pool.map(design_job, djobs, chunksize=16)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    for r in rows:
        r["group"] = agent_group[(r["scale"], r["agent"])]
        r["hidden"] = hidden[r["scale"]][r["design"]]
    with open(HERE / "records.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    rng = np.random.default_rng(20261108)
    res = {"n_exact_builds": len(rows), "seeds_per_build": SEEDS, "by_scale": {}}
    stable = lambda r: r["final_class"] == "stable"
    for scale in SPECS:
        rs = [r for r in rows if r["scale"] == scale]
        out = {"groups": {}}
        for g in sorted({r["group"] for r in rs}):
            gr = [r for r in rs if r["group"] == g]
            gh = [r for r in gr if r["hidden"]]
            gn = [r for r in gr if not r["hidden"]]
            out["groups"][g] = {
                "n_builds": len(gr),
                "class_counts": {c: sum(r["final_class"] == c for r in gr) for c in ("stable", "marginal", "unstable")},
                "order_counts": {c: sum(r["order_class"] == c for r in gr) for c in ("stable", "marginal", "unstable")},
                "robustly_stable_all": boot_share(gr, stable, None, rng),
                "robustly_stable_hidden": boot_share(gh, stable, None, rng),
                "not_robust_hidden": boot_share(gh, lambda r: not stable(r), None, rng),
                "robustly_stable_no_hidden": boot_share(gn, stable, None, rng),
                "buildable_in_order": boot_share(gr, lambda r: r["order_class"] == "stable", None, rng),
            }
            # MuJoCo share over all replays (seeds pooled within each build)
            by_d = {}
            for r in gr:
                by_d.setdefault(r["design"], []).extend(r["mujoco"])
            ds = list(by_d)
            sums = np.array([sum(by_d[d]) for d in ds], dtype=float)
            cnts = np.array([len(by_d[d]) for d in ds], dtype=float)
            idx = rng.integers(0, len(ds), size=(BOOT, len(ds)))
            bs = sums[idx].sum(1) / cnts[idx].sum(1)
            out["groups"][g]["mujoco_stands"] = {"share": float(sums.sum() / cnts.sum()),
                                                  "lo": float(np.quantile(bs, 0.025)),
                                                  "hi": float(np.quantile(bs, 0.975)), "n_replays": int(cnts.sum())}
        # P4: MuJoCo against the order classification
        p4 = {}
        for c in ("stable", "marginal", "unstable"):
            rep = [s for r in rs if r["order_class"] == c for s in r["mujoco"]]
            p4[c] = {"n_replays": len(rep), "stand_share": float(np.mean(rep)) if rep else None}
        out["mujoco_by_order_class"] = p4
        # designs as generated (the generator's own bricks), descriptive
        dr = [r for r in drows if r["scale"] == scale]
        out["designs_generator_bricks"] = {
            "n": len(dr), "share_hidden": float(np.mean(hidden[scale])),
            "class_counts": {c: sum(r["final_class"] == c for r in dr) for c in ("stable", "marginal", "unstable")},
            "order_counts": {c: sum(r["order_class"] == c for r in dr) for c in ("stable", "marginal", "unstable")},
        }
        res["by_scale"][scale] = out
    small = [r for r in rows if r["scale"] == "small"]
    res["P3_claude_minus_baseline_robustly_stable"] = boot_paired(
        [r for r in small if r["group"] == "baseline"], [r for r in small if r["group"] == "claude"], stable, rng)
    res["P3_claude_minus_baseline_hidden_only"] = boot_paired(
        [r for r in small if r["group"] == "baseline" and r["hidden"]],
        [r for r in small if r["group"] == "claude" and r["hidden"]], stable, rng)
    large = [r for r in rows if r["scale"] == "large"]
    res["large_best_minus_baseline_robustly_stable"] = boot_paired(
        [r for r in large if r["group"] == "baseline"], [r for r in large if r["group"] == "best_transfer"], stable, rng)
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
