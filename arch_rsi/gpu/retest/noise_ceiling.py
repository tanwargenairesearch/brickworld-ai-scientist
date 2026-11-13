"""Descriptive, outside PROTOCOL_R.md (added 9 Nov 2026 after the report): is the low rank agreement across recipes
(Spearman 0.25) just seed noise? Split-half rank agreement of the 25 distinct picks within each recipe, over all 35
splits of their 8 confirmation seeds into two halves (seed 1001 fixed in the first half). No new runs.
usage (project root): .venv/bin/python -I arch_rsi/gpu/retest/noise_ceiling.py
"""
import itertools
import json
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parents[2]


def ranks(v):
    v = np.asarray(v, float)
    r = np.empty(len(v))
    r[v.argsort()] = np.arange(len(v), dtype=float)
    for x in np.unique(v):
        r[v == x] = r[v == x].mean()
    return r


def sp(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


R = json.loads((A / "gpu/retest/runs/results.json").read_text())
picks = {s["sha"]: s for arm in ("g0", "nofb", "menu") for s in R["per_arm"][arm]["searches"] if s["sha"] != "764c7fb529694085"}
shas = sorted(picks)
new = np.array([picks[h]["per_seed"] for h in shas])
old = np.array([[(lambda r: r["dev_exact"] if r["status"] == "ok" and r["dev_exact"] is not None else 0.0)(
    json.loads((A / f"campaigns/stage1/confirm/{h}/seed{s}/result.json").read_text())) for s in range(1001, 1009)] for h in shas])


def split_half(M):
    vals = [sp(M[:, list(a)].mean(1), M[:, [i for i in range(8) if i not in a]].mean(1))
            for a in itertools.combinations(range(8), 4) if 0 in a]
    return {"mean": round(float(np.mean(vals)), 3), "min": round(min(vals), 3), "max": round(max(vals), 3), "n_splits": len(vals)}


out = {"n_picks": len(shas), "across_recipes_8_seed_means": round(sp(old.mean(1), new.mean(1)), 3),
       "split_half_old_recipe": split_half(old), "split_half_new_recipe": split_half(new)}
sb = {k: round(2 * v["mean"] / (1 + v["mean"]), 3) for k, v in (("old", out["split_half_old_recipe"]), ("new", out["split_half_new_recipe"]))}
out["spearman_brown_8_seed_reliability"] = sb
out["approx_ceiling_across_recipes"] = round(float(np.sqrt(sb["old"] * sb["new"])), 3)
(A / "gpu/retest/noise_ceiling.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
