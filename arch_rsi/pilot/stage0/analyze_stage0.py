"""Summarise Stage 0. Reads dev designs and per-design dev results only; never reads holdout/.
usage: .venv/bin/python -I arch_rsi/pilot/stage0/analyze_stage0.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
C = HERE / "campaign"
DEV = {"large": ROOT / "targets" / "large_dev.pt", "xl": ROOT / "arch_rsi" / "targets" / "xl_dev.pt"}


def hidden(p):
    v = torch.load(p).bool()
    above = (v.flip(1).cumsum(1).flip(1) - v.int()) > 0
    return ((~v) & above).flatten(1).any(1).numpy()


H = {w: hidden(p) for w, p in DEV.items()}


def score(rdir, w):
    r = json.loads((rdir / "result.json").read_text())
    if r["status"] != "ok":
        return {"status": r["status"], "exact": 0.0, "hidden": 0.0, "wall": r["wall_s"]}
    p = np.array(json.loads((rdir / "dev.json").read_text())["per_design"]["exact"], bool)
    return {"status": "ok", "exact": float(p.mean()), "hidden": float(p[H[w]].mean()), "nohidden": float(p[~H[w]].mean()), "wall": r["wall_s"]}


out = {"floor_no_hidden_share": {w: float(1 - h.mean()) for w, h in H.items()}, "noise": {}}
for w in H:
    rows = [score(C / "baseline" / w / f"seed{s}", w) | {"seed": s} for s in range(4)]
    rows += [score(C / "noise" / w / f"seed{s}", w) | {"seed": s} for s in range(200, 206)]
    ex = np.array([r["exact"] for r in rows]); hd = np.array([r["hidden"] for r in rows])
    out["noise"][w] = {"seeds": [r["seed"] for r in rows], "exact": [round(x, 3) for x in ex], "hidden": [round(x, 3) for x in hd],
                       "exact_mean": round(float(ex.mean()), 3), "exact_sd": round(float(ex.std(ddof=1)), 3), "min": round(float(ex.min()), 3), "max": round(float(ex.max()), 3),
                       "seeds_at_floor(hidden<2%)": int((hd < .02).sum()), "seeds_hidden>=50%": int((hd >= .5).sum()),
                       "statuses": [r["status"] for r in rows], "wall_s_mean": round(float(np.mean([r["wall"] for r in rows])), 1)}
st = C / "Q" / "g0" / "s1" / "state.json"
if st.exists():
    s = json.loads(st.read_text()); T = []
    for t in s["trials"]:
        r = t["result"]; row = {"label": t["label"], "name": t["name"][:70], "status": r["status"], "dev": r.get("dev_exact"), "per_seed": r.get("dev_per_seed"),
                                "flops": r.get("flops_ratio"), "params": r.get("params"), "steps": r.get("train_steps")}
        if r["status"] == "ok":
            d = C / "Q" / "g0" / "s1" / "trials" / t["label"]
            row["hidden_per_seed"] = [round(score(d / f"seed{i}", "xl")["hidden"], 3) for i in (0, 1)]
        T.append(row)
    ok = [t for t in T if t["status"] == "ok"]
    out["ceiling_probe"] = {"trials": T, "best_dev_mean": max(t["dev"] for t in ok) if ok else None, "best_single_seed": max(max(t["per_seed"]) for t in ok) if ok else None,
                            "incumbent": s["incumbent"], "invalid": sum(t["status"] == "invalid" for t in T), "failed": sum(t["status"] not in ("ok", "invalid") for t in T)}
(HERE / "summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
