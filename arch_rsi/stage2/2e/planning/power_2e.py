"""Planning power for Stage 2e's paired sign-flip test. A simulation with stated assumptions, not a result.

d_i = seen_i - hidden_i for pair i. Test: two-sided sign-flip test on mean(d) at 0.05 (Monte Carlo flips here; the
real analysis is exact). Scenarios for the spread of d:
  - normal with SD 0.17 / 0.22 / 0.26 (optimistic / central / pessimistic; see the protocol);
  - "2b": seen and hidden drawn independently from Stage 2b's 16 g0 and 16 nofb confirmed search scores, each arm
    centred, so the spread and shape are 2b's own (8 confirmation seeds, SD of a difference about 0.235).
Also: the chance, when the true gain is 0, that the 95% interval's upper end falls below +10 points; and the
two-stage rule (32 pairs, 16 more if the root mean square of the first 32 differences is above 0.20).
"""
import json
import sys
from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261110)
R = json.loads(Path("/path/to/brickworld-ai-scientist/arch_rsi/campaigns/stage2b/results.json").read_text())
sc = lambda a: np.array([np.mean([round(x * 512) / 512 for x in s["per_seed"]]) for s in R["per_arm"][a]["searches"]])
G, H = sc("g0") - sc("g0").mean(), sc("nofb") - sc("nofb").mean()
SIMS, FLIPS = 4000, 2000


def draw(kind, n, sims):
    if kind == "2b":
        return rng.choice(G, (sims, n)) - rng.choice(H, (sims, n))
    return rng.normal(0, kind, (sims, n))


def pvals(d):
    """two-sided sign-flip p for each row of d (sims x n)"""
    obs = np.abs(d.mean(1))
    out = np.empty(len(d))
    for i in range(0, len(d), 500):
        blk = np.abs(d[i:i + 500])
        s = rng.choice([-1.0, 1.0], (FLIPS, blk.shape[1]))
        t = np.abs(blk @ s.T) / blk.shape[1]                       # block x flips
        out[i:i + 500] = (1 + (t >= obs[i:i + 500, None] - 1e-12).sum(1)) / (1 + FLIPS)
    return out


res = {}
print(f"{'spread of d':>12s} {'pairs':>5s} | size  | power +5  +10  +15 | P(upper end < +10 | gain 0)")
for kind in (0.17, 0.22, 0.26, "2b"):
    for n in (24, 32, 40, 48, 64):
        row = {}
        for delta in (0.0, 0.05, 0.10, 0.15):
            d = draw(kind, n, SIMS) + delta
            p = pvals(d)
            row[delta] = float(((p < 0.05) & (d.mean(1) > 0)).mean()) if delta else float((p < 0.05).mean())
            if delta == 0:
                up = d.mean(1) + 1.96 * d.std(1, ddof=1) / np.sqrt(n)
                row["bounded"] = float((up < 0.10).mean())
        res[f"{kind}|{n}"] = row
        print(f"{str(kind):>12s} {n:5d} | {row[0.0]:.3f} | {row[0.05]:9.2f} {row[0.10]:4.2f} {row[0.15]:4.2f} | {row['bounded']:.2f}")

print("\ntwo-stage rule: 32 pairs; 16 more if rms(d over the first 32) > 0.20; sign-flip test on all pairs run")
for kind in (0.17, 0.22, 0.26, "2b"):
    out = {}
    for delta in (0.0, 0.05, 0.10, 0.15):
        d = draw(kind, 48, SIMS) + delta
        ext = np.sqrt((d[:, :32] ** 2).mean(1)) > 0.20
        p = np.where(ext, pvals(d), pvals(d[:, :32]))
        m = np.where(ext, d.mean(1), d[:, :32].mean(1))
        out[delta] = float(((p < 0.05) & (m > 0)).mean()) if delta else float((p < 0.05).mean())
        if delta in (0.0, 0.10):
            out[f"ext{delta}"] = float(ext.mean())
    res[f"two-stage|{kind}"] = out
    print(f"{str(kind):>12s} | size {out[0.0]:.3f} | power +5 {out[0.05]:.2f}  +10 {out[0.10]:.2f}  +15 {out[0.15]:.2f} | extended in {out['ext0.0']:.0%} of runs at gain 0, {out['ext0.1']:.0%} at +10")
print("\nSD of d in the 2b scenario:", round(float(np.sqrt(G.var(ddof=1) + H.var(ddof=1))), 4))
json.dump({str(k): v for k, v in res.items()}, open(Path(__file__).with_name("power_2e.json"), "w"), indent=1, default=str)
