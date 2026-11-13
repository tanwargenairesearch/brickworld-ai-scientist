"""Two extra planning figures quoted in PROTOCOL_2E.md: power at the second-stage threshold, and the pairs a 5-point
gain would need. Same simulated sign-flip test as power_2e.py (normal differences). Not a result."""
import numpy as np

rng = np.random.default_rng(7)


def power(sd, n, delta, sims=4000, flips=2000):
    d = rng.normal(delta, sd, (sims, n))
    obs, hit = np.abs(d.mean(1)), 0
    for i in range(0, sims, 250):
        blk = np.abs(d[i:i + 250])
        s = rng.choice([-1.0, 1.0], (flips, n))
        t = np.abs(blk @ s.T) / n
        p = (1 + (t >= obs[i:i + 250, None] - 1e-12).sum(1)) / (1 + flips)
        hit += ((p < 0.05) & (d[i:i + 250].mean(1) > 0)).sum()
    return hit / sims


print("sd 0.20, 32 pairs, +10:", power(0.20, 32, 0.10))
for n in (130, 150, 160):
    print("sd 0.22,", n, "pairs, +5:", power(0.22, n, 0.05, sims=2000, flips=1000))
