"""Read-only measurements on the finished Stage 2b campaign, for planning Stage 2e. Trains nothing, calls nothing."""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path("/path/to/brickworld-ai-scientist")
C = ROOT / "arch_rsi/campaigns/stage2b"
R = json.loads((C / "results.json").read_text())
assert R["primary_family"]["status"] == "final"
ARMS = ("g0", "nofb", "netonly", "reconly", "random")
N = 512


def ex(x):
    k = round(x * N)
    assert abs(k / N - x) < 6e-5, x
    return k / N


def load(arm, i):
    d = C / "W" / arm / f"s{i}"
    return json.loads((d / "state.json").read_text()), json.loads((d / "pick.json").read_text()), d


out = {}
BAR = json.loads((C / "baseline/summary.json").read_text())["w"]
print("start seeds 0-3:", BAR["devs"], "shown (seeds 0-1):", BAR["shown_dev"], "4-seed mean:", BAR["mean"])

# (a) seed noise of a try's score
print("\n(a) seed noise of a try (2 training seeds per candidate)")
for arm in ARMS:
    diffs2, within_tot, n_tr, both_low = [], [], 0, 0
    for i in range(1, 17):
        s, p, d = load(arm, i)
        means = []
        for t in s["trials"]:
            r = t["result"]
            if r["status"] != "ok":
                continue
            a, b = (ex(v) for v in r["dev_per_seed"])
            diffs2.append((a - b) ** 2 / 2)      # unbiased variance of one seed, from a pair
            means.append((a + b) / 2)
            n_tr += 1
        within_tot.append(st.pvariance(means) * len(means) / (len(means) - 1))
    s_w2 = st.mean(diffs2)
    v_tot = st.mean(within_tot)
    v_true = max(0.0, v_tot - s_w2 / 2)
    rel = lambda k: v_true / (v_true + s_w2 / k)
    out[arm] = {"seed_sd": s_w2 ** 0.5, "v_tot": v_tot, "v_true": v_true, "rel2": rel(2), "rel3": rel(3), "rel4": rel(4), "n": n_tr}
    print(f"  {arm:8s} n={n_tr:3d}  SD between seeds of one candidate {s_w2 ** 0.5:.3f} | SD of 2-seed means among a search's candidates "
          f"{v_tot ** 0.5:.3f} | true SD among candidates {v_true ** 0.5:.3f} | reliability 2 seeds {rel(2):.2f}, 3 seeds {rel(3):.2f}, 4 seeds {rel(4):.2f}")

# retest: the top 3's first two seeds against their two new seeds
print("\n    the top 3 retested: first-2-seed mean vs the 2 new seeds (regression to the mean)")
for arm in ARMS:
    first, second = [], []
    for i in range(1, 17):
        s, p, d = load(arm, i)
        for row in p["retested"]:
            if not row["all_ok"]:
                continue
            v = [ex(x) for x in row["dev_all"]]
            first.append(st.mean(v[:2])); second.append(st.mean(v[2:]))
    c = st.correlation(first, second) if len(first) > 2 else float("nan")
    print(f"  {arm:8s} n={len(first)}  first two seeds {st.mean(first) * 100:.1f}%  two new seeds {st.mean(second) * 100:.1f}%  drop {100 * (st.mean(first) - st.mean(second)):.1f} pts  corr {c:.2f}")

# (b) confirmation noise
print("\n(b) noise in a pick's confirmed score (8 fresh seeds)")
for arm in ARMS:
    ss = R["per_arm"][arm]["searches"]
    wv = [st.variance([ex(x) for x in s["per_seed"]]) for s in ss]
    prim = [st.mean([ex(x) for x in s["per_seed"]]) for s in ss]
    vb = st.variance(prim)
    se2 = st.mean(wv) / 8
    out[arm] |= {"confirm_seed_sd": st.mean(wv) ** 0.5, "confirm_se": se2 ** 0.5, "between_sd": vb ** 0.5,
                 "between_true_sd": max(0, vb - se2) ** 0.5}
    print(f"  {arm:8s} SD between a pick's seeds {st.mean(wv) ** 0.5:.3f} | SE of an 8-seed mean {se2 ** 0.5:.3f} (12 seeds {(st.mean(wv) / 12) ** 0.5:.3f}, 16 seeds {(st.mean(wv) / 16) ** 0.5:.3f})"
          f" | SD between searches {vb ** 0.5:.3f} | share of that variance from confirmation noise {se2 / vb:.2f}")

# (c) how much of the final score is set by round 1
print("\n(c) round 1 and the final confirmed score")
pool_b, pool_y = [], []
for arm in ARMS:
    ss = {s["search"]: s for s in R["per_arm"][arm]["searches"]}
    b1, y, r1pick, rounds, b1best_is_pick = [], [], 0, [], 0
    for i in range(1, 17):
        s, p, d = load(arm, i)
        r1 = [t for t in s["trials"] if t["round"] == 1 and t["result"]["status"] == "ok"]
        best1 = max([t["result"]["dev_exact"] for t in r1], default=0.0)
        b1.append(max(best1, BAR["shown_dev"]))
        y.append(st.mean([ex(x) for x in ss[i]["per_seed"]]))
        lab = p["label"]
        rounds.append(0 if lab == "start" else int(lab[1]))
        r1pick += lab.startswith("r1")
    c = st.correlation(b1, y)
    pool_b += [v - st.mean(b1) for v in b1]; pool_y += [v - st.mean(y) for v in y]
    out[arm] |= {"r1_best_mean": st.mean(b1), "corr_r1_final": c, "picks_from_r1": r1pick,
                 "pick_round_counts": {k: rounds.count(k) for k in sorted(set(rounds))}}
    print(f"  {arm:8s} round-1 best (2 seeds, floored at the bar) mean {st.mean(b1) * 100:.1f}% SD {st.stdev(b1):.3f} | corr with final confirmed score {c:+.2f} (R2 {c * c:.2f})"
          f" | picks by round {out[arm]['pick_round_counts']}")
g, n = out["g0"], out["nofb"]
cb = st.correlation(pool_b[:32], pool_y[:32])
print(f"  g0 and nofb pooled (arm means removed, n=32): corr {cb:+.2f}, R2 {cb * cb:.2f}")
# the same index in two arms shares nothing: a sanity check that should be near 0
gy = [st.mean([ex(x) for x in s["per_seed"]]) for s in R["per_arm"]["g0"]["searches"]]
ny = [st.mean([ex(x) for x in s["per_seed"]]) for s in R["per_arm"]["nofb"]["searches"]]
print(f"  sanity: corr of g0 search i with nofb search i (independent searches) {st.correlation(gy, ny):+.2f}")
d = [a - b for a, b in zip(gy, ny)]
print(f"  unpaired design: SD of (g0 search i - nofb search i) {st.stdev(d):.3f}; sqrt(sd_g0^2 + sd_nofb^2) {(st.variance(gy) + st.variance(ny)) ** 0.5:.3f}")
out["unpaired_diff_sd"] = (st.variance(gy) + st.variance(ny)) ** 0.5
json.dump(out, open(Path(__file__).with_name("measure_2b.json"), "w"), indent=1)
