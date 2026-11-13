"""Read-only: run times, LLM call costs and prompt growth in Stage 2b; later rounds vs the final score."""
import json
import statistics as st
from datetime import datetime
from pathlib import Path

C = Path("/path/to/brickworld-ai-scientist/arch_rsi/campaigns/stage2b")
R = json.loads((C / "results.json").read_text())
ARMS = ("g0", "nofb", "netonly", "reconly", "random")
BAR = json.loads((C / "baseline/summary.json").read_text())["w"]["shown_dev"]
ex = lambda x: round(x * 512) / 512

# ---- (d) training runs
print("(d) training runs")
tot_runs = tot_wall = 0
per_arm = {}
for arm in ARMS:
    walls = [json.loads(f.read_text()) for f in (C / "W" / arm).glob("s*/trials/*/seed*/result.json")]
    w = [r["wall_s"] for r in walls]
    tw = [r.get("train_wall_s") or 0 for r in walls]
    per_arm[arm] = {"runs": len(w), "sum_h": sum(w) / 3600, "mean_s": st.mean(w), "median_s": st.median(w), "max_s": max(w)}
    tot_runs += len(w); tot_wall += sum(w)
    print(f"  {arm:8s} trial+retest runs {len(w):4d}  wall mean {st.mean(w):6.1f} s  median {st.median(w):6.1f}  max {max(w):6.1f}  sum {sum(w) / 3600:5.1f} h  (train part mean {st.mean(tw):.1f} s)")
conf = [json.loads(f.read_text()) for f in (C / "confirm").glob("*/seed*/result.json")]
cw = [r["wall_s"] for r in conf]
print(f"  confirm+comparator runs {len(cw)}  wall mean {st.mean(cw):.1f} s  sum {sum(cw) / 3600:.1f} h")
base = [json.loads(f.read_text())["wall_s"] for f in (C / "baseline").glob("w/seed*/result.json")]
tot_runs += len(cw) + len(base); tot_wall += sum(cw) + sum(base)
# confirm runs by arm (g0 and nofb picks only)
shas = {a: {s["sha"] for s in R["per_arm"][a]["searches"]} for a in ARMS}
for a in ("g0", "nofb"):
    ws = [json.loads(f.read_text())["wall_s"] for h in shas[a] for f in (C / "confirm" / h).glob("seed*/result.json")]
    per_arm[a]["confirm_mean_s"] = st.mean(ws)
    print(f"  {a} picks' confirm runs: {len(ws)} mean {st.mean(ws):.1f} s")
lines = (C / "progress.log").read_text().splitlines()
t = lambda s: datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
elapsed = (t(lines[-1]) - t(lines[0])).total_seconds()
print(f"  all runs on disk: {tot_runs}  summed wall {tot_wall / 3600:.1f} h  | elapsed {elapsed / 3600:.2f} h (incl. ~25 min stopped) | workers {json.loads((C / 'manifest.json').read_text())['config']['workers']}")
run_elapsed = elapsed - 25 * 60
print(f"  elapsed seconds per run on disk (excl. the stop): {run_elapsed / tot_runs:.1f} s | summed wall / 4 slots = {tot_wall / 4 / 3600:.1f} h -> elapsed is {run_elapsed / (tot_wall / 4):.2f}x that")
gn = [r for a in ("g0", "nofb") for r in [json.loads(f.read_text())["wall_s"] for f in (C / "W" / a).glob("s*/trials/*/seed*/result.json")]]
print(f"  g0+nofb trial/retest runs: mean wall {st.mean(gn):.1f} s (4 at once) -> {st.mean(gn) / 4:.1f} s of machine time per run")

# ---- LLM calls
print("\n    LLM calls (the four AI-scientist arms)")
by_round = {}
n_calls = usd = 0
lat = []
for arm in ("g0", "nofb", "netonly", "reconly"):
    for f in (C / "W" / arm).glob("s*/round_*.json"):
        if f.name.endswith(".failed.json"):
            continue
        rec = json.loads(f.read_text())
        rep = "repair" in f.name
        r = int(f.name.split("_")[1].split(".")[0])
        u = rec["usage"]
        key = (arm, r, rep)
        by_round.setdefault(key, []).append((u.get("notional_cost_usd") or 0, len(rec["prompt"]), rec["latency_s"],
                                             (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0), u.get("output_tokens") or 0,
                                             rec["attempt"]))
        n_calls += 1; usd += u.get("notional_cost_usd") or 0; lat.append(rec["latency_s"])
print(f"  call logs on disk: {n_calls}, notional ${usd:.2f}; results.json says {sum(c['claude_calls'] for c in R['costs'].values())} calls, ${sum(c['claude_notional_usd'] for c in R['costs'].values()):.2f}")
print(f"  latency per call: mean {st.mean(lat):.0f} s, median {st.median(lat):.0f} s, max {max(lat):.0f} s")
for arm in ("g0", "nofb"):
    for r in (1, 2, 3, 4):
        v = by_round.get((arm, r, False), [])
        print(f"  {arm:5s} round {r}: calls {len(v):2d}  mean ${st.mean(x[0] for x in v):.3f}  prompt {st.mean(x[1] for x in v) / 1000:6.1f}k chars  input tokens {st.mean(x[3] for x in v):8.0f}  output tokens {st.mean(x[4] for x in v):7.0f}  latency {st.mean(x[2] for x in v):5.0f} s")
    reps = [x for k, v in by_round.items() if k[0] == arm and k[2] for x in v]
    print(f"  {arm:5s} repair calls: {len(reps)}" + (f"  mean ${st.mean(x[0] for x in reps):.3f}" if reps else ""))
# candidate code size
sizes = [len(t["code"]) for a in ("g0", "nofb") for i in range(1, 17) for t in json.loads((C / "W" / a / f"s{i}" / "state.json").read_text())["trials"]]
print(f"  candidate file size (g0, nofb): mean {st.mean(sizes) / 1000:.1f}k chars, max {max(sizes) / 1000:.1f}k")

# ---- later rounds and the final score
print("\n(c2) running best after each round vs the final confirmed score; spread of the running best")
for arm in ("g0", "nofb", "netonly"):
    fin = {s["search"]: st.mean(ex(x) for x in s["per_seed"]) for s in R["per_arm"][arm]["searches"]}
    row = []
    for r in (1, 2, 3, 4):
        b = []
        for i in range(1, 17):
            trials = json.loads((C / "W" / arm / f"s{i}" / "state.json").read_text())["trials"]
            b.append(max([BAR] + [t["result"]["dev_exact"] for t in trials if t["round"] <= r and t["result"]["status"] == "ok"]))
        row.append(f"r{r}: mean {st.mean(b) * 100:.1f} SD {st.stdev(b):.3f} corr {st.correlation(b, [fin[i] for i in range(1, 17)]):+.2f}")
    print(f"  {arm:8s} " + " | ".join(row))
# how often the final pick was already among the candidates after round r
for arm in ("g0", "nofb"):
    picks = [json.loads((C / "W" / arm / f"s{i}" / "pick.json").read_text())["label"] for i in range(1, 17)]
    print(f"  {arm}: picks {picks}")
