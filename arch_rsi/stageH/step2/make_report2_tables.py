"""Print REPORT_STEP2.md's tables from campaigns/step2/results.json (or a path given as the argument)."""
import json
import sys
from pathlib import Path

P = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "campaigns" / "step2" / "results.json"
R = json.loads(P.read_text())
ARMS = ("told", "readouts", "scores", "hidden")
NAME = {"told": "Told the rule", "readouts": "Readouts", "scores": "Scores only", "hidden": "Scores hidden"}


def pct(x):
    return f"{100 * x:.1f}%"


def pv(p):
    return "< 0.000001" if p < 1.5e-6 else (f"{p:.6f}" if p < 0.001 else f"{p:.3f}")


print("| Arm | Found (all) | Shift: found | Mirror: found | Mean score | Shift mean | Mirror mean | Picks = start | Violations | Judge yes / partly / no |")
print("|---|---|---|---|---|---|---|---|---|---|")
for a in ARMS:
    pa = R["per_arm"][a]
    rows = [r for r in R["rows"] if r["arm"] == a]
    fw = {w: sum(r["found"] for r in rows if r["world"] == w) for w in ("shift", "mirror")}
    nw = {w: sum(1 for r in rows if r["world"] == w) for w in ("shift", "mirror")}
    j = pa["judge"]
    print(f"| {NAME[a]} | {pa['found']}/{pa['n']} | {fw['shift']}/{nw['shift']} | {fw['mirror']}/{nw['mirror']} | {pct(pa['mean'])} | "
          f"{pct(pa['by_world']['shift'])} | {pct(pa['by_world']['mirror'])} | {pa['picks_start']} | {pa['violations']} | "
          f"{j['yes']} / {j['partly']} / {j['no']} |")
print()
print("| Test | Arms | Difference in mean score | 95% bootstrap interval | Permutation p | Holm p | Passes |")
print("|---|---|---|---|---|---|---|")
for grp in ("primary", "secondary"):
    for k, v in R[grp].items():
        lo, hi = v["ci95_boot"]
        passes = ("yes" if v["passes"] else "no") if "passes" in v else "(secondary, no pass rule)"
        print(f"| {k} | {NAME[v['arms'][0]]} − {NAME[v['arms'][1]].lower()} | {100 * v['diff']:+.1f} points | "
              f"{100 * lo:+.1f} to {100 * hi:+.1f} | {pv(v['p'])} | {pv(v['p_holm'])} | {passes} |")
print()
print("| Arm and world | Confirmed score of each search's pick (search 0 to 11) |")
print("|---|---|")
for w in ("shift", "mirror"):
    for a in ARMS:
        rows = sorted((r for r in R["rows"] if r["arm"] == a and r["world"] == w), key=lambda r: r["search"])
        print(f"| {NAME[a]}, {w} | {', '.join(f'{100 * r['score']:.1f}' for r in rows)} |")
print()
print("| Calls | Number | Notional cost |")
print("|---|---|---|")
tot_n = tot_c = 0
for k, v in R["calls"].items():
    tot_n += v["calls"]; tot_c += v["cost_usd"]
    print(f"| {NAME.get(k, k.capitalize())} | {v['calls']} | ${v['cost_usd']:.2f} |")
print(f"| **All** | **{tot_n}** | **${tot_c:.2f}** |")
print()
c = R["check"]
print(f"check: told found {c['found']}/{c['n']}, needed {c['needed']}, passes {c['passes']}; ending {R['ending']}; "
      f"discovery claim allowed {R['discovery_claim_allowed']}; all reviewed {R['all_reviewed']}; wall {R['total_wall_h']} h; "
      f"preflight ok {R['preflight']['ok']} changed {R['preflight']['changed']}; postflight ok {R['postflight']['ok']} changed {R['postflight']['changed']}")
print("ending text:", R["ending_text"])
