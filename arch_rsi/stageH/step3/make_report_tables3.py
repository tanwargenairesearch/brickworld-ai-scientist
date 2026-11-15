"""Prints the report's tables for Step 3 from a campaign's results.json only (no number is typed by hand).
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3/make_report_tables3.py [campaigns/step3/results.json]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "campaigns" / "step3" / "results.json").read_text())


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def pts(x):
    return "n/a" if x is None else f"{100 * x:+.1f}"


def ci(c):
    return "n/a" if not c else f"{100 * c[0]:+.1f} to {100 * c[1]:+.1f}"


def p(x):
    return "n/a" if x is None else (f"{x:.4f}" if x >= 0.0001 else f"{x:.1e}")


print(f"Status: {R['status']}; preset {R.get('preset')}; written {R.get('written_utc')} UTC; {R.get('total_wall_h')} h.")
print(f"Preflight: {R.get('preflight')}\nPostflight: {R.get('postflight')}\n")
print(f"**Ending {R['ending']}:** {R['ending_text']}\n")
print("| Test | Arms | Difference, points | 95% interval | p | Holm p | Passes |")
print("|---|---|---|---|---|---|---|")
for k, v in R["primary"].items():
    extra = f" (lineages {', '.join(v['lineages']) or 'none'})" if k == "P4" else ""
    print(f"| {k}{extra} | {v['arms'][0]} − {v['arms'][1]} | {pts(v['diff'])} | {ci(v['ci95_boot'])} | {p(v['p'])} "
          f"| {p(v['p_holm'])} | **{'yes' if v['passes'] else 'no'}** |")
c = R["check"]
print(f"\nCheck (g0 with readouts): {c['found']} of {c['n']} searches at 38.3% or more; {c['needed']} needed: "
      f"{'passes' if c['passes'] else 'fails'}. A kept rewrite changed the header: {'yes' if R['kept_rewrite_changed_header'] else 'no'}.\n")
print("| Arm | Searches | Mean | By world | Found | Found by world | Picks the start | Violations | Judge yes/partly/no |")
print("|---|---|---|---|---|---|---|---|---|")
for a, v in R["per_arm"].items():
    bw = ", ".join(f"{w} {pct(x)}" for w, x in v["by_world"].items())
    fw = ", ".join(f"{w} {x}" for w, x in v["found_by_world"].items())
    j = v["judge"]
    print(f"| {a} | {v['n']} | {pct(v['mean'])} | {bw} | {v['found']} | {fw} | {v['picks_start']} | {v['violations']} "
          f"| {j['yes']}/{j['partly']}/{j['no']} |")
print("\n| Secondary | Difference, points | 95% interval | Holm p |")
print("|---|---|---|---|")
for k, v in R["secondary"].items():
    print(f"| {k} | {pts(v['diff'])} | {ci(v['ci95_boot'])} | {p(v['p_holm'])} |")
print("\nMeta-training, per lineage and generation:\n")
print("| Lineage | Generation | Program | Header (R, K, seeds) | Mean confirmed | Per search | Kept | Parent header used |")
print("|---|---|---|---|---|---|---|---|")
for L, lin in R["lineages"].items():
    for g in lin["generations"]:
        s = g.get("settings") or {}
        hdr = f"{s.get('R')}, {s.get('K')}, {len(s.get('trial_seeds', []))}" if s else "n/a"
        kept = "start" if g["kept"] is None else ("yes" if g["kept"] else "no")
        per = ", ".join(f"{x:.3f}" for x in g.get("search_means") or [])
        print(f"| {L} | {g['generation']} | {g['program']} | {hdr} | {pct(g['mean'])} | {per} | {kept} | {'yes' if g.get('header_fallback') else ''} |")
    print(f"| {L} | final | {lin['final']} | | | | | |")
print(f"\nNotes: {R.get('notes')}\n")
print("| Program | Words | Kinds of change named (counts) | Program review |")
print("|---|---|---|---|")
rev = {}
for h, v in R.get("program_review", {}).items():
    for prog in v.get("programs") or []:
        rev[prog] = v["verdict"]
for k, v in R["programs"].items():
    named = ", ".join(f"{w} {n}" for w, n in v["named"].items() if n)
    print(f"| {k} | {v['words']} | {named or 'none'} | {rev.get(k, 'n/a')} |")
print("\n| Calls | Count | Notional cost |")
print("|---|---|---|")
for k, v in sorted(R["calls"].items()):
    print(f"| {k} | {v['calls']} | ${v['cost_usd']:.2f} |")
tot = sum(v["cost_usd"] for v in R["calls"].values())
print(f"| total | {sum(v['calls'] for v in R['calls'].values())} | ${tot:.2f} |")
