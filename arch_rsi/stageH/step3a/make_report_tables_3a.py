"""Prints the report's tables for Step 3a from runs/results.json only (no number is typed by hand).
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3a/make_report_tables_3a.py [results.json] > tables.md
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import rules3a as R3  # noqa: E402

R = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "runs" / "results.json").read_text())
EV = R["evaluation"]


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def pts(x, sign=True):
    return "n/a" if x is None else (f"{100 * x:+.1f}" if sign else f"{100 * x:.1f}")


def ci(c):
    return "n/a" if not c else f"{100 * c[0]:+.1f} to {100 * c[1]:+.1f}"


def p(x):
    return "n/a" if x is None else (f"{x:.4f}" if x >= 0.0001 else f"{x:.1e}")


def yes(b):
    return "yes" if b else "no"


print(f"Status: {R.get('status')}; written {R.get('written_utc')} UTC; {R['n_runs']} runs recorded; "
      f"{R.get('total_wall_min')} min from the tests to the end of the analysis.")
print(f"Guard: {R.get('guard')}\nPreflight: {R.get('preflight')}\nPostflight: {R.get('postflight')}\n")
print(f"Fallback triggered for: {R['fallback_triggered'] or 'none'}\n")
print("| Rule | Family | Budget | Plain, ref A | Plain, fix | Rule, ref A | Rule, fix | Damage, points | Repair, points (95% interval) | p | Holm p | Rule-specific gain, points (95% interval) | Legal | Complete | Passes |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r, n in R3.NAMES.items():
    d = EV["rules"][r]
    m = d["means"]
    print(f"| {n} | {d['family']} | {d['budget']:,} | {pct(m['pA'])} | {pct(m['pF'])} | {pct(m['rA'])} | {pct(m['rF'])} "
          f"| {pts(d['damage'], False)} | {pts(d['repair'])} ({ci(d['repair_ci95_boot'])}) | {p(d['p_used'])} | {p(d['p_holm'])} "
          f"| {pts(d['interaction'])} ({ci(d['interaction_ci95_boot'])}) | {yes(d['legal'])} | {yes(d['complete'])} | **{yes(d['passes'])}** |")
print(f"\n{EV['n_pass']} of 8 new rules pass: {EV['passing_ids'] or 'none'}.\n")
print("| Family | Members | Passing (H1 and H2 from Step 1) | Usable for Step 3 |")
print("|---|---|---|---|")
for f, v in EV["families"].items():
    print(f"| {f} | {', '.join(v['members'])} | {', '.join(v['passing']) or 'none'} | {yes(v['usable'])} |")
print(f"\nDecision: **{EV['decision']}** (usable families: {', '.join(EV['usable_families']) or 'none'}).\n")
print("| Rule | Repair at least 25 points with Holm p < 0.05 | Rule-specific gain at least 20 points | Legal | Complete | Recovery fraction |")
print("|---|---|---|---|---|---|")
for r, n in R3.NAMES.items():
    d = EV["rules"][r]
    rf = d["recovery_fraction"]
    print(f"| {n} | {yes(d['pass_repair'])} | {yes(d['pass_specific'])} | {yes(d['pass_legal'])} | {yes(d['pass_complete'])} "
          f"| {'n/a' if rf is None else f'{rf:.2f}'} |")
print("\nLegality checks (fix in its rule world, check designs, GPU):\n")
print("| Rule | Passed | Untrained: exact | Untrained: clean fill | After the short run: exact | Short-run updates | Error |")
print("|---|---|---|---|---|---|---|")
for r, n in R3.NAMES.items():
    c = EV["rules"][r]["check"] or {}
    u0, u1 = c.get("check_untrained") or {}, c.get("check_short_training") or {}
    print(f"| {n} | {yes(c.get('ok'))} | {pct(u0.get('exact'))} | {pct(u0.get('clean_fill'))} | {pct(u1.get('exact'))} "
          f"| {u1.get('short_run_updates', 'n/a')} | {c.get('error') or ''} |")
print("\nCells (per-seed scores in seed order 4000-4005; exact rate on designs with hidden cells; median training time):\n")
print("| Cell | Runs ok | Mean | SD | Per seed | Exact on hidden-cell designs | Median training s | Attempts |")
print("|---|---|---|---|---|---|---|---|")
for k, c in R["cells"].items():
    ps = ", ".join(pct(v) for v in c["per_seed"].values())
    print(f"| {k} | {c['n_ok']}/{c['n']} | {pct(c['mean'])} | {pts(c['sd'], False)} | {ps} | {pct(c['exact_hidden_mean'])} "
          f"| {c['train_wall_s_median']} | {c['attempts']} |")
print("\nWrong cells by row (0-7) and by column (0-7), all seeds:\n")
print("| Cell | By row | By column |")
print("|---|---|---|")
for k, c in R["cells"].items():
    print(f"| {k} | {c['wrong_by_row']} | {c['wrong_by_col']} |")
idt = R["identity"]
pr = idt["plain_reproduces_step1"]
print(f"\nThe plain cell against Step 1's (same seeds): all per-design results identical: {pr['all_identical']}")
for row in pr["rows"]:
    print(f"- seed {row['seed']}: " + (f"here {pct(row['here'])}, Step 1 {pct(row['step1'])}, same per design {row['same_per_design']}"
                                     if row["comparable"] else "not comparable (a run failed)"))
print("\nEach booklet rule's fixed cell against the plain cell (same seeds): all per-design results identical:")
for r, v in idt["booklet_fix_matches_plain"].items():
    print(f"- {R3.NAMES[r]}: {v['all_identical']}")
print("\nLearning curves (mean training-episode exact rate at the last logged update):\n")
for k, c in R["cells"].items():
    lc = c["learning_curve"]
    if lc:
        print(f"- {k}: {pct(lc[-1]['train_exact'])} at {lc[-1]['steps']:,} steps")
