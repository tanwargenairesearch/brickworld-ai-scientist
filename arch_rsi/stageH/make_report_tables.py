"""Prints REPORT.md's tables for Stage H1 from runs/results.json only (no number is typed by hand).
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/make_report_tables.py > /path/tables.md
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
R = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "runs" / "results.json").read_text())
EV = R["evaluation"]
NAMES = {"shift": "H1, shifted booklet", "mirror": "H2, mirrored hands", "torus": "H3, wraparound edges", "late": "H4, late feedback"}


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


print(f"Status: {R['status']}; written {R['written_utc']} UTC; {R['n_runs']} runs recorded; "
      f"{R['total_wall_min']} min from the checks to the end of the analysis.")
print(f"Preflight: {R['preflight']}\nPostflight: {R['postflight']}\n")
print(f"Fallback triggered for: {R['fallback_triggered'] or 'none'}\n")
print("| Rule | Budget | Plain, ref A | Plain, fix | Rule, ref A | Rule, fix | Damage, points | Repair, points (95% interval) | p | Holm p | Rule-specific gain, points (95% interval) | Legal | Complete | Passes |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r, n in NAMES.items():
    d = EV["rules"][r]
    m = d["means"]
    print(f"| {n} | {d['budget']:,} | {pct(m['pA'])} | {pct(m['pF'])} | {pct(m['rA'])} | {pct(m['rF'])} | {pts(d['damage'], False)} "
          f"| {pts(d['repair'])} ({ci(d['repair_ci95_boot'])}) | {p(d['p_used'])} | {p(d['p_holm'])} "
          f"| {pts(d['interaction'])} ({ci(d['interaction_ci95_boot'])}) | {yes(d['legal'])} | {yes(d['complete'])} | **{yes(d['passes'])}** |")
print(f"\nDecision: **{EV['decision']}**; {EV['n_pass']} of 4 pass: {EV['passing'] or 'none'}.\n")
print("| Rule | Repair at least 25 points with Holm p < 0.05 | Rule-specific gain at least 20 points | Legal | Complete | Recovery fraction |")
print("|---|---|---|---|---|---|")
for r, n in NAMES.items():
    d = EV["rules"][r]
    rf = d["recovery_fraction"]
    print(f"| {n} | {yes(d['pass_repair'])} | {yes(d['pass_specific'])} | {yes(d['pass_legal'])} | {yes(d['pass_complete'])} "
          f"| {'n/a' if rf is None else f'{rf:.2f}'} |")
print("\nLegality checks (fix in its rule world, check designs, GPU):\n")
print("| Rule | Passed | Untrained: exact | Untrained: clean fill | After the short run: exact | Short-run updates | Error |")
print("|---|---|---|---|---|---|---|")
for r, n in NAMES.items():
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
print("\nWhere the wrong cells are (share in edge rows and columns 0 and 7; the edges are 43.8% of the board):\n")
print("| Cell | Wrong cells (all seeds) | Share on the edges |")
print("|---|---|---|")
for k, c in R["cells"].items():
    e = c["edge"] or {}
    print(f"| {k} | {e.get('wrong_total', 'n/a')} | {pct(e.get('edge_share'))} |")
h = R["h1_identity"]
print(f"\nH1 identity (shift fix against plain reference A, same seeds): all per-design results identical: {h['all_identical']}")
for row in h["rows"]:
    print(f"- seed {row['seed']}: " + (f"plain_A {pct(row['plain_A'])}, shift_F {pct(row['shift_F'])}, same per design {row['same_per_design']}"
                                     if row["comparable"] else "not comparable (a run failed)"))
v = R["plain_A_vs_2c"]
print(f"\nReference A on plain W_s: {pct(v['plain_A_mean'])} here (hplain_dev, GPU) against {pct(v['ref_2c_mean'])} in 2c's calibration "
      f"(ws_dev, CPU); permutation p {p(v['perm_p'])}. {v['note']}")
print("\nLearning curves (mean training-episode exact rate at the last logged update):\n")
for k, c in R["cells"].items():
    lc = c["learning_curve"]
    if lc:
        print(f"- {k}: {pct(lc[-1]['train_exact'])} at {lc[-1]['steps']:,} steps")
