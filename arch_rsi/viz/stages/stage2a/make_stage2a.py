"""Stage 2a article video: calibrating a harder world (W). Visualization only.

World W replaces world Q, and a pre-registered budget ladder picks the training budget at which W leaves room above
the start without being out of reach (arch_rsi/stage2/2a). Nothing here trains, scores or selects.

Beats (about 20 s, 1920x1080, 12 fps):
  1. The ladder: which budgets ran, what the start and the two reference networks scored at each (dots per seed, a
     bar for the mean), and which rule fired. Three frames: the empty ladder with its rules, the first rung, the rest.
  2. The chosen budget: the start's 20 seeds in its two seed ranges, reference A and reference B, as dots and means.
  3. Takeaway card.

Data (read-only; every number shown is read here and cross-checked against arch_rsi/stage2/2a/REPORT.md):
  arch_rsi/stage2/2a/PROTOCOL_2A.md            world sizes, the ladder's budgets, fallback and rules (parsed)
  arch_rsi/stage2/2a/run_2a.py                 the thresholds as coded (parsed, must match the protocol)
  arch_rsi/stage2/2a/runs/manifest.json        budgets, fallback, seeds, network files
  arch_rsi/stage2/2a/runs/decisions.json       the ladder and the chosen budget
  arch_rsi/stage2/2a/runs/summary.json         the noise step at the chosen budget
  arch_rsi/stage2/2a/runs/run.log              the ladder's logged readings (cross-check)
  arch_rsi/stage2/2a/runs/b<budget>/<cell>/    each run's result.json and dev.json (scores re-read from here)
  arch_rsi/gpu/retest/runs/results.json        the reference networks' retest scores on Q (card line)

Outputs (this directory): stage2a.mp4, stage2a.png (the last chart frame, beat 2), stage2a_data.json.

usage (project root, low priority while Stage 2b runs):
  nice -n 19 ./.venv/bin/python -I arch_rsi/viz/stages/stage2a/make_stage2a.py              # png + json + mp4
  nice -n 19 ./.venv/bin/python -I arch_rsi/viz/stages/stage2a/make_stage2a.py --no-video   # png + json only
  ... --frames DIR   also writes one png per beat to DIR, for checking layout
"""
import sys

sys.dont_write_bytecode = True  # write nothing outside the output directory

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
A = ROOT / "arch_rsi"
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from arch_rsi.viz.stages import vidkit as vk  # noqa: E402

S2A = A / "stage2" / "2a"
RUNS = S2A / "runs"
STAGE = "Stage 2a: calibrating world W"
NETS = ("start_g", "refA", "refB")
CELL = {"start_g": ("start", "g"), "start_f": ("start", "f"), "refA": ("refA", "g"), "refB": ("refB", "g")}
SRC = {"protocol": "arch_rsi/stage2/2a/PROTOCOL_2A.md", "runner": "arch_rsi/stage2/2a/run_2a.py",
       "manifest": "arch_rsi/stage2/2a/runs/manifest.json", "decisions": "arch_rsi/stage2/2a/runs/decisions.json",
       "summary": "arch_rsi/stage2/2a/runs/summary.json", "log": "arch_rsi/stage2/2a/runs/run.log",
       "runs": "arch_rsi/stage2/2a/runs/b<budget>/<net>_<recipe>_s<seed>/{result,dev}.json",
       "retest": "arch_rsi/gpu/retest/runs/results.json"}


def jread(p):
    return json.loads(Path(p).read_text())


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def num(s):
    return int(s.replace(",", ""))


def load():
    proto = (S2A / "PROTOCOL_2A.md").read_text()
    runner = (S2A / "run_2a.py").read_text()
    rep = (S2A / "REPORT.md").read_text()
    M = jread(RUNS / "manifest.json")
    DEC = jread(RUNS / "decisions.json")
    SUM = jread(RUNS / "summary.json")
    log = (RUNS / "run.log").read_text()
    assert SUM["decisions"] == DEC

    # --- the protocol's world and rules, parsed, and matched against the runner and the manifest
    w = re.search(r"\*\*W:\*\* (\d+)×(\d+)×(\d+), built from (\d+)–(\d+) bricks", proto).groups()
    q = re.search(r"On Q it's (\d+)×(\d+)×(\d+) with (\d+)–(\d+) bricks", proto).groups()
    ladder = [num(x) for x in re.search(r"\*\*Budgets:\*\* ([\d,]+), then ([\d,]+), then ([\d,]+) steps", proto).groups()]
    fallback = num(re.search(r"\*\*The smallest was too easy:\*\* run ([\d,]+) steps once", proto).group(1))
    easy_s, easy_r = (int(x) for x in re.search(r"\*\*Too easy:\*\* S > (\d+)% or R ≥ (\d+)%", proto).groups())
    hard = int(re.search(r"\*\*Too hard:\*\* both S and R below (\d+)%", proto).group(1))
    coded = [float(x) for x in re.search(r"EASY_S, EASY_R, HARD = ([\d.]+), ([\d.]+), ([\d.]+)", runner).groups()]
    assert [round(100 * c) for c in coded] == [easy_s, easy_r, hard]
    assert M["config"]["budgets"] == ladder and M["config"]["fallback"] == fallback

    def reading(S, R):
        return "too easy" if (S > easy_s / 100 or R >= easy_r / 100) else ("too hard" if max(S, R) < hard / 100 else "passes")

    # --- every run re-read from its directory
    def runs(budget, key, seeds):
        net, rk = CELL[key]
        out = []
        for s in seeds:
            d = RUNS / f"b{budget}" / f"{net}_{rk}_s{s}"
            r, dv = jread(d / "result.json"), jread(d / "dev.json")
            assert r["status"] == "ok" and r["budget"] == budget and abs(r["dev_exact"] - dv["exact"]) < 1e-12
            out.append({"seed": s, "score": dv["exact"], "learned": r["learned"], "steps": r["steps"],
                        "flops_ratio": r["flops_ratio"], "per_design": dv["per_design"]["exact"], "n": dv["n"]})
        return out

    seeds_lad = {"start_g": M["config"]["lad_start_g"], "start_f": M["config"]["lad_start_f"],
                 "refA": M["config"]["lad_ref"], "refB": M["config"]["lad_ref"]}
    rungs = {}
    for rd in DEC["ladder"]:
        b = rd["budget"]
        cells = {k: runs(b, k, seeds_lad[k]) for k in CELL}
        for k, rr in cells.items():
            c = rd["cells"][k]
            assert all(abs(x["score"] - v) < 6e-5 for x, v in zip(rr, c["per_seed"])), (b, k)
            assert abs(np.mean([x["score"] for x in rr]) - c["mean"]) < 6e-5 and sum(x["learned"] for x in rr) == c["learned"]
        S = float(np.mean([x["score"] for x in cells["start_g"]]))
        R = max(float(np.mean([x["score"] for x in cells[k]])) for k in ("refA", "refB"))
        assert abs(S - rd["S"]) < 6e-5 and abs(R - rd["R"]) < 6e-5 and reading(S, R) == rd["reading"], b
        rungs[b] = {"reading": rd["reading"], "S": S, "R": R, "cells": cells}
    tried = [rd["budget"] for rd in DEC["ladder"]]
    assert tried == [ladder[0], fallback] and rungs[ladder[0]]["reading"] == "too easy"  # stopped at the first rung
    assert rungs[fallback]["reading"] == "passes" and DEC["status"] == "chosen" and DEC["chosen_budget"] == fallback
    for b, tag in ((ladder[0], "ladder"), (fallback, "ladder fallback")):
        m = re.search(rf"{tag} b{b}: start[^\n]*?([\d.]+), better reference ([\d.]+) -> ([a-z ]+)\n", log)
        assert m and m.group(3) == rungs[b]["reading"] and abs(float(m.group(2)) - rungs[b]["R"]) < 6e-5, b
    assert re.search(rf"chosen budget {fallback};", log)

    # --- the noise step at the chosen budget
    B = DEC["chosen_budget"]
    nz = SUM["noise"]
    assert nz["budget"] == B
    seeds_noise = {"start_g": M["config"]["noise_start_g"], "refA": M["config"]["noise_ref"], "refB": M["config"]["noise_ref"]}
    noise = {k: runs(B, k, seeds_noise[k]) for k in NETS}
    for k in NETS:
        assert all(abs(x["score"] - v) < 6e-5 for x, v in zip(noise[k], nz[k]["per_seed"])), k
        assert abs(np.mean([x["score"] for x in noise[k]]) - nz[k]["mean"]) < 6e-5 and \
            sum(x["learned"] for x in noise[k]) == nz[k]["learned"], k
    ranges = []
    for key in ("start_g_range1", "start_g_range2"):
        sd = nz[key]["seeds"]
        v = [x["score"] for x in noise["start_g"] if x["seed"] in sd]
        assert len(v) == len(sd) and abs(np.mean(v) - nz[key]["mean"]) < 6e-5
        ranges.append({"seeds": sd, "mean": float(np.mean(v)), "scores": v})
    floor_score = min(x["score"] for x in noise["start_g"])
    floor = [x for x in noise["start_g"] if x["score"] == floor_score]
    assert all(x["per_design"] == floor[0]["per_design"] for x in floor)  # exactly the same designs
    floor_designs = sum(floor[0]["per_design"])
    n_dev = noise["start_g"][0]["n"]
    refB_steps = {x["steps"] for x in noise["refB"]}
    refB_flops = {x["flops_ratio"] for x in noise["refB"]}
    assert len(refB_steps) == 1 and len(refB_flops) == 1

    # --- the references' retest scores on Q (card), matched by the network file's sha
    RT = jread(A / "gpu" / "retest" / "runs" / "results.json")
    q_score = {}
    for k in ("refA", "refB"):
        sha = Path(M["nets"][k]).parent.name
        hits = {s["score"] for arm in ("g0", "nofb", "menu") for s in RT["per_arm"][arm]["searches"] if s["sha"] == sha}
        assert len(hits) == 1
        q_score[k] = hits.pop()
    ref_src = {k: re.search(rf"\*\*Reference {k[-1]}:\*\* ([^\n]+?)\.\n", rep).group(1) for k in ("refA", "refB")}
    all_picks = [s["score"] for arm in ("g0", "nofb", "menu") for s in RT["per_arm"][arm]["searches"]]
    assert q_score["refA"] == max(all_picks)  # reference A: the best Stage 1 pick in the retest

    lm = re.search(r'\*\*"Learned"\*\* means the seed builds at least (\d+)% of the (\d+) dev designs that have hidden', rep)
    learned_pct, n_hidden = int(lm.group(1)), int(lm.group(2))
    for d in sorted(RUNS.glob("b*/*/result.json")):
        r = jread(d)
        k = r["hidden"] * n_hidden
        assert abs(k - round(k)) < 1e-6 and r["learned"] == (r["hidden"] >= learned_pct / 100 - 1e-12), d
    D = dict(w=w, q=q, learned_pct=learned_pct, n_hidden=n_hidden, n_lad={k: len(v) for k, v in seeds_lad.items()}, ladder=ladder, fallback=fallback, easy_s=easy_s, easy_r=easy_r, hard=hard, rungs=rungs, B=B,
             noise=noise, ranges=ranges, perm_p=nz["ranges_perm_p"], headroom=nz["headroom"], floor_n=len(floor),
             floor_designs=floor_designs, n_dev=n_dev, refB_steps=refB_steps.pop(), refB_flops=refB_flops.pop(),
             q_score=q_score, ref_src=ref_src,
             means={k: float(np.mean([x["score"] for x in noise[k]])) for k in NETS})
    check_report(D, rep, proto)
    return D


def check_report(D, rep, proto):
    """Every number shown, formatted as shown, must appear where REPORT.md (or the protocol) states it."""
    def has(s, text=rep):
        assert s in text, f"not stated: {s!r}"

    lo, hi = D["rungs"][D["ladder"][0]], D["rungs"][D["fallback"]]

    def cell_mean(r, k):
        return pct(np.mean([x["score"] for x in r["cells"][k]]))

    def learned(r, k):
        return f"{sum(x['learned'] for x in r['cells'][k])} of {len(r['cells'][k])} learned"

    has(f"| {D['ladder'][0]:,} | {cell_mean(lo, 'start_g')} ({learned(lo, 'start_g')}) | {cell_mean(lo, 'start_f')} | "
        f"{cell_mean(lo, 'refA')} | {cell_mean(lo, 'refB')} | too easy: a reference reached {D['easy_r']}% |")
    has(f"| {D['fallback']:,} (fallback) | {cell_mean(hi, 'start_g')} ({learned(hi, 'start_g')}) | {cell_mean(hi, 'start_f')} | "
        f"{cell_mean(hi, 'refA')} | {cell_mean(hi, 'refB')} | passes |")
    has(f"## Outcome: the chosen budget is {D['B']:,} steps")
    n, m = D["noise"], D["means"]
    r1, r2 = D["ranges"]
    has(f"| Start, Stage G's recipe | {len(n['start_g'])} ({r1['seeds'][0]}–{r1['seeds'][-1]}, {r2['seeds'][0]}–{r2['seeds'][-1]}) | "
        f"{pct(m['start_g'])} |")
    lr_s = f"{sum(x['learned'] for x in n['start_g'])} of {len(n['start_g'])}"
    assert re.search(rf"\| Start, Stage G's recipe \| {len(n['start_g'])} \(.*?\) \| {re.escape(pct(m['start_g']))} \| "
                     rf"[\d.]+ \| {lr_s} \|", rep)
    for r in (r1, r2):
        has(f"| — seeds {r['seeds'][0]}–{r['seeds'][-1]} | {len(r['seeds'])} | {pct(r['mean'])} |")
    for k, name in (("refA", "Reference A"), ("refB", "Reference B")):
        has(f"| {name} | {len(n[k])} | {pct(m[k])} |")
        assert re.search(rf"\| {name} \| {len(n[k])} \| {re.escape(pct(m[k]))} \| [\d.]+ \| "
                         rf"{sum(x['learned'] for x in n[k])} of {len(n[k])} \|", rep), name
    has(f"An exact permutation test gives p = {D['perm_p']:.2f}")
    has(f"{D['floor_n']} of the {len(n['start_g'])} seeds built exactly the same {D['floor_designs']} designs")
    has(f"the start leaves {100 * D['headroom']:.0f} points")
    has(f"compute matching trains it for {D['refB_steps']:,} steps instead of {D['B']:,}")
    has(f"Reference B's FLOP ratio is {D['refB_flops']:.2f} on W")
    has(f"({pct(D['q_score']['refA'])} and {pct(D['q_score']['refB'])} in the retest)")
    has("**Reference A:** no-feedback search 9's pick.")
    has("**Reference B:** no-feedback search 5's pick.")
    has(f"averages {pct(m['start_g'])} and learns the skill on {lr_s} seeds")
    has(f"On W they're {100 * (m['refA'] - m['refB']):.0f} points apart")
    has(f'**"Learned"** means the seed builds at least {D["learned_pct"]}% of the {D["n_hidden"]} dev designs that have '
        "hidden cells")
    has(f"{100 * m['refA']:.0f}%", rep)
    w, q = D["w"], D["q"]
    has(f"**W:** {w[0]}×{w[1]}×{w[2]}, built from {w[3]}–{w[4]} bricks", proto)


def pill(ax, x, y, text, kind):
    style = {"too easy": dict(fc=vk.SURFACE, ec=vk.INK2, color=vk.INK),
             "chosen": dict(fc=vk.INK, ec=vk.INK, color=vk.SURFACE),
             "not run": dict(fc=vk.SURFACE, ec=vk.AXIS, color=vk.MUTED)}[kind]
    ax.text(x, y, text, ha="center", va="center", fontsize=14.5, color=style["color"], zorder=8,
            bbox=dict(boxstyle="round,pad=0.45", fc=style["fc"], ec=style["ec"], lw=1.3))


OFF = {"start_g": -0.25, "refA": 0.0, "refB": 0.25}
MARK = {"start_g": dict(marker="o", ms=9, color=vk.GREY, mec=vk.SURFACE, mew=1.0),
        "refA": dict(marker="D", ms=9, color=vk.GREEN, mec=vk.SURFACE, mew=1.0),
        "refB": dict(marker="D", ms=9, mfc=vk.SURFACE, mec=vk.GREEN, mew=1.8)}


def legend(fig, D, y):
    handles = [Line2D([], [], lw=0, label="the start (Stage G's recipe)", **MARK["start_g"]),
               Line2D([], [], lw=0, label=f"reference A: {D['ref_src']['refA']}", **MARK["refA"]),
               Line2D([], [], lw=0, label=f"reference B: {D['ref_src']['refB']}", **MARK["refB"]),
               Line2D([], [], color=vk.INK, lw=3, label="mean")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, y), ncol=4, frameon=False, fontsize=13,
                     handletextpad=0.4, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(vk.INK2)


def ladder_fig(D, step):
    """step 0: the empty ladder and its rules; 1: the first rung; 2: the larger rungs skipped and the fallback."""
    fig = vk.new_fig()
    w, q = D["w"], D["q"]
    vk.header(fig, "Choosing a training budget for world W",
              f"W is {w[0]}×{w[1]}×{w[2]}, built from {w[3]}–{w[4]} bricks; world Q was {q[0]}×{q[1]}×{q[2]} with "
              f"{q[3]}–{q[4]}. Each dot is one training run, scored on W's {D['n_dev']} dev designs.", stage=STAGE)
    legend(fig, D, 0.815)
    ax = fig.add_axes([0.075, 0.235, 0.575, 0.48])
    vk.style_axes(ax)
    ax.set_ylim(-4, 118)
    ax.set_ylabel("built exactly, dev designs", fontsize=14, color=vk.INK2, labelpad=10)
    cols = [D["fallback"]] + D["ladder"]
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_xticks([c + o for c in range(len(cols)) for o in OFF.values()])
    shown_b = [D["ladder"][0]] * (step >= 1) + [D["fallback"]] * (step >= 2)
    ticks = []
    for b in cols:
        for k, nm in zip(NETS, ("start", "A", "B")):
            ticks.append(f"{nm}\n{pct(np.mean([x['score'] for x in D['rungs'][b]['cells'][k]]))}" if b in shown_b else nm)
    ax.set_xticklabels(ticks, fontsize=12, color=vk.INK2)
    ax.axhspan(-4, D["hard"], color=vk.GRID, alpha=0.55, lw=0, zorder=0)
    ax.text(len(cols) - 0.54, D["hard"] - 1.5, f"too hard if the start and both\nreferences are below {D['hard']}%",
            ha="right", va="top", fontsize=11.5, color=vk.INK2, zorder=2, linespacing=1.3)
    tag = {D["fallback"]: "fallback"}
    tag.update({b: f"rung {i + 1}" for i, b in enumerate(D["ladder"])})
    for c, b in enumerate(cols):
        if c:
            ax.axvline(c - 0.5, color=vk.AXIS, lw=1, zorder=0)
        ax.text(c, -0.2, f"{b:,} steps", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=15,
                fontweight="bold", color=vk.INK)
        ax.text(c, -0.26, tag[b], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12.5,
                color=vk.MUTED)
        ax.plot([c - 0.37, c - 0.13], [D["easy_s"]] * 2, color=vk.INK2, lw=1.6, ls=(0, (4, 2.5)), zorder=2)
        ax.plot([c - 0.12, c + 0.37], [D["easy_r"]] * 2, color=vk.INK2, lw=1.6, ls=(0, (4, 2.5)), zorder=2)
        shown = (b == D["ladder"][0] and step >= 1) or (b == D["fallback"] and step >= 2)
        if shown:
            r = D["rungs"][b]
            for k in NETS:
                xs = c + OFF[k] + np.linspace(-0.05, 0.05, len(r["cells"][k]))
                ax.plot(xs, [100 * x["score"] for x in r["cells"][k]], lw=0, zorder=5, alpha=0.9, **MARK[k])
                mu = 100 * np.mean([x["score"] for x in r["cells"][k]])
                ax.plot([c + OFF[k] - 0.1, c + OFF[k] + 0.1], [mu, mu], color=vk.INK, lw=3, zorder=6)
            pill(ax, c, 110, "too easy" if r["reading"] == "too easy" else "passes: chosen",
                 "too easy" if r["reading"] == "too easy" else "chosen")
        elif step >= 2 and b in D["ladder"][1:]:
            pill(ax, c, 110, "not run", "not run")
            ax.text(c, 70, "a larger budget\nis only easier", ha="center", va="center", fontsize=13, color=vk.MUTED,
                    linespacing=1.4)
    ax.text(len(cols) - 0.5 + 0.02, D["easy_r"], f"  {D['easy_r']}%: too easy\n  for a reference", ha="left",
            va="center", fontsize=11.5, color=vk.INK2, clip_on=False)
    ax.text(len(cols) - 0.5 + 0.02, D["easy_s"], f"  {D['easy_s']}%: too easy\n  for the start", ha="left",
            va="center", fontsize=11.5, color=vk.INK2, clip_on=False)

    # the rules, from the protocol
    x0, y0 = 0.75, 0.70
    fig.text(x0, y0, "The rules, fixed before any training", fontsize=16, fontweight="bold", color=vk.INK, ha="left")
    lines = [("Too easy:", f"the start above {D['easy_s']}%, or a"), ("", f"reference at {D['easy_r']}% or more."),
             ("Too hard:", f"the start and both references"), ("", f"below {D['hard']}%."),
             ("Order:", f"{D['ladder'][0]:,}, {D['ladder'][1]:,}, {D['ladder'][2]:,}"),
             ("", "steps, stopping at the first that's"), ("", "too easy. If the first is too easy,"),
             ("", f"try {D['fallback']:,} once."), ("Chosen:", "the largest budget that passes.")]
    y = y0 - 0.05
    for head, t in lines:
        if head:
            y -= 0.012
            fig.text(x0, y, head, fontsize=14, fontweight="bold", color=vk.INK, ha="left")
        fig.text(x0 + 0.062, y, t, fontsize=14, color=vk.INK2, ha="left")
        y -= 0.03
    fig.text(x0, y - 0.03, f"At each budget: the start on {D['n_lad']['start_g']} seeds,\neach reference on "
                           f"{D['n_lad']['refA']}. Mean under each column.", fontsize=12.5, color=vk.MUTED, ha="left",
             va="top", linespacing=1.4)
    lo, hi = D["rungs"][D["ladder"][0]], D["rungs"][D["fallback"]]
    if step == 0:
        vk.caption(fig, "The protocol fixed the ladder and its rules before any training on W.")
    elif step == 1:
        vk.caption(fig, f"First rung, {D['ladder'][0]:,} steps: too easy. Reference A reached {pct(lo['R'])}, past the "
                        f"{D['easy_r']}% line; the start was at {pct(lo['S'])}.")
    else:
        vk.caption(fig, f"Larger budgets would only be easier, so they weren't run. The fallback, {D['fallback']:,} steps, "
                        f"passed: the start {pct(hi['S'])}, the better reference {pct(hi['R'])}.")
    return fig


def noise_fig(D):
    fig = vk.new_fig()
    m, n = D["means"], D["noise"]
    r1, r2 = D["ranges"]
    vk.header(fig, f"At {D['B']:,} steps: room above the start, and above the best known network",
              f"Each dot is one training seed. The start on {len(n['start_g'])} seeds from two ranges; each reference on "
              f"{len(n['refA'])}. Scored on W's {D['n_dev']} dev designs.", stage=STAGE)
    legend(fig, D, 0.815)
    ax = fig.add_axes([0.075, 0.235, 0.56, 0.485])
    vk.style_axes(ax)
    ax.set_ylabel("built exactly, dev designs", fontsize=14, color=vk.INK2, labelpad=10)
    X = {"r1": 0.0, "r2": 1.0, "refA": 2.6, "refB": 3.6}
    ax.set_xlim(-0.6, 4.2)
    ax.set_xticks(list(X.values()))
    cnt = {k: f"{sum(v['learned'] for v in n[k])} of {len(n[k])}" for k in NETS}
    lr = {k: f"learned the skill: {cnt[k]}" for k in NETS}
    ax.set_xticklabels([f"seeds {r['seeds'][0]}–{r['seeds'][-1]}\nmean {pct(r['mean'])}" for r in (r1, r2)]
                       + [f"reference A\n{lr['refA']}", f"reference B\n{lr['refB']}"], fontsize=13, color=vk.INK)
    ax.text(0.5, -0.175, f"the start learned the skill on {cnt['start_g']} seeds", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=13, color=vk.INK)
    rng = np.random.default_rng(0)  # horizontal jitter only

    def dots(x, vals, key):
        xs = x + rng.uniform(-0.16, 0.16, len(vals))
        ax.plot(xs, 100 * np.asarray(vals), lw=0, zorder=5, alpha=0.9, **MARK[key])

    for key, r in (("r1", r1), ("r2", r2)):
        dots(X[key], r["scores"], "start_g")
        ax.plot([X[key] - 0.28, X[key] + 0.28], [100 * r["mean"]] * 2, color=vk.GREY, lw=2.2, zorder=6)
    ax.plot([X["r1"] - 0.4, X["r2"] + 0.4], [100 * m["start_g"]] * 2, color=vk.INK, lw=3, zorder=6)
    ax.text(0.5, 100 * m["start_g"] + 25, f"start, all {len(n['start_g'])} seeds: {pct(m['start_g'])}", ha="center",
            va="bottom", fontsize=15, fontweight="bold", color=vk.INK, zorder=7)
    ax.annotate("", (0.5, 100 * m["start_g"] + 1.5), xytext=(0.5, 100 * m["start_g"] + 24.5),
                arrowprops=dict(arrowstyle="-", color=vk.INK, lw=1.0))
    for key in ("refA", "refB"):
        dots(X[key], [x["score"] for x in n[key]], key)
        ax.plot([X[key] - 0.3, X[key] + 0.3], [100 * m[key]] * 2, color=vk.INK, lw=3, zorder=6)
        ax.text(X[key] + 0.33, 100 * m[key], pct(m[key]), ha="left", va="center", fontsize=15, fontweight="bold",
                color=vk.INK, zorder=7)
    # headroom above the start
    xh = 1.8
    y0 = 100 * m["start_g"]
    ax.annotate("", (xh, 100), xytext=(xh, y0), arrowprops=dict(arrowstyle="<->", color=vk.INK2, lw=1.2,
                                                                shrinkA=0, shrinkB=0))
    ax.text(xh, 103, f"headroom: {100 * D['headroom']:.0f} points", ha="center", va="bottom", fontsize=12.5,
            color=vk.INK2)

    x0 = 0.685
    fig.text(x0, 0.69, "What the dots say", fontsize=16, fontweight="bold", color=vk.INK, ha="left")
    notes = [f"No seed-range effect on W: {pct(r1['mean'])} against",
             f"{pct(r2['mean'])} (permutation test p = {D['perm_p']:.2f}).",
             "",
             f"The start is stuck on a floor: {D['floor_n']} of {len(n['start_g'])} seeds",
             f"built exactly the same {D['floor_designs']} designs.",
             "",
             f"In the retest on Q, the references scored {pct(D['q_score']['refA'])}",
             f"and {pct(D['q_score']['refB'])}. On W they're {100 * (m['refA'] - m['refB']):.0f} points apart.",
             "",
             f"Reference B costs more per step, so compute",
             f"matching trained it for {D['refB_steps']:,} steps. Whether",
             "that explains its score hasn't been tested."]
    y = 0.64
    for t in notes:
        fig.text(x0, y, t, fontsize=14.5, color=vk.INK2, ha="left")
        y -= 0.026 if not t else 0.032
    vk.caption(fig, f"At {D['B']:,} steps the start builds {pct(m['start_g'])} of W's designs and reference A "
                    f"{pct(m['refA'])}: a search has room above both.")
    fig.text(0.05, 0.088, f"Learned the skill: built at least {D['learned_pct']}% of the {D['n_hidden']} dev designs with "
                          "hidden cells. "
                          f"Start: Stage G's recipe; references trained the same way.", fontsize=12.5, color=vk.MUTED,
             ha="left")
    return fig


def takeaway_fig(D):
    m = D["means"]
    fig = vk.new_fig()
    fig.text(0.08, 0.84, STAGE.upper(), fontsize=15, color=vk.MUTED, ha="left", fontweight="bold")
    fig.text(0.08, 0.74, "A world with headroom:", fontsize=40, fontweight="bold", color=vk.INK, ha="left")
    fig.text(0.08, 0.665, f"the start builds {100 * m['start_g']:.0f}% of designs,", fontsize=40, fontweight="bold",
             color=vk.INK, ha="left")
    fig.text(0.08, 0.59, f"the best Stage 1 pick {100 * m['refA']:.0f}%.", fontsize=40, fontweight="bold", color=vk.INK,
             ha="left")
    lines = [f"Reference B scored {pct(D['q_score']['refB'])} on world Q in the retest; on W it builds {pct(m['refB'])}.",
             f"The chosen budget: {D['B']:,} steps, the only one that passed."]
    for k, t in enumerate(lines):
        fig.text(0.08, 0.44 - 0.07 * k, t, fontsize=22, color=vk.INK2, ha="left")
    fig.text(0.08, 0.08, f"\"Best Stage 1 pick\": reference A, {D['ref_src']['refA']}, the top retest score on Q "
                         f"({pct(D['q_score']['refA'])}). Calibration only: nothing was searched or selected; W's "
                         "held-out designs were never read.", fontsize=14, color=vk.MUTED, ha="left")
    return fig


def summary(D, secs):
    m, n = D["means"], D["noise"]
    r4 = lambda x: round(float(x), 4)  # noqa: E731
    return {
        "sources": SRC,
        "world_W": {"size": "×".join(D["w"][:3]), "bricks": f"{D['w'][3]}–{D['w'][4]}"},
        "world_Q": {"size": "×".join(D["q"][:3]), "bricks": f"{D['q'][3]}–{D['q'][4]}"},
        "rules_pct": {"too_easy_start_above": D["easy_s"], "too_easy_reference_at_least": D["easy_r"],
                      "too_hard_all_below": D["hard"]},
        "ladder_budgets": D["ladder"], "fallback": D["fallback"], "chosen_budget": D["B"],
        "beat1_rungs": {str(b): {"reading": r["reading"], "S_start_mean": r4(r["S"]), "R_better_reference_mean": r4(r["R"]),
                                 "cells": {k: {"per_seed": [r4(x["score"]) for x in r["cells"][k]],
                                               "mean": r4(np.mean([x["score"] for x in r["cells"][k]]))} for k in NETS}}
                        for b, r in D["rungs"].items()},
        "beat1_not_run": D["ladder"][1:],
        "beat2_chosen_budget": {
            "start_g": {"n": len(n["start_g"]), "mean": r4(m["start_g"]), "learned": sum(x["learned"] for x in n["start_g"]),
                        "ranges": [{"seeds": f"{r['seeds'][0]}–{r['seeds'][-1]}", "mean": r4(r["mean"]),
                                    "per_seed": [r4(v) for v in r["scores"]]} for r in D["ranges"]],
                        "perm_p": D["perm_p"], "floor": {"seeds": D["floor_n"], "designs": D["floor_designs"]}},
            "refA": {"n": len(n["refA"]), "mean": r4(m["refA"]), "learned": sum(x["learned"] for x in n["refA"]),
                     "per_seed": [r4(x["score"]) for x in n["refA"]], "source": D["ref_src"]["refA"],
                     "retest_on_Q": D["q_score"]["refA"]},
            "refB": {"n": len(n["refB"]), "mean": r4(m["refB"]), "learned": sum(x["learned"] for x in n["refB"]),
                     "per_seed": [r4(x["score"]) for x in n["refB"]], "source": D["ref_src"]["refB"],
                     "retest_on_Q": D["q_score"]["refB"], "steps": D["refB_steps"], "flops_ratio": D["refB_flops"]},
            "headroom_points": round(100 * D["headroom"], 1), "refA_minus_refB_points": round(100 * (m["refA"] - m["refB"]), 1)},
        "dev_designs": D["n_dev"],
        "video_seconds": secs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--frames", help="also write one png per beat to this directory")
    args = ap.parse_args()
    D = load()
    beats = [(lambda: ladder_fig(D, 0), 3.0, "1a_ladder_rules"), (lambda: ladder_fig(D, 1), 3.0, "1b_ladder_rung1"),
             (lambda: ladder_fig(D, 2), 3.5, "1c_ladder_fallback"), (lambda: noise_fig(D), 6.5, "2_chosen_budget"),
             (lambda: takeaway_fig(D), 4.0, "3_takeaway")]
    if args.frames:
        out = Path(args.frames)
        out.mkdir(parents=True, exist_ok=True)
        for make, _, name in beats:
            fig = make()
            fig.savefig(out / f"{name}.png", facecolor=vk.SURFACE, dpi=100)
            plt.close(fig)
    fig = noise_fig(D)  # the last chart frame
    fig.savefig(HERE / "stage2a.png", facecolor=vk.SURFACE, dpi=100)
    plt.close(fig)
    secs = None
    if not args.no_video:
        with vk.Writer(HERE / "stage2a.mp4") as w:
            for make, s, _ in beats:
                w.add(make(), seconds=s)
        secs = round(w.seconds, 1)
    (HERE / "stage2a_data.json").write_text(json.dumps(summary(D, secs), indent=1, ensure_ascii=False))
    print(f"ok: chosen {D['B']}, start {D['means']['start_g']:.4f}, refA {D['means']['refA']:.4f}, "
          f"refB {D['means']['refB']:.4f}, video {secs} s")


if __name__ == "__main__":
    main()
