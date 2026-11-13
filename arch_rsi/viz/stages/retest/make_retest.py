"""Retest article video: do Stage 1's picks still beat the start under a better recipe? Visualization only.

Stage 1's 25 distinct picks (and the start) were retrained under Stage G's recipe at twice Stage 1's budget, on the
same fresh seeds (arch_rsi/gpu/retest). Nothing here trains, scores or selects.

Beats (about 25 s, 1920x1080, 12 fps):
  1. The lead over the start per arm (g0, no feedback, random menu), Stage 1 against the retest, beside each arm's
     mean and the start's, so the shrink is visible: the start gained more than the picks.
  2. Each distinct pick's Stage 1 score against its retest score, coloured by arm, with Spearman's rho and the
     report's worked examples.
  3. The noise check: split-half rank agreement within each recipe against agreement across recipes.
  4. Takeaway card.

Data (read-only; every number shown is read here and cross-checked against arch_rsi/gpu/retest/REPORT.md):
  arch_rsi/gpu/retest/runs/results.json         per-arm means, picks, Stage 1 scores, contrasts, rank agreement
  arch_rsi/gpu/retest/runs/manifest.json        seeds and budget; per-seed scores re-read from runs/<sha>/seed*/dev.json
  arch_rsi/gpu/retest/noise_ceiling.json        split-half rank agreement (descriptive, added after the report)
  arch_rsi/gpu/retest/seedcheck/result.json     the bit-for-bit rerun of one seed from each seed range
  arch_rsi/campaigns/stage1/results.json        Stage 1 per-search scores (cross-check only)
  arch_rsi/campaigns/stage1/confirm/764c7fb529694085/seed1001/result.json   Stage 1's training steps for the start
  arch_rsi/gpu/stageG/runs/cal/xl/start_b262144_s300..309/                  the start on Stage G's seeds (caveat line)

Outputs (this directory): retest.mp4, retest.png (the last chart frame, beat 3), retest_data.json.

usage (project root, low priority while Stage 2b runs):
  nice -n 19 ./.venv/bin/python -I arch_rsi/viz/stages/retest/make_retest.py              # png + json + mp4
  nice -n 19 ./.venv/bin/python -I arch_rsi/viz/stages/retest/make_retest.py --no-video   # png + json only
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

import matplotlib.colors as mc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from arch_rsi.viz.stages import vidkit as vk  # noqa: E402

RT = A / "gpu" / "retest"
START_SHA = "764c7fb529694085"
ARMS = ("g0", "nofb", "menu")
ARM_COLOR = {"g0": vk.BLUE, "nofb": vk.GREEN, "menu": vk.ORANGE}
ARM_LONG = {"g0": "AI scientist sees its scores (g0)", "nofb": "Scores hidden from the AI scientist", "menu": "Random menu (no AI scientist)"}
ARM_TICK = {"g0": "AI scientist sees\nits scores (g0)", "nofb": "Scores hidden from\nthe AI scientist", "menu": "Random menu\n(no AI scientist)"}
ARM_SHORT = {"g0": "g0", "nofb": "no feedback", "menu": "menu"}
STAGE = "Retest of Stage 1's picks"
SRC = {"results": "arch_rsi/gpu/retest/runs/results.json", "manifest": "arch_rsi/gpu/retest/runs/manifest.json",
       "noise": "arch_rsi/gpu/retest/noise_ceiling.json", "seedcheck": "arch_rsi/gpu/retest/seedcheck/result.json",
       "stage1": "arch_rsi/campaigns/stage1/results.json",
       "stage1_steps": f"arch_rsi/campaigns/stage1/confirm/{START_SHA}/seed1001/result.json",
       "stageG": "arch_rsi/gpu/stageG/runs/cal/xl/start_b262144_s{300..309}/dev.json"}


def jread(p):
    return json.loads(Path(p).read_text())


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def pts(x):
    return f"+{100 * x:.1f}" if x >= 0 else f"−{-100 * x:.1f}"


def ranks(v):
    v = np.asarray(v, float)
    r = np.empty(len(v))
    r[v.argsort(kind="stable")] = np.arange(len(v), dtype=float)
    for x in np.unique(v):
        r[v == x] = r[v == x].mean()
    return r


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def load():
    R = jread(RT / "runs" / "results.json")
    M = jread(RT / "runs" / "manifest.json")
    NC = jread(RT / "noise_ceiling.json")
    SC = jread(RT / "seedcheck" / "result.json")
    S1 = jread(A / "campaigns" / "stage1" / "results.json")
    rep = (RT / "REPORT.md").read_text()
    pa = R["per_arm"]

    # --- per-seed scores re-read from the run directories
    def seed_scores(sha, seeds):
        return [jread(RT / "runs" / sha / f"seed{s}" / "dev.json")["exact"] for s in seeds]

    pick_seeds, comp_seeds = M["config"]["pick_seeds"], M["config"]["comp_seeds"]
    for arm in ARMS:
        for s in pa[arm]["searches"]:
            per = seed_scores(s["sha"], pick_seeds)
            assert all(abs(a - b) < 6e-5 for a, b in zip(per, s["per_seed"])), (arm, s["search"])
            assert abs(np.mean(per) - s["score"]) < 6e-5, (arm, s["search"])
            s1 = next(e for e in S1["per_arm"][arm]["searches"] if e["search"] == s["search"])
            assert s1["sha"] == s["sha"] and abs(s1["score"] - s["stage1_score"]) < 6e-5, (arm, s["search"])
        assert abs(np.mean([s["score"] for s in pa[arm]["searches"]]) - pa[arm]["mean"]) < 6e-5, arm
        assert abs(pa[arm]["stage1_mean"] - S1["per_arm"][arm]["mean"]) < 6e-5, arm
    start_seeds = seed_scores(START_SHA, comp_seeds)
    assert abs(np.mean(start_seeds) - pa["start"]["mean"]) < 6e-5
    assert abs(pa["start"]["stage1_mean"] - S1["per_arm"]["start"]["mean"]) < 6e-5

    # --- budgets and recipe
    budget_new = R["budget"]
    assert M["config"]["budget"] == budget_new
    steps_old = jread(A / "campaigns" / "stage1" / "confirm" / START_SHA / "seed1001" / "result.json")
    assert steps_old["flops_ratio"] == 1.0
    budget_old = steps_old["train_steps"]
    assert jread(RT / "runs" / START_SHA / "seed1001" / "result.json")["steps"] == budget_new
    assert budget_new == 2 * budget_old  # "twice the steps"

    # --- beat 1: leads over the start
    lead = {}
    for arm in ARMS:
        lo = R["leads_over_start"][arm]
        assert abs(lo["stage1"] - (pa[arm]["stage1_mean"] - pa["start"]["stage1_mean"])) < 2e-4, arm
        assert abs(lo["retest"] - (pa[arm]["mean"] - pa["start"]["mean"])) < 2e-4, arm
        ci = (R["primary_family"]["contrasts"][f"{arm} - start"]["ci95_boot"] if arm != "menu"
              else R["secondary_family"]["menu - start"]["ci95_boot"])
        diff = (R["primary_family"]["contrasts"][f"{arm} - start"]["diff"] if arm != "menu"
                else R["secondary_family"]["menu - start"]["diff"])
        assert abs(diff - lo["retest"]) < 2e-4, arm
        lead[arm] = {"stage1": lo["stage1"], "retest": lo["retest"], "retest_ci95": ci}
    means = {k: {"stage1": pa[k]["stage1_mean"], "retest": pa[k]["mean"]} for k in ARMS + ("start",)}
    assert R["primary_family"]["reading"] == "lead survives"

    # --- the start on Stage G's seeds (caveat line), and the rerun check
    g_seeds = list(range(300, 310))
    g_dir = A / "gpu" / "stageG" / "runs" / "cal" / "xl"
    for s in g_seeds:
        spec = jread(g_dir / f"start_b262144_s{s}" / "run.json")
        assert spec["budget_steps"] == budget_new and spec["recipe"] == R["recipe"], s
    start_g = float(np.mean([jread(g_dir / f"start_b262144_s{s}" / "dev.json")["exact"] for s in g_seeds]))
    assert set(SC) == {"300", "1001"} and 300 in g_seeds and 1001 in comp_seeds
    assert all(v["same_weights"] and v["same_per_design"] for v in SC.values())

    # --- beat 2: distinct picks
    picks = {}
    for arm in ARMS:
        for s in pa[arm]["searches"]:
            if s["sha"] == START_SHA:
                continue
            if s["sha"] in picks:
                p = picks[s["sha"]]
                assert p["arm"] == arm and p["stage1"] == s["stage1_score"] and p["retest"] == s["score"]
                p["searches"].append(s["search"])
                continue
            picks[s["sha"]] = {"arm": arm, "searches": [s["search"]], "name": s["name"], "stage1": s["stage1_score"],
                               "retest": s["score"], "seeds_learned": s["seeds_learned"]}
    P = list(picks.values())
    assert len(P) == R["rank_agreement"]["n_distinct_picks"] == NC["n_picks"]
    rho = spearman([p["stage1"] for p in P], [p["retest"] for p in P])
    assert abs(rho - R["rank_agreement"]["spearman"]) < 1e-3 and abs(rho - NC["across_recipes_8_seed_means"]) < 1e-3
    n_up = sum(p["retest"] > p["stage1"] for p in P)
    n_arm = {arm: sum(p["arm"] == arm for p in P) for arm in ARMS}

    def find(arm, search):
        return next(p for p in P if p["arm"] == arm and search in p["searches"])

    examples = [("nofb", 1, "conv + cell self-attention"), ("g0", 6, "GroupNorm body"), ("nofb", 5, "layer-shared per-z encoder")]
    ex = [dict(find(a, i), label=f"{ARM_SHORT[a]}, search {i}: {lbl}") for a, i, lbl in examples]
    claude = sorted((p for p in P if p["arm"] != "menu"), key=lambda p: p["stage1"])
    key = lambda p: (p["arm"], tuple(p["searches"]))  # noqa: E731
    assert {key(claude[0]), key(claude[1])} == {key(ex[0]), key(ex[1])}  # caption: "the two weakest AI-scientist picks"
    fall = min(P, key=lambda p: p["retest"] - p["stage1"])  # the report's "moved down" example
    assert fall["arm"] == "menu"
    ex.append(dict(fall, label="menu pick: conv, width 32, depth 3, tanh"))
    best = max(P, key=lambda p: p["retest"])

    # --- beat 3: noise check
    noise = {"old": NC["split_half_old_recipe"], "new": NC["split_half_new_recipe"], "across": NC["across_recipes_8_seed_means"]}
    assert noise["old"]["n_splits"] == noise["new"]["n_splits"]

    D = dict(R=R, budget_old=budget_old, budget_new=budget_new, lead=lead, means=means, start_g=start_g,
             n_start_seeds=len(comp_seeds), n_pick_seeds=len(pick_seeds), picks=P, rho=rho, n_up=n_up, n_arm=n_arm,
             examples=ex, best=best, noise=noise, n_dev=jread(RT / "runs" / START_SHA / "seed1001" / "dev.json")["n"])
    check_report(D, rep)
    return D


def check_report(D, rep):
    """Every number shown, formatted as shown, must appear where REPORT.md states it."""
    def has(s):
        assert s in rep, f"REPORT.md does not state: {s!r}"

    L, Mn = D["lead"], D["means"]
    has(f"Stage 1 trained at {D['budget_old']:,} steps")
    has(f"the retest trains at {D['budget_new']:,} steps")
    has(f"Exact-build success on Q's {D['n_dev']} dev designs".replace("Exact", "exact"))
    for arm, name in (("g0", "g0"), ("nofb", "No feedback"), ("menu", "Menu")):
        has(f"| {name} | {pts(L[arm]['stage1'])} | {pts(L[arm]['retest'])} |")
    for arm, row in (("g0", "**R1:** g0 − start"), ("nofb", "**R2:** no feedback − start"), ("menu", "Menu − start")):
        lo, hi = L[arm]["retest_ci95"]
        has(f"| {row} | {pts(L[arm]['retest'])} points | {pts(lo)} to {pts(hi)} |")
    for k, row in (("g0", "g0 (sees its results)"), ("nofb", "No feedback (scores hidden)"), ("menu", "Random menu"),
                   ("start", "Start (12 seeds)")):
        assert re.search(rf"\| {re.escape(row)} \| {re.escape(pct(Mn[k]['retest']))} \|.*\| "
                         rf"{re.escape(pct(Mn[k]['stage1']))} \|", rep), row
    has(f"{pct(Mn['start']['retest'])} on this retest's seeds (1001–1012)")
    has(f"{pct(D['start_g'])} on Stage G's seeds (300–309)")
    has("Each reproduced its original run bit for bit")
    has(f"Spearman's ρ = {D['rho']:.2f} across the {len(D['picks'])} distinct picks")
    for e in D["examples"]:
        has(f"from {pct(e['stage1'])} to {pct(e['retest'])}")
    has(f"only {D['examples'][3]['seeds_learned']} of its {D['n_pick_seeds']} seeds learned the skill")
    has(f"{D['n_up']} of the {len(D['picks'])} picks scored higher than in Stage 1")
    nz = D["noise"]
    for k in ("old", "new"):
        has(f"| {nz[k]['mean']:.2f} (range {nz[k]['min']:.2f}–{nz[k]['max']:.2f}) |")
    has(f"| Old recipe against new recipe, 8-seed means | {nz['across']:.2f} |")
    has(f"mean of {nz['old']['n_splits']} splits")
    has(f"The best pick averaged {pct(D['best']['retest'])} (no-feedback search 9)")
    assert D["best"]["arm"] == "nofb" and 9 in D["best"]["searches"]


def tint(c, a=0.38):
    r, s = np.array(mc.to_rgb(c)), np.array(mc.to_rgb(vk.SURFACE))
    return tuple(s + a * (r - s))


def spread(vals, gap):
    """Nudge label positions apart (upwards) so no two are closer than gap."""
    out = np.array(vals, float)
    order = np.argsort(out)
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        if out[i] - out[j] < gap:
            out[i] = out[j] + gap
    return out


def foot(fig, text, y=0.088):
    fig.text(0.05, y, text, fontsize=12.5, color=vk.MUTED, ha="left")


# ---------------------------------------------------------------- beat 1
def lead_fig(D, retest):
    fig = vk.new_fig()
    vk.header(fig, "The AI scientist's picks still beat the start, by less" if retest else "Stage 1: the AI scientist's picks beat the start",
              f"Lead over the starting network, in points of exact builds on {D['n_dev']} dev designs. The same fresh seeds "
              f"in both: {D['n_pick_seeds']} per pick, {D['n_start_seeds']} for the start.", stage=STAGE)
    ax = fig.add_axes([0.075, 0.215, 0.50, 0.55])
    vk.style_axes(ax, ylim=(0, 60), yticks=(0, 10, 20, 30, 40, 50, 60), pct=False)
    ax.set_yticklabels([f"+{v}" if v else "0" for v in (0, 10, 20, 30, 40, 50, 60)], fontsize=13)
    ax.set_ylabel("lead over the start (points)", fontsize=14, color=vk.INK2, labelpad=10)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([ARM_TICK[a] for a in ARMS], fontsize=14, color=vk.INK)
    w = 0.34
    for j, arm in enumerate(ARMS):
        L, c = D["lead"][arm], ARM_COLOR[arm]
        v1 = 100 * L["stage1"]
        ax.bar(j - 0.19, v1, w, color=tint(c), edgecolor=c, lw=1.2, zorder=3)
        ax.text(j - 0.19, v1 + 1.2, pts(L["stage1"]), ha="center", va="bottom", fontsize=15, color=vk.INK2,
                fontweight="bold", zorder=5)
        if retest:
            v2 = 100 * L["retest"]
            lo, hi = (100 * v for v in L["retest_ci95"])
            ax.bar(j + 0.19, v2, w, color=c, zorder=3)
            ax.plot([j + 0.19] * 2, [lo, hi], color=vk.INK, lw=1.4, zorder=4)
            for y in (lo, hi):
                ax.plot([j + 0.19 - 0.05, j + 0.19 + 0.05], [y, y], color=vk.INK, lw=1.4, zorder=4)
            ax.text(j + 0.19 + 0.07, v2 + 0.6, pts(L["retest"]), ha="left", va="bottom", fontsize=15, color=vk.INK,
                    fontweight="bold", zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc=vk.SURFACE, ec="none", alpha=0.85))
    handles = [Patch(fc=tint(vk.GREY), ec=vk.GREY, lw=1.2,
                     label=f"Stage 1: old recipe, {D['budget_old']:,} steps"),
               Patch(fc=vk.GREY, label=f"Retest: Stage G's recipe, {D['budget_new']:,} steps") if retest else None,
               Line2D([], [], color=vk.INK, lw=1.4, label="retest 95% interval") if retest else None]
    leg = ax.legend(handles=[h for h in handles if h], loc="upper right", frameon=False, fontsize=12.5, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(vk.INK2)

    # right: the means behind the leads
    ax2 = fig.add_axes([0.70, 0.215, 0.22, 0.55])
    vk.style_axes(ax2)
    ax2.set_xlim(-0.35, 1.35)
    ax2.set_xticks([0, 1] if retest else [0])
    ax2.set_xticklabels(["Stage 1", "Retest"] if retest else ["Stage 1"], fontsize=14, color=vk.INK)
    ax2.set_title("Mean score behind each lead", fontsize=15, fontweight="bold", color=vk.INK, loc="left", pad=12)
    keys = ARMS + ("start",)
    col = dict(ARM_COLOR, start=vk.GREY)
    name = dict(ARM_SHORT, start="start")
    for x, stage in ((0, "stage1"), (1, "retest")):
        if stage == "retest" and not retest:
            continue
        vals = [100 * D["means"][k][stage] for k in keys]
        ys = spread(vals, 4.2)
        for k, v, y in zip(keys, vals, ys):
            ax2.plot([x], [v], "o", ms=9, color=col[k], mec=vk.SURFACE, mew=1.2, zorder=5)
            if x == 0 and retest:
                ax2.text(x - 0.09, y, f"{v:.1f}%", ha="right", va="center", fontsize=12.5, color=col[k], fontweight="bold")
            elif x == 0:
                ax2.text(x + 0.09, y, f"{v:.1f}%  {name[k]}", ha="left", va="center", fontsize=12.5, color=col[k],
                         fontweight="bold")
            else:
                ax2.text(x + 0.09, y, f"{v:.1f}%  {name[k]}", ha="left", va="center", fontsize=12.5, color=col[k],
                         fontweight="bold")
    if retest:
        for k in keys:
            ax2.plot([0, 1], [100 * D["means"][k]["stage1"], 100 * D["means"][k]["retest"]], color=col[k], lw=2.2,
                     ls=(0, (4, 2)) if k == "start" else "-", zorder=3)
    ax2.tick_params(labelleft=False)
    if retest:
        vk.caption(fig, "The lead survives, smaller: the start gained more from the new recipe and budget than the picks did.")
        foot(fig, f"Caveat: the start scored {pct(D['means']['start']['retest'])} on these seeds but {pct(D['start_g'])} on "
                  "another seed range under the same recipe (a rerun of one seed from each matched bit for bit), so the lead's "
                  "size is less certain than its interval.")
    else:
        L = D["lead"]
        vk.caption(fig, f"In Stage 1, the AI scientist's picks beat the start by {100 * L['g0']['stage1']:.1f} and "
                        f"{100 * L['nofb']['stage1']:.1f} points; random menu picks by {100 * L['menu']['stage1']:.1f}.")
    return fig


# ---------------------------------------------------------------- beat 2
def scatter_fig(D, annotate):
    fig = vk.new_fig()
    vk.header(fig, "A pick's Stage 1 score says little about its retest score",
              f"Each dot is one of the {len(D['picks'])} distinct picks: its score on {D['n_pick_seeds']} fresh seeds under "
              "the old recipe (across) and on the same seeds under the new one (up).", stage=STAGE)
    ax = fig.add_axes([0.075, 0.165, 0.355, 0.63])
    ticks = (20, 40, 60, 80, 100)
    vk.style_axes(ax, ylim=(20, 100), yticks=ticks)
    for x in ticks:
        ax.axvline(x, color=vk.GRID, lw=1, zorder=0)
    ax.set_xlim(16, 104)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{v}%" for v in ticks], fontsize=13)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel(f"Stage 1 score (old recipe, {D['budget_old']:,} steps)", fontsize=14, color=vk.INK2, labelpad=10)
    ax.set_ylabel(f"retest score (new recipe, {D['budget_new']:,} steps)", fontsize=14, color=vk.INK2, labelpad=10)
    ax.plot([16, 104], [16, 104], color=vk.AXIS, lw=1.4, ls=(0, (5, 3)), zorder=1)
    ax.text(33, 35, "same score in both", ha="left", va="bottom", fontsize=11.5, color=vk.MUTED, rotation=45,
            rotation_mode="anchor")
    s0, s1 = (100 * D["means"]["start"][k] for k in ("stage1", "retest"))
    ax.plot([s0], [s1], "o", ms=11, mfc="none", mec=vk.GREY, mew=2, zorder=4)
    ax.annotate(f"the start ({D['n_start_seeds']} seeds)", (s0, s1), xytext=(0, -16), textcoords="offset points",
                ha="center", va="top", fontsize=11.5, color=vk.GREY)
    for p in D["picks"]:
        ax.plot([100 * p["stage1"]], [100 * p["retest"]], "o", ms=11, color=ARM_COLOR[p["arm"]], mec=vk.SURFACE,
                mew=1.2, alpha=0.95, zorder=5)
    if annotate:
        spots = [(21.5, 94.5, "left", "top"), (21.5, 76.0, "left", "top"), (21.5, 103.5, "left", "top"),
                 (73.5, 33.0, "left", "top")]
        for e, (tx, ty, ha, va) in zip(D["examples"], spots):
            x, y = 100 * e["stage1"], 100 * e["retest"]
            head, _, what = e["label"].partition(": ")
            txt = (f"{head}: {what}\n" if e["arm"] == "menu" else f"{head}\n{what}\n") + f"{pct(e['stage1'])} → {pct(e['retest'])}"
            if e["arm"] == "menu":
                txt += f"; {e['seeds_learned']} of {D['n_pick_seeds']} seeds learned"
            ax.annotate(txt, (x, y), xytext=(tx, ty), textcoords="data", ha=ha, va=va, fontsize=11.5, color=vk.INK,
                        zorder=7, arrowprops=dict(arrowstyle="-", color=vk.INK2, lw=0.9, shrinkA=2, shrinkB=6),
                        bbox=dict(boxstyle="round,pad=0.2", fc=vk.SURFACE, ec="none", alpha=0.9))
            ax.plot([x], [y], "o", ms=15, mfc="none", mec=vk.INK, mew=1.4, zorder=6)

    # right-hand panel
    x0 = 0.50
    if annotate:
        fig.text(x0, 0.70, f"Spearman ρ = {D['rho']:.2f}", fontsize=40, fontweight="bold", color=vk.INK, ha="left")
        fig.text(x0, 0.655, f"rank agreement across the {len(D['picks'])} distinct picks", fontsize=17, color=vk.INK2,
                 ha="left")
        fig.text(x0, 0.585, "A pick's Stage 1 rank says little about its retest rank.", fontsize=17, color=vk.INK, ha="left")
        fig.text(x0, 0.545, f"{D['n_up']} of the {len(D['picks'])} scored higher in the retest, partly from the doubled "
                            "budget.", fontsize=17, color=vk.INK, ha="left")
    ly = 0.40
    fig.text(x0, ly, "Picks by arm", fontsize=14, fontweight="bold", color=vk.INK2, ha="left")
    rows = [(ARM_COLOR[a], f"{ARM_LONG[a]}: {D['n_arm'][a]} {'distinct ' if a == 'menu' else ''}picks") for a in ARMS]
    for k, (c, t) in enumerate(rows):
        y = ly - 0.045 * (k + 1)
        fig.lines.append(Line2D([x0 + 0.006], [y + 0.006], marker="o", ms=11, color=c, mec=vk.SURFACE,
                                transform=fig.transFigure, figure=fig))
        fig.text(x0 + 0.018, y, t, fontsize=14.5, color=vk.INK2, ha="left")
    y = ly - 0.045 * 4
    fig.lines.append(Line2D([x0 + 0.006], [y + 0.006], marker="o", ms=11, mfc="none", mec=vk.GREY, mew=2,
                            transform=fig.transFigure, figure=fig))
    fig.text(x0 + 0.018, y, "The start (not a pick): the same comparison", fontsize=14.5, color=vk.INK2, ha="left")
    if annotate:
        vk.caption(fig, "Rankings barely carry across recipes: the two weakest AI-scientist picks caught up with the rest, and one "
                        "menu pick fell.")
    else:
        vk.caption(fig, "Above the dashed line: the pick scored higher in the retest than in Stage 1.")
    return fig


# ---------------------------------------------------------------- beat 3
def noise_fig(D):
    nz = D["noise"]
    fig = vk.new_fig()
    vk.header(fig, "Is the reshuffle just seed noise? Mostly not",
              f"Rank agreement of the {len(D['picks'])} distinct picks (Spearman ρ). Within a recipe: one half of each "
              f"pick's {D['n_pick_seeds']} seeds against the other half.", stage=STAGE)
    ax = fig.add_axes([0.285, 0.235, 0.53, 0.50])
    rows = [("Within the old recipe", "one half of the seeds against the other", nz["old"], vk.GREY),
            ("Within the new recipe", "the same split-half test", nz["new"], vk.GREY),
            ("Across recipes", "old against new, all 8 seeds", {"mean": nz["across"]}, vk.INK)]
    ax.set_facecolor(vk.SURFACE)
    for x in (0, 0.25, 0.5, 0.75, 1.0):
        ax.axvline(x, color=vk.GRID, lw=1, zorder=0)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(2.6, -0.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=13)
    ax.set_xlabel("Spearman rank agreement (1 = same order)", fontsize=14, color=vk.INK2, labelpad=10)
    ax.set_yticks([])
    ax.tick_params(length=0, pad=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(vk.AXIS)
    for j, (title, sub, v, c) in enumerate(rows):
        ax.barh(j, v["mean"], 0.52, color=c, zorder=3)
        lab = f"{v['mean']:.2f}"
        if "min" in v:
            ax.plot([v["min"], v["max"]], [j, j], color=vk.INK, lw=1.5, zorder=4)
            for x in (v["min"], v["max"]):
                ax.plot([x, x], [j - 0.1, j + 0.1], color=vk.INK, lw=1.5, zorder=4)
            ax.text(v["max"] + 0.015, j, lab, ha="left", va="center", fontsize=20, fontweight="bold", color=vk.INK)
            ax.text(v["max"] + 0.085, j, f"range {v['min']:.2f}–{v['max']:.2f}", ha="left", va="center", fontsize=12.5,
                    color=vk.MUTED)
        else:
            ax.text(v["mean"] + 0.015, j, lab, ha="left", va="center", fontsize=20, fontweight="bold", color=vk.INK)
        fig.text(0.05, ax.transData.transform((0, j))[1] / vk.H_PX + 0.012, title, fontsize=17, fontweight="bold",
                 color=vk.INK, ha="left", va="center")
        fig.text(0.05, ax.transData.transform((0, j))[1] / vk.H_PX - 0.022, sub, fontsize=14, color=vk.INK2, ha="left",
                 va="center")
    vk.caption(fig, "Within each recipe the order mostly holds; across recipes it nearly vanishes. Seed noise doesn't "
                    "explain most of the reshuffle.")
    foot(fig, f"A descriptive check added after the report; no new runs. Split-half bars: the mean over {nz['old']['n_splits']} "
              "ways to split the seeds in half; whiskers: lowest to highest.")
    return fig


# ---------------------------------------------------------------- beat 4
def takeaway_fig(D):
    fig = vk.new_fig()
    fig.text(0.08, 0.84, STAGE.upper(), fontsize=15, color=vk.MUTED, ha="left", fontweight="bold")
    for k, t in enumerate(("Architecture still matters,", "but which network looks best", "depends on how it's trained.")):
        fig.text(0.08, 0.74 - 0.075 * k, t, fontsize=40, fontweight="bold", color=vk.INK, ha="left")
    L = D["lead"]
    lines = [f"Under a better recipe and budget, the AI scientist's picks still beat the start by {100 * L['g0']['retest']:.1f} "
             f"and {100 * L['nofb']['retest']:.1f} points.",
             f"Their order barely carried over (ρ = {D['rho']:.2f}), and seed noise doesn't explain most of that."]
    for k, t in enumerate(lines):
        fig.text(0.08, 0.46 - 0.07 * k, t, fontsize=22, color=vk.INK2, ha="left")
    fig.text(0.08, 0.08, "A retrain, not a new search: these picks were chosen under the old recipe. "
                         f"World Q, {D['n_dev']} dev designs; the held-out designs were never read.",
             fontsize=14, color=vk.MUTED, ha="left")
    return fig


def summary(D, secs):
    L, Mn, nz = D["lead"], D["means"], D["noise"]
    r4 = lambda x: round(float(x), 4)  # noqa: E731
    return {
        "sources": SRC,
        "beat1_lead_over_start_points": {a: {"stage1": round(100 * L[a]["stage1"], 1), "retest": round(100 * L[a]["retest"], 1),
                                             "retest_ci95": [round(100 * v, 1) for v in L[a]["retest_ci95"]]} for a in ARMS},
        "beat1_means": {k: {"stage1": r4(Mn[k]["stage1"]), "retest": r4(Mn[k]["retest"])} for k in Mn},
        "beat1_caveat_start_mean_stageG_seeds_300_309": r4(D["start_g"]),
        "budgets": {"stage1": D["budget_old"], "retest": D["budget_new"]},
        "seeds": {"per_pick": D["n_pick_seeds"], "start": D["n_start_seeds"]}, "dev_designs": D["n_dev"],
        "beat2_picks": [{"arm": p["arm"], "searches": p["searches"], "name": p["name"], "stage1": p["stage1"],
                         "retest": p["retest"]} for p in D["picks"]],
        "beat2_spearman": round(D["rho"], 4), "beat2_n_distinct": len(D["picks"]), "beat2_n_scored_higher": D["n_up"],
        "beat2_n_per_arm": D["n_arm"],
        "beat2_examples": [{"label": e["label"], "stage1": e["stage1"], "retest": e["retest"], "seeds_learned": e["seeds_learned"]}
                           for e in D["examples"]],
        "beat3_rank_agreement": {"split_half_old": nz["old"], "split_half_new": nz["new"], "across_recipes": nz["across"]},
        "video_seconds": secs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--frames", help="also write one png per beat to this directory")
    args = ap.parse_args()
    D = load()
    beats = [(lambda: lead_fig(D, False), 3.0, "1a_lead_stage1"), (lambda: lead_fig(D, True), 5.5, "1b_lead_retest"),
             (lambda: scatter_fig(D, False), 3.0, "2a_scatter"), (lambda: scatter_fig(D, True), 5.0, "2b_scatter_annot"),
             (lambda: noise_fig(D), 4.5, "3_noise"), (lambda: takeaway_fig(D), 4.0, "4_takeaway")]
    if args.frames:
        out = Path(args.frames)
        out.mkdir(parents=True, exist_ok=True)
        for make, _, name in beats:
            fig = make()
            fig.savefig(out / f"{name}.png", facecolor=vk.SURFACE, dpi=100)
            plt.close(fig)
    fig = noise_fig(D)  # the last chart frame
    fig.savefig(HERE / "retest.png", facecolor=vk.SURFACE, dpi=100)
    plt.close(fig)
    secs = None
    if not args.no_video:
        with vk.Writer(HERE / "retest.mp4") as w:
            for make, s, _ in beats:
                w.add(make(), seconds=s)
        secs = round(w.seconds, 1)
    (HERE / "retest_data.json").write_text(json.dumps(summary(D, secs), indent=1, ensure_ascii=False))
    print(f"ok: {len(D['picks'])} picks, rho {D['rho']:.4f}, video {secs} s")


if __name__ == "__main__":
    main()
