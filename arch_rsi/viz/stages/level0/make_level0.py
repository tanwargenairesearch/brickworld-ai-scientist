"""Level 0 article video: Claude tunes a PPO training recipe in a small brick-assembly world, against seeded random
search on the same budget. Visualization of finished data only: nothing here trains, selects, scores or calls a model.

What it shows (about 28 s, 1920x1080, 12 fps):
  1. Title card.
  2. Build-up, round by round, for the arms sonnet, opus and random (proxy_sonnet and random_topview are left out:
     they test the planted flawed checker, a separate question). One panel per arm. Each dot is one search trial's dev
     exact success (1 seed, 512 dev designs); a cross at 0 is a trial that hit the 600 s limit. Step lines are each of
     the 5 runs' running best, by the campaign's incumbent rule (protocol.md, orchestrate.py: the highest dev exact
     among baseline seed 0 and the run's first k trials, ties to the earlier one, timed-out trials not selectable),
     starting from baseline seed 0's dev exact, the bar every run had to beat.
  3. The answer: held-out (final split) exact success of each run's incumbent at k = 32, retrained on 2 fresh seeds
     (dots: the 5 runs; bar: their mean), next to the starting recipe's final exact (dots: its 5 seeds).
  4. A takeaway card drawn from REPORT.md.

Data (all read-only, project root campaigns/main/):
  journal.jsonl      one line per trial; search trials are "<arm>-r<run>-t<i>", i = 0..31 in proposal order,
                     round = i // 4 + 1 (manifest per_round). Dev exact per trial, status, recipe (config_key).
  trials/<id>/result.json   dev exact again, cross-checked against the journal.
  incumbents.json    the campaign's own incumbents at k = 4, 8, 16, 32; the recomputed running best must match.
  results.json       toy_r[arm].fresh_final_exact_by_k["32"] (final-split scores the campaign already reported),
                     baseline.final_exact, baseline.dev_exact, toy_r_comparisons, trial_status.
  manifest.json      runs, rounds, trials per round, budgets, step budget, timeout, fresh seeds, model names.
  REPORT.md          every number shown that the report states is parsed from it and asserted (within rounding).
The script never touches the held-out design files.

Outputs (this directory): level0.mp4, level0.png (the final chart frame), level0_data.json (every number shown).

usage (project root; one render at a time, low priority):
  nice -n 19 ./.venv/bin/python -I arch_rsi/viz/stages/level0/make_level0.py            # png + data json
  nice -n 19 ./.venv/bin/python -I arch_rsi/viz/stages/level0/make_level0.py --video    # + level0.mp4
"""
import sys

sys.dont_write_bytecode = True  # write nothing outside this directory
import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from arch_rsi.viz.stages import vidkit as vk  # noqa: E402

C = ROOT / "campaigns" / "main"
ARMS = ("sonnet", "opus", "random")
CLAUDE = ("sonnet", "opus")
COLOR = {"sonnet": vk.BLUE, "opus": vk.BLUE, "random": vk.ORANGE, "start": vk.GREY}


# ---------------------------------------------------------------- data
def model_name(model_id: str) -> str:
    """claude-sonnet-5-5 -> Claude Sonnet 5.5"""
    parts = model_id.split("-")
    return f"{parts[0].capitalize()} {parts[1].capitalize()} {'.'.join(parts[2:])}"


def report_num(rep: str, pattern: str, n: int = 1):
    m = re.search(pattern, rep, re.M)
    assert m, f"REPORT.md: no match for {pattern!r}"
    vals = [float(g.replace(",", "")) for g in m.groups()[:n]]
    return vals[0] if n == 1 else vals


def close(a: float, b: float, tol: float = 0.051) -> bool:
    return abs(a - b) <= tol


def load() -> dict:
    man = json.loads((C / "manifest.json").read_text())
    S = man["settings"]
    res = json.loads((C / "results.json").read_text())
    inc = json.loads((C / "incumbents.json").read_text())
    rep = (C / "REPORT.md").read_text()
    lines = (C / "journal.jsonl").read_text().splitlines()
    J = {}
    for ln in lines:
        r = json.loads(ln)
        assert r["trial"] not in J, r["trial"]
        J[r["trial"]] = r
    runs, rounds, per = S["runs_per_arm"], S["rounds"], S["per_round"]
    ntr = rounds * per
    assert S["budgets_k"][-1] == ntr

    # the bar every run had to beat: baseline seed 0's dev exact (orchestrate.py: pool = [base0] + trials[:k])
    b0 = J["baseline-s0"]
    bar = b0["dev"]["exact"]
    assert bar == json.loads((C / "trials" / "baseline-s0" / "result.json").read_text())["dev"]["exact"]
    assert abs(bar - res["baseline"]["dev_exact"]["values"][0]) < 6e-5
    lo, hi = report_num(rep, r"^- Dev exact: [\d.]+% \(5 seeds, ([\d.]+)–([\d.]+)%\)", 2)
    assert lo - 0.05 <= 100 * bar <= hi + 0.05, (bar, lo, hi)
    base_cfg = json.loads(b0["config_key"])
    assert len(base_cfg) == 22 and "22-knob" in rep

    arms = {}
    for arm in ARMS:
        a = S["arms"][arm]
        assert a["metric"] == "exact", arm
        rows_all = []
        for r in range(runs):
            rows, best, best_id, run_best, new = [], bar, "baseline-s0", [], []
            for i in range(ntr):
                tid = f"{arm}-r{r}-t{i:02d}"
                t = J[tid]
                ok = t["status"] == "ok"
                assert ok == bool(t.get("dev")), tid
                d = t["dev"]["exact"] if ok else None
                if ok:
                    assert d == json.loads((C / "trials" / tid / "result.json").read_text())["dev"]["exact"], tid
                is_new = ok and d > best  # strict: ties go to the earlier trial (or the baseline)
                if is_new:
                    best, best_id = d, tid
                rows.append({"trial": tid, "round": i // per + 1, "status": t["status"], "dev_exact": d,
                             "new_best": is_new, "running_best": best})
                run_best.append(best)
                if i + 1 in S["budgets_k"]:
                    assert best_id == inc[f"{arm}|{r}|{i + 1}"], (arm, r, i + 1, best_id, inc[f"{arm}|{r}|{i + 1}"])
            rows_all.append({"run": r, "trials": rows, "incumbent_k32": best_id, "dev_k32": best})
        devk = res["toy_r"][arm]["dev_exact_by_k_selection_biased"][str(ntr)]["values"]
        assert all(abs(x["dev_k32"] - v) < 6e-5 for x, v in zip(rows_all, devk)), arm
        st = res["trial_status"][arm]
        n_to = sum(x["status"] == "timeout" for rr in rows_all for x in rr["trials"])
        assert n_to == st.get("timeout", 0) and runs * ntr - n_to == st["ok"], arm
        m = re.search(rf"^\| {arm} \| (\d+) \| (\d+) \| (\d+) \|$", rep, re.M)
        assert m and (int(m[1]), int(m[2])) == (st["ok"], st.get("timeout", 0)), arm

        # the answer: held-out exact of each run's k = 32 incumbent, retrained on fresh seeds (results.json)
        f = res["toy_r"][arm]["fresh_final_exact_by_k"][str(ntr)]
        assert f["n"] == runs and len(f["values"]) == runs
        assert abs(np.mean(f["values"]) - f["mean"]) < 1e-4, arm
        row = re.search(rf"^\| {arm} \|( \d+\.\d \[.*)\|$", rep, re.M)
        assert row, arm
        cells = [c.strip() for c in row[1].split("|")]
        rep32 = float(cells[S["budgets_k"].index(ntr)].split()[0])
        assert close(100 * f["mean"], rep32), (arm, f["mean"], rep32)
        arms[arm] = {"title": {"sonnet": "AI scientist: smaller LLM", "opus": "AI scientist: larger LLM"}[arm] if a["kind"] == "llm" else "Random search (seeded)",
                     "runs": rows_all, "final_k32_per_run": f["values"], "final_k32_mean": f["mean"],
                     "final_k32_ci95": f["ci95"]}

    # the values the brief names
    for arm, want in (("sonnet", 96.8), ("opus", 96.1), ("random", 60.2)):
        assert close(100 * arms[arm]["final_k32_mean"], want), (arm, arms[arm]["final_k32_mean"])
    bf = res["baseline"]["final_exact"]
    assert abs(np.mean(bf["values"]) - bf["mean"]) < 1e-4
    assert close(100 * bf["mean"], 29.1) and close(100 * bf["mean"], report_num(rep, r"^- Final exact: ([\d.]+)%"))
    lo_s, hi_s, lo_o, hi_o = report_num(
        rep, r"Sonnet − random 95% interval \[([\d.]+), ([\d.]+)\] pp; Opus − random \[([\d.]+), ([\d.]+)\] pp", 4)
    gap = {a: res["toy_r_comparisons"][f"{a}_minus_random"]["level_k32_diff_ci95"] for a in CLAUDE}
    for a, (lo_, hi_) in (("sonnet", (lo_s, hi_s)), ("opus", (lo_o, hi_o))):
        assert close(100 * gap[a][0], lo_) and close(100 * gap[a][1], hi_), (a, gap[a])

    # takeaway facts, recomputed from the 10 Claude k = 32 incumbents' recipes and checked against REPORT.md
    recs = [json.loads(J[arms[a]["runs"][r]["incumbent_k32"]]["config_key"]) for a in CLAUDE for r in range(runs)]
    shared = [k for k in base_cfg if all(c[k] != base_cfg[k] for c in recs)]
    assert shared == ["lr"], shared
    lrs = [c["lr"] for c in recs]
    assert all(v > base_cfg["lr"] for v in lrs)
    n_rec = len(recs)
    r_n, r_from, r_lo, r_hi = report_num(rep, r"all (\d+) raised `lr`, from ([\d.]+) to ([\d.]+)–([\d.]+)", 4)
    assert (r_n, r_from, r_lo, r_hi) == (n_rec, base_cfg["lr"], min(lrs), max(lrs)), (r_n, r_from, r_lo, r_hi)
    n_ppo = sum(c["ppo_epochs"] > base_cfg["ppo_epochs"] for c in recs)
    ppo_to = sorted({c["ppo_epochs"] for c in recs if c["ppo_epochs"] > base_cfg["ppo_epochs"]})
    p_n, p_from, p_to = report_num(rep, r"(\d+) raised `ppo_epochs` from (\d+) to (\d+)", 3)
    assert (p_n, p_from, [p_to]) == (n_ppo, base_cfg["ppo_epochs"], ppo_to), (p_n, p_from, p_to)
    claude_dev100 = all(x["dev_k32"] == 1.0 for a in CLAUDE for x in arms[a]["runs"])
    assert claude_dev100 and "reached 100% in every Claude run by k = 32" in rep

    t0 = J[f"{ARMS[0]}-r0-t00"]
    assert t0["train"]["steps"] == S["small"]["budget_steps"] and t0["dev"]["n"] == 512
    return {"S": S, "bar": bar, "arms": arms, "base_final": bf, "gap": gap, "runs": runs, "rounds": rounds,
            "per": per, "ntr": ntr, "steps": S["small"]["budget_steps"], "timeout_s": S["small"]["timeout_s"],
            "n_dev": t0["dev"]["n"], "fresh_seeds": S["fresh_seeds"], "n_knobs": len(base_cfg),
            "lr_from": base_cfg["lr"], "lr_lo": min(lrs), "lr_hi": max(lrs), "n_rec": n_rec, "n_ppo": n_ppo,
            "ppo_from": base_cfg["ppo_epochs"], "ppo_to": ppo_to[0], "baseline_seeds": S["baseline_seeds"]}


# ---------------------------------------------------------------- drawing
def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def search_panel(ax, D, arm, upto, show_y):
    A, ntr, per = D["arms"][arm], D["ntr"], D["per"]
    col = COLOR[arm]
    vk.style_axes(ax)
    for rb in np.arange(per + 0.5, ntr, per):
        ax.axvline(rb, color=vk.GRID, lw=1, zorder=0)
    ax.set_xlim(-0.6, ntr + 0.8)
    ax.set_xticks([per * j + (per + 1) / 2 for j in range(D["rounds"])])
    ax.set_xticklabels([str(j + 1) for j in range(D["rounds"])], fontsize=13, color=vk.INK2)
    ax.set_xlabel(f"round ({per} trials per run)", fontsize=13, color=vk.INK2, labelpad=8)
    ax.tick_params(labelleft=show_y)
    ax.set_title(A["title"], fontsize=17, fontweight="bold", color=vk.INK, loc="left", pad=10)
    x0, bar = 0.0, 100 * D["bar"]
    ax.plot([x0], [bar], "o", ms=10, color=vk.GREY, mec=vk.SURFACE, mew=1.5, zorder=7)
    for rr in A["runs"]:
        off = (rr["run"] - (D["runs"] - 1) / 2) * 0.09
        sx, sy = [x0], [bar]
        for i, t in enumerate(rr["trials"][:upto]):
            sx += [i + 1 + off, i + 1 + off]
            sy += [sy[-1], 100 * t["running_best"]]
        if upto:
            sx.append(upto + 0.45 + off)
            sy.append(sy[-1])
        ax.plot(sx, sy, color=col, lw=1.7, alpha=0.75, solid_joinstyle="miter", zorder=4)
        for i, t in enumerate(rr["trials"][:upto]):
            x = i + 1 + off
            if t["status"] != "ok":
                ax.plot(x, 0, "x", ms=7, mew=1.6, color=vk.INK2, zorder=5)
            elif t["new_best"]:
                ax.plot(x, 100 * t["dev_exact"], "o", ms=7.5, color=col, mec=vk.SURFACE, mew=0.8, zorder=6)
            else:
                ax.plot(x, 100 * t["dev_exact"], "o", ms=5, color=vk.OTHER, mec=vk.SURFACE, mew=0.5, alpha=0.8,
                        zorder=3)


def answer_panel(ax, D, show):
    order = list(ARMS) + ["start"]
    vk.style_axes(ax)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["smaller\nLLM", "larger\nLLM", "random\nsearch", f"start recipe\n({D['baseline_seeds']} seeds)"], fontsize=12.5,
                       color=vk.INK2)
    ax.axvline(len(ARMS) - 0.5, color=vk.AXIS, lw=1, zorder=0)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("built exactly, held-out designs", fontsize=13, color=vk.INK2, labelpad=10)
    ax.set_title(f"Held-out test, after {D['ntr']} trials", fontsize=17, fontweight="bold", color=vk.INK,
                 loc="left", pad=10)
    if not show:
        return
    for j, a in enumerate(order):
        vals = D["arms"][a]["final_k32_per_run"] if a in D["arms"] else D["base_final"]["values"]
        mu = D["arms"][a]["final_k32_mean"] if a in D["arms"] else D["base_final"]["mean"]
        col = COLOR[a]
        ax.bar(j, 100 * mu, width=0.62, color=col, alpha=0.18, zorder=1)
        ax.plot([j - 0.31, j + 0.31], [100 * mu] * 2, color=col, lw=3, zorder=2)
        xs = j + np.linspace(-0.18, 0.18, len(vals))
        ax.plot(xs, 100 * np.array(vals), "o", ms=8, color=col, mec=vk.SURFACE, mew=1.0, zorder=4)
        ax.text(j, 100 * mu / 2, pct(mu), ha="center", va="center", fontsize=17, fontweight="bold",
                color=vk.INK if a != "start" else vk.INK2, zorder=5)


def legend(fig, D, y):
    h = [Line2D([], [], color=vk.BLUE, marker="o", ms=8, lw=0, mec=vk.SURFACE, label="AI scientist trial: new best in its run"),
         Line2D([], [], color=vk.ORANGE, marker="o", ms=8, lw=0, mec=vk.SURFACE, label="random trial: new best"),
         Line2D([], [], color=vk.OTHER, marker="o", ms=6, lw=0, mec=vk.SURFACE, label="not a new best"),
         Line2D([], [], color=vk.INK2, marker="x", ms=7, lw=0, mew=1.6, label=f"timed out ({D['timeout_s']} s)"),
         Line2D([], [], color=vk.INK2, lw=1.7, label="a run's best so far"),
         Line2D([], [], color=vk.GREY, marker="o", ms=9, lw=0, mec=vk.SURFACE,
                label=f"start recipe, seed 0: {pct(D['bar'])}"),
         Patch(facecolor=vk.INK2, alpha=0.25, label="held-out: mean (bar), each run (dot)")]
    leg = fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.045, y), ncol=7, frameon=False, fontsize=12.5,
                     handletextpad=0.45, columnspacing=1.5)
    for t in leg.get_texts():
        t.set_color(vk.INK2)


def chart_fig(D, upto, answer, caption=None, final=False):
    fig = vk.new_fig()
    n_search = len(ARMS) * D["runs"]
    vk.header(fig, f"An AI scientist tunes a training recipe: {n_search} searches, {n_search * D['ntr']} trials",
              f"Each dot is one trial: a recipe trained once ({D['steps']:,} steps) and scored on {D['n_dev']} dev "
              f"designs. Lines: each run's best so far, starting from the start recipe's score.", stage="Level 0")
    legend(fig, D, 0.826)
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.9], left=0.06, right=0.93, top=0.735, bottom=0.17,
                  wspace=0.09)
    for j, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[0, j])
        search_panel(ax, D, arm, upto, show_y=(j == 0))
        if j == 0:
            ax.set_ylabel(f"built exactly, {D['n_dev']} dev designs", fontsize=13, color=vk.INK2, labelpad=10)
            ax.text(0.5, 100 * D["bar"] - 3.5, f"bar to beat: {pct(D['bar'])}", ha="left", va="top", fontsize=12,
                    color=vk.INK2, zorder=8, bbox=dict(boxstyle="round,pad=0.2", fc=vk.SURFACE, ec="none", alpha=0.9))
    axa = fig.add_subplot(gs[0, 3])
    answer_panel(axa, D, answer)
    axa.set_position(axa.get_position().translated(0.02, 0))
    if final:
        a = D["arms"]
        vk.caption(fig, f"Held-out: smaller LLM {pct(a['sonnet']['final_k32_mean'])}, larger LLM "
                        f"{pct(a['opus']['final_k32_mean'])}, random search {pct(a['random']['final_k32_mean'])}, "
                        f"start recipe {pct(D['base_final']['mean'])}.", y=0.068)
        gs_, go_ = D["gap"]["sonnet"], D["gap"]["opus"]
        vk.caption(fig, f"Lead over random, 95% interval over {D['runs']} runs: smaller LLM +{100 * gs_[0]:.1f} to "
                        f"+{100 * gs_[1]:.1f} points, larger LLM +{100 * go_[0]:.1f} to +{100 * go_[1]:.1f}. Dev scores "
                        f"(left) are what the searches selected on.", y=0.03, bold=False)
    elif caption:
        vk.caption(fig, caption)
    return fig


def title_card(D):
    return vk.card("Level 0: an AI scientist tunes a training recipe",
                   ["An LLM picks the agent's training settings from a fixed menu; random search gets the same budget.",
                    "The agent: PPO, learning to build target designs in a small brick-assembly world.",
                    f"Each arm: {D['runs']} runs × {D['ntr']} trials ({D['rounds']} rounds of {D['per']}). "
                    f"Score: share of designs built exactly."],
                   stage="Learning in public")


def takeaway_card(D):
    return vk.card("That was tuning, not research.",
                   [f"The AI scientist chose from a fixed menu of {D['n_knobs']} settings. It wrote no code.",
                    f"The one change all {D['n_rec']} final AI-scientist recipes shared: a higher learning rate "
                    f"({D['lr_from']:g} → {D['lr_lo']:g}–{D['lr_hi']:g}).",
                    f"{D['n_ppo']} of {D['n_rec']} also raised PPO epochs from {D['ppo_from']} to {D['ppo_to']}.",
                    "The small world was too easy: every AI-scientist run reached 100% on dev.",
                    f"{D['runs']} runs per arm, so the intervals are wide."],
                   stage="Level 0")


SCENES_NOTE = "title, bar, 8 rounds, answer, final, takeaway"


def scenes(D):
    out = [("title", lambda: title_card(D), 3.5),
           ("bar", lambda: chart_fig(D, 0, False, "Every run starts from the start recipe's dev score: the bar to "
                                                  "beat."), 2.5)]
    for rd in range(1, D["rounds"] + 1):
        out.append((f"round{rd}", (lambda rd=rd: chart_fig(
            D, rd * D["per"], False, f"Round {rd} of {D['rounds']}: {D['per']} new trials in each of the "
                                     f"{D['runs']} runs per arm.")), 0.9))
    out.append(("answer", lambda: chart_fig(
        D, D["ntr"], True, f"Each run's pick after {D['ntr']} trials, retrained on {D['fresh_seeds']} fresh seeds, "
                           f"scored once on held-out designs."), 3.5))
    out.append(("final", lambda: chart_fig(D, D["ntr"], True, final=True), 6.5))
    out.append(("takeaway", lambda: takeaway_card(D), 5.5))
    return out


def summary(D, secs=None):
    a = D["arms"]
    return {
        "source": "campaigns/main (journal.jsonl, incumbents.json, results.json, manifest.json; checked vs REPORT.md)",
        "arms_shown": list(ARMS), "arms_left_out": ["proxy_sonnet", "random_topview"],
        "bar_to_beat_dev_exact_baseline_seed0": D["bar"],
        "settings": {"runs_per_arm": D["runs"], "rounds": D["rounds"], "trials_per_round": D["per"],
                     "trials_per_run": D["ntr"], "budget_steps": D["steps"], "timeout_s": D["timeout_s"],
                     "dev_designs": D["n_dev"], "fresh_seeds": D["fresh_seeds"], "recipe_knobs": D["n_knobs"]},
        "search": {arm: {"title": a[arm]["title"], "runs": [
            {"run": rr["run"], "incumbent_k32": rr["incumbent_k32"], "dev_exact_k32": rr["dev_k32"],
             "trials": [{"trial": t["trial"], "round": t["round"], "status": t["status"],
                         "dev_exact": t["dev_exact"], "new_best": t["new_best"], "running_best": t["running_best"]}
                        for t in rr["trials"]]} for rr in a[arm]["runs"]]} for arm in ARMS},
        "heldout_k32_fresh_seeds": {arm: {"per_run": a[arm]["final_k32_per_run"], "mean": a[arm]["final_k32_mean"],
                                          "ci95": a[arm]["final_k32_ci95"],
                                          "key": f"results.json toy_r.{arm}.fresh_final_exact_by_k.32"} for arm in ARMS},
        "start_recipe_final_exact": {"per_seed": D["base_final"]["values"], "mean": D["base_final"]["mean"],
                                     "key": "results.json baseline.final_exact"},
        "lead_over_random_k32_ci95": {k: {"ci95": v, "key": f"results.json toy_r_comparisons.{k}_minus_random."
                                                            "level_k32_diff_ci95"} for k, v in D["gap"].items()},
        "takeaway": {"recipes": D["n_rec"], "lr_from": D["lr_from"], "lr_to_range": [D["lr_lo"], D["lr_hi"]],
                     "only_knob_all_changed": "lr", "ppo_epochs_raised": D["n_ppo"],
                     "ppo_epochs_from_to": [D["ppo_from"], D["ppo_to"]], "claude_runs_dev_100pct_at_k32": True},
        "video_seconds": secs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    D = load()
    fig = chart_fig(D, D["ntr"], True, final=True)
    fig.savefig(HERE / "level0.png", facecolor=vk.SURFACE, dpi=100)
    plt.close(fig)
    secs = None
    if args.video:
        with vk.Writer(HERE / "level0.mp4") as w:
            for _, make, s in scenes(D):
                w.add(make(), seconds=s)
        secs = round(w.seconds, 2)
    (HERE / "level0_data.json").write_text(json.dumps(summary(D, secs), indent=1))
    print(json.dumps({k: v for k, v in summary(D, secs).items() if k != "search"}, indent=1))


if __name__ == "__main__":
    main()
