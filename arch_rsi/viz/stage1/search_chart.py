"""Best-so-far progress charts for Stage 1 (dev designs only; the final split stays locked). Visualization only.

1. search_overview.png/.svg, and its animation search_overview.mp4: all 30 searches, one panel per arm.
   - x: candidate, in the order its search proposed it (4 rounds of 3). y: exact success on Q's 512 dev designs.
   - Dot: a candidate's 2-seed mean, the score the search selects on. Blue: it became its search's new best. Grey: it
     didn't. A cross at 0: training hit the 600 s limit, which counts as a failure.
   - Thin lines: each search's running best, by the frozen incumbent rule (a strictly higher 2-seed mean), starting
     from the start's seeds 0-1 mean, the bar every search had to beat. Thick line: the mean of the 10 running bests.
   - Right-hand column: each search's pick retrained on 8 fresh seeds (the confirmation), and the arm's mean.
   - Horizontal line: the starting network on 12 fresh seeds.
2. search_closeup.png/.svg: one search from each Claude arm, chosen by a fixed rule (ranked 6th of 10 by confirmed
   score, the lower of the two middle searches), with each new best named, like a best-so-far progress chart.

Every score is read from the frozen evaluator's dev.json files and cross-checked against state.json and results.json.
usage (project root): .venv/bin/python -I arch_rsi/viz/stage1/search_chart.py
"""
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
A = HERE.parents[1]
C = A / "campaigns" / "stage1"
RES = json.loads((C / "results.json").read_text())
START_SHA = "764c7fb529694085"
ARMS = [("g0", "AI scientist sees its scores"), ("nofb", "Scores hidden from the AI scientist"), ("menu", "Random picks from a menu (no AI scientist)")]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, OTHER = "#2a78d6", "#a3a29b"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial"], "font.size": 13, "text.color": INK,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "svg.fonttype": "none"})


def dev(d: Path) -> float:
    return json.loads((d / "dev.json").read_text())["exact"]


BASE = json.loads((C / "baseline" / "summary.json").read_text())["xl"]
BAR = sum(dev(C / "baseline" / "xl" / f"seed{s}") for s in (0, 1)) / 2
assert abs(BAR - BASE["shown_dev"]) < 1e-12
START12 = [dev(C / "confirm" / START_SHA / f"seed{s}") for s in range(1001, 1013)]
START_MEAN = float(np.mean(START12))
assert abs(START_MEAN - RES["per_arm"]["start"]["mean"]) < 6e-5


def search(arm: str, i: int) -> dict:
    sdir = C / "Q" / arm / f"s{i}"
    st = json.loads((sdir / "state.json").read_text())
    rows = []
    for t in st["trials"]:
        ok = t["result"]["status"] == "ok"
        per = [dev(sdir / "trials" / t["label"] / f"seed{s}") for s in (0, 1)] if ok else []
        m = sum(per) / 2 if ok else None
        if ok:
            assert abs(m - t["result"]["dev_exact"]) < 6e-5, (arm, i, t["label"])
        rows.append({"label": t["label"], "round": t["round"], "name": t["name"], "per_seed": per, "mean": m,
                     "stored": t["result"]["dev_exact"], "status": t["result"]["status"]})
    best, inc, kept, run = BAR, None, [], []
    for r in rows:  # the frozen incumbent rule, on the stored 2-seed mean
        if r["status"] == "ok" and r["stored"] > best:
            best, inc = r["stored"], r["label"]
            kept.append(r["label"])
        run.append(best)
    assert inc == st["incumbent"], (arm, i, inc, st["incumbent"])
    entry = next(s for s in RES["per_arm"][arm]["searches"] if s["search"] == i)
    conf = [dev(C / "confirm" / entry["sha"] / f"seed{s}") for s in range(1001, 1009)]
    assert all(abs(a - b) < 6e-5 for a, b in zip(conf, entry["per_seed"]))
    return {"arm": arm, "search": i, "rows": rows, "kept": kept, "run": run, "pick": entry["pick"],
            "pick_name": entry["name"], "confirm": conf, "confirm_mean": float(np.mean(conf))}


DATA = {arm: [search(arm, i) for i in range(1, 11)] for arm, _ in ARMS}
ARM_MEAN = {arm: float(np.mean([s["confirm_mean"] for s in DATA[arm]])) for arm, _ in ARMS}
for arm, _ in ARMS:
    assert abs(ARM_MEAN[arm] - RES["per_arm"][arm]["mean"]) < 6e-5
P1 = RES["primary_family"]["contrasts"]["g0 - nofb"]
NCAND = 12
XCONF = NCAND + 1.9


def style(ax, show_y):
    ax.set_facecolor(SURFACE)
    for y in (0, 20, 40, 60, 80, 100):
        ax.axhline(y, color=GRID, lw=1, zorder=0)
    for rb in (3.5, 6.5, 9.5):
        ax.axvline(rb, color=GRID, lw=1, zorder=0)
    ax.axvline(NCAND + 0.9, color=AXIS, lw=1, zorder=0)
    ax.set_xlim(-0.1, XCONF + 1.6)
    ax.set_ylim(-4, 104)
    ax.set_xticks([2, 5, 8, 11, XCONF])
    ax.set_xticklabels(["round 1", "round 2", "round 3", "round 4", "pick,\n8 fresh seeds"], fontsize=12, color=INK2)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 20, 40, 60, 80, 100)], fontsize=12)
    ax.tick_params(length=0, pad=8, labelleft=show_y)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def overview_panel(ax, arm, title, upto=NCAND, confirm=True, show_y=True):
    style(ax, show_y)
    ax.set_title(title, fontsize=15, fontweight="bold", color=INK, loc="left", pad=12)
    ax.axhline(100 * START_MEAN, color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=1)
    ax.plot([0.35], [100 * BAR], "o", ms=8, color=INK2, mec=SURFACE, mew=1.5, zorder=4)
    runs = []
    for s in DATA[arm]:
        off = (s["search"] - 5.5) * 0.055
        sx, sy = [0.35], [100 * BAR]
        for x, b in zip(range(1, upto + 1), s["run"][:upto]):
            sx += [x + off, x + off]
            sy += [sy[-1], 100 * b]
        ax.plot(sx, sy, color=BLUE, lw=0.9, alpha=0.28, zorder=2)
        runs.append(s["run"])
        for x, r in zip(range(1, upto + 1), s["rows"][:upto]):
            if r["status"] != "ok":
                ax.plot(x + off, 0, "x", ms=7, mew=1.6, color=INK2, zorder=4)
                continue
            new = r["label"] in s["kept"]
            ax.plot(x + off, 100 * r["mean"], "o", ms=7 if new else 5.5, color=BLUE if new else OTHER,
                    mec=SURFACE, mew=0.8, zorder=5 if new else 3)
    if upto:
        mean_run = 100 * np.mean([r[:upto] for r in runs], axis=0)
        mx, my = [0.35], [100 * BAR]
        for x, b in zip(range(1, upto + 1), mean_run):
            mx += [x, x]
            my += [my[-1], b]
        ax.plot(mx, my, color=BLUE, lw=3, solid_joinstyle="round", zorder=6)
        ax.annotate(f"{my[-1]:.0f}%", (mx[-1], my[-1]), xytext=(4, 9), textcoords="offset points", ha="left",
                    va="bottom", fontsize=12, color=BLUE, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    if confirm:
        for s in DATA[arm]:
            off = (s["search"] - 5.5) * 0.07
            ax.plot(XCONF + off, 100 * s["confirm_mean"], "D", ms=6.5, color=INK2, mec=SURFACE, mew=0.8, zorder=5)
        ax.plot([XCONF - 0.55, XCONF + 0.55], [100 * ARM_MEAN[arm]] * 2, color=INK, lw=3, zorder=6)
        ax.text(XCONF + 0.62, 100 * ARM_MEAN[arm], f"{100 * ARM_MEAN[arm]:.1f}%", ha="left", va="center",
                fontsize=13, color=INK, fontweight="bold", zorder=7)


def legend(fig, y):
    handles = [Line2D([], [], color=BLUE, marker="o", ms=8, lw=0, mec=SURFACE, label="new best in its search"),
               Line2D([], [], color=OTHER, marker="o", ms=7, lw=0, mec=SURFACE, label="not a new best"),
               Line2D([], [], color=INK2, marker="x", ms=7, lw=0, mew=1.6, label="training timed out"),
               Line2D([], [], color=BLUE, lw=3, label="mean running best of 10 searches"),
               Line2D([], [], color=INK2, marker="D", ms=6, lw=0, mec=SURFACE, label="a search's pick, 8 fresh seeds"),
               Line2D([], [], color=INK2, lw=1.4, ls=(0, (5, 3)), label=f"start, 12 fresh seeds: {100 * START_MEAN:.1f}%")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, y), ncol=6, frameon=False, fontsize=11.5,
                     handletextpad=0.4, columnspacing=1.3)
    for t in leg.get_texts():
        t.set_color(INK2)


def overview_fig(upto=NCAND, confirm=True, outcome=True, caption=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    for j, (ax, (arm, title)) in enumerate(zip(axes, ARMS)):
        overview_panel(ax, arm, title, upto, confirm, show_y=(j == 0))
    axes[0].set_ylabel("built exactly, 512 dev designs", fontsize=13, color=INK2, labelpad=10)
    axes[0].text(0.15, 100 * BAR - 3, f"bar to beat: {100 * BAR:.0f}%", ha="left", va="top", fontsize=10.5, color=INK2,
                 zorder=8, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.9))
    fig.text(0.055, 0.952, "Stage 1: 30 architecture searches, 360 network designs", fontsize=22, fontweight="bold",
             color=INK, ha="left")
    fig.text(0.055, 0.912, "Each dot is one design, trained on 2 seeds. The AI scientist wrote them as PyTorch code; "
             "the menu arm drew them at random. 8×8×5 world, dev designs only.",
             fontsize=13.5, color=INK2, ha="left")
    legend(fig, 0.895)
    if outcome:
        d = 100 * (ARM_MEAN["g0"] - ARM_MEAN["nofb"])
        lo, hi = (100 * v for v in P1["ci95_boot"])
        msg = (f"Seeing its scores added {d:+.1f} points on fresh seeds (95% interval {lo:+.1f} to {hi:+.1f}; p = "
               f"{P1['p_holm']:.2f}): nothing measurable. Both AI-scientist arms beat the start by about 50 points.")
        fig.text(0.055, 0.035, msg, fontsize=13, color=INK, ha="left", fontweight="bold")
    elif caption:
        fig.text(0.055, 0.035, caption, fontsize=13, color=INK, ha="left", fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.79, bottom=0.14, wspace=0.08)
    return fig


def closeup_fig():
    picks = {}
    for arm in ("g0", "nofb"):
        ranked = sorted(DATA[arm], key=lambda s: -s["confirm_mean"])
        picks[arm] = ranked[5]
    fig, axes = plt.subplots(1, 2, figsize=(16, 10.4), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    for j, (ax, arm) in enumerate(zip(axes, ("g0", "nofb"))):
        s = picks[arm]
        style(ax, j == 0)
        ax.set_xticks(list(range(1, NCAND + 1)) + [XCONF])
        ax.set_xticklabels([r["label"] for r in s["rows"]] + ["pick,\n8 seeds"], fontsize=10.5, color=INK2)
        title = dict(ARMS)[arm]
        ax.set_title(f"{title}: search {s['search']} (ranked 6th of 10)", fontsize=15, fontweight="bold", color=INK,
                     loc="left", pad=12)
        ax.axhline(100 * START_MEAN, color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=1)
        sx, sy = [0.35], [100 * BAR]
        for x, b in zip(range(1, NCAND + 1), s["run"]):
            sx += [x, x]
            sy += [sy[-1], 100 * b]
        sx.append(NCAND + 0.5)
        sy.append(sy[-1])
        ax.plot(sx, sy, color=BLUE, lw=2.4, zorder=2)
        ax.plot([0.35], [100 * BAR], "o", ms=9, color=INK2, mec=SURFACE, mew=2, zorder=4)
        for x, r in zip(range(1, NCAND + 1), s["rows"]):
            new = r["label"] in s["kept"]
            if r["status"] != "ok":
                ax.plot(x, 0, "x", ms=9, mew=2, color=INK2, zorder=4)
                continue
            for v in r["per_seed"]:
                ax.plot(x, 100 * v, "o", ms=8, mfc="none", mec=BLUE if new else OTHER, mew=1.2, alpha=0.5, zorder=3)
            ax.plot(x, 100 * r["mean"], "o", ms=12 if new else 9, color=BLUE if new else OTHER, mec=SURFACE, mew=2,
                    zorder=5)
            if new:
                ax.annotate(f"{100 * r['mean']:.0f}%", (x, 100 * r["mean"]), xytext=(0, 10), textcoords="offset points",
                            ha="center", va="bottom", fontsize=11.5, color=INK, fontweight="bold", zorder=6)
        for v in s["confirm"]:
            ax.plot(XCONF, 100 * v, "o", ms=7, mfc="none", mec=INK2, mew=1.1, alpha=0.6, zorder=3)
        ax.plot([XCONF - 0.5, XCONF + 0.5], [100 * s["confirm_mean"]] * 2, color=INK, lw=3, zorder=6)
        ax.text(XCONF + 0.6, 100 * s["confirm_mean"], f"{100 * s['confirm_mean']:.0f}%", ha="left", va="center",
                fontsize=12, color=INK, fontweight="bold")
        key = ["New best, in the order found:"] + [
            f"{r['label']}  {100 * r['mean']:.0f}%  {r['name'][:46]}" for r in s["rows"] if r["label"] in s["kept"]]
        key.append(f"Pick after retesting the top 3: {s['pick']}")
        ax.text(0.0, -0.13, "\n".join(key), transform=ax.transAxes, ha="left", va="top", fontsize=11, color=INK2,
                linespacing=1.5, zorder=8)
    axes[0].set_ylabel("built exactly, 512 dev designs", fontsize=13, color=INK2, labelpad=10)
    fig.text(0.055, 0.955, "Two Stage 1 searches up close", fontsize=22, fontweight="bold", color=INK, ha="left")
    fig.text(0.055, 0.915, "Chosen by a fixed rule: the search ranked 6th of 10 by confirmed score in each AI-scientist arm. "
             "Big dot = 2-seed mean, rings = single seeds.", fontsize=13.5, color=INK2, ha="left")
    fig.text(0.055, 0.885, "In the right panel the AI scientist never saw these scores: the harness recorded them, and the final "
             "pick still used them.", fontsize=13.5, color=INK2, ha="left", style="italic")
    handles = [Line2D([], [], color=BLUE, marker="o", ms=10, lw=0, mec=SURFACE, label="new best (2-seed mean)"),
               Line2D([], [], color=OTHER, marker="o", ms=8, lw=0, mec=SURFACE, label="not a new best"),
               Line2D([], [], color=BLUE, lw=2.4, label="running best"),
               Line2D([], [], color=INK, lw=3, label="pick on 8 fresh seeds (rings: each seed)"),
               Line2D([], [], color=INK2, lw=1.4, ls=(0, (5, 3)), label=f"start, 12 fresh seeds: {100 * START_MEAN:.1f}%")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.87), ncol=5, frameon=False,
                     fontsize=11.5, handletextpad=0.4, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.subplots_adjust(left=0.07, right=0.975, top=0.79, bottom=0.25, wspace=0.08)
    return fig, picks


def to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def animate(path, fps=12):
    states = [(0, False, False, "Each search starts from the start's score on seeds 0–1, the bar to beat.", 2.5)]
    for k in range(1, NCAND + 1):
        rnd = (k - 1) // 3 + 1
        states.append((k, False, False, f"Round {rnd}: candidate {k} of 12 in every search.", 1.1))
    states.append((NCAND, True, False, "Each search's pick is retrained on 8 fresh seeds. The drop is selection "
                   "optimism: picks look best partly by luck.", 4.0))
    states.append((NCAND, True, True, None, 6.0))
    W, H = 1600, 900
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                           "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    n = 0
    for upto, conf, out, cap, secs in states:
        fig = overview_fig(upto, conf, out, cap)
        buf = to_rgb(fig)
        plt.close(fig)
        assert buf.shape == (H, W, 3), buf.shape
        for _ in range(int(round(secs * fps))):
            ff.stdin.write(buf.tobytes())
            n += 1
    ff.stdin.close()
    ff.wait()
    return n / fps


if __name__ == "__main__":
    fig = overview_fig()
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"search_overview.{ext}", facecolor=SURFACE, dpi=100)
    plt.close(fig)
    fig, picks = closeup_fig()
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"search_closeup.{ext}", facecolor=SURFACE, dpi=100)
    plt.close(fig)
    secs = animate(HERE / "search_overview.mp4")
    summary = {
        "bar_to_beat_start_seeds_0_1": round(BAR, 4), "start_12_fresh_seeds": [round(v, 4) for v in START12],
        "start_mean": round(START_MEAN, 4),
        "arms": {arm: {"confirm_mean": round(ARM_MEAN[arm], 4),
                       "mean_running_best_by_candidate": [round(float(v), 4) for v in np.mean([s["run"] for s in DATA[arm]], 0)],
                       "new_bests_per_search": [len(s["kept"]) for s in DATA[arm]],
                       "timeouts": sum(r["status"] != "ok" for s in DATA[arm] for r in s["rows"])} for arm, _ in ARMS},
        "closeup": {arm: {"search": s["search"], "rule": "ranked 6th of 10 by 8-seed confirmed score",
                          "kept": s["kept"], "pick": s["pick"], "pick_name": s["pick_name"],
                          "confirm_mean": round(s["confirm_mean"], 4)} for arm, s in picks.items()},
        "video_seconds": round(secs, 1)}
    (HERE / "search_data.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
