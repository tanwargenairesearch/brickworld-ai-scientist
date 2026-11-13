"""Best-so-far progress charts for the two Stage 0 ceiling probes (pilot data, dev designs only).

Each chart: x = candidate in the order the search proposed it, y = exact-success on the 512 xl dev designs.
Every candidate is a dot (its 2-seed mean); light rings are the two single seeds; the step line is the
search's running best, which starts at the starting network's 4-seed mean (the incumbent rule in
orchestrate_arch.py: a candidate becomes the incumbent only if its 2-seed mean is strictly higher).
The horizontal reference line is the starting network's 10-seed mean at the same budget.
Every number is read from the per-run dev.json files written by the frozen evaluator; nothing is copied.

usage (project root): .venv/bin/python -I arch_rsi/viz/progress_chart.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
S0 = HERE.parent / "pilot" / "stage0"
OUT = HERE

# dataviz reference palette, light mode
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"          # categorical slot 1: new best + running best
OTHER = "#898781"         # candidates that did not beat the running best
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial"], "font.size": 13, "text.color": INK,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "svg.fonttype": "none"})


def dev(run_dir: Path) -> float:
    return json.loads((run_dir / "dev.json").read_text())["exact"]


def search_points(sdir: Path):
    st = json.loads((sdir / "state.json").read_text())
    rows = []
    for t in st["trials"]:
        seeds = sorted((sdir / "trials" / t["label"]).glob("seed*"))
        per = [dev(s) for s in seeds]
        rows.append({"label": t["label"], "round": t["round"], "name": t["name"], "per_seed": per,
                     "mean": sum(per) / len(per), "status": t["result"]["status"]})
    return rows, st["incumbent"]


PROBES = [
    {"key": "q131k", "steps": "131,072", "sdir": S0 / "probe2" / "Q" / "g0" / "s1",
     "base4": [S0 / "probe2" / "baseline" / "xl" / f"seed{s}" for s in range(4)],
     "base10_extra": [S0 / "sweep" / f"start_131072_s{s}" for s in range(300, 306)],
     "title": "Probe 2: an AI-scientist search for network designs, 131,072 training steps",
     "key_xy": (0.012, 0.95)},
    {"key": "q524k", "steps": "524,288", "sdir": S0 / "campaign" / "Q" / "g0" / "s1",
     "base4": [S0 / "campaign" / "baseline" / "xl" / f"seed{s}" for s in range(4)],
     "base10_extra": [S0 / "campaign" / "noise" / "xl" / f"seed{s}" for s in range(200, 206)],
     "title": "Probe 1: an AI-scientist search at the full budget, 524,288 training steps",
     "key_xy": (0.53, 0.38)},
]
FLOOR_XL = json.loads((S0 / "summary.json").read_text())["floor_no_hidden_share"]["xl"]


def draw(p):
    rows, incumbent = search_points(p["sdir"])
    base4 = [dev(d) for d in p["base4"]]
    base10 = base4 + [dev(d) for d in p["base10_extra"]]
    start4, start10 = sum(base4) / 4, sum(base10) / len(base10)

    # running best, exactly as the search's incumbent rule
    best, best_label, kept, run = start4, None, [], []
    for r in rows:
        if r["status"] == "ok" and r["mean"] > best:
            best, best_label = r["mean"], r["label"]
            kept.append(r["label"])
        run.append(best)
    assert best_label == incumbent, (best_label, incumbent)

    fig, ax = plt.subplots(figsize=(16, 9.6), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = list(range(1, len(rows) + 1))

    for y in (0, 20, 40, 60, 80, 100):
        ax.axhline(y, color=GRID, lw=1, zorder=0)
    for rb in (3.5, 6.5, 9.5):
        ax.axvline(rb, color=GRID, lw=1, zorder=0)
    ax.axhline(100 * start10, color=INK2, lw=1.5, zorder=1)
    ax.text(len(rows) + 0.45, 100 * start10 + 0.6, f"starting network,\n10-seed mean {100 * start10:.0f}%",
            va="bottom", ha="left", fontsize=12, color=INK2)
    ax.axhline(100 * FLOOR_XL, color=AXIS, lw=1, zorder=1)
    ax.text(len(rows) + 0.45, 100 * FLOOR_XL - 0.6, f"floor {100 * FLOOR_XL:.0f}%: share of designs\nwith no hidden cells",
            va="top", ha="left", fontsize=11, color=MUTED)

    # running-best step line from x=0 (the starting network's 4-seed mean, the search's starting incumbent)
    sx, sy = [0.4], [100 * start4]
    for x, b in zip(xs, run):
        sx += [x, x]
        sy += [sy[-1], 100 * b]
    sx.append(len(rows) + 0.3)
    sy.append(100 * run[-1])
    ax.plot(sx, sy, color=BLUE, lw=2.2, solid_joinstyle="round", solid_capstyle="round", zorder=2)
    ax.plot([0.4], [100 * start4], "o", ms=9, color=INK2, mec=SURFACE, mew=2, zorder=4)
    ax.text(0.4, 100 * start4 - 4, f"bar to beat:\nstart, 4 seeds\n{100 * start4:.0f}%", ha="center", va="top",
            fontsize=11, color=INK2)

    for x, r in zip(xs, rows):
        for v in r["per_seed"]:
            ax.plot(x, 100 * v, "o", ms=8, mfc="none", mec=BLUE if r["label"] in kept else OTHER, mew=1.3,
                    alpha=0.45, zorder=3)
        is_kept = r["label"] in kept
        ax.plot(x, 100 * r["mean"], "o", ms=13 if is_kept else 10, color=BLUE if is_kept else OTHER,
                mec=SURFACE, mew=2, zorder=5)
    # direct labels: the percentage above each new best; names go in a key (labels would collide)
    for x, r in zip(xs, rows):
        if r["label"] in kept:
            ax.annotate(f"{100 * r['mean']:.0f}%", (x, 100 * r["mean"]), xytext=(0, 11), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold", zorder=6)
    key = ["New best, in the order found"] + [
        f"{r['label']}   {100 * r['mean']:.0f}%   {r['name']}" for r in rows if r["label"] in kept]
    kx, ky = p["key_xy"]
    ax.text(kx, ky, "\n".join(key), transform=ax.transAxes, ha="left", va="top", fontsize=11.5, color=INK2,
            linespacing=1.55, zorder=6, bbox=dict(boxstyle="round,pad=0.6", fc=SURFACE, ec=GRID, lw=1))

    ax.set_xlim(-0.2, len(rows) + 0.6)
    ax.set_ylim(0, 105)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=12)
    for i, rnd in enumerate((1, 2, 3, 4)):
        ax.text(2 + 3 * i, -10.5, f"round {rnd}", ha="center", va="top", fontsize=12, color=INK2,
                transform=ax.transData)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 20, 40, 60, 80, 100)], fontsize=12)
    ax.set_ylabel("built exactly, 512 dev designs", fontsize=13, color=INK2, labelpad=10)
    ax.tick_params(length=0, pad=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    n_kept = len(kept)
    fig.text(0.06, 0.955, p["title"], fontsize=21, fontweight="bold", color=INK, ha="left")
    fig.text(0.06, 0.915, f"{len(rows)} candidates proposed by the AI scientist in 4 rounds of 3, {n_kept} became the new best. "
             f"8×8×5 world, dev designs only.", fontsize=14, color=INK2, ha="left")
    fig.text(0.06, 0.885, "Pilot, one search, 2 training seeds per candidate: an illustration, not evidence.",
             fontsize=14, color=INK2, ha="left", style="italic")

    handles = [Line2D([], [], color=BLUE, marker="o", ms=11, lw=0, mec=SURFACE, mew=2, label="new best (2-seed mean)"),
               Line2D([], [], color=OTHER, marker="o", ms=9, lw=0, mec=SURFACE, mew=2, label="did not beat the best"),
               Line2D([], [], color=OTHER, marker="o", ms=8, lw=0, mfc="none", mew=1.3, alpha=0.6, label="single seed"),
               Line2D([], [], color=BLUE, lw=2.2, label="running best")]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.075), ncol=4, frameon=False,
                    fontsize=12, handletextpad=0.5, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.subplots_adjust(left=0.08, right=0.81, top=0.80, bottom=0.15)
    fig.text(0.06, 0.03, "Exact success = the final build matches the target design cell for cell. "
             "Brick-assembly world, lattice rules; PPO training compute-matched across designs.",
             fontsize=11, color=MUTED, ha="left")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"progress_{p['key']}.{ext}", facecolor=SURFACE, dpi=100)
    plt.close(fig)
    return {"probe": p["key"], "start_4seed": round(start4, 4), "start_10seed": round(start10, 4),
            "start_10seed_values": [round(v, 4) for v in base10], "kept": kept, "final_best": round(run[-1], 4),
            "incumbent": incumbent,
            "candidates": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]}


if __name__ == "__main__":
    data = [draw(p) for p in PROBES]
    (OUT / "progress_data.json").write_text(json.dumps(data, indent=1))
    for d in data:
        print(d["probe"], "start4", d["start_4seed"], "start10", d["start_10seed"], "kept", d["kept"], "best", d["final_best"])
