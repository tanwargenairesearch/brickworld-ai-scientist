"""Stage 0 article video: one pilot search, candidate by candidate (the article's Figure 2). Visualization only.

What it shows: Stage 0's probe 2, one search on world Q (8x8x5) at 131,072 training steps, the probe the article's
Figure 2 draws. Claude wrote 12 networks in 4 rounds of 3; each trained on 2 seeds (0 and 1).
  - Dot: a network's 2-seed mean exact success on the 512 dev designs. Rings: its single seeds.
  - Step line: the running best by the incumbent rule (a network becomes the best only if its 2-seed mean is strictly
    higher), starting from the bar to beat, the starting network's 4-seed mean (seeds 0-3), as in Figure 2.
  - Reference line: the starting network's 10-seed mean at this budget (seeds 0-3 plus the sweep's seeds 300-305).
    Floor line: the share of dev designs with no hidden cells.
  - Captions carry the article's caveat; a takeaway card closes the video.

Data sources (read-only):
  arch_rsi/pilot/stage0/probe2/Q/g0/s1/state.json            trials, names, rounds, stored 2-seed means, incumbent
  arch_rsi/pilot/stage0/probe2/Q/g0/s1/trials/*/seed{0,1}/    per-seed dev.json (frozen evaluator)
  arch_rsi/pilot/stage0/probe2/baseline/xl/seed{0..3}/        the start's 4 seeds; baseline/summary.json cross-check
  arch_rsi/pilot/stage0/sweep/start_131072_s{300..305}/       the start's 6 more seeds at this budget
  arch_rsi/pilot/stage0/probe2/manifest.json                  budget, rounds, candidates per round
  arch_rsi/targets/xl_dev.pt                                  dev designs only: the hidden-cell mask (floor, skill)
  arch_rsi/pilot/stage0/summary.json, REPORT.md               cross-checks (asserts)
  ../linkedin/rsi-article-v14.md (if present)                 cross-check that this is Figure 2's probe
Every number shown is computed here from those files and asserted against REPORT.md (within rounding).

usage (project root, low priority while Stage 2b runs):
  nice -n 19 ./.venv/bin/python -I arch_rsi/viz/stages/stage0/make_stage0.py           # png + data json + mp4
  nice -n 19 ./.venv/bin/python -I arch_rsi/viz/stages/stage0/make_stage0.py --still   # png + data json only
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402  (only to read the dev design tensor; nothing is trained)
from matplotlib.lines import Line2D  # noqa: E402

from arch_rsi.viz.stages import vidkit as vk  # noqa: E402

A = ROOT / "arch_rsi"
S0 = A / "pilot" / "stage0"
P2 = S0 / "probe2"
SDIR = P2 / "Q" / "g0" / "s1"
ARTICLE = ROOT.parent / "linkedin" / "rsi-article-v14.md"
NAME = "stage0"


def dev_json(d: Path) -> dict:
    return json.loads((d / "dev.json").read_text())


def hidden_mask() -> tuple:
    """Dev designs with at least one empty cell under a filled one (the rule in analyze_stage0.py / run_stageG.py)."""
    v = torch.load(A / "targets" / "xl_dev.pt").bool()
    above = (v.flip(1).cumsum(1).flip(1) - v.int()) > 0
    n, z, y, x = v.shape
    return ((~v) & above).flatten(1).any(1).numpy(), (x, y, z), n


def pct(v: float) -> int:
    return int(round(100 * v))


def load() -> dict:
    H, dims, n_dev = hidden_mask()
    man = json.loads((P2 / "manifest.json").read_text())
    cfg = man["config"]
    budget, R, K = cfg["steps"]["xl"], cfg["R"], cfg["K"]
    trial_seeds, base_seeds = cfg["trial_seeds"], cfg["base_seeds"]

    st = json.loads((SDIR / "state.json").read_text())
    rows = []
    for t in st["trials"]:
        per = [dev_json(SDIR / "trials" / t["label"] / f"seed{s}")["exact"] for s in trial_seeds]
        m = float(np.mean(per))
        assert t["result"]["status"] == "ok"
        assert abs(m - t["result"]["dev_exact"]) < 6e-5, t["label"]
        assert dev_json(SDIR / "trials" / t["label"] / "seed0")["n"] == n_dev
        rows.append({"label": t["label"], "round": t["round"], "name": t["name"], "per_seed": per, "mean": m})
    assert len(rows) == R * K and [r["round"] for r in rows] == [k // K + 1 for k in range(R * K)]

    base4 = [dev_json(P2 / "baseline" / "xl" / f"seed{s}")["exact"] for s in base_seeds]
    bsum = json.loads((P2 / "baseline" / "summary.json").read_text())["xl"]
    assert all(abs(a - b) < 1e-12 for a, b in zip(base4, bsum["devs"]))
    extra = sorted((S0 / "sweep").glob(f"start_{budget}_s*"))
    assert [p.name for p in extra] == [f"start_{budget}_s{s}" for s in range(300, 306)]
    pe10 = [np.array(dev_json(d)["per_design"]["exact"], bool)
            for d in [P2 / "baseline" / "xl" / f"seed{s}" for s in base_seeds] + extra]
    base10 = [float(p.mean()) for p in pe10]
    assert all(abs(a - b) < 1e-12 for a, b in zip(base10, base4 + [dev_json(d)["exact"] for d in extra]))
    learned10 = sum(float(p[H].mean()) >= 0.10 for p in pe10)  # "learned the skill": >= 10% of hidden-cell designs
    start4, start10 = float(np.mean(base4)), float(np.mean(base10))
    floor = float(1 - H.mean())
    assert abs(floor - json.loads((S0 / "summary.json").read_text())["floor_no_hidden_share"]["xl"]) < 1e-12

    # running best by the incumbent rule (strictly higher 2-seed mean), from the start's 4-seed mean (Figure 2's bar)
    best, inc, kept, run = start4, None, [], []
    for r in rows:
        if r["mean"] > best:
            best, inc = r["mean"], r["label"]
            kept.append(r["label"])
        run.append(best)
    assert inc == st["incumbent"]
    # the orchestrator's own bar is the seeds 0-1 mean (shown_dev); the path of new bests is the same under either bar
    b2, k2 = bsum["shown_dev"], []
    for r in rows:
        if r["mean"] > b2:
            b2 = r["mean"]
            k2.append(r["label"])
    assert k2 == kept

    D = {"rows": rows, "kept": kept, "run": run, "start4": start4, "base4": base4, "start10": start10,
         "base10": base10, "learned10": learned10, "floor": floor, "budget": budget, "R": R, "K": K,
         "dims": dims, "n_dev": n_dev, "incumbent": inc, "shown_dev_bar": bsum["shown_dev"]}
    check_report(D)
    check_article(D)
    return D


def check_report(D: dict) -> None:
    """Cross-check against arch_rsi/pilot/stage0/REPORT.md (rounded figures in its probe 2 section)."""
    rep = (S0 / "REPORT.md").read_text()
    m = re.search(r"best result per round rose ([\d%→ ]+)\.", rep)
    per_round = [int(x) for x in re.findall(r"\d+", m.group(1))]
    rows = D["rows"]
    assert per_round == [pct(max(r["mean"] for r in rows if r["round"] == k)) for k in range(1, D["R"] + 1)]
    m = re.search(r"separate critic, scored (\d+)% and (\d+)% on its two seeds", rep)
    win = next(r for r in rows if r["label"] == D["incumbent"])
    assert [int(m.group(1)), int(m.group(2))] == [pct(v) for v in win["per_seed"]]
    m = re.search(r"seeds 0–3 scored (\d+), (\d+), (\d+) and (\d+)%", rep)
    assert [int(g) for g in m.groups()] == [pct(v) for v in D["base4"]]
    m = re.search(r"mean of (\d+)% over 10 seeds, and (\d+) of the 10 learned the skill", rep)
    assert int(m.group(1)) == pct(D["start10"]) and int(m.group(2)) == D["learned10"] and len(D["base10"]) == 10
    m = re.search(r"22% on P and (\d+)% on Q", rep)
    assert int(m.group(1)) == pct(D["floor"])
    assert "**Probe 2: one g0 search on Q at 131,072 steps.**" in rep and D["budget"] == 131072


def check_article(D: dict) -> None:
    """Figure 2's alt text and the caveat list in the article draft, if it is on this machine."""
    if not ARTICLE.exists():
        print("article not found; skipped the Figure 2 cross-check")
        return
    art = ARTICLE.read_text()
    m = re.search(r"running best rises from (\d+)% to (\d+)%, (\d+)%, (\d+)% and then (\d+)%", art)
    steps = [pct(v) for v in dict.fromkeys(D["run"])]
    assert [int(g) for g in m.groups()] == steps, (m.groups(), steps)
    m = re.search(r"starting network's 4-seed mean of (\d+)%; its 10-seed mean of (\d+)%", art)
    assert [int(g) for g in m.groups()] == [pct(D["start4"]), pct(D["start10"])]
    m = re.search(r"floor line marks the (\d+)% of designs with no hidden cells", art)
    assert int(m.group(1)) == pct(D["floor"])
    m = re.search(r"10 seeds at this budget ranged from (\d+)% to (\d+)%. Only (\d+) of the 10 learned", art)
    assert [int(g) for g in m.groups()] == [pct(min(D["base10"])), pct(max(D["base10"])), D["learned10"]]
    print("article Figure 2 cross-check passed")


# ---------------------------------------------------------------- drawing
X0 = 0.15  # where the bar to beat sits


def chart(D: dict, upto: int, caption: str = None, focus: bool = False):
    rows, kept, run = D["rows"], D["kept"], D["run"]
    n = len(rows)
    fig = vk.new_fig()
    x, y, z = D["dims"]
    vk.header(fig, "One search, round by round",
              f"{n} networks the AI scientist wrote in {D['R']} rounds of {D['K']}, each trained on 2 seeds. "
              f"{x}×{y}×{z} brick-assembly world, {D['budget']:,} training steps, {D['n_dev']} dev designs.",
              stage="Stage 0 · pilot search (the article's Figure 2)")
    ax = fig.add_axes([0.065, 0.175, 0.615, 0.555])
    vk.style_axes(ax)
    for rb in np.arange(D["K"] + 0.5, n, D["K"]):
        ax.axvline(rb, color=vk.GRID, lw=1, zorder=0)
    ax.set_xlim(-0.55, n + 0.6)
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels([r["label"] for r in rows], fontsize=13, color=vk.INK2)
    for k in range(D["R"]):
        ax.text((D["K"] + 1) / 2 + D["K"] * k, -0.085, f"round {k + 1}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=14, color=vk.INK2, fontweight="bold")
    ax.set_ylabel(f"built exactly, {D['n_dev']} dev designs", fontsize=14, color=vk.INK2, labelpad=10)

    # reference lines, labelled in the right margin
    s10, fl = 100 * D["start10"], 100 * D["floor"]
    ax.axhline(s10, color=vk.INK2, lw=1.6, zorder=1)
    ax.axhline(fl, color=vk.AXIS, lw=1.2, zorder=1)
    ax.text(n + 0.75, s10 + 0.8, f"start, 10 seeds: {s10:.0f}%", ha="left", va="bottom", fontsize=13,
            color=vk.INK2, clip_on=False)
    ax.text(n + 0.75, fl - 0.8, f"floor {fl:.0f}%: designs with\nno hidden cells", ha="left", va="top", fontsize=12,
            color=vk.MUTED, clip_on=False, linespacing=1.3)

    # the bar to beat
    s4 = 100 * D["start4"]
    ax.plot([X0], [s4], "o", ms=10, color=vk.INK2, mec=vk.SURFACE, mew=2, zorder=4)
    ax.text(X0, s4 + 4, f"bar to beat:\nstart's 4 seeds\n{s4:.0f}%", ha="center", va="bottom", fontsize=12,
            color=vk.INK2, linespacing=1.3, zorder=6)

    # running best
    sx, sy = [X0], [s4]
    for xx, b in zip(range(1, upto + 1), run[:upto]):
        sx += [xx, xx]
        sy += [sy[-1], 100 * b]
    if upto == n:
        sx.append(n + 0.35)
        sy.append(sy[-1])
    ax.plot(sx, sy, color=vk.BLUE, lw=2.6, solid_joinstyle="round", solid_capstyle="round", zorder=2)

    for xx, r in zip(range(1, upto + 1), rows[:upto]):
        new = r["label"] in kept
        for v in r["per_seed"]:
            ax.plot(xx, 100 * v, "o", ms=9, mfc="none", mec=vk.BLUE if new else vk.OTHER, mew=1.4, alpha=0.5, zorder=3)
        ax.plot(xx, 100 * r["mean"], "o", ms=15 if new else 11, color=vk.BLUE if new else vk.OTHER,
                mec=vk.SURFACE, mew=2, zorder=5)
        if new:
            ax.annotate(f"{100 * r['mean']:.0f}%", (xx, 100 * r["mean"]), xytext=(0, 17), textcoords="offset points",
                        ha="center", va="bottom", fontsize=14, color=vk.INK, fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.12", fc=vk.SURFACE, ec="none", alpha=0.85))
    if focus and upto:
        r = rows[upto - 1]
        ax.plot(upto, 100 * r["mean"], "o", ms=30, mfc="none", mec=vk.INK, mew=1.4, alpha=0.55, zorder=6)

    # key of new bests, in the right margin
    found = [r for r in rows[:upto] if r["label"] in kept]
    fig.text(0.722, 0.728, "New best, in the order found", ha="left", va="top", fontsize=14, color=vk.INK,
             fontweight="bold")
    for j, r in enumerate(found):
        yy = 0.688 - 0.034 * j
        fig.text(0.722, yy, f"{r['label']}  {100 * r['mean']:.0f}%", ha="left", va="top", fontsize=12.5, color=vk.BLUE,
                 fontweight="bold")
        fig.text(0.776, yy, r["name"], ha="left", va="top", fontsize=12.5, color=vk.INK2)
    if not found:
        fig.text(0.722, 0.688, "none yet", ha="left", va="top", fontsize=12.5, color=vk.MUTED)

    handles = [Line2D([], [], color=vk.BLUE, marker="o", ms=12, lw=0, mec=vk.SURFACE, mew=2, label="new best (2-seed mean)"),
               Line2D([], [], color=vk.OTHER, marker="o", ms=10, lw=0, mec=vk.SURFACE, mew=2, label="not a new best"),
               Line2D([], [], color=vk.OTHER, marker="o", ms=9, lw=0, mfc="none", mew=1.4, label="single seed"),
               Line2D([], [], color=vk.BLUE, lw=2.6, label="running best")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.815), ncol=4, frameon=False,
                     fontsize=13.5, handletextpad=0.5, columnspacing=1.8)
    for t in leg.get_texts():
        t.set_color(vk.INK2)
    if caption:
        vk.caption(fig, caption)
    return fig


def takeaway(D: dict):
    n = len(D["rows"])
    x, y, z = D["dims"]
    lo, hi = pct(min(D["base10"])), pct(max(D["base10"]))
    return vk.card("Why this chart proves almost nothing", [
        f"It's the best of {n} tries, each on 2 seeds, scored on the designs used to choose it.",
        f"Seeds swing: the start's {len(D['base10'])} seeds at this budget ranged from {lo}% to {hi}%.",
        f"Only {D['learned10']} of those {len(D['base10'])} learned to build designs with cells hidden under an overhang.",
        "Skill, selection luck, or learning from results: only controls can tell them apart."],
        stage="Stage 0 · pilot search",
        foot=f"One pilot search, {x}×{y}×{z} brick-assembly world, {D['budget']:,} training steps, dev designs only.")


def captions(D: dict) -> list:
    rows, kept, n = D["rows"], D["kept"], len(D["rows"])
    states = [(0, f"The search starts from the start's 4-seed mean, {100 * D['start4']:.0f}%: the bar each network "
                  f"has to beat.", 2.6)]
    for k, r in enumerate(rows, 1):
        a, b = (100 * v for v in r["per_seed"])
        tail = " New best." if r["label"] in kept else ""
        states.append((k, f"Round {r['round']}, {r['label']}: {r['name']}. Seeds {a:.0f}% and {b:.0f}%, "
                          f"mean {100 * r['mean']:.0f}%.{tail}", 1.15))
    states.append((n, "Each dot is a 2-seed score on the same dev designs the search chose on.", 2.6))
    states.append((n, "A running best rises even under pure noise. An illustration, not evidence.", 2.8))
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", action="store_true", help="write the png and data json only")
    args = ap.parse_args()
    D = load()
    n = len(D["rows"])
    states = captions(D)
    fig = chart(D, n, states[-1][1])
    fig.savefig(HERE / f"{NAME}.png", facecolor=vk.SURFACE, dpi=100)
    vk.plt.close(fig)
    if args.still:
        fig = takeaway(D)
        fig.savefig(HERE / f"{NAME}_card_preview.png", facecolor=vk.SURFACE, dpi=100)
        vk.plt.close(fig)
    secs = None
    if not args.still:
        (HERE / f"{NAME}_card_preview.png").unlink(missing_ok=True)
        with vk.Writer(HERE / f"{NAME}.mp4") as w:
            for k, (upto, cap, s) in enumerate(states):
                w.add(chart(D, upto, cap, focus=0 < k <= n), seconds=s)
            chart_s = w.seconds
            w.add(takeaway(D), seconds=5.0)
        secs = {"chart": round(chart_s, 2), "card": round(w.seconds - chart_s, 2), "total": round(w.seconds, 2)}
    r4 = lambda v: round(float(v), 4)  # noqa: E731
    data = {
        "what": "Stage 0 probe 2 (the article's Figure 2): one AI-scientist search on world Q at 131,072 steps, dev designs",
        "sources": {"trials": str(SDIR.relative_to(ROOT)), "start_seeds_0_3": str((P2 / "baseline" / "xl").relative_to(ROOT)),
                    "start_seeds_300_305": str((S0 / "sweep").relative_to(ROOT)) + f"/start_{D['budget']}_s*",
                    "hidden_mask": "arch_rsi/targets/xl_dev.pt", "report": "arch_rsi/pilot/stage0/REPORT.md"},
        "world_dims_xyz": D["dims"], "n_dev_designs": D["n_dev"], "budget_steps": D["budget"],
        "rounds": D["R"], "per_round": D["K"],
        "candidates": [{"label": r["label"], "round": r["round"], "name": r["name"],
                        "per_seed": [r4(v) for v in r["per_seed"]], "mean_2seed": r4(r["mean"]),
                        "new_best": r["label"] in D["kept"]} for r in D["rows"]],
        "running_best": [r4(v) for v in D["run"]], "new_bests": D["kept"], "incumbent": D["incumbent"],
        "bar_to_beat_start_4seed_mean": r4(D["start4"]), "start_seeds_0_3": [r4(v) for v in D["base4"]],
        "orchestrator_bar_seeds_0_1_mean_not_shown": r4(D["shown_dev_bar"]),
        "start_10seed_mean": r4(D["start10"]), "start_10seed_values": [r4(v) for v in D["base10"]],
        "start_10seed_range": [r4(min(D["base10"])), r4(max(D["base10"]))],
        "start_10seed_learned_skill": D["learned10"], "floor_no_hidden_share": r4(D["floor"]),
        "captions": [c for _, c, _ in states], "video_seconds": secs}
    (HERE / f"{NAME}_data.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(json.dumps({k: data[k] for k in ("running_best", "new_bests", "bar_to_beat_start_4seed_mean",
                                           "start_10seed_mean", "start_10seed_learned_skill", "video_seconds")}))


if __name__ == "__main__":
    main()
