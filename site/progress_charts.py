"""Progress charts for the article, in the 'every experiment is a dot, the step line is the best kept' style.

Two charts, both drawn from the campaigns' own files (nothing is typed in by hand except the short labels
for what each rewrite changed, which paraphrase the rewrite's own "changes" list in programs/<L>/call_<g>.json):

1. Discovery (Step 2): every trial of every search, for the readouts and scores-only groups in both worlds.
   Selection rule, fixed before drawing: no selection. All 48 searches of the two groups are embedded; the page
   opens on search 1 (s0) of the shifted-booklet world, the first search in run order.
   Dot = a trial's 2-seed mean; rings = its single seeds. Step line = the search's running best, starting at
   the starting network's 2-seed score (0 in both rule worlds), replaced only by a strictly higher mean.
   The script checks that this rule reproduces each search's recorded pick (state.json "incumbent").
   Confirmed score = the pick retrained on seeds 5001-5004, from results.json.
2. Self-rewrites (Step 3, training worlds): every meta-training search of both lineages. Dot = a search's
   confirmed score; step line = the mean confirmed score of the lineage's kept program.

Outputs: the JSON block between the PROGRESS_DATA markers in site/article.html, and two PNGs in site/assets
for the LinkedIn post. usage (project root): .venv/bin/python -I site/progress_charts.py
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
S2 = ROOT / "arch_rsi/stageH/step2/campaigns/step2"
S3 = ROOT / "arch_rsi/stageH/step3/campaigns/step3"
SITE = ROOT / "site"

WORLD_NAMES = {  # display names, as in the Stage H and Step 3a reports
    "shift": "Shifted booklet", "mirror": "Mirrored hands", "vmirror": "Hands mirrored top to bottom",
    "bookmirror": "Booklet mirrored left to right", "transpose": "Booklet transposed",
    "antitranspose": "Transposed on the other diagonal", "rot90": "Quarter turn", "rot270": "Three-quarter turn"}
TRAIN = ("mirror", "vmirror", "bookmirror", "transpose", "antitranspose", "rot90", "rot270")  # s3common.TRAIN_WORLDS

# What each kept rewrite changed: short paraphrases of the "changes" list in programs/<L>/call_<g>.json.
CHANGES = {
    ("A", "g1"): "Header 4 × 3 × 2 → 4 × 6 × 1: twice the ideas per search, plus a list of idea families not yet tried",
    ("A", "g2"): "Leads with matching the hidden action layout; generic fixes are listed as having failed everywhere",
    ("A", "g3"): "Header → 3 × 4 × 2: judge candidates on both seeds, to cut losses from seed collapse",
    ("B", "g1"): "Header 4 × 3 × 2 → 4 × 6 × 1; test hedges over the layout from round 1",
    ("B", "g2"): "Round 1 tests two sets of 4 grid symmetries; wide hedges over 16 or 24 layouts dropped",
}


def step2():
    rows = json.loads((S2 / "results.json").read_text())["rows"]
    conf = {(r["world"], r["arm"], r["search"]): r for r in rows}
    out = {}
    for world in ("shift", "mirror"):
        for arm in ("readouts", "scores"):
            searches = []
            for i in range(12):
                st = json.loads((S2 / world / arm / f"s{i}" / "state.json").read_text())
                trials, best, best_label, steps = [], 0.0, "start", []
                for t in st["trials"]:
                    res = t.get("result") or {}
                    ok = res.get("status") == "ok"
                    m = res.get("dev_exact") if ok else None
                    seeds = res.get("dev_per_seed") if ok else None
                    new = ok and m is not None and m > best
                    if new:
                        best, best_label = m, t["label"]
                    trials.append({"l": t["label"], "n": t.get("name", ""), "m": m, "s": seeds, "new": new})
                    steps.append(best)
                rec = conf[(world, arm, i)]
                pick = st.get("incumbent") or "start"
                assert pick == best_label == rec["label"], (world, arm, i, pick, best_label, rec["label"])
                searches.append({"trials": trials, "steps": steps, "pick": pick, "conf": rec["score"], "found": rec["found"]})
            out[f"{world}/{arm}"] = searches
    return out


def step3():
    out = {}
    for L in ("A", "B"):
        gens = []
        for gdir in sorted((S3 / "meta" / L).glob("gen*"), key=lambda p: int(p.name[3:])):
            gfile = gdir / "generation.json"
            if not gfile.exists():
                continue
            g = json.loads(gfile.read_text())
            worlds = g["worlds"]
            assert sorted(worlds) == sorted(TRAIN), worlds
            pts = []
            for w, s in zip(worlds, g["search_means"]):
                e = json.loads((gdir / w / "entry.json").read_text())
                assert abs(e["score"] - s) < 1e-9, (L, gdir.name, w)
                pts.append({"w": WORLD_NAMES[w], "v": s, "pick": e.get("name") or "starting architecture"})
            prog = g["program"]
            gens.append({"g": prog, "mean": g["mean"], "pts": pts, "change": CHANGES.get((L, prog), "")})
        lin = S3 / "meta" / L / "lineage.json"
        kept = {}
        if lin.exists():
            for gg in json.loads(lin.read_text())["generations"]:
                kept[gg["program"]] = gg.get("kept")
        # kept is recorded in lineage.json; while a lineage is still running, read it from the progress log
        for line in (S3 / "progress.log").read_text().splitlines():
            m = re.search(rf"lineage {L} generation \d+ \((g\d)\): mean confirmed [\d.]+ vs [\d.]+ for g\d: (kept|rejected)", line)
            if m:
                kept[m.group(1)] = m.group(2) == "kept"
        for gen in gens:
            gen["kept"] = kept.get(gen["g"])
        out[L] = gens
    return out


def step3end():
    """How each lineage ended, from its lineage.json (written when the lineage finishes)."""
    out = {}
    for L in ("A", "B"):
        lin = S3 / "meta" / L / "lineage.json"
        if lin.exists():
            j = json.loads(lin.read_text())
            out[L] = {"final": j["final"], "departure": bool(j.get("departure"))}
    return out


def write_block(data):
    art = SITE / "article.html"
    t = art.read_text(encoding="utf-8")
    block = ('<script id="progress-data" type="application/json">' + json.dumps(data, separators=(",", ":")) + "</script>")
    new, n = re.subn(r'<script id="progress-data" type="application/json">.*?</script>', lambda _: block, t, flags=re.S)
    assert n == 1, "marker block not found in article.html"
    art.write_text(new, encoding="utf-8")
    return len(block)


def pngs(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, INK, INK2, MUTED, GRID = "#FFFFFF", "#1A1B1E", "#3B3E45", "#6A6E76", "#E5E6EA"
    ACC, OK, WALL, GRAY = "#5560CC", "#26814F", "#C93D37", "#9AA0AA"
    LIN = {"A": "#5560CC", "B": "#0B8A65"}
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial"], "font.size": 13, "text.color": INK,
                         "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})

    # 1. discovery: shifted booklet, search 1, readouts vs scores only
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4), dpi=110, facecolor=BG, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.14, wspace=0.08)
    fig.text(0.07, 0.94, "Same hidden rule, same budget: one search shown its own runs, one shown scores", fontsize=22, weight="bold")
    fig.text(0.07, 0.885, "Shifted-booklet world, search 1 of 12 in each group. Every dot is a design the AI scientist trained "
             "(mean of 2 seeds); the line is its best so far.", fontsize=13.5, color=INK2)
    for ax, arm, title in ((axes[0], "readouts", "Shown readouts of its own runs"), (axes[1], "scores", "Shown scores only")):
        s = data["step2"]["shift/" + arm][0]
        ax.set_facecolor(BG)
        xs = list(range(1, 13))
        for x, t in zip(xs, s["trials"]):
            if t["s"]:
                ax.scatter([x, x], [v * 100 for v in t["s"]], s=46, facecolors="none", edgecolors=(ACC if t["new"] else GRAY), linewidths=1.2, alpha=.7, zorder=2)
            if t["m"] is not None:
                ax.scatter([x], [t["m"] * 100], s=90 if t["new"] else 50, color=(ACC if t["new"] else MUTED), zorder=3)
        ax.step([0] + xs, [0] + [v * 100 for v in s["steps"]], where="post", color=ACC, linewidth=2.4, zorder=1)
        ax.axhline(38.3, color=OK, linestyle=(0, (4, 4)), linewidth=1.2)
        ax.text(12.4, 39.5, "working fix, 38.3%", color=OK, fontsize=11, ha="right")
        for r in (3.5, 6.5, 9.5):
            ax.axvline(r, color=GRID, linewidth=1)
        ax.set_xticks(xs, [t["l"] for t in s["trials"]], fontsize=10.5)
        ax.set_xlim(0.3, 12.7); ax.set_ylim(-3, 103)
        ax.set_yticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=30)
        verdict = f"pick: {s['pick']} · confirmed on 4 new seeds: {s['conf'] * 100:.1f}%"
        ax.text(0, 1.03, verdict, transform=ax.transAxes, fontsize=12.5, color=(ACC if s["found"] else WALL))
    axes[0].set_ylabel("designs built exactly, 512 dev designs")
    fig.text(0.07, 0.045, "Rings: single seeds. Rounds of 3 candidates are separated by vertical lines. Source: Step 2 campaign files "
             "(state.json per search, results.json for confirmed scores).", fontsize=11, color=MUTED)
    fig.savefig(SITE / "assets" / "progress-discovery.png", facecolor=BG)
    plt.close(fig)

    # 2. self-rewrites: both lineages, training worlds
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4), dpi=110, facecolor=BG, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.24, wspace=0.08)
    fig.text(0.07, 0.94, "The AI scientist rewriting its own research procedure", fontsize=22, weight="bold")
    fig.text(0.07, 0.885, "Training worlds only. Each dot is one search's confirmed score (7 worlds per generation); "
             "the line is the kept program's mean. Held-out results are separate.", fontsize=13.5, color=INK2)
    for ax, L in zip(axes, ("A", "B")):
        gens = data["step3"][L]
        ax.set_facecolor(BG)
        for gi, gen in enumerate(gens):
            for k, p in enumerate(gen["pts"]):
                ax.scatter([gi + (k - 3) * 0.06], [p["v"] * 100], s=48, color=LIN[L], alpha=.45, zorder=2)
        means = [g["mean"] * 100 for g in gens]
        ax.step(range(len(gens)), means, where="post", color=LIN[L], linewidth=2.4, zorder=3)
        ax.scatter(range(len(gens)), means, s=110, color=LIN[L], zorder=4)
        for gi, m in enumerate(means):
            ax.text(gi + 0.08, m + 3.5, f"{m:.1f}%", fontsize=12.5, weight="bold", color=INK)
        n = len(gens)
        ax.set_xticks(range(4), ["g0", "g1", "g2", "g3"]); ax.set_xlim(-0.4, 3.6); ax.set_ylim(-3, 103)
        end = data.get("step3end", {}).get(L)
        for gi in range(n, 4):
            ax.text(gi, 6, "not written" if end else "pending", fontsize=12, color=MUTED, ha="center")
        ax.set_yticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"Lineage {L}", loc="left", fontsize=15, weight="bold", pad=12)
        lines = [f"{g['g']}: {g['change']}" for g in gens if g["change"]]
        ax.text(0, -0.13, "\n".join(_wrap(l, 74) for l in lines), transform=ax.transAxes, fontsize=10.5, color=INK2, va="top", linespacing=1.4)
    axes[0].set_ylabel("confirmed score, training worlds")
    fig.savefig(SITE / "assets" / "progress-rewrites.png", facecolor=BG)
    plt.close(fig)


def _wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = "    " + w
        else:
            cur = (cur + " " + w).strip() if cur else w
    lines.append(cur)
    return "\n".join(lines)


if __name__ == "__main__":
    data = {"step2": step2(), "step3": step3(), "step3end": step3end()}
    n2 = sum(len(v) for v in data["step2"].values())
    print("step2 searches:", n2, "| step3 generations:", {L: [g["g"] for g in v] for L, v in data["step3"].items()})
    print("data block bytes:", write_block(data))
    pngs(data)
    print("wrote", SITE / "assets" / "progress-discovery.png", "and", SITE / "assets" / "progress-rewrites.png")
