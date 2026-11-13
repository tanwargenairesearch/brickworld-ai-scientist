"""Stage 2b video, scenes 2-4, and the article's Figure 7. Visualization only: nothing here selects, scores or tests.

Scene 2: every search in every arm building up candidate by candidate (2-seed dev means, running bests by the frozen
         incumbent rule, starting from the start's seeds 0-1 mean, the bar every search had to beat).
Scene 3: each search's pick on 8 fresh seeds (primary scores: a pick the rule review judged non-compliant scores as the
         start), the comparators on 12 fresh seeds, and the pre-registered gate (g0 - no feedback) with P2.
Scene 4: the limits card.

Every number is read from the campaign's own files and cross-checked against results.json (written by analyze_2b.py
after review_2b.py). The script refuses to run if results.json is missing or its primary family isn't final, or if W's
final split has been opened. Dev designs only.

usage (project root):
  .venv/bin/python -I arch_rsi/viz/stage2b/search_chart_2b.py                      # Figure 7 still + data json
  .venv/bin/python -I arch_rsi/viz/stage2b/search_chart_2b.py --video              # + search_2b.mp4 (scenes 2-4)
  .venv/bin/python -I arch_rsi/viz/stage2b/search_chart_2b.py --campaign arch_rsi/campaigns/stage2b_dry --out /tmp/x
"""
import argparse
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
ROOT = A.parent
TITLES = {"g0": "AI scientist sees its scores", "nofb": "Scores hidden from the AI scientist",
          "netonly": "AI scientist, network only (recipe fixed)", "reconly": "AI scientist, recipe only (network fixed)",
          "random": "Random draws (no AI scientist)"}
SHORT = {"g0": "sees\nscores", "nofb": "scores\nhidden", "netonly": "network\nonly", "reconly": "recipe\nonly",
         "random": "random", "refA": "retest's\nbest", "start_g": "start"}
COMPS = ("refA", "start_g")

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, OTHER = "#2a78d6", "#a3a29b"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial"], "font.size": 13, "text.color": INK,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "svg.fonttype": "none"})
W_PX, H_PX = 1920, 1080


def dev(d: Path) -> float:
    return json.loads((d / "dev.json").read_text())["exact"]


def load(cdir: Path) -> dict:
    if (A / "stage2" / "FINAL_OPENED_W").exists():
        raise SystemExit("FINAL_OPENED_W exists: this script reads dev results only")
    res = json.loads((cdir / "results.json").read_text())
    if res["primary_family"]["status"] != "final":
        raise SystemExit("results.json is not final (run review_2b.py, then analyze_2b.py)")
    man = json.loads((cdir / "manifest.json").read_text())
    arms = man["config"]["arms"]
    summ = json.loads((cdir / "confirm" / "summary.json").read_text())
    bar = json.loads((cdir / "baseline" / "summary.json").read_text())["w"]["shown_dev"]
    data = {}
    for arm in arms:
        rows_by_search = []
        for entry in res["per_arm"][arm]["searches"]:
            i = entry["search"]
            sdir = cdir / "W" / arm / f"s{i}"
            st = json.loads((sdir / "state.json").read_text())
            rows = []
            for t in st["trials"]:
                r = t["result"]
                ok = r["status"] == "ok"
                if ok:
                    per = [dev(sdir / "trials" / t["label"] / f"seed{s}") for s in (0, 1)]
                    assert abs(sum(per) / 2 - r["dev_exact"]) < 6e-5, (arm, i, t["label"])
                rows.append({"label": t["label"], "round": t["round"], "name": t.get("name", ""),
                             "status": r["status"], "mean": r["dev_exact"] if ok else None})
            best, inc, kept, run = bar, None, [], []
            for r in rows:  # the frozen incumbent rule
                if r["status"] == "ok" and r["mean"] > best:
                    best, inc = r["mean"], r["label"]
                    kept.append(r["label"])
                run.append(best)
            assert inc == st["incumbent"], (arm, i, inc, st["incumbent"])
            se = next(e for e in summ if e["kind"] == "search" and e["arm"] == arm and e["search"] == i)
            conf = [dev(cdir / "confirm" / se["sha"] / f"seed{s}") if se["status"][str(s)] == "ok" else 0.0
                    for s in se["seeds"]]
            assert all(abs(a - b) < 6e-5 for a, b in zip(conf, entry["per_seed"])), (arm, i)
            rows_by_search.append({"search": i, "rows": rows, "kept": kept, "run": run, "pick": entry["pick"],
                                   "pick_name": entry["name"], "verdict": entry["verdict"],
                                   "score": entry["score"], "primary": entry["primary_score"]})
        data[arm] = rows_by_search
        assert abs(np.mean([s["primary"] for s in rows_by_search]) - res["per_arm"][arm]["primary_mean"]) < 6e-5
    comps = {c: res["per_arm"][c]["per_seed"] for c in COMPS}
    for c in COMPS:
        assert abs(np.mean(comps[c]) - res["per_arm"][c]["mean"]) < 6e-5
    ncand = man["config"]["R"] * man["config"]["K"]
    n_conf = {len(e["seeds"]) for e in summ if e["kind"] == "search"}
    assert len(n_conf) == 1
    return {"res": res, "arms": arms, "data": data, "bar": bar, "comps": comps, "ncand": ncand, "K": man["config"]["K"],
            "n_conf": n_conf.pop(), "n_comp": len(comps["start_g"]), "n_search": len(data[arms[0]]),
            "campaign": str(cdir.relative_to(ROOT))}


def pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def pval(p: float) -> str:
    return "below 0.001" if p < 0.001 else f"{p:.3f}"


def outcome_lines(D: dict) -> list:
    """The gate (P1) and P2, recomputed from unrounded primary scores; wording follows the pre-written endings."""
    res, data = D["res"], D["data"]
    m = {a: float(np.mean([s["primary"] for s in data[a]])) for a in ("g0", "nofb")}
    m["refA"] = float(np.mean(D["comps"]["refA"]))
    c = res["primary_family"]["contrasts"]
    p1, p2 = c["g0 - nofb"], c["g0 - refA"]
    d1, d2 = m["g0"] - m["nofb"], m["g0"] - m["refA"]
    assert abs(d1 - p1["diff"]) < 6e-5 and abs(d2 - p2["diff"]) < 6e-5
    lo, hi = p1["ci95_boot"]
    passed = res["primary_family"]["gate_passed"]
    l1 = (f"Seeing its scores vs hiding them: {pts(d1)} points on fresh seeds (95% interval {pts(lo)} to {pts(hi)}; "
          f"Holm-adjusted p {pval(p1['p_holm'])}). " + ("The gate passed." if passed else "The gate failed again."))
    l2 = f"Against the best network from the retest, trained the same way: {pts(d2)} points (Holm-adjusted p {pval(p2['p_holm'])})."
    return [l1, l2]


def style(ax, ncand, k, n_conf, show_y, xconf):
    ax.set_facecolor(SURFACE)
    for y in (0, 20, 40, 60, 80, 100):
        ax.axhline(y, color=GRID, lw=1, zorder=0)
    for rb in np.arange(k + 0.5, ncand, k):
        ax.axvline(rb, color=GRID, lw=1, zorder=0)
    ax.axvline(ncand + 0.9, color=AXIS, lw=1, zorder=0)
    ax.set_xlim(-0.1, xconf + 1.9)
    ax.set_ylim(-4, 104)
    ax.set_xticks([(k + 1) / 2 + k * r for r in range(ncand // k)] + [xconf])
    ax.set_xticklabels([f"round {r + 1}" for r in range(ncand // k)] + [f"pick,\n{n_conf} seeds"], fontsize=11, color=INK2)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 20, 40, 60, 80, 100)], fontsize=11)
    ax.tick_params(length=0, pad=6, labelleft=show_y)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def search_panel(ax, D, arm, upto, confirm, show_y):
    ncand, bar, n = D["ncand"], D["bar"], D["n_search"]
    xconf = ncand + 1.9
    style(ax, ncand, D["K"], D["n_conf"], show_y, xconf)
    ax.set_title(TITLES[arm], fontsize=14, fontweight="bold", color=INK, loc="left", pad=8)
    ax.axhline(100 * float(np.mean(D["comps"]["start_g"])), color=INK2, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.plot([0.35], [100 * bar], "o", ms=7, color=INK2, mec=SURFACE, mew=1.5, zorder=4)
    runs = []
    for s in D["data"][arm]:
        off = (s["search"] - (n + 1) / 2) * (0.5 / n)
        sx, sy = [0.35], [100 * bar]
        for x, b in zip(range(1, upto + 1), s["run"][:upto]):
            sx += [x + off, x + off]
            sy += [sy[-1], 100 * b]
        ax.plot(sx, sy, color=BLUE, lw=0.8, alpha=0.22, zorder=2)
        runs.append(s["run"] + [s["run"][-1]] * (ncand - len(s["run"])))
        for x, r in zip(range(1, upto + 1), s["rows"][:upto]):
            if r["status"] == "timeout":
                ax.plot(x + off, 0, "x", ms=6, mew=1.4, color=INK2, zorder=4)
            elif r["status"] != "ok":
                ax.plot(x + off, 0, "o", ms=5, mfc="none", mec=INK2, mew=1.0, zorder=4)
            else:
                new = r["label"] in s["kept"]
                ax.plot(x + off, 100 * r["mean"], "o", ms=6 if new else 4.5, color=BLUE if new else OTHER,
                        mec=SURFACE, mew=0.6, zorder=5 if new else 3)
    if upto:
        mean_run = 100 * np.mean([r[:upto] for r in runs], axis=0)
        mx, my = [0.35], [100 * bar]
        for x, b in zip(range(1, upto + 1), mean_run):
            mx += [x, x]
            my += [my[-1], b]
        ax.plot(mx, my, color=BLUE, lw=2.6, solid_joinstyle="round", zorder=6)
        ax.annotate(f"{my[-1]:.0f}%", (mx[-1], my[-1]), xytext=(3, 7), textcoords="offset points", ha="left",
                    va="bottom", fontsize=11, color=BLUE, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    if confirm:
        vals = [s["primary"] for s in D["data"][arm]]
        for s in D["data"][arm]:
            off = (s["search"] - (n + 1) / 2) * (0.9 / n)
            hollow = s["verdict"] == "violation"
            ax.plot(xconf + off, 100 * s["primary"], "D", ms=5, mfc="none" if hollow else INK2, mec=INK2 if hollow else SURFACE,
                    mew=0.8, zorder=5)
        mu = 100 * float(np.mean(vals))
        ax.plot([xconf - 0.6, xconf + 0.6], [mu] * 2, color=INK, lw=3, zorder=6)
        ax.text(xconf + 0.7, mu, f"{mu:.1f}%", ha="left", va="center", fontsize=12, color=INK, fontweight="bold", zorder=7)


def answer_panel(ax, D, show):
    order = list(D["arms"]) + list(COMPS)
    ax.set_facecolor(SURFACE)
    for y in (0, 20, 40, 60, 80, 100):
        ax.axhline(y, color=GRID, lw=1, zorder=0)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-4, 104)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT[a] for a in order], fontsize=10.5, color=INK2)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([])
    ax.tick_params(length=0, pad=6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.axvline(len(D["arms"]) - 0.5, color=AXIS, lw=1, zorder=0)
    ax.set_title(f"The answer: each pick on {D['n_conf']} fresh seeds", fontsize=14, fontweight="bold", color=INK, loc="left", pad=8)
    if not show:
        return
    rng = np.random.default_rng(0)  # jitter only
    for j, a in enumerate(order):
        vals = [s["primary"] for s in D["data"][a]] if a in D["data"] else D["comps"][a]
        xs = j + rng.uniform(-0.2, 0.2, len(vals))
        hi = a in ("g0", "nofb")
        if a in D["data"]:
            ax.plot(xs, 100 * np.array(vals), "D", ms=5, color=BLUE if hi else INK2, mec=SURFACE, mew=0.6, alpha=0.8, zorder=3)
        else:
            ax.plot(xs, 100 * np.array(vals), "o", ms=5, mfc="none", mec=INK2, mew=1.0, alpha=0.7, zorder=3)
        mu = 100 * float(np.mean(vals))
        ax.plot([j - 0.32, j + 0.32], [mu] * 2, color=BLUE if hi else INK, lw=3.2, zorder=5)
        ax.text(j, mu + 3, f"{mu:.0f}%", ha="center", va="bottom", fontsize=11.5, fontweight="bold",
                color=BLUE if hi else INK, zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85))


def legend(fig, D, y):
    start = 100 * float(np.mean(D["comps"]["start_g"]))
    handles = [Line2D([], [], color=BLUE, marker="o", ms=7, lw=0, mec=SURFACE, label="new best in its search"),
               Line2D([], [], color=OTHER, marker="o", ms=6, lw=0, mec=SURFACE, label="not a new best"),
               Line2D([], [], color=INK2, marker="x", ms=6, lw=0, mew=1.4, label="timed out"),
               Line2D([], [], color=INK2, marker="o", ms=5, lw=0, mfc="none", label="rejected before training"),
               Line2D([], [], color=BLUE, lw=2.6, label=f"mean running best of {D['n_search']} searches"),
               Line2D([], [], color=INK2, marker="D", ms=5, lw=0, mec=SURFACE, label=f"a search's pick, {D['n_conf']} fresh seeds"),
               Line2D([], [], color=INK2, lw=1.2, ls=(0, (5, 3)), label=f"start, {D['n_comp']} fresh seeds: {start:.1f}%")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, y), ncol=7, frameon=False, fontsize=11,
                     handletextpad=0.4, columnspacing=1.1)
    for t in leg.get_texts():
        t.set_color(INK2)


def overview_fig(D, upto=None, confirm=True, answer=True, caption=None):
    upto = D["ncand"] if upto is None else upto
    fig, axes = plt.subplots(2, 3, figsize=(W_PX / 100, H_PX / 100), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    slots = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    for j, (ax, arm) in enumerate(zip(slots, D["arms"])):
        search_panel(ax, D, arm, upto, confirm, show_y=(j in (0, 3)))
    answer_panel(axes[1, 2], D, answer)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("built exactly, 512 dev designs", fontsize=12, color=INK2, labelpad=8)
    axes[0, 0].text(0.15, 100 * D["bar"] + 3, f"bar to beat: {100 * D['bar']:.0f}%", ha="left", va="bottom",
                    fontsize=10, color=INK2, zorder=8, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.9))
    total = sum(len(s["rows"]) for a in D["arms"] for s in D["data"][a])
    fig.text(0.045, 0.958, f"Stage 2b: {D['n_search'] * len(D['arms'])} searches, {total} candidates, in a harder world",
             fontsize=22, fontweight="bold", color=INK, ha="left")
    fig.text(0.045, 0.925, "Each dot is one candidate (network and training recipe), trained on 2 seeds at equal compute. "
             "10×10×6 brick-assembly world, 512 dev designs; held-out designs locked.", fontsize=13, color=INK2, ha="left")
    legend(fig, D, 0.912)
    if answer:
        for k, line in enumerate(outcome_lines(D)):
            fig.text(0.045, 0.045 - 0.028 * k, line, fontsize=13.5, color=INK, ha="left", fontweight="bold" if k == 0 else None)
    elif caption:
        fig.text(0.045, 0.045, caption, fontsize=13.5, color=INK, ha="left", fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.12, wspace=0.10, hspace=0.38)
    return fig


def limits_fig(D):
    passed = D["res"]["primary_family"]["gate_passed"]
    fig = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.08, 0.80, "What this does and doesn't show", fontsize=34, fontweight="bold", color=INK, ha="left")
    lines = ["One toy world, one AI scientist (an LLM), dev designs only. The held-out designs are still locked.",
             "The AI scientist's research program and weights never changed. This tests a search, not a self-improving system.",
             ("Next: the AI scientist rewrites its own research program, against controls (Stage 2c)." if passed else
              "The gate failed, so the self-revision stage doesn't run.")]
    for k, t in enumerate(lines):
        fig.text(0.08, 0.64 - 0.11 * k, t, fontsize=22, color=INK2, ha="left")
    return fig


def to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def animate(D, path, fps=12):
    ncand = D["ncand"]
    states = [("o", 0, False, False, "Every search starts from the start's score on seeds 0-1: the bar to beat.", 3.0)]
    for k in range(1, ncand + 1):
        states.append(("o", k, False, False, f"Round {(k - 1) // D['K'] + 1}: candidate {k} of {ncand} in every search.", 1.0))
    states.append(("o", ncand, True, False, f"Each search's pick is retrained on {D['n_conf']} fresh seeds.", 4.0))
    states.append(("o", ncand, True, True, None, 12.0))
    states.append(("limits", None, None, None, None, 8.0))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W_PX}x{H_PX}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "18", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    n = 0
    for kind, upto, conf, ans, cap, secs in states:
        fig = limits_fig(D) if kind == "limits" else overview_fig(D, upto, conf, ans, cap)
        buf = to_rgb(fig)
        plt.close(fig)
        assert buf.shape == (H_PX, W_PX, 3), buf.shape
        for _ in range(int(round(secs * fps))):
            ff.stdin.write(buf.tobytes())
            n += 1
    ff.stdin.close()
    assert ff.wait() == 0
    return n / fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="arch_rsi/campaigns/stage2b")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    D = load((ROOT / args.campaign).resolve())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig = overview_fig(D)
    for ext in ("png", "svg"):
        fig.savefig(out / f"search_2b.{ext}", facecolor=SURFACE, dpi=100)
    plt.close(fig)
    summary = {"campaign": D["campaign"], "bar_to_beat": round(D["bar"], 4), "outcome": outcome_lines(D),
               "gate_passed": D["res"]["primary_family"]["gate_passed"],
               "arms": {a: {"primary_mean": round(float(np.mean([s["primary"] for s in D["data"][a]])), 4),
                            "mean_running_best_by_candidate": [round(float(v), 4) for v in np.mean(
                                [s["run"] + [s["run"][-1]] * (D["ncand"] - len(s["run"])) for s in D["data"][a]], 0)],
                            "timeouts": sum(r["status"] == "timeout" for s in D["data"][a] for r in s["rows"]),
                            "rejected": sum(r["status"] not in ("ok", "timeout") for s in D["data"][a] for r in s["rows"]),
                            "violations": sum(s["verdict"] == "violation" for s in D["data"][a])}
                        for a in D["arms"]},
               "comparators": {c: round(float(np.mean(D["comps"][c])), 4) for c in COMPS}}
    if args.video:
        summary["video_seconds"] = round(animate(D, out / "search_2b.mp4"), 1)
    (out / "search_2b_data.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
