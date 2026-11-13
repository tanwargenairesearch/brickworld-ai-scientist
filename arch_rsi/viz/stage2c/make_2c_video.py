"""Stage 2c video (PROTOCOL_2C.md, "Video build"): 5 scenes, 1920x1080, 12 fps, captions on screen, no audio.
Visualization only: nothing here selects, scores or tests. Written 10 November 2026, before any Stage 2c result.

Scene 1 (8 s):  the three-loop system diagram, Loop 3 highlighted, its status chip updated.
Scene 2 (25 s): lineage A's program, generation by generation: each rewrite's changes and evidence (quoted from the
                revision files, cut with "…" at a fixed length, never reworded), stamped kept or rejected; beside it an
                autoresearch-style chart, one dot per generation (mean confirmed score on W_s), kept rewrites labelled,
                the current program's score as a step line. Lineage A always, whatever its result.
Scene 3 (25 s): one search under the last kept program, in the single-frame view (viz/stage2b/search_build.py): the
                final-arm searches whose pick isn't the start and was judged compliant, sorted by primary score, the
                one at 0-based position floor(m/2); the same rule as Stage 2b's video. If none is eligible, a card says so.
Scene 4 (15 s): every arm on W with each search's pick, the arm mean and a 95% bootstrap interval of the mean
                (descriptive); P1-P4 with intervals and Holm-adjusted p; the caption is the ending ENDINGS_2C.md
                assigns to the tier in results.json, with its numbers filled in.
Scene 5 (6 s):  the limits.

Checks (the script stops on any mismatch): each program shown is byte-identical to the saved program (its hash in
lineage.json, generation.json and the revision file); each generation's mean is recomputed from its confirmation runs
and must match the meta-training logs; each arm's searches are recomputed from their confirmation runs and must
match results.json, as must the arm means; scene 3's build must match its run's per-design result (search_build.py).
It refuses to run unless results.json's primary family is final. It never reads W's final split.

usage (project root):
  nice -n 19 .venv/bin/python -I arch_rsi/viz/stage2c/make_2c_video.py            # stills + video_2c_data.json
  nice -n 19 .venv/bin/python -I arch_rsi/viz/stage2c/make_2c_video.py --video    # + stage2c.mp4
  ... --campaign DIR --out DIR --skip-build   (scene 3 as a card: for tests on campaigns without checkpoints)
"""
import argparse
import importlib.util
import json
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parents[1]
ROOT = A.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(A / "stage2" / "2c"))

import matplotlib.pyplot as plt  # noqa: E402

import prog2c as P  # noqa: E402
from arch_rsi.sandbox import sha  # noqa: E402
from arch_rsi.viz.stages import vidkit as vk  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SYS = _load("make_system", A / "viz" / "stages" / "system" / "make_system.py")
ENDINGS = A / "stage2" / "2c" / "ENDINGS_2C.md"
CAMP_2B = A / "campaigns" / "stage2b"
ORDER = ["g0", "g1", "final", "facts", "shuffled"]
LABEL = {"g0": "original\nprogram (g0)", "g1": "first\nrewrite (g1)", "final": "last kept\nprogram",
         "facts": "facts-only\nrewrite", "shuffled": "rewrite from\nshuffled scores"}
PNAME = {"P1": "first rewrite vs original", "P2": "vs facts-only rewrite", "P3": "vs shuffled-score rewrite",
         "P4": "last kept vs first rewrite"}
SECONDS = {1: 8, 2: 25, 3: 25, 4: 15, 5: 6}
CUT = 220  # characters of a quoted change or evidence shown before "…"


def mean(x):
    return sum(x) / len(x)


def run_score(f: Path) -> float:
    r = json.loads(f.read_text())
    return r["dev_exact"] if r["status"] == "ok" else 0.0


# ---------------------------------------------------------------- data and checks
def load(cdir: Path) -> dict:
    res = json.loads((cdir / "results.json").read_text())
    if res["primary_family"]["status"] != "final":
        raise SystemExit("results.json is not final (run review_2c.py, then analyze_2c.py)")
    summ = json.loads((cdir / "confirm" / "summary.json").read_text())
    lins = {}
    for lj_file in sorted((cdir / "meta").glob("*/lineage.json")):
        lj = json.loads(lj_file.read_text())
        L = lj["lineage"]
        gens = []
        for row in lj["generations"]:
            g, name = row["generation"], row["program"]
            gj = json.loads((cdir / "meta" / L / f"gen{g}" / "generation.json").read_text())
            txt = (cdir / "programs" / L / f"{name}.md").read_text()
            assert sha(txt) == row["program_sha"] == gj["program_sha"], (L, name, "program differs from the saved one")
            means = [mean([run_score(cdir / "confirm" / "ws" / e["sha"] / f"seed{s}" / "result.json") for s in e["seeds"]])
                     for e in gj["searches"]]
            m = mean(means)
            assert abs(m - gj["mean"]) < 1e-9 and abs(m - row["mean"]) < 1e-9, (L, g, "generation mean mismatch")
            gen = {"g": g, "name": name, "mean": m, "kept": row.get("kept"), "parent": row.get("parent"),
                   "parent_mean": row.get("parent_mean"), "header": P.parse_program(txt)[0], "changes": [],
                   "n_searches": len(gj["searches"]), "n_seeds": len(gj["searches"][0]["seeds"]), "fallback": False}
            if g > 0:
                rv = json.loads((cdir / "programs" / L / f"revision_{name}.json").read_text())
                assert rv["sha"] == sha(txt), (L, name, "revision file differs from the saved program")
                gen["changes"] = rv["result"]["changes"]
                gen["fallback"] = bool(rv["result"].get("header_fallback"))
            gens.append(gen)
        lins[L] = {"gens": gens, "final": lj["final"]}
    arms = {}
    for arm in ORDER:
        d = res["per_arm"].get(arm)
        if not d:
            continue
        for s in d["searches"]:
            e = next(e for e in summ if e["arm"] == arm and e.get("lineage") == s["lineage"] and e["search"] == s["search"])
            base = (CAMP_2B / "confirm" / e["sha"]) if e.get("reused_from") == "stage2b" else (cdir / "confirm" / "w" / e["sha"])
            x = mean([run_score(base / f"seed{q}" / "result.json") for q in e["seeds"]])
            assert abs(x - s["score"]) < 6e-5, (arm, s["lineage"], s["search"], "search score mismatch")
            s["path"] = e.get("path")
        prim = [s["primary_score"] for s in d["searches"]]
        assert abs(mean(prim) - d["primary_mean"]) < 6e-5, (arm, "arm mean mismatch")
        arms[arm] = d
    return {"res": res, "lins": lins, "arms": arms, "summary": summ, "cdir": cdir}


# ---------------------------------------------------------------- endings
def pv(p) -> str:
    return "below 0.001" if p < 0.001 else f"{p:.3f}"


def ending(res: dict) -> tuple:
    """The ending ENDINGS_2C.md assigns to this result, with its numbers filled in: (key, video line, article text)."""
    t, c = res["tiers"], res["primary_family"]["contrasts"]
    p1 = c["P1"]
    if t["words_allowed"] == "recursive":
        key = "recursive"
    elif t["words_allowed"] == "Level 2":
        key = "level2"
    elif p1["tested"] and p1["diff"] > 0 and p1["p_holm"] < 0.05:
        key = "transfer_only"
    else:
        key = "none"
    vals = {}
    for k in ("P1", "P2", "P3", "P4"):
        ck = c[k]
        vals[k] = f"{100 * ck['diff']:+.1f}" if ck["tested"] else "not tested"
        vals[k + "p"] = pv(ck["p_holm"]) if ck["tested"] else "not tested"
    vals["P4txt"] = (f"{vals['P4']} points, Holm-adjusted p {vals['P4p']}" if c["P4"]["tested"]
                     else "not tested: no lineage kept a rewrite")
    sec = re.search(rf"^## {key}\nvideo: (.*)\narticle: (.*)$", ENDINGS.read_text(), re.M)
    video, article = sec.group(1), sec.group(2)
    fill = lambda s: re.sub(r"\{(\w+)\}", lambda m: vals[m.group(1)], s)  # noqa: E731
    out = fill(video), fill(article)
    assert "{" not in out[0] + out[1]
    return key, out[0], out[1]


# ---------------------------------------------------------------- scenes
def scene1(D):
    gens = next(iter(D["lins"].values()))["gens"]
    status = f"STAGE 2c · RAN: {len(D['lins'])} LINEAGES × {len(gens) - 1} REWRITES"
    return SYS.render("STAGES 1–2b · DONE", status, True, 2)


def cut(s: str, n: int = CUT) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def versus(a: float, b: float) -> str:
    """Two scores as percentages: one decimal, or two when one decimal would hide a real difference."""
    for d in (1, 2, 3):
        sa, sb = f"{100 * a:.{d}f}%", f"{100 * b:.{d}f}%"
        if sa != sb or a == b:
            break
    return f"{sa} vs {sb}" + (" (equal)" if a == b else "")


def header_change(old: dict, new: dict) -> str:
    if old == new:
        return "Search settings unchanged."
    f = lambda h: (f"{h['rounds']} rounds × {h['candidates_per_round']} candidates, {plural(h['seeds_per_candidate'], 'seed')}, "  # noqa: E731
                   f"retest {h['retest_top']}; split network {h['split']['network']} / recipe {h['split']['recipe']} / "
                   f"both {h['split']['both']}")
    return f"Search settings: {f(old)}  →  {f(new)}"


def scene2_fig(D, L, upto):
    gens = D["lins"][L]["gens"]
    g0 = gens[0]
    fig = vk.new_fig()
    vk.header(fig, f"The AI scientist rewrites its own research program (lineage {L})",
              f"Each generation runs {g0['n_searches']} searches under the program in the smaller 8×8×6 world; each "
              f"search's pick is confirmed on {g0['n_seeds']} fresh seeds. A rewrite is kept only if it scores higher.",
              stage="Stage 2c · rewriting on a smaller world")
    # left: the history so far, the current generation expanded
    y = 0.74
    cur = {g["name"]: g for g in gens}
    for g in gens[:upto + 1]:
        if g["g"] == 0:
            line = f"g0 · the starting program · {100 * g['mean']:.1f}%"
            col = vk.GREY
        else:
            stamp = "KEPT" if g["kept"] else "REJECTED"
            line = f"{g['name']} · written from {g['parent']} · {stamp} · {versus(g['mean'], g['parent_mean'])}"
            col = vk.BLUE if g["kept"] else vk.MUTED
        fig.text(0.05, y, line, fontsize=17, fontweight="bold", color=col, ha="left", va="top")
        y -= 0.04
        if g["g"] == upto and g["g"] > 0:
            par = cur[g["parent"]]
            hc = header_change(par["header"], g["header"]) + ("  (the parent's settings, after 3 invalid headers)" if g["fallback"] else "")
            for ln in textwrap.wrap(hc, 92):
                fig.text(0.065, y, ln, fontsize=13.5, color=vk.INK2, ha="left", va="top")
                y -= 0.028
            for ch in g["changes"][:3]:
                for k, ln in enumerate(textwrap.wrap("• " + cut(ch.get("change", "")), 88)):
                    fig.text(0.065, y, ln, fontsize=14, color=vk.INK, ha="left", va="top")
                    y -= 0.028
                for ln in textwrap.wrap("evidence: " + cut(ch.get("evidence", "")), 92):
                    fig.text(0.08, y, ln, fontsize=12.5, color=vk.INK2, ha="left", va="top", style="italic")
                    y -= 0.025
                y -= 0.008
            if len(g["changes"]) > 3:
                fig.text(0.065, y, f"… and {len(g['changes']) - 3} more change(s), in the program files", fontsize=12.5,
                         color=vk.MUTED, ha="left", va="top")
                y -= 0.03
            y -= 0.012
    # right: one dot per generation, the current program's score as a step line
    ax = fig.add_axes([0.58, 0.14, 0.38, 0.58])
    vals = [100 * g["mean"] for g in gens]
    lo = max(0, int(np.floor((min(vals) - 8) / 10)) * 10)
    hi = min(100, int(np.ceil((max(vals) + 8) / 10)) * 10)
    step = 10 if hi - lo <= 60 else 20
    vk.style_axes(ax, ylim=(lo, hi), yticks=tuple(range(lo, hi + 1, step)))
    ax.set_xlim(-0.5, len(gens) - 0.5)
    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels([g["name"] if g["g"] <= upto else "" for g in gens], fontsize=13, color=vk.INK2)
    ax.set_xlabel("generation", fontsize=13, color=vk.INK2)
    ax.set_ylabel("mean confirmed success, 8×8×6 world", fontsize=13, color=vk.INK2)
    shown = gens[:upto + 1]
    best, xs, ys = gens[0]["mean"], [0], [100 * gens[0]["mean"]]
    for g in shown[1:]:
        xs.append(g["g"])
        ys.append(100 * best)
        if g["kept"]:
            best = g["mean"]
            xs.append(g["g"])
            ys.append(100 * best)
    ax.plot(xs, ys, color=vk.BLUE, lw=2.6, zorder=2)
    for g in shown:
        v = 100 * g["mean"]
        if g["g"] == 0:
            ax.plot(0, v, "o", ms=12, color=vk.INK2, mec=vk.SURFACE, mew=2, zorder=4)
            ax.annotate("start: g0", (0, v), xytext=(8, -18), textcoords="offset points", fontsize=12, color=vk.INK2)
        elif g["kept"]:
            ax.plot(g["g"], v, "o", ms=13, color=vk.BLUE, mec=vk.SURFACE, mew=2, zorder=5)
            first = g["changes"][0].get("change", "") if g["changes"] else ""
            ax.annotate(f"{g['name']}: {cut(first, 34)}", (g["g"], v), xytext=(8, 8), textcoords="offset points",
                        fontsize=12, color=vk.BLUE, rotation=28, rotation_mode="anchor")
        else:
            ax.plot(g["g"], v, "o", ms=10, color=vk.OTHER, mec=vk.SURFACE, mew=1.5, zorder=3)
            ax.annotate(f"{g['name']} (rejected)", (g["g"], v), xytext=(8, -16), textcoords="offset points", fontsize=11,
                        color=vk.MUTED)
    fig.text(0.58, 0.765, "kept rewrites in blue, rejected in grey; the line is the current program's score",
             fontsize=13, color=vk.INK2, ha="left")
    vk.caption(fig, "Changes are quoted from the rewrite's own record, cut with … where long. Tested later on the "
               "larger 10×10×6 world, which the rewriting never saw.")
    return fig


def boot_ci(vals, draws=10_000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    m = rng.choice(v, size=(draws, len(v)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def scene4_fig(D, video_line):
    res = D["res"]
    fig = vk.new_fig()
    n_conf = len(D["summary"][0]["seeds"])
    vk.header(fig, "The answer on the larger world: each search's pick, retrained on fresh seeds",
              f"10×10×6 world, 512 dev designs; each dot is one search's pick on {n_conf} fresh seeds (a pick that broke "
              "a rule scores as the start). Bars: arm mean; whiskers: 95% bootstrap interval.", stage="Stage 2c · the answer")
    ax = fig.add_axes([0.06, 0.20, 0.50, 0.56])
    vk.style_axes(ax)
    arms = [a for a in ORDER if a in D["arms"]]
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([LABEL[a] for a in arms], fontsize=12.5, color=vk.INK2)
    rng = np.random.default_rng(0)  # jitter only
    for j, a in enumerate(arms):
        vals = [s["primary_score"] for s in D["arms"][a]["searches"]]
        hi_arm = a in ("g1", "final")
        col = vk.BLUE if hi_arm else vk.INK2
        ax.plot(j + rng.uniform(-0.18, 0.18, len(vals)), 100 * np.array(vals), "D", ms=6, color=col, mec=vk.SURFACE,
                mew=0.6, alpha=0.75, zorder=3)
        mu, (lo, hi) = 100 * mean(vals), boot_ci(vals)
        ax.plot([j, j], [100 * lo, 100 * hi], color=col, lw=2, zorder=4)
        ax.plot([j - 0.3, j + 0.3], [mu, mu], color=col, lw=3.4, zorder=5)
        ax.text(j + 0.33, mu, f"{mu:.0f}%", ha="left", va="center", fontsize=13, fontweight="bold", color=col, zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc=vk.SURFACE, ec="none", alpha=0.85))
    ax.set_ylabel("built exactly, 512 dev designs", fontsize=13, color=vk.INK2)
    # P1-P4: differences in points with their intervals and Holm-adjusted p
    fx = fig.add_axes([0.63, 0.20, 0.33, 0.56])
    fx.set_facecolor(vk.SURFACE)
    c = res["primary_family"]["contrasts"]
    fin = D["arms"].get("final", {}).get("searches", [])
    same = bool(fin) and all(s.get("shared_with") == "g1" for s in fin)
    box = dict(boxstyle="round,pad=0.15", fc=vk.SURFACE, ec="none")
    tested = [c[k] for k in ("P1", "P2", "P3", "P4") if c[k]["tested"]]
    lo = min([100 * t["ci95_boot"][0] for t in tested] + [0]) - 5
    hi = max([100 * t["ci95_boot"][1] for t in tested] + [0]) + 5
    fx.set_xlim(lo, hi)
    fx.set_ylim(0.3, 4.9)
    fx.axvline(0, color=vk.AXIS, lw=1.2)
    fx.set_yticks([])
    for s in ("top", "right", "left"):
        fx.spines[s].set_visible(False)
    fx.spines["bottom"].set_color(vk.AXIS)
    fx.tick_params(length=0, colors=vk.MUTED)
    fx.set_xlabel("difference, points of success", fontsize=13, color=vk.INK2)
    for k, y in zip(("P1", "P2", "P3", "P4"), (4, 3, 2, 1)):
        ck = c[k]
        if not ck["tested"]:
            fx.text(lo + 1, y, f"{k} {PNAME[k]}: not tested (no lineage kept a rewrite)", fontsize=12.5, color=vk.MUTED,
                    va="center", bbox=box)
            continue
        good = ck["diff"] > 0 and ck["p_holm"] < 0.05
        col = vk.BLUE if good else vk.INK2
        fx.plot([100 * ck["ci95_boot"][0], 100 * ck["ci95_boot"][1]], [y, y], color=col, lw=2.4)
        fx.plot(100 * ck["diff"], y, "o", ms=10, color=col, mec=vk.SURFACE, mew=1.5)
        fx.text(lo + 1, y + 0.32, f"{k} {PNAME[k]}: {100 * ck['diff']:+.1f} points, Holm-adjusted p {pv(ck['p_holm'])}",
                fontsize=12.5, color=col, va="center", fontweight="bold" if good else None, bbox=box)
        if k == "P4" and same:
            fx.text(lo + 1, y - 0.3, "the same searches: the last kept program is the first rewrite", fontsize=11.5,
                    color=vk.MUTED, va="center", bbox=box)
    for k, ln in enumerate(textwrap.wrap(video_line, 150)[:2]):
        fig.text(0.05, 0.075 - 0.032 * k, ln, fontsize=16, color=vk.INK, ha="left", fontweight="bold" if k == 0 else None)
    return fig


def scene5_fig(D):
    gens = next(iter(D["lins"].values()))["gens"]
    marker = A / "stage2" / "FINAL_OPENED_W"
    held = ("Held-out designs: opened once, after every confirmation run." if marker.exists()
            else "Held-out designs: still locked. Every number here is on dev designs.")
    lines = ["One toy brick world: rewrites made on 8×8×6, tested on 10×10×6.",
             f"{len(D['lins'])} lineages of {len(gens) - 1} rewrites; one AI scientist (an LLM).",
             "Its weights never changed: only its written research program did.",
             held, "Toy scale, on one laptop GPU."]
    return vk.card("What this does and doesn't show", lines, stage="Stage 2c")


def pick_scene3(D) -> dict:
    """Stage 2b's rule on the final arm: eligible picks (not the start, judged compliant), by primary score, floor(m/2)."""
    rows = D["arms"].get("final", {}).get("searches", [])
    elig = [s for s in rows if s["pick"] != "start" and s["verdict"] in ("compliant", "compliant (by construction)")]
    elig = sorted(elig, key=lambda s: -s["primary_score"])  # stable: ties keep results.json order
    if not elig:
        return {"none": True, "why": "no final-arm search has an eligible pick" if rows else "the final arm wasn't run "
                                     "(no lineage kept a rewrite)"}
    s = elig[len(elig) // 2]
    return {"none": False, "row": s, "m": len(elig), "position": len(elig) // 2}


def scene3_frames(D, skip_build):
    ch = pick_scene3(D)
    if ch["none"] or skip_build:
        why = ch.get("why") or "skipped: this campaign has no trained checkpoints (a test run)"
        return [(vk.to_rgb(vk.card("One search under the last kept program", [why.capitalize() + "."], stage="Stage 2c")), SECONDS[3] * vk.FPS)], ch
    SB = _load("search_build", A / "viz" / "stage2b" / "search_build.py")
    import torch
    designs = torch.load(A / "stage2" / "targets" / "w_dev.pt")
    di = int(SB.R.has_hidden_cells(designs).nonzero()[0, 0])
    sdir = D["cdir"] / ch["row"]["path"]
    base = json.loads((D["cdir"] / "baseline" / "w" / "summary.json").read_text())
    Db = SB.load_search(sdir, base, "w", designs, di, SB.R.SPECS["w"])
    st = json.loads((sdir / "state.json").read_text())["settings"]
    title = f"One search under lineage {ch['row']['lineage']}'s last kept program: {Db['n']} experiments, {Db['n_kept']} kept"
    sub = (f"Its header set {st['R']} rounds of {st['K']} candidates, {plural(len(st['trial_seeds']), 'training seed')} each. "
           f"10×10×6 world; the search shown is the middle one by confirmed score.")
    out = [(np.asarray(img.convert("RGB")), times) for img, times in SB.video_frames(Db, title, sub, seconds=SECONDS[3])]
    short = SECONDS[3] * vk.FPS - sum(times for _, times in out)
    if short > 0:  # a short search: hold its last frame so the scene keeps its length
        out[-1] = (out[-1][0], out[-1][1] + short)
    return out, ch


# ---------------------------------------------------------------- output
def write_frames(w, frames):
    for buf, times in frames:
        assert buf.shape == (vk.H_PX, vk.W_PX, 3), buf.shape
        for _ in range(times):
            w.ff.stdin.write(buf.tobytes())
            w.frames += 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="arch_rsi/campaigns/stage2c")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--allow-small", action="store_true", help="tests only: fewer than 16 searches per arm")
    a = ap.parse_args(argv)
    cdir = Path(a.campaign) if Path(a.campaign).is_absolute() else ROOT / a.campaign
    D = load(cdir.resolve())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_arm = {arm: d["n"] for arm, d in D["arms"].items() if arm in ("g0", "g1", "facts", "shuffled")}
    if set(n_arm.values()) != {16} and not a.allow_small:
        raise SystemExit(f"ENDINGS_2C.md says 16 searches per arm; this campaign has {n_arm}")
    key, video_line, article = ending(D["res"])
    L = sorted(D["lins"])[0]
    G = len(D["lins"][L]["gens"]) - 1
    scene1(D).save(out / "scene1_system.png")
    for name, fig in (("scene2_program", scene2_fig(D, L, G)), ("scene4_answer", scene4_fig(D, video_line)),
                      ("scene5_limits", scene5_fig(D))):
        fig.savefig(out / f"{name}.png", facecolor=vk.SURFACE, dpi=100)
        plt.close(fig)
    data = {"campaign": str(cdir), "ending": key, "video_line": video_line, "article": article, "lineage_shown": L,
            "generations": [{k: v for k, v in g.items() if k != "changes"} for g in D["lins"][L]["gens"]],
            "arm_primary_means": {arm: D["arms"][arm]["primary_mean"] for arm in D["arms"]},
            "checks": "passed: programs, generation means, search scores and arm means"}
    if a.video:
        with vk.Writer(out / "stage2c.mp4") as w:
            write_frames(w, [(np.asarray(scene1(D).convert("RGB")), SECONDS[1] * vk.FPS)])
            first = 3
            per = (SECONDS[2] - first - 3) / max(G, 1)
            w.add(scene2_fig(D, L, 0), first)
            for g in range(1, G + 1):
                w.add(scene2_fig(D, L, g), per)
            w.add(scene2_fig(D, L, G), 3)
            frames, ch = scene3_frames(D, a.skip_build)
            write_frames(w, frames)
            w.add(scene4_fig(D, video_line), SECONDS[4])
            w.add(scene5_fig(D), SECONDS[5])
        data["video_seconds"] = round(w.seconds, 1)
        data["scene3"] = {k: v for k, v in ch.items() if k != "row"} | (
            {"search": {k: ch["row"][k] for k in ("lineage", "search", "pick", "primary_score", "path")}} if not ch["none"] else {})
    (out / "video_2c_data.json").write_text(json.dumps(data, indent=1))
    print(json.dumps({k: data[k] for k in ("ending", "video_line") if k in data} | {"seconds": data.get("video_seconds")}, indent=1))
    return data


if __name__ == "__main__":
    main()
