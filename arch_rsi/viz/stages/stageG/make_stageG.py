"""Stage G article video: fixing the training recipe. Visualization only: nothing is trained, scored or selected here.

What it shows: the starting network on world Q (8x8x5) at 262,144 training steps, on the same seeds 300-305.
  - Left panel: exact success on the 512 dev designs, per seed, under the frozen CPU recipe (64 worlds per update,
    Stage 0's sweep) and under Stage G's pick (256 worlds per update, 32 minibatches, on the GPU), joined seed by seed;
    the means; and the extra check, Stage G's recipe on the CPU (seeds 300-302). Filled dot: the seed learned the
    hidden-cell skill (it builds at least 10% of the dev designs with hidden cells, as in PROTOCOL_G.md).
  - Right panel: success on training designs as training runs (the trainers' own logged curves), mean over seeds.
  - A takeaway card, with G6a's device-only check.

Data sources (read-only):
  arch_rsi/pilot/stage0/sweep/start_262144_s{300..305}/          frozen recipe: dev.json, run/train_log.json
  arch_rsi/gpu/stageG/runs/cal/xl/start_b262144_s{300..305}/     new recipe on the GPU: dev.json, result.json,
                                                                 run.json, run/train_log.json
  arch_rsi/gpu/stageG/extra_cpu_recipe/start_b262144_s{300..302}/ new recipe on the CPU (extra check)
  arch_rsi/gpu/stageG/runs/decisions.json, summary.json           G3 and G6a numbers (cross-checked)
  arch_rsi/gpu/stageG/runs/manifest.json                          hashes of the frozen recipe file
  arch_rsi/worlds.py                                              the frozen RECIPE (parsed, not imported)
  arch_rsi/targets/xl_dev.pt                                      dev designs only: the hidden-cell mask
  arch_rsi/gpu/stageG/REPORT.md, PROTOCOL_G.md                    cross-checks (asserts)
Every number shown is computed here from those files and asserted against REPORT.md (within rounding).

usage (project root, low priority while Stage 2b runs):
  nice -n 19 ./.venv/bin/python -I arch_rsi/viz/stages/stageG/make_stageG.py           # png + data json + mp4
  nice -n 19 ./.venv/bin/python -I arch_rsi/viz/stages/stageG/make_stageG.py --still   # png + data json only
"""
import argparse
import ast
import hashlib
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
G = A / "gpu" / "stageG"
NAME = "stageG"
SEEDS = list(range(300, 306))
CPU_SEEDS = list(range(300, 303))
OLD_C, NEW_C = vk.GREY, vk.ORANGE


def frozen_recipe() -> dict:
    """worlds.RECIPE, read from the source without importing it; the file must be the one Stage G hashed."""
    p = A / "worlds.py"
    man = json.loads((G / "runs" / "manifest.json").read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest() == man["hashes"]["arch_rsi/worlds.py"]
    tree = ast.parse(p.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "RECIPE")
    return ast.literal_eval(node.value)


def hidden_mask() -> tuple:
    """Dev designs with at least one empty cell under a filled one (the rule in run_stageG.py)."""
    v = torch.load(A / "targets" / "xl_dev.pt").bool()
    above = (v.flip(1).cumsum(1).flip(1) - v.int()) > 0
    n, z, y, x = v.shape
    return ((~v) & above).flatten(1).any(1).numpy(), (x, y, z), n


def run_dir(d: Path, H) -> dict:
    dv = json.loads((d / "dev.json").read_text())
    pe = np.array(dv["per_design"]["exact"], bool)
    assert abs(pe.mean() - dv["exact"]) < 1e-12 and dv["split"] == "dev"
    tl = json.loads((d / "run" / "train_log.json").read_text())
    assert tl["status"] == "ok"
    hid = float(pe[H].mean())
    return {"dir": d, "exact": float(dv["exact"]), "hidden": hid, "learned": hid >= 0.10, "log": tl,
            "curve_steps": [c["steps"] for c in tl["curve"]], "curve": [c["train_exact"] for c in tl["curve"]]}


def grad_steps(n_updates: int, rc: dict) -> tuple:
    batch = rc["n_envs"] * rc["rollout_len"]
    return n_updates * rc["ppo_epochs"] * rc["n_minibatches"], batch // rc["n_minibatches"], batch


def load() -> dict:
    H, dims, n_dev = hidden_mask()
    rec0 = frozen_recipe()
    budget = 262144
    old = [run_dir(A / "pilot" / "stage0" / "sweep" / f"start_{budget}_s{s}", H) for s in SEEDS]
    new = [run_dir(G / "runs" / "cal" / "xl" / f"start_b{budget}_s{s}", H) for s in SEEDS]
    cpu = [run_dir(G / "extra_cpu_recipe" / f"start_b{budget}_s{s}", H) for s in CPU_SEEDS]
    for r, s in zip(old, SEEDS):
        spec = json.loads((r["dir"] / "run.json").read_text())
        assert spec["seed"] == s and spec["world"] == "xl" and spec["budget_steps"] == budget and "recipe" not in spec
        assert r["log"]["steps"] == budget
    for grp, dev_name in ((new, "mps"), (cpu, "cpu")):
        for r in grp:
            spec = json.loads((r["dir"] / "run.json").read_text())
            assert spec["world"] == "xl" and spec["budget_steps"] == budget and spec["device"] == dev_name
            assert r["log"]["device"] == dev_name and r["log"]["steps"] == budget
    for r in new:  # recomputed skill and score match the runner's own result.json
        res = json.loads((r["dir"] / "result.json").read_text())
        assert abs(res["dev_exact"] - r["exact"]) < 1e-12 and abs(res["hidden"] - r["hidden"]) < 1e-9
        assert res["learned"] == r["learned"] and res["status"] == "ok"
    for r in cpu:
        assert abs(json.loads((r["dir"] / "result.json").read_text())["dev_exact"] - r["exact"]) < 1e-12

    # the two recipes: same gradient steps, same minibatch size (G's recipe overrides only these keys)
    rec1 = new[0]["log"]["recipe"]
    assert all(r["log"]["recipe"] == rec1 for r in new + cpu)
    assert all(rec1[k] == rec0[k] for k in ("rollout_len", "ppo_epochs"))
    gs0, mb0, b0 = grad_steps(old[0]["log"]["n_updates"], rec0)
    gs1, mb1, b1 = grad_steps(new[0]["log"]["n_updates"], rec1)
    assert all(r["log"]["n_updates"] == old[0]["log"]["n_updates"] for r in old)
    assert all(r["log"]["n_updates"] == new[0]["log"]["n_updates"] for r in new + cpu)
    assert (gs0, mb0) == (gs1, mb1)

    # G3 row in decisions.json (and the same in summary.json)
    dec = json.loads((G / "runs" / "decisions.json").read_text())
    assert dec == json.loads((G / "runs" / "summary.json").read_text())["decisions"]
    g3 = next(r for r in dec["G3"]["rows"] if r["budget"] == budget)
    assert dec["G3"]["budget"] == budget and dec["G3"]["world"] == "xl"
    assert all(abs(a - b) < 6e-5 for a, b in zip(g3["start_per_seed"], [r["exact"] for r in new]))
    assert abs(g3["start_mean"] - np.mean([r["exact"] for r in new])) < 6e-5
    assert g3["start_learned"] == sum(r["learned"] for r in new)
    g6 = dec["G6"]["a"]
    g6_budget = json.loads((G / "runs" / "manifest.json").read_text())["config"]["g6_budget"]

    def curve(grp):
        st = grp[0]["curve_steps"]
        assert all(r["curve_steps"] == st for r in grp)
        return np.array(st), np.mean([r["curve"] for r in grp], 0)

    D = {"dims": dims, "n_dev": n_dev, "budget": budget, "old": old, "new": new, "cpu": cpu,
         "rec0": rec0, "rec1": rec1, "upd0": old[0]["log"]["n_updates"], "upd1": new[0]["log"]["n_updates"],
         "grad_steps": gs0, "minibatch": mb0, "batch0": b0, "batch1": b1,
         "m_old": float(np.mean([r["exact"] for r in old])), "m_new": float(np.mean([r["exact"] for r in new])),
         "m_cpu": float(np.mean([r["exact"] for r in cpu])),
         "l_old": sum(r["learned"] for r in old), "l_new": sum(r["learned"] for r in new),
         "l_cpu": sum(r["learned"] for r in cpu),
         "g6": {"gpu_mean": g6["gpu_mean"], "cpu_mean": g6["cpu_mean"], "p": g6["exact_perm_p_two_sided"],
                "n": len(g6["gpu_per_seed"]), "n_cpu": len(g6["cpu_stage1_per_seed"]), "splits": g6["splits"],
                "budget": g6_budget, "stop": g6["stop"]},
         "c_old": curve(old), "c_new": curve(new), "c_cpu": curve(cpu)}
    assert abs(np.mean(g6["gpu_per_seed"]) - g6["gpu_mean"]) < 6e-5 and not g6["stop"]
    assert abs(np.mean(g6["cpu_stage1_per_seed"]) - g6["cpu_mean"]) < 6e-5
    check_curve_claim(D)
    check_report(D)
    return D


def check_curve_claim(D: dict) -> None:
    """The caption says the new recipe starts slower, then pulls ahead: check it on the logged curves."""
    b = D["budget"]
    (x0, y0), (x1, y1) = D["c_old"], D["c_new"]
    early0, early1 = y0[x0 <= b / 4].mean(), y1[x1 <= b / 4].mean()
    late0, late1 = y0[x0 >= 3 * b / 4].mean(), y1[x1 >= 3 * b / 4].mean()
    assert early0 > early1 and late1 > late0, (early0, early1, late0, late1)
    D["curve_check"] = {"first_quarter_mean": {"frozen": round(float(early0), 4), "new_gpu": round(float(early1), 4)},
                        "last_quarter_mean": {"frozen": round(float(late0), 4), "new_gpu": round(float(late1), 4)}}


def ints(s: str) -> list:
    return [int(x) for x in re.findall(r"\d+", s)]


def p100(v: float) -> int:
    return int(round(100 * v))


def check_report(D: dict) -> None:
    rep = (G / "REPORT.md").read_text()
    assert re.search(r"the same budget \(([\d,]+) steps\)", rep).group(1).replace(",", "") == str(D["budget"])
    row = r"\| {} \| (\w+) \| ([\d, ]+)% \| ([\d.]+)% \| (\d+) of (\d+) \|"
    m = re.search(row.format(r"Frozen: (\d+) worlds per update, (\d+) minibatches \(Stage 0's sweep\)"), rep)
    assert (int(m.group(1)), int(m.group(2))) == (D["rec0"]["n_envs"], D["rec0"]["n_minibatches"]) and m.group(3) == "CPU"
    assert ints(m.group(4)) == [p100(r["exact"]) for r in D["old"]]
    assert float(m.group(5)) == round(100 * D["m_old"], 1) and ints(m.group(6) + " " + m.group(7)) == [D["l_old"], 6]
    m = re.search(row.format(r"Stage G's pick: (\d+) worlds per update, (\d+) minibatches \(G3\)"), rep)
    assert (int(m.group(1)), int(m.group(2))) == (D["rec1"]["n_envs"], D["rec1"]["n_minibatches"]) and m.group(3) == "GPU"
    assert ints(m.group(4)) == [p100(r["exact"]) for r in D["new"]]
    assert float(m.group(5)) == round(100 * D["m_new"], 1) and ints(m.group(6) + " " + m.group(7)) == [D["l_new"], 6]
    m = re.search(row.format(r"Stage G's pick \(extra check, seeds (\d+)–(\d+)\)"), rep)
    assert [int(m.group(1)), int(m.group(2))] == [CPU_SEEDS[0], CPU_SEEDS[-1]] and m.group(3) == "CPU"
    assert ints(m.group(4)) == [p100(r["exact"]) for r in D["cpu"]]
    assert float(m.group(5)) == round(100 * D["m_cpu"], 1) and ints(m.group(6) + " " + m.group(7)) == [D["l_cpu"], 3]
    m = re.search(r"Both take ([\d,]+) gradient steps of (\d+) samples", rep)
    assert int(m.group(1).replace(",", "")) == D["grad_steps"] and int(m.group(2)) == D["minibatch"]
    m = re.search(r"collects ([\d,]+) samples instead of ([\d,]+), and the policy is refreshed (\d+) times instead "
                  r"of (\d+)", rep)
    assert [int(g.replace(",", "")) for g in m.groups()] == [D["batch1"], D["batch0"], D["upd1"], D["upd0"]]
    g6 = D["g6"]
    m = re.search(r"seeds (\d+)–(\d+)\) averaged ([\d.]+)%.*?same seeds averaged ([\d.]+)%.*?over ([\d,]+) splits: "
                  r"p = ([\d.]+)", rep)
    assert float(m.group(3)) == round(100 * g6["gpu_mean"], 1) and float(m.group(4)) == round(100 * g6["cpu_mean"], 1)
    assert int(m.group(5).replace(",", "")) == g6["splits"] and float(m.group(6)) == round(g6["p"], 3)
    assert int(m.group(2)) - int(m.group(1)) + 1 == g6["n"] == g6["n_cpu"]
    assert "131,072 steps, seeds 1001–1012" in (G / "PROTOCOL_G.md").read_text() and g6["budget"] == 131072
    assert "builds at least 10% of the dev designs that have hidden cells" in (G / "PROTOCOL_G.md").read_text()


# ---------------------------------------------------------------- drawing
XC = (0.0, 1.0, 2.2)  # columns: frozen, new on GPU, new on CPU


def dot(ax, x, y, learned, color, ms=14):
    ax.plot(x, 100 * y, "o", ms=ms, mfc=color if learned else vk.SURFACE, mec=color if learned else color,
            mew=2.2, zorder=5)


def mean_bar(ax, x, m, learned, n, color, side):
    ax.plot([x - 0.22, x + 0.22], [100 * m] * 2, color=vk.INK, lw=3.4, zorder=6, solid_capstyle="butt")
    ha, xx = ("right", x - 0.3) if side == "left" else ("left", x + 0.3)
    ax.text(xx, 100 * m + 1.2, f"{100 * m:.1f}%", ha=ha, va="bottom", fontsize=19, color=vk.INK, fontweight="bold",
            zorder=7)
    ax.text(xx, 100 * m - 1.2, f"{learned} of {n} learned", ha=ha, va="top", fontsize=13, color=color, zorder=7)


def chart(D: dict, show: int, curve_frac: float = 0.0, caption: str = None):
    """show: 0 = axes only, 1 = + frozen recipe, 2 = + new recipe on the GPU, 3 = + new recipe on the CPU."""
    fig = vk.new_fig()
    x, y, z = D["dims"]
    vk.header(fig, "Fixing the training recipe",
              f"The starting network on world Q ({x}×{y}×{z} brick-assembly world), {D['budget']:,} training steps, "
              f"the same seeds. Both recipes take {D['grad_steps']:,} gradient steps of {D['minibatch']} samples.",
              stage="Stage G · re-choosing the training recipe")

    # left: per-seed dev scores
    ax = fig.add_axes([0.07, 0.205, 0.475, 0.5])
    vk.style_axes(ax)
    ax.set_xlim(-0.95, 3.05)
    ax.set_xticks(XC)
    r0, r1 = D["rec0"], D["rec1"]
    ax.set_xticklabels([f"Frozen recipe\n{r0['n_envs']} worlds per update\nCPU · seeds {SEEDS[0]}–{SEEDS[-1]}",
                        f"New recipe\n{r1['n_envs']} worlds per update\nGPU · seeds {SEEDS[0]}–{SEEDS[-1]}",
                        f"New recipe on the CPU\nextra check\nseeds {CPU_SEEDS[0]}–{CPU_SEEDS[-1]}"],
                       fontsize=13.5, color=vk.INK2, linespacing=1.35)
    ax.set_ylabel(f"built exactly, {D['n_dev']} dev designs", fontsize=14, color=vk.INK2, labelpad=10)
    ax.set_title(f"Each seed's score on the {D['n_dev']} dev designs", fontsize=17, color=vk.INK, loc="left", pad=14,
                 fontweight="bold")
    off = [(i - (len(SEEDS) - 1) / 2) * 0.035 for i in range(len(SEEDS))]
    if show >= 2:
        for o, a, b in zip(off, D["old"], D["new"]):
            ax.plot([XC[0] + o, XC[1] + o], [100 * a["exact"], 100 * b["exact"]], color=vk.INK2, lw=1.4, alpha=0.35,
                    zorder=2)
    if show >= 1:
        for o, r in zip(off, D["old"]):
            dot(ax, XC[0] + o, r["exact"], r["learned"], OLD_C)
        mean_bar(ax, XC[0], D["m_old"], D["l_old"], len(D["old"]), OLD_C, "left")
    if show >= 2:
        for o, r in zip(off, D["new"]):
            dot(ax, XC[1] + o, r["exact"], r["learned"], NEW_C)
        mean_bar(ax, XC[1], D["m_new"], D["l_new"], len(D["new"]), NEW_C, "right")
    if show >= 3:
        ax.axvline(XC[2] - 0.25, color=vk.GRID, lw=1.2, zorder=0)
        for i, r in enumerate(D["cpu"]):
            dot(ax, XC[2] + (i - 1) * 0.05, r["exact"], r["learned"], NEW_C)
        mean_bar(ax, XC[2], D["m_cpu"], D["l_cpu"], len(D["cpu"]), NEW_C, "right")

    # right: training curves
    bx = fig.add_axes([0.635, 0.205, 0.335, 0.5])
    vk.style_axes(bx)
    b = D["budget"]
    bx.set_xlim(0, b * 1.02)
    ticks = [0, b // 4, b // 2, 3 * b // 4, b]
    bx.set_xticks(ticks)
    bx.set_xticklabels([f"{round(t / 1000)}k" if t else "0" for t in ticks], fontsize=13)
    bx.set_xlabel("training steps", fontsize=14, color=vk.INK2, labelpad=8)
    bx.tick_params(axis="x", length=0, pad=8)
    bx.set_title("Success on training designs, during training", fontsize=17, color=vk.INK, loc="left", pad=14,
                 fontweight="bold")
    lines = [(D["c_old"], OLD_C, "-", 2.8, f"frozen recipe, CPU (mean of {len(D['old'])} seeds)"),
             (D["c_new"], NEW_C, "-", 3.0, f"new recipe, GPU (mean of {len(D['new'])} seeds)"),
             (D["c_cpu"], NEW_C, (0, (4, 2.5)), 2.2, f"new recipe, CPU (mean of {len(D['cpu'])} seeds)")]
    for (xs, ys), c, ls, lw, _ in lines:
        k = xs <= curve_frac * b
        if k.any():
            bx.plot(xs[k], 100 * ys[k], color=c, ls=ls, lw=lw, solid_joinstyle="round", zorder=3)
    leg = bx.legend(handles=[Line2D([], [], color=c, ls=ls, lw=lw, label=lab) for _, c, ls, lw, lab in lines],
                    loc="upper left", frameon=False, fontsize=13, handlelength=2.6)
    for t in leg.get_texts():
        t.set_color(vk.INK2)

    handles = [Line2D([], [], color=OLD_C, marker="o", ms=12, lw=0, mew=2.2, label="frozen recipe"),
               Line2D([], [], color=NEW_C, marker="o", ms=12, lw=0, mew=2.2, label="new recipe"),
               Line2D([], [], color=vk.INK2, marker="o", ms=12, lw=0, mfc=vk.SURFACE, mew=2.2,
                      label="hollow: the seed didn't learn the hidden-cell skill (built under 10% of the designs with hidden cells)"),
               Line2D([], [], color=vk.INK, lw=3.4, label="mean")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.815), ncol=4, frameon=False,
                     fontsize=13.5, handletextpad=0.5, columnspacing=1.8)
    for t in leg.get_texts():
        t.set_color(vk.INK2)
    if caption:
        vk.caption(fig, caption)
    return fig


def states(D: dict) -> list:
    r0, r1 = D["rec0"], D["rec1"]
    curve_cap = "Success on training designs: the new recipe starts slower, then pulls ahead."
    return [
        (0, 0.0, "Same network, same world, same seeds, same number of gradient steps. Only the recipe changes.", 2.2),
        (1, 0.0, f"Frozen CPU recipe, {r0['n_envs']} worlds per update: {100 * D['m_old']:.1f}% of dev designs. Only "
                 f"{D['l_old']} of {len(D['old'])} seeds learned to build designs with hidden cells.", 3.0),
        (2, 0.0, f"New recipe, {r1['n_envs']} worlds per update: {100 * D['m_new']:.1f}%, on the same seeds. All "
                 f"{D['l_new']} seeds learned the skill.", 3.0),
        (3, 0.0, f"The new recipe on the CPU, seeds {CPU_SEEDS[0]}–{CPU_SEEDS[-1]}: {100 * D['m_cpu']:.1f}%. The jump "
                 f"comes from the recipe, not the GPU.", 3.2),
        (3, 0.25, curve_cap, 0.6), (3, 0.5, curve_cap, 0.6), (3, 0.75, curve_cap, 0.6), (3, 1.0, curve_cap, 1.4),
        (3, 1.0, f"A small check: {len(D['new'])} seeds per recipe, dev designs only. Next: retest Stage 1's picks "
                 f"with the new recipe.", 2.6)]


def takeaway(D: dict):
    x, y, z = D["dims"]
    g6 = D["g6"]
    return vk.card("It's the recipe, not the device", [
        f"Same starting network, same seeds: {100 * D['m_old']:.1f}% → {100 * D['m_new']:.1f}% of dev designs.",
        f"The new recipe on the CPU did the same: {100 * D['m_cpu']:.1f}% ({len(D['cpu'])} seeds, an extra check).",
        f"The device alone showed no detectable shift: {100 * g6['gpu_mean']:.1f}% on the GPU vs "
        f"{100 * g6['cpu_mean']:.1f}% on the CPU, p = {g6['p']:.3f}.",
        "The frozen recipe had been holding the starting network back.",
        "Next: retrain Stage 1's picks with the new recipe, to see how much of their lead survives."],
        stage="Stage G · re-choosing the training recipe",
        foot=f"World Q ({x}×{y}×{z} brick-assembly world), dev designs only, no LLM calls. {D['budget']:,} training "
             f"steps; the device check used the frozen recipe, {g6['budget']:,} steps, {g6['n']} seeds per device.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", action="store_true", help="write the png and data json only")
    args = ap.parse_args()
    D = load()
    st = states(D)
    fig = chart(D, 3, 1.0, st[-1][2])
    fig.savefig(HERE / f"{NAME}.png", facecolor=vk.SURFACE, dpi=100)
    vk.plt.close(fig)
    secs = None
    if args.still:
        fig = takeaway(D)
        fig.savefig(HERE / f"{NAME}_card_preview.png", facecolor=vk.SURFACE, dpi=100)
        vk.plt.close(fig)
    else:
        (HERE / f"{NAME}_card_preview.png").unlink(missing_ok=True)
        with vk.Writer(HERE / f"{NAME}.mp4") as w:
            for show, frac, cap, s in st:
                w.add(chart(D, show, frac, cap), seconds=s)
            chart_s = w.seconds
            w.add(takeaway(D), seconds=5.0)
        secs = {"chart": round(chart_s, 2), "card": round(w.seconds - chart_s, 2), "total": round(w.seconds, 2)}

    r4 = lambda v: round(float(v), 4)  # noqa: E731
    rel = lambda p: str(p.relative_to(ROOT))  # noqa: E731

    def runs(grp):
        return [{"seed": int(r["dir"].name.rsplit("_s", 1)[1]), "dev_exact": r4(r["exact"]), "hidden": r4(r["hidden"]),
                 "learned_skill": bool(r["learned"]), "dir": rel(r["dir"])} for r in grp]

    def crv(c):
        return {"steps": [int(s) for s in c[0]], "train_exact_mean": [r4(v) for v in c[1]]}

    g6 = D["g6"]
    data = {
        "what": "Stage G: the starting network on world Q at 262,144 steps, frozen recipe vs Stage G's recipe, dev designs",
        "world_dims_xyz": D["dims"], "n_dev_designs": D["n_dev"], "budget_steps": D["budget"],
        "recipes": {"frozen": {k: D["rec0"][k] for k in ("n_envs", "n_minibatches", "lr", "ppo_epochs", "rollout_len")}
                    | {"updates": D["upd0"], "samples_per_update": D["batch0"], "source": "arch_rsi/worlds.py RECIPE"},
                    "new": D["rec1"] | {"updates": D["upd1"], "samples_per_update": D["batch1"],
                                        "source": "train_log.json recipe"},
                    "gradient_steps_each": D["grad_steps"], "samples_per_minibatch": D["minibatch"]},
        "frozen_recipe_cpu": {"runs": runs(D["old"]), "mean": r4(D["m_old"]), "learned": D["l_old"],
                              "source": "arch_rsi/pilot/stage0/sweep/start_262144_s300..305"},
        "new_recipe_gpu": {"runs": runs(D["new"]), "mean": r4(D["m_new"]), "learned": D["l_new"],
                           "source": "arch_rsi/gpu/stageG/runs/cal/xl/start_b262144_s300..305 (= decisions.json G3)"},
        "new_recipe_cpu_extra_check": {"runs": runs(D["cpu"]), "mean": r4(D["m_cpu"]), "learned": D["l_cpu"],
                                       "source": "arch_rsi/gpu/stageG/extra_cpu_recipe/start_b262144_s300..302"},
        "device_only_check_G6a": {"gpu_mean": g6["gpu_mean"], "cpu_mean": g6["cpu_mean"], "p_two_sided": g6["p"],
                                  "seeds_each": g6["n"], "budget_steps": g6["budget"],
                                  "source": "arch_rsi/gpu/stageG/runs/decisions.json G6.a"},
        "training_curves": {"what": "train_exact from each run's train_log.json, mean over seeds",
                            "frozen_cpu": crv(D["c_old"]), "new_gpu": crv(D["c_new"]), "new_cpu": crv(D["c_cpu"]),
                            "caption_check": D["curve_check"]},
        "skill_rule": "learned = builds >= 10% of the dev designs with hidden cells (PROTOCOL_G.md)",
        "captions": [c for _, _, c, _ in st], "video_seconds": secs}
    (HERE / f"{NAME}_data.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(json.dumps({"means": [r4(D["m_old"]), r4(D["m_new"]), r4(D["m_cpu"])],
                      "learned": [D["l_old"], D["l_new"], D["l_cpu"]], "grad_steps": D["grad_steps"],
                      "curve_check": D["curve_check"], "video_seconds": secs}))


if __name__ == "__main__":
    main()
