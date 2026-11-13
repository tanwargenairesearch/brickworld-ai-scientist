"""VISUALIZATION ONLY. One search in one frame, in the style of an autoresearch progress chart:
  left   - the current experiment's trained network building one dev design, brick by brick;
  bottom - its evaluation: every dev design, filled if built exactly, for both training seeds;
  right  - the search's progress: one dot per experiment, kept improvements labelled, the running best as a step line.

Nothing here feeds any score, selection or test. Numbers come from the search's own files and are checked:
- each experiment's 2-seed mean is recomputed from its dev.json files and must match state.json;
- the incumbent is recomputed by the frozen rule (a strictly higher stored 2-seed mean, starting from the start's
  seeds 0-1 bar) and must equal state.json's incumbent; a dot is drawn as "kept" only if its exact mean is strictly
  above the running best, so a tie with the bar that passes the frozen rule by rounding isn't shown as a new best;
- the build shown is a greedy replay of the seed-0 checkpoint on one design (the first dev design with hidden
  cells); whether it is built exactly must match that run's own per-design result, or the script stops.

Prototype on Stage 1 data (CPU, low priority; safe while Stage 2b runs):
    nice -n 19 .venv/bin/python -I arch_rsi/viz/stage2b/search_build.py --stills 7 12
On W (Stage 2b or 2c; W is registered here, in this process only):
    ... --world w --targets arch_rsi/stage2/targets/w_dev.pt --search <search dir> --baseline <baseline summary.json>
Seeds per experiment come from the search's state.json ("settings", Stage 2c's header), else 2 (seeds 0 and 1). The
bar to beat is the start's mean over those seeds: from a {world: {"shown_dev"}} summary (Stage 1, 2b) or from a
2c summary's per-seed "all". video_frames() yields the frames, optionally fitted to a time budget (Stage 2c scene 3).
"""
import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("r1", HERE.parent / "stage1" / "render_learning_stage1.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)  # puts the project root on sys.path; reuses Stage 1's isometric drawing and replay

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from arch_rsi.stage2 import w_world  # noqa: E402

w_world.register()

W, H, FPS = 1920, 1080, 12
INK, INK2, MUTED, GRIDC, SURF = R.INK, R.INK2, R.MUTED, R.GRIDC, R.SURF
ARM_COL = (42, 120, 214)       # kept improvements and the running best
OTHER = (190, 189, 182)        # discarded experiments
EMPTY = (232, 231, 225)        # evaluation cell: not built exactly
text = R.text


def dev_json(d: Path) -> dict:
    return json.loads((d / "dev.json").read_text())


def build_one(src, state, target, spec):
    """Greedy rollout of one trained network on one design (the frozen evaluator's rule, batch of 1, CPU)."""
    aspec = R.arch_spec(spec)
    torch.manual_seed(0)
    policy = R.CandidatePolicy(R.load_candidate(src)(aspec), aspec)
    policy.load_state_dict(state)
    policy.eval()
    world = R.BrickWorld(spec, 1, "cpu", targets=target.unsqueeze(0), auto_reset=False)
    seq = []
    with torch.no_grad():
        for _ in range(spec.max_steps + 1):
            if bool(world.ended.all()):
                break
            logits, _ = policy(R.encode(world, "voxels"), world.valid_mask())
            a = logits.argmax(dim=1)
            seq.append(int(a[0]))
            world.step(a)
    return seq


def bar_for(baseline: dict, world: str, seeds: list) -> float:
    """The start's mean over the given seeds: Stage 1 / 2b summaries hold shown_dev (seeds 0-1), 2c's hold every seed."""
    if world in baseline:
        assert seeds == [0, 1], "a {world: shown_dev} summary only covers seeds 0 and 1"
        return baseline[world]["shown_dev"]
    vals = [baseline["all"][str(s)] for s in seeds]
    return sum(vals) / len(vals)


def load_search(sdir: Path, baseline: dict, world: str, designs, di: int, spec) -> dict:
    st = json.loads((sdir / "state.json").read_text())
    seeds = st.get("settings", {}).get("trial_seeds", [0, 1])
    bar = bar_for(baseline, world, seeds)
    target = designs[di]
    rows, best, inc, shown_best = [], bar, None, bar
    for k, t in enumerate(st["trials"], start=1):
        ok = t["result"]["status"] == "ok"
        row = {"i": k, "label": t["label"], "round": t["round"], "name": t["name"], "ok": ok, "kept": False}
        if ok:
            devs = [dev_json(sdir / "trials" / t["label"] / f"seed{s}") for s in seeds]
            row["per_seed"] = [d["exact"] for d in devs]
            row["grid"] = [d["per_design"]["exact"] for d in devs]
            row["mean"] = sum(row["per_seed"]) / len(seeds)
            assert abs(row["mean"] - t["result"]["dev_exact"]) < 6e-5, t["label"]
            if t["result"]["dev_exact"] > best:  # the frozen incumbent rule, on the stored mean over the seeds
                best, inc = t["result"]["dev_exact"], t["label"]
                # The stored mean is rounded to 4 decimals and the bar isn't, so a candidate that ties the bar exactly
                # (0.0684 against 0.068359...) passes the frozen rule. It's the runner's incumbent, but it isn't drawn
                # as a new best: the chart marks only a strictly higher exact mean.
                row["kept"] = row["mean"] > shown_best + 1e-9
                shown_best = max(shown_best, row["mean"])
            run = sdir / "trials" / t["label"] / "seed0" / "run"
            src = (run / "candidate.py").read_text()
            ck = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
            assert ck["candidate_sha"] == R.sha(src), t["label"]
            bricks, end, occ = R.replay(build_one(src, ck["state_dict"], target, spec), target, spec)
            exact = bool((occ == target).all())
            assert exact == bool(row["grid"][0][di]), f"{t['label']}: replay disagrees with its seed-0 dev.json"
            row.update(bricks=bricks, end=end, exact=exact, missing=int((target & ~occ).sum()),
                       extra=int((occ & ~target).sum()))
        row["best"] = best
        rows.append(row)
        print(row["label"], row["name"], "mean", round(row.get("mean", 0), 4), "kept", row["kept"],
              "design exact", row.get("exact"), flush=True)
    assert inc == st["incumbent"], (inc, st["incumbent"])
    return {"rows": rows, "bar": bar, "target": target, "di": di, "n": len(rows), "spec": spec, "seeds": seeds,
            "n_kept": sum(r["kept"] for r in rows), "n_designs": len(designs),
            "recipes": any("RECIPE" in t.get("code", "") for t in st["trials"])}


def seeds_txt(n: int) -> str:
    return f"{n} seed{'' if n == 1 else 's'}"


# ---------------------------------------------------------------- drawing
def rot_text(img, xy, s, size, col, angle=28):
    """Rotated label rising from xy, outlined in the surface colour. If it would run off the right edge, it is drawn
    level instead, ending just left of xy."""
    f = R.font(size)
    x0, y0, x1, y1 = f.getbbox(s)
    tmp = Image.new("RGBA", (x1 + 12, y1 + 12), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((6, 6), s, font=f, fill=col + (255,), stroke_width=3, stroke_fill=SURF + (255,))
    rot = tmp.rotate(angle, expand=True, resample=Image.BICUBIC)
    if xy[0] + rot.size[0] > img.size[0] - 8:
        img.paste(tmp, (int(xy[0] - 24 - tmp.size[0]), int(xy[1] - 4 - tmp.size[1])), tmp)
    else:
        img.paste(rot, (int(xy[0]), int(xy[1] - rot.size[1])), rot)


def short(s, n=30):
    return s if len(s) <= n else s[:n - 1] + "…"


def chart(img, d, D, cur, show_cur):
    """Right panel. cur = index of the experiment being shown (1-based); its dot appears only when show_cur."""
    x0, y0, x1, y1 = 1080, 120, 1880, 1000
    text(d, (x0, y0), "Search progress", 20, INK, bold=True)
    text(d, (x0, y0 + 28), f"Each dot is one experiment: {'a network and recipe' if D.get('recipes') else 'a network'} written, trained on {seeds_txt(len(D['seeds']))} and "
         "scored. Higher is better.", 15, INK2)
    px0, py0, px1, py1 = x0 + 60, y0 + 120, x1 - 30, y1 - 60
    n = D["n"]
    X = lambda i: px0 + (px1 - px0) * i / (n + 0.5)  # noqa: E731
    Y = lambda v: py1 - (py1 - py0) * v  # noqa: E731
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        d.line([(px0, Y(v)), (px1, Y(v))], fill=GRIDC, width=1)
        text(d, (px0 - 10, Y(v)), f"{v * 100:.0f}%", 14, MUTED, anchor="rm")
    text(d, (px0 - 52, py0 - 34), "Built exactly (share of dev designs)", 14, INK2)
    rounds = sorted({r["round"] for r in D["rows"]})
    for rd in rounds:
        idx = [r["i"] for r in D["rows"] if r["round"] == rd]
        text(d, ((X(idx[0]) + X(idx[-1])) / 2, py1 + 12), f"round {rd}", 14, MUTED, anchor="ma")
        if rd != rounds[-1]:
            xs = X(idx[-1] + 0.5)
            d.line([(xs, py0), (xs, py1)], fill=GRIDC, width=1)
    for i in range(0, n + 1):
        text(d, (X(i), py1 + 34), "start" if i == 0 else str(i), 13, MUTED, anchor="ma")
    text(d, ((px0 + px1) / 2, py1 + 54), "Experiment #", 14, INK2, anchor="ma")
    shown = [r for r in D["rows"] if r["i"] < cur or (r["i"] == cur and show_cur)]
    # running best: a step line from the start's bar, extended to the latest experiment shown
    pts, best = [(X(0), Y(D["bar"]))], D["bar"]
    for r in shown:
        if r["kept"]:
            pts += [(X(r["i"]), Y(best)), (X(r["i"]), Y(r["mean"]))]
            best = r["mean"]
    pts.append((X(shown[-1]["i"]) if shown else X(0), Y(best)))
    d.line(pts, fill=ARM_COL, width=3)
    for r in shown:
        if not r["ok"]:
            xx = X(r["i"])
            d.line([(xx - 6, Y(0) - 6), (xx + 6, Y(0) + 6)], fill=MUTED, width=2)
            d.line([(xx - 6, Y(0) + 6), (xx + 6, Y(0) - 6)], fill=MUTED, width=2)
        elif not r["kept"]:
            xx, yy = X(r["i"]), Y(r["mean"])
            d.ellipse([xx - 6, yy - 6, xx + 6, yy + 6], fill=OTHER)
    d.ellipse([X(0) - 8, Y(D["bar"]) - 8, X(0) + 8, Y(D["bar"]) + 8], fill=INK2, outline=SURF, width=2)
    text(d, (X(0) - 6, Y(D["bar"]) + 14), "start network", 14, INK2)
    for r in shown:
        if r["kept"]:
            xx, yy = X(r["i"]), Y(r["mean"])
            d.ellipse([xx - 9, yy - 9, xx + 9, yy + 9], fill=ARM_COL, outline=SURF, width=2)
            rot_text(img, (xx + 10, yy - 8), short(r["name"]), 14, ARM_COL)
    if show_cur:
        r = D["rows"][cur - 1]
        yy = Y(r["mean"]) if r["ok"] else Y(0)
        d.ellipse([X(cur) - 15, yy - 15, X(cur) + 15, yy + 15], outline=INK, width=2)
    # legend
    lx, ly = x0, y0 + 58
    d.ellipse([lx, ly + 2, lx + 14, ly + 16], fill=ARM_COL)
    text(d, (lx + 22, ly), "kept: a new best", 14, INK)
    d.ellipse([lx + 160, ly + 3, lx + 172, ly + 15], fill=OTHER)
    text(d, (lx + 182, ly), "discarded", 14, INK)
    d.line([(lx + 280, ly + 9), (lx + 300, ly + 9)], fill=ARM_COL, width=3)
    text(d, (lx + 308, ly), "running best", 14, INK)


def grids(d, D, r, fill):
    """Bottom-left: every dev design as a cell, filled if built exactly; the shown design is ringed."""
    x0, y0 = 40, 772
    text(d, (x0, y0), f"Evaluation: all {D['n_designs']} dev designs, filled = built exactly", 17, INK, bold=True)
    n_s = len(D["seeds"])
    cols, cell, gap, step = (32, 12, 1, 480) if n_s <= 2 else (32, 9, 1, 340)
    for s in range(n_s):
        gx, gy = x0 + s * step, y0 + 58
        g = r["grid"][s] if r["ok"] else [0] * D["n_designs"]
        shown = min(fill, len(g))
        got = sum(g[:shown])
        sd = D["seeds"][s]
        lab = f"Seed {sd} run: {100 * got / len(g):.1f}%" if r["ok"] else f"Seed {sd} run: failed"
        text(d, (gx, gy - 26), lab, 15, INK2)
        for k in range(len(g)):
            cy, cx = divmod(k, cols)
            xx, yy = gx + cx * (cell + gap), gy + cy * (cell + gap)
            col = ARM_COL if (k < shown and g[k]) else EMPTY
            d.rectangle([xx, yy, xx + cell - 1, yy + cell - 1], fill=col)
            if s == 0 and k == D["di"] and k < shown:
                d.rectangle([xx - 2, yy - 2, xx + cell + 1, yy + cell + 1], outline=INK, width=2)
    if r["ok"] and fill >= D["n_designs"]:
        what = "the seed's score" if n_s == 1 else ("mean of the two seeds" if n_s == 2 else f"mean of the {n_s} seeds")
        text(d, (x0, y0 + 274), f"Score = {what}: {100 * r['mean']:.1f}%. The ringed cell is the "
             "design built above.", 15, INK)


def frame(D, cur, nb, fill, show_cur, title, subtitle):
    img = Image.new("RGB", (W, H), SURF)
    d = ImageDraw.Draw(img)
    text(d, (40, 26), title, 30, INK, bold=True)
    text(d, (40, 68), subtitle, 17, INK2)
    r = D["rows"][cur - 1]
    text(d, (40, 120), f"Experiment {cur} of {D['n']}  ·  round {r['round']}", 20, INK, bold=True)
    text(d, (40, 148), r["name"], 16, INK2)
    tgt = D["target"]
    sp = D["spec"]
    text(d, (880, 120), "Target design", 14, INK2)
    st = 120 / ((sp.X + sp.Y) * 0.866)
    iso_t = R.Iso(sp, st, 940, 150 + sp.Z * st + 10)
    iso_t.ground(d)
    iso_t.bricks(d, R.target_bricks(tgt.numpy()), lambda b: R.TARGET_COL)
    sc = min(860 / ((sp.X + sp.Y) * 0.866), 470 / ((sp.X + sp.Y) * 0.5 + sp.Z))
    iso = R.Iso(sp, sc, 500, 200 + sp.Z * sc)
    iso.ground(d, footprint=tgt.any(0).numpy())
    if r["ok"]:
        bricks = r["bricks"][:nb]
        iso.bricks(d, bricks, lambda b: R.WRONG if b["wrong"] else ARM_COL)
        text(d, (40, 694), f"Seed-0 network building this design greedily: bricks placed {len(bricks)}", 15, INK2)
        if nb >= len(r["bricks"]) and fill > 0:
            R.badge(d, 52, 734, r["exact"])
            pl = lambda n: f"{n} cell{'' if n == 1 else 's'}"  # noqa: E731
            msg = "built exactly" if r["exact"] else f"not exact: {pl(r['missing'])} missing, {pl(r['extra'])} extra"
            text(d, (72, 724), msg, 16, INK)
    else:
        text(d, (40, 694), "This experiment failed (time limit or crash) and scores as failed.", 15, INK2)
    grids(d, D, r, fill)
    chart(img, d, D, cur, show_cur)
    return img


def titles(D, title=None, subtitle=None):
    sp = D["spec"]
    world_txt = "×".join(str(v) for v in (sp.X, sp.Y, sp.Z))
    title = title or f"An AI scientist's search: {D['n']} experiments, {D['n_kept']} kept improvements"
    wrote = "a network and its training recipe" if D.get("recipes") else "a network"
    subtitle = subtitle or (f"One search in the {world_txt} brick-assembly world. Each experiment: the AI scientist writes "
                            f"{wrote}, PPO trains it on {seeds_txt(len(D['seeds']))}, then it builds {D['n_designs']} dev "
                            "designs it never saw.")
    return title, subtitle


def video_frames(D, title, subtitle, seconds=None):
    """(image, repeat count) pairs: each experiment builds brick by brick, its evaluation fills in, then it holds.
    With seconds, bricks are skipped evenly and holds shortened so the whole fits the budget (at least 1 frame each)."""
    plan = []
    for r in D["rows"]:
        nb = len(r.get("bricks", []))
        plan.append((r, nb, 4, 8, 24 if r["kept"] else 10))
    natural = sum(4 + nb + 8 + hold for _, nb, _, _, hold in plan)
    f = 1.0 if seconds is None else min(1.0, seconds * FPS / natural)
    for r, nb, pre, fills, hold in plan:
        k = r["i"]
        yield frame(D, k, 0, 0, False, title, subtitle), max(1, round(pre * f))
        n_b = max(1, round(nb * f)) if nb else 0
        for j in range(1, n_b + 1):
            yield frame(D, k, round(j * nb / n_b), 0, False, title, subtitle), 1
        n_f = max(1, round(fills * f))
        for j in range(1, n_f + 1):
            yield frame(D, k, nb, D["n_designs"] * j // n_f, False, title, subtitle), 1
        yield frame(D, k, nb, D["n_designs"], True, title, subtitle), max(1, round(hold * f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", default=str(ROOT / "arch_rsi/campaigns/stage1/Q/g0/s2"))
    ap.add_argument("--baseline", default=str(ROOT / "arch_rsi/campaigns/stage1/baseline/summary.json"))
    ap.add_argument("--world", default="xl")
    ap.add_argument("--targets", default=str(ROOT / "arch_rsi/targets/xl_dev.pt"))
    ap.add_argument("--out", default=str(HERE / "proto"))
    ap.add_argument("--stills", type=int, nargs="*", default=[7, 12])
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = R.SPECS[a.world]
    designs = torch.load(a.targets)
    di = int(R.has_hidden_cells(designs).nonzero()[0, 0])  # fixed rule: first dev design with hidden cells
    D = load_search(Path(a.search), json.loads(Path(a.baseline).read_text()), a.world, designs, di, spec)
    title, subtitle = titles(D)
    for k in a.stills:
        r = D["rows"][k - 1]
        frame(D, k, len(r.get("bricks", [])), D["n_designs"], True, title, subtitle).save(out / f"still_exp{k:02d}.png")
    (out / "search_build_data.json").write_text(json.dumps(
        {"search": a.search, "design_index": di, "bar": D["bar"],
         "rows": [{k: v for k, v in r.items() if k not in ("bricks", "grid")} for r in D["rows"]]}, indent=1))
    if not a.video:
        return
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-threads", "2", "-pix_fmt",
                           "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out / "search_build.mp4")],
                          stdin=subprocess.PIPE)

    for img, times in video_frames(D, title, subtitle):
        for _ in range(times):
            ff.stdin.write(img.tobytes())
    ff.stdin.close()
    assert ff.wait() == 0


if __name__ == "__main__":
    main()
