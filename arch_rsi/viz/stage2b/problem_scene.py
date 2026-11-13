"""VISUALIZATION ONLY: the problem, in one scene. A target design, and two trained networks building it brick by brick:
the starting network, and the pick of the search the video shows next. Nothing here scores, selects or tests.

It is an illustration, and the frame says so. The design is chosen to show one success beside one failure:
  the first dev design with hidden cells that the pick's seed-0 network builds exactly and the start's doesn't,
read from each run's own per-design results. Both builds are greedy replays of the saved seed-0 checkpoints and must
agree with those results, or the script stops. The rates printed are each run's own (seed 0, all dev designs) and the
confirmed ones from results.json (fresh seeds).

usage (project root):
  nice -n 19 .venv/bin/python -I arch_rsi/viz/stage2b/problem_scene.py --search 15 [--video]
"""
import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAMP = ROOT / "arch_rsi" / "campaigns" / "stage2b"
_spec = importlib.util.spec_from_file_location("search_build", HERE / "search_build.py")
SB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SB)  # registers world W; reuses Stage 1's isometric drawing and replay
R, text = SB.R, SB.text
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

W, H, FPS = SB.W, SB.H, SB.FPS
INK, INK2, MUTED, SURF = SB.INK, SB.INK2, SB.MUTED, SB.SURF
START_COL, PICK_COL = (217, 130, 43), SB.ARM_COL   # the learning agent's orange for the start; blue for the pick


def load_run(run_dir: Path, target, spec, di: int) -> dict:
    """One saved seed-0 run: its score on all dev designs, and a replay of it building the chosen design."""
    dev = SB.dev_json(run_dir)
    src = (run_dir / "run" / "candidate.py").read_text()
    ck = torch.load(run_dir / "run" / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ck["candidate_sha"] == R.sha(src), run_dir
    bricks, end, occ = R.replay(SB.build_one(src, ck["state_dict"], target, spec), target, spec)
    exact = bool((occ == target).all())
    assert exact == bool(dev["per_design"]["exact"][di]), f"{run_dir}: the replay disagrees with its dev.json"
    return {"exact_all": dev["exact"], "bricks": bricks, "end": end, "exact": exact,
            "missing": int((target & ~occ).sum()), "extra": int((occ & ~target).sum())}


def load(search: int) -> dict:
    res = json.loads((CAMP / "results.json").read_text())
    assert res["primary_family"]["status"] == "final", "results.json isn't final"
    assert not (ROOT / "arch_rsi" / "stage2" / "FINAL_OPENED_W").exists()
    row = next(s for s in res["per_arm"]["g0"]["searches"] if s["search"] == search)
    spec = R.SPECS["w"]
    designs = torch.load(ROOT / "arch_rsi" / "stage2" / "targets" / "w_dev.pt")
    start_dir = CAMP / "baseline" / "w" / "seed0"
    pick_dir = CAMP / "W" / "g0" / f"s{search}" / "trials" / row["pick"] / "seed0"
    assert R.sha((pick_dir / "run" / "candidate.py").read_text()) == row["sha"], "the trial isn't the search's pick"
    s_ok = SB.dev_json(start_dir)["per_design"]["exact"]
    p_ok = SB.dev_json(pick_dir)["per_design"]["exact"]
    hidden = R.has_hidden_cells(designs)
    di = next(i for i in range(len(designs)) if bool(hidden[i]) and p_ok[i] and not s_ok[i])   # the stated rule
    target = designs[di]
    return {"spec": spec, "target": target, "di": di, "cells": int(target.sum()), "n_designs": len(designs),
            "start": load_run(start_dir, target, spec, di), "pick": load_run(pick_dir, target, spec, di),
            "start_fresh": res["per_arm"]["start_g"]["mean"], "start_seeds": res["per_arm"]["start_g"]["seeds"],
            "pick_fresh": row["primary_score"], "pick_seeds": len(row["per_seed"]), "search": search,
            "pick_label": row["pick"]}


def column(d, D, cx, head, sub, run, nb, col, done):
    sp = D["spec"]
    text(d, (cx, 138), head, 20, INK, bold=True, anchor="ma")
    text(d, (cx, 168), sub, 15, INK2, anchor="ma")
    sc = min(540 / ((sp.X + sp.Y) * 0.866), 480 / ((sp.X + sp.Y) * 0.5 + sp.Z))
    iso = R.Iso(sp, sc, cx, 206 + sp.Z * sc)
    if run is None:
        iso.ground(d)
        iso.bricks(d, R.target_bricks(D["target"].numpy()), lambda b: R.TARGET_COL)
        text(d, (cx, 700), f"{D['cells']} cells, some hidden under other bricks", 16, INK2, anchor="ma")
        return
    iso.ground(d, footprint=D["target"].any(0).numpy())
    bricks = run["bricks"][:nb]
    iso.bricks(d, bricks, lambda b: R.WRONG if b["wrong"] else col)
    text(d, (cx, 700), f"bricks placed: {len(bricks)}", 16, INK2, anchor="ma")
    if done:
        pl = lambda n: f"{n} cell{'' if n == 1 else 's'}"  # noqa: E731
        msg = "built exactly" if run["exact"] else f"not exact: {pl(run['missing'])} missing, {pl(run['extra'])} extra"
        w = R.font(17).getlength(msg)
        R.badge(d, cx - w / 2 - 16, 742, run["exact"])
        text(d, (cx - w / 2 + 4, 731), msg, 17, INK)


def frame(D, nb, done):
    img = Image.new("RGB", (W, H), SURF)
    d = ImageDraw.Draw(img)
    sp = D["spec"]
    text(d, (40, 26), "The problem: build the target design exactly, one brick at a time", 30, INK, bold=True)
    text(d, (40, 68), f"A {sp.X}×{sp.Y}×{sp.Z} brick-assembly world. A network sees the target and what is built so far, then "
                      "places one brick or presses DONE. Only an exact match counts.", 17, INK2)
    column(d, D, 330, "Target design", "one of the dev designs", None, 0, None, False)
    column(d, D, 960, "The starting network", "trained by reinforcement learning (PPO)", D["start"], nb, START_COL, done)
    column(d, D, 1590, "A network the AI scientist wrote", "PPO again, with its own training recipe, at equal compute", D["pick"], nb,
           PICK_COL, done)
    if done:
        y = 820
        d.line([(40, y - 22), (W - 40, y - 22)], fill=(225, 224, 217), width=1)
        text(d, (40, y), "How often each builds a design exactly, over all "
                         f"{D['n_designs']} dev designs:", 18, INK, bold=True)
        text(d, (40, y + 34), f"These two runs (seed 0): the start {D['start']['exact_all'] * 100:.1f}%, the AI scientist's "
                              f"network {D['pick']['exact_all'] * 100:.1f}%.", 17, INK)
        text(d, (40, y + 62), f"On fresh seeds: the start {D['start_fresh'] * 100:.1f}% ({D['start_seeds']} seeds), the AI "
                              f"scientist's network {D['pick_fresh'] * 100:.1f}% ({D['pick_seeds']} seeds).", 17, INK)
        text(d, (40, y + 106), "The AI scientist's job: write a network and a training recipe that build more designs "
                               "exactly, at the same compute.", 17, INK2)
        text(d, (40, y + 142), "An illustration, not evidence: this is the first dev design with hidden cells that this "
                               "network builds and the start doesn't. The test comes next.", 15, MUTED)
    return img


def frames(D):
    """(image, repeat count) pairs: the target alone, both builds brick by brick, then the result held."""
    n = max(len(D["start"]["bricks"]), len(D["pick"]["bricks"]))
    out = [(frame(D, 0, False), 2 * FPS)]
    out += [(frame(D, k, False), 4) for k in range(1, n + 1)]
    out.append((frame(D, n, True), 7 * FPS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=int, required=True)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    D = load(a.search)
    fr = frames(D)
    fr[-1][0].save(out / "problem_still.png")
    secs = sum(t for _, t in fr) / FPS
    data = {"search": a.search, "pick": D["pick_label"], "design_index": D["di"], "cells": D["cells"],
            "rule": "first dev design with hidden cells that the pick's seed-0 run builds exactly and the start's doesn't",
            "start": {k: v for k, v in D["start"].items() if k != "bricks"} | {"bricks": len(D["start"]["bricks"])},
            "pick_run": {k: v for k, v in D["pick"].items() if k != "bricks"} | {"bricks": len(D["pick"]["bricks"])},
            "start_fresh": D["start_fresh"], "pick_fresh": D["pick_fresh"], "seconds": round(secs, 1)}
    (out / "problem_data.json").write_text(json.dumps(data, indent=1))
    print(json.dumps(data, indent=1))
    if not a.video:
        return
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-crf",
                           "18", "-movflags", "+faststart", str(out / "problem.mp4")], stdin=subprocess.PIPE)
    for img, times in fr:
        for _ in range(times):
            ff.stdin.write(img.tobytes())
    ff.stdin.close()
    assert ff.wait() == 0


if __name__ == "__main__":
    main()
