"""VISUALIZATION ONLY: a video of three Stage 1 networks learning to build, from snapshots saved by
../train_snapshots.py (a line-for-line copy of the frozen CPU training loop that also keeps 11 weight snapshots).

Fixed rules, written before anything was rendered:
- Networks: the starting network, and one pick from each Claude arm: the search ranked 6th of 10 by its 8-seed
  confirmed score (the lower of the two middle searches): no-feedback search 3's r3c1 and g0 search 2's r4c1.
- Seed 1001, the first confirmation seed, on the 8x8x5 world (xl) at Stage 1's 131,072-step budget
  (compute-matched, so the two picks train for fewer steps).
- Design shown: the FIRST design in arch_rsi/targets/xl_dev.pt (dev split) that has hidden cells, as in the pilot video.
- Each snapshot builds with the frozen evaluator's rule (greedy argmax, all 512 dev designs at once). The dev score
  printed for each snapshot is exact success over those 512 designs.
- Check: each network's last snapshot must score exactly what its Stage 1 confirmation run (seed 1001) scored.

usage (project root): .venv/bin/python -I arch_rsi/viz/stage1/render_learning_stage1.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import load_candidate, sha  # noqa: E402
from arch_rsi.worlds import SPECS  # noqa: E402
from brickworld.env import BRICK_TYPES, BrickWorld, has_hidden_cells, rest_heights  # noqa: E402
from brickworld.model import encode  # noqa: E402
from evaluate import check  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
CONF = HERE.parents[1] / "campaigns" / "stage1" / "confirm"
NETS = [
    {"key": "start_s1001", "title": "Starting network", "short": "Start", "color": (110, 119, 129),
     "frozen_dev": CONF / "764c7fb529694085" / "seed1001" / "dev.json"},
    {"key": "nofbs3_s1001", "title": "Scores hidden: search 3's pick", "short": "Scores hidden", "color": (27, 175, 122),
     "frozen_dev": CONF / "f549c2a4e143595a" / "seed1001" / "dev.json"},
    {"key": "g0s2_s1001", "title": "Sees its scores: search 2's pick", "short": "Sees its scores", "color": (42, 120, 214),
     "frozen_dev": CONF / "20631875717dac9e" / "seed1001" / "dev.json"},
]
WRONG = (208, 59, 59)
TARGET_COL = (150, 158, 168)
GOOD, BAD = (12, 163, 12), (208, 59, 59)
INK, INK2, MUTED, GRIDC, SURF = (11, 11, 11), (82, 81, 78), (137, 135, 129), (225, 224, 217), (252, 252, 251)
W, H, FPS = 1920, 1080, 12
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
SYMBOLS = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def badge(d, cx, cy, ok, r=10):
    """Status icon (filled circle + tick or cross) so the colour never carries the meaning alone."""
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GOOD if ok else BAD)
    d.text((cx, cy + 1), "✓" if ok else "✕", font=ImageFont.truetype(SYMBOLS, int(r * 1.5)), fill=(255, 255, 255),
           anchor="mm")


# ---------------------------------------------------------------- evaluation (frozen evaluator's rule)
def evaluate_snapshot(src, state, designs, spec, di):
    aspec = arch_spec(spec)
    torch.manual_seed(0)
    policy = CandidatePolicy(load_candidate(src)(aspec), aspec)
    policy.load_state_dict(state)
    policy.eval()
    world = BrickWorld(spec, len(designs), "cpu", targets=designs, auto_reset=False)
    acts = []
    with torch.no_grad():
        for _ in range(spec.max_steps + 1):
            if bool(world.ended.all()):
                break
            logits, _ = policy(encode(world, "voxels"), world.valid_mask())
            a = logits.argmax(dim=1)
            acts.append((int(a[di]), bool(world.ended[di])))
            world.step(a)
    sc = check(world.occ.numpy(), designs.numpy())
    seq = [a for a, was_ended in acts if not was_ended]
    return float(sc["exact"].mean()), seq, world.occ[di].clone()


def replay(seq, target, spec):
    """Replay one design's actions; returns the bricks placed, how the episode ended and the final grid."""
    world = BrickWorld(spec, 1, "cpu", targets=target.unsqueeze(0), auto_reset=False)
    bricks, end = [], "no moves"
    for a in seq:
        if a == spec.done_action:
            world.step(torch.tensor([a]))
            end = "pressed DONE"
            break
        t, rem = divmod(a, spec.Y * spec.X)
        y, x = divmod(rem, spec.X)
        z = int(rest_heights(world.height, spec)[0, a])
        info = world.step(torch.tensor([a]))
        if bool(info["illegal"][0]):
            end = "illegal move"
            break
        dy, dx = BRICK_TYPES[t]
        cells = [(z, yy, xx) for yy in range(y, y + dy) for xx in range(x, x + dx)]
        bricks.append({"cells": cells, "wrong": any(not bool(target[c]) for c in cells)})
        if bool(world.ended[0]):
            end = "step limit"
            break
    return bricks, end, world.occ[0].clone()


# ---------------------------------------------------------------- isometric drawing
def shade(c, f):
    return tuple(int(max(0, min(255, v * f))) for v in c)


def mix(c, w, t):
    return tuple(int(a * (1 - t) + b * t) for a, b in zip(c, w))


class Iso:
    def __init__(self, spec, s, ox, oy):
        self.spec, self.s, self.ox, self.oy = spec, s, ox, oy
        self.a, self.b = s * 0.8660254, s * 0.5

    def p(self, x, y, z):
        return (self.ox + (x - y) * self.a, self.oy + (x + y) * self.b - z * self.s)

    def ground(self, d, footprint=None):
        X, Y = self.spec.X, self.spec.Y
        d.polygon([self.p(0, 0, 0), self.p(X, 0, 0), self.p(X, Y, 0), self.p(0, Y, 0)], fill=(240, 241, 238))
        if footprint is not None:
            for y in range(Y):
                for x in range(X):
                    if footprint[y, x]:
                        d.polygon([self.p(x, y, 0), self.p(x + 1, y, 0), self.p(x + 1, y + 1, 0), self.p(x, y + 1, 0)],
                                  fill=(226, 230, 236))
        for i in range(X + 1):
            d.line([self.p(i, 0, 0), self.p(i, Y, 0)], fill=GRIDC, width=1)
        for j in range(Y + 1):
            d.line([self.p(0, j, 0), self.p(X, j, 0)], fill=GRIDC, width=1)

    def bricks(self, d, bricks, color_of, lift=None):
        """bricks: list of dicts with cells (z, y, x); lift: (brick index, layers) draws that brick raised
        (one frame of a drop animation). Painter's order by x+y+z (true isometric view along -(1,1,1))."""
        cellsets = [set(b["cells"]) for b in bricks]
        owner = {}
        for i, b in enumerate(bricks):
            if lift and lift[0] == i:
                continue
            for c in b["cells"]:
                owner[c] = i
        cubes = []
        for i, b in enumerate(bricks):
            dz = lift[1] if lift and lift[0] == i else 0.0
            for (z, y, x) in b["cells"]:
                cubes.append((x + y + z + dz, z + dz, x, y, z, i, dz))
        cubes.sort(key=lambda c: (c[0], c[1]))
        p = self.p
        ew = max(1, round(self.s / 16))
        for _, _, x, y, z, i, dz in cubes:
            col = color_of(bricks[i])
            top, left, right, edge = mix(col, (255, 255, 255), 0.28), col, shade(col, 0.78), shade(col, 0.55)
            own = cellsets[i]

            def same(cx, cy, cz):
                return (cz, cy, cx) in own

            def occ(cx, cy, cz):
                return same(cx, cy, cz) if dz else (cz, cy, cx) in owner

            zb = z + dz
            zt = zb + 1
            show_top, show_r, show_l = not occ(x, y, z + 1), not occ(x + 1, y, z), not occ(x, y + 1, z)
            if show_r:
                d.polygon([p(x + 1, y, zt), p(x + 1, y + 1, zt), p(x + 1, y + 1, zb), p(x + 1, y, zb)], fill=right)
            if show_l:
                d.polygon([p(x, y + 1, zt), p(x + 1, y + 1, zt), p(x + 1, y + 1, zb), p(x, y + 1, zb)], fill=left)
            if show_top:
                d.polygon([p(x, y, zt), p(x + 1, y, zt), p(x + 1, y + 1, zt), p(x, y + 1, zt)], fill=top)
            # brick outlines: skip edges shared with another cell of the same brick
            if show_top:
                if not same(x, y - 1, z):
                    d.line([p(x, y, zt), p(x + 1, y, zt)], fill=edge, width=ew)
                if not same(x - 1, y, z):
                    d.line([p(x, y, zt), p(x, y + 1, zt)], fill=edge, width=ew)
                if not same(x + 1, y, z):
                    d.line([p(x + 1, y, zt), p(x + 1, y + 1, zt)], fill=edge, width=ew)
                if not same(x, y + 1, z):
                    d.line([p(x, y + 1, zt), p(x + 1, y + 1, zt)], fill=edge, width=ew)
            if show_r:
                d.line([p(x + 1, y, zb), p(x + 1, y + 1, zb)], fill=edge, width=ew)
                if not same(x, y + 1, z):
                    d.line([p(x + 1, y + 1, zt), p(x + 1, y + 1, zb)], fill=edge, width=ew)
                if not same(x, y - 1, z):
                    d.line([p(x + 1, y, zt), p(x + 1, y, zb)], fill=edge, width=ew)
            if show_l:
                d.line([p(x, y + 1, zb), p(x + 1, y + 1, zb)], fill=edge, width=ew)
                if not same(x - 1, y, z):
                    d.line([p(x, y + 1, zt), p(x, y + 1, zb)], fill=edge, width=ew)
                if not same(x + 1, y, z):
                    d.line([p(x + 1, y + 1, zt), p(x + 1, y + 1, zb)], fill=edge, width=ew)


def target_bricks(target):
    """The target as unit cells (its brick boundaries are not stored), drawn as one-cell blocks."""
    Z, Y, X = target.shape
    return [{"cells": [(z, y, x)], "wrong": False} for z in range(Z) for y in range(Y) for x in range(X) if target[z, y, x]]


# ---------------------------------------------------------------- frame composition
def text(d, xy, s, size, col=INK, bold=False, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=col, anchor=anchor)


def curve_panel(d, box, data, cur, floor):
    x0, y0, x1, y1 = box
    text(d, (x0, y0), "Built exactly, 512 dev designs, at each checkpoint", 17, INK2)
    px0, py0, px1, py1 = x0 + 52, y0 + 34, x1 - 250, y1 - 46
    for v in (0, 25, 50, 75, 100):
        yy = py1 - (py1 - py0) * v / 100
        d.line([(px0, yy), (px1, yy)], fill=GRIDC, width=1)
        text(d, (px0 - 8, yy), f"{v}%", 13, MUTED, anchor="rm")
    yf = py1 - (py1 - py0) * floor * 100 / 100
    d.line([(px0, yf), (px1, yf)], fill=(195, 194, 183), width=1)
    text(d, (px1 + 8, yf), f"floor {floor * 100:.0f}%: designs with no hidden cells", 13, MUTED, anchor="lm")
    for k in range(0, 11, 2):
        xx = px0 + (px1 - px0) * k / 10
        text(d, (xx, py1 + 6), f"{k * 10}%", 13, MUTED, anchor="ma")
    xc = px0 + (px1 - px0) * cur / 10
    d.line([(xc, py0 - 4), (xc, py1)], fill=(195, 194, 183), width=2)
    for n, net in zip(data, NETS):
        pts = [(px0 + (px1 - px0) * k / 10, py1 - (py1 - py0) * v) for k, v in enumerate(n["dev"][:cur + 1])]
        if len(pts) > 1:
            d.line(pts, fill=net["color"], width=3, joint="curve")
        for (xx, yy) in pts:
            d.ellipse([xx - 5, yy - 5, xx + 5, yy + 5], fill=net["color"], outline=SURF, width=2)
        xx, yy = pts[-1]
        d.ellipse([xx - 7, yy - 7, xx + 7, yy + 7], fill=net["color"], outline=SURF, width=2)
    # legend with current values (text in ink, colour on the swatch)
    ly = py0 + 6
    for n, net in zip(data, NETS):
        d.line([(px1 + 10, ly + 9), (px1 + 34, ly + 9)], fill=net["color"], width=3)
        text(d, (px1 + 42, ly), f"{net['short']}  {100 * n['dev'][cur]:.0f}%", 15, INK)
        ly += 26
    text(d, (px0, py1 + 26), "share of the training budget  (compute-matched: 100% = 131,072 steps for the start, "
         f"{data[1]['steps'][-1]:,} and {data[2]['steps'][-1]:,} for the two picks)", 13, MUTED)


def frame(data, target, k, nb, lift=None, show_result=False):
    img = Image.new("RGB", (W, H), SURF)
    d = ImageDraw.Draw(img)
    text(d, (40, 26), "Stage 1: three networks learning to build", 30, INK, bold=True)
    text(d, (40, 66), "One dev design (8×8×5 world), checkpoints every 10% of training, seed 1001 (the first fresh "
         "confirmation seed). At each checkpoint the network builds it greedily, brick by brick.", 17, INK2)
    # target panel
    text(d, (40, 112), "Target design", 18, INK, bold=True)
    text(d, (40, 136), f"{int(target.sum())} cells, with hidden cells under an overhang", 14, INK2)
    iso_t = Iso(SPEC, 22, 40 + 170, 330)
    iso_t.ground(d)
    iso_t.bricks(d, target_bricks(target.numpy()), lambda b: TARGET_COL)
    fp = target.any(0).numpy()
    for j, (n, net) in enumerate(zip(data, NETS)):
        bx = 420 + j * 500
        text(d, (bx, 112), net["title"], 18, INK, bold=True)
        steps = n["steps"][k]
        text(d, (bx, 136), f"checkpoint {k + 1} of 11: after {steps:,} training steps ({k * 10}%)", 14, INK2)
        iso = Iso(SPEC, 28, bx + 235, 360)
        iso.ground(d, footprint=fp)
        snap = n["snaps"][k]
        bricks = snap["bricks"][:nb]
        iso.bricks(d, bricks, lambda b, c=net["color"]: WRONG if b["wrong"] else c, lift=lift)
        text(d, (bx, 668), f"bricks placed: {len(bricks)}", 15, INK2)
        if show_result:
            ok = snap["exact"]
            msg = "built exactly" if ok else f"not exact: {snap['missing']} cells missing, {snap['extra']} extra"
            badge(d, bx + 150, 677, ok)
            text(d, (bx + 168, 668), msg, 15, INK)
            text(d, (bx + 168, 690), f"({snap['end']})", 14, INK2)
    text(d, (420, 722), "Red brick = it puts a cell outside the target. Visualization only, one seed: on its 12 fresh seeds the "
         "start learned the hidden-cell skill on 2 (mean 22.7%). On 8 fresh seeds, picks averaged 73.6% with scores hidden and 75.2% with them seen.", 13, MUTED)
    curve_panel(d, (40, 760, W - 40, H - 6), data, k, FLOOR)
    return img


# ---------------------------------------------------------------- main
SPEC = SPECS["xl"]
FLOOR = 0.0


def main():
    global FLOOR
    designs = torch.load(HERE.parents[1] / "targets" / "xl_dev.pt")
    hidden = has_hidden_cells(designs)
    di = int(hidden.nonzero()[0, 0])          # fixed rule: first dev design with hidden cells
    FLOOR = float((~hidden).float().mean())
    target = designs[di]
    data = []
    for net in NETS:
        rdir = RUNS / net["key"]
        src = (rdir / "candidate.py").read_text()
        ck = torch.load(rdir / "snapshots.pt")
        assert ck["candidate_sha"] == sha(src)
        log = json.loads((rdir / "train_log.json").read_text())
        n = {"dev": [], "steps": [], "snaps": [], "snap_at": log["snap_at"]}
        for u in log["snap_at"]:
            dev, seq, occ_batch = evaluate_snapshot(src, ck["snapshots"][u], designs, SPEC, di)
            bricks, end, occ = replay(seq, target, SPEC)
            assert torch.equal(occ, occ_batch), "replay differs from the batched evaluation"
            ex = bool((occ == target).all())
            n["dev"].append(dev)
            n["steps"].append(u * log["batch"])
            n["snaps"].append({"bricks": bricks, "end": end, "exact": ex,
                               "missing": int((target & ~occ).sum()), "extra": int((occ & ~target).sum())})
            print(net["key"], "update", u, "steps", u * log["batch"], "dev", round(dev, 4), "design", di,
                  "bricks", len(bricks), "exact", ex, end, flush=True)
        frozen = json.loads(net["frozen_dev"].read_text())["exact"]
        n["frozen_dev"] = frozen
        n["matches_frozen"] = abs(n["dev"][-1] - frozen) < 1e-9
        print(net["key"], "final dev", n["dev"][-1], "frozen run dev", frozen, "match", n["matches_frozen"])
        assert n["matches_frozen"], f"{net['key']}: last snapshot does not reproduce the Stage 1 confirmation run"
        data.append(n)

    summary = {"design_index": di, "design_rule": "first xl dev design with hidden cells", "floor": FLOOR,
               "nets": [{"key": net["key"], "snap_updates": n["snap_at"], "steps": n["steps"],
                         "dev_exact_512": [round(v, 4) for v in n["dev"]],
                         "design_exact": [s["exact"] for s in n["snaps"]],
                         "bricks": [len(s["bricks"]) for s in n["snaps"]], "end": [s["end"] for s in n["snaps"]],
                         "frozen_dev": n["frozen_dev"], "final_matches_frozen": n["matches_frozen"]}
                        for net, n in zip(NETS, data)]}
    (HERE / "learning_data.json").write_text(json.dumps(summary, indent=1))

    # video: per checkpoint, an intro hold, each brick dropping in (2 frames), a result hold
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                           "-movflags", "+faststart", str(HERE / "stage1_learning.mp4")], stdin=subprocess.PIPE)
    n_frames = 0

    def emit(img, times=1):
        nonlocal n_frames
        for _ in range(times):
            ff.stdin.write(img.tobytes())
            n_frames += 1

    for k in range(11):
        emit(frame(data, target, k, 0), 8)
        nmax = max(len(n["snaps"][k]["bricks"]) for n in data)
        for i in range(nmax):
            emit(frame(data, target, k, i + 1, lift=(i, 0.6)))
            emit(frame(data, target, k, i + 1))
        last = frame(data, target, k, nmax, show_result=True)
        emit(last, 30 if k == 10 else 18)
        if k in (0, 3, 6, 10):
            last.save(HERE / f"stage1_learning_ckpt{k + 1:02d}.png")
    ff.stdin.close()
    ff.wait()
    print("video frames", n_frames, "seconds", round(n_frames / FPS, 1))
    contact_sheet(data, target)


def contact_sheet(data, target):
    cw, ch = 168, 190
    L = 480  # left: target column (24-280) and row labels (300-470)
    Wc, Hc = L + 11 * cw + 20, 140 + 3 * ch + 40
    img = Image.new("RGB", (Wc, Hc), SURF)
    d = ImageDraw.Draw(img)
    text(d, (24, 18), "Stage 1: what each network builds at every 10% of training (one dev design, seed 1001)", 24, INK, bold=True)
    text(d, (24, 54), "Dev score under each build = exact success on all 512 dev designs at that checkpoint. "
         "Red = a brick with a cell outside the target. Visualization only.", 15, INK2)
    text(d, (24, 100), "Target", 16, INK, bold=True)
    iso = Iso(SPEC, 15, 24 + 128, 140 + ch + 20)
    iso.ground(d)
    iso.bricks(d, target_bricks(target.numpy()), lambda b: TARGET_COL)
    fp = target.any(0).numpy()
    for j, (n, net) in enumerate(zip(data, NETS)):
        y0 = 140 + j * ch
        for k in range(11):
            x0 = L + k * cw
            if j == 0:
                text(d, (x0 + cw / 2, 96), f"{k * 10}%", 15, INK2, anchor="ma")
            iso = Iso(SPEC, 9.6, x0 + cw / 2, y0 + 50)
            iso.ground(d, footprint=fp)
            s = n["snaps"][k]
            iso.bricks(d, s["bricks"], lambda b, c=net["color"]: WRONG if b["wrong"] else c)
            badge(d, x0 + cw / 2 - 44, y0 + 159, s["exact"], r=8)
            text(d, (x0 + cw / 2 - 30, y0 + 150), f"dev {100 * n['dev'][k]:.0f}%", 14, INK2)
        text(d, (300, y0 + 70), net["title"].replace(": ", ":\n"), 15, INK, bold=True)
    img.save(HERE / "stage1_learning_contact_sheet.png")


if __name__ == "__main__":
    main()
