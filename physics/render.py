"""Replay videos and stills of brick-world builds in MuJoCo.

Illustration only: bricks are lowered kinematically to 0.5 mm above their lattice rest
position, then either snapped and welded there ("welded", what the lattice rules assume)
or released as loose rigid bodies with the same offsets as the physics check ("loose").
  .venv/bin/python render.py
"""
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import brickphys as bp

HERE = Path(__file__).parent
OUT = HERE / "media"
SPECS = {"small": {"X": 5, "Y": 5, "Z": 3}, "large": {"X": 6, "Y": 6, "Z": 4}}

W, H = 600, 540           # px per panel
HEAD, FOOT = 104, 46     # px header and footer
FPS = 30
SLOWMO = 3               # video plays at 1/3 real time
FRAME_DT = 1 / (FPS * SLOWMO)
DESCEND, DESCEND_SPEED = 0.018, 0.15  # m, m/s
SETTLE_WELD = 0.12

INK, INK2, MUTED = (31, 35, 40), (87, 96, 106), (128, 136, 145)
BG = (255, 255, 255)
SONNET = "0.165 0.471 0.839 1"     # #2a78d6
START = "0.431 0.467 0.506 1"      # #6E7781
LARGE = "0.922 0.408 0.204 1"      # #eb6834 (used for the large-world clip)
WRONG = "0.816 0.231 0.231 1"      # #d03b3b, a brick with a cell outside the target

FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
SYMBOLS = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def seed_of(scale, agent, di, k=0):
    return int.from_bytes(hashlib.sha256(f"{scale}/{agent}/{di}/{k}".encode()).digest()[:4], "little")


def grid_lines(spec, z=0.00003, w=0.00018, rgba="0.72 0.75 0.79 1"):
    X, Y = spec["X"], spec["Y"]
    hx, hy = X * bp.PITCH / 2, Y * bp.PITCH / 2
    g = [f'<geom type="box" pos="0 0 -0.0004" size="{hx + 0.002} {hy + 0.002} 0.0004" rgba="0.88 0.89 0.91 1" contype="0" conaffinity="0"/>']
    for i in range(X + 1):
        x = -hx + i * bp.PITCH
        g.append(f'<geom type="box" pos="{x} 0 {z}" size="{w/2} {hy} {z}" rgba="{rgba}" contype="0" conaffinity="0"/>')
    for j in range(Y + 1):
        y = -hy + j * bp.PITCH
        g.append(f'<geom type="box" pos="0 {y} {z}" size="{hx} {w/2} {z}" rgba="{rgba}" contype="0" conaffinity="0"/>')
    return "\n    ".join(g)


class Sim:
    """One build replayed step by step, rendered to frames on a fixed sim-time clock."""

    def __init__(self, bricks, spec, target, color, welded, offsets, ghost=True):
        self.bricks, self.spec, self.welded = bricks, spec, welded
        tgt = np.array(target, dtype=bool)
        colors = []
        base = np.array([float(v) for v in color.split()[:3]])
        for i, b in enumerate(bricks):
            cells = tgt[b["z"], b["y"]:b["y"] + b["dy"], b["x"]:b["x"] + b["dx"]]
            f = 0.3 * (1 - i / max(len(bricks) - 1, 1))
            c = base * (1 - f) + f
            colors.append(" ".join(f"{v:.3f}" for v in c) + " 1" if cells.all() else WRONG)
        xml = bp.scene_xml(bricks, spec, colors=colors, ghost=target if ghost else None, welds=welded,
                           floor_rgba="0.97 0.97 0.98 1")
        xml = xml.replace('<geom name="floor"', grid_lines(spec) + '\n    <geom name="floor"', 1)
        xml = xml.replace('type="plane" size="1 1 0.01" rgba="0.97 0.97 0.98 1"', 'type="plane" size="1 1 0.01" rgba="1 1 1 0"', 1)
        disc = 0.16
        xml = xml.replace('<geom name="floor"', f'<geom type="cylinder" pos="0 0 -0.0009" size="{disc} 0.0005" rgba="1 1 1 1" contype="0" conaffinity="0"/>\n    <geom name="floor"', 1)
        xml = xml.replace('<visual>', '<visual><rgba haze="1 1 1 1"/>', 1)
        xml = xml.replace('<light pos=', '<light directional="true" pos=', 1)
        xml = xml.replace('<worldbody>', '<asset><texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/></asset>\n  <worldbody>', 1)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.r = mujoco.Renderer(self.m, H, W)
        self.cam = mujoco.MjvCamera()
        span = max(spec["X"], spec["Y"]) * bp.PITCH
        self.cam.lookat[:] = [0, 0, spec["Z"] * bp.HEIGHT * 0.22]
        self.cam.distance = span * 2.25
        self.cam.azimuth, self.cam.elevation = 128, -30
        self.offsets = offsets
        self.targets = [bp.lattice_pos(b, spec) for b in bricks]
        self.frames, self.next = [], 0.0
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[1] = 1

    def snap(self):
        self.r.update_scene(self.d, self.cam, self.opt)
        self.frames.append(self.r.render().copy())

    def step(self, hold=None):
        if hold is not None:
            qa, va, p = hold
            self.d.qpos[qa:qa + 3] = p
            self.d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
            self.d.qvel[va:va + 6] = 0
        mujoco.mj_step(self.m, self.d)
        if self.d.time >= self.next:
            self.snap()
            self.next += FRAME_DT

    def run(self, seconds):
        for _ in range(int(round(seconds / bp.TIMESTEP))):
            self.step()

    def place(self, i):
        jid = self.m.body(f"b{i}").jntadr[0]
        qa, va = self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid]
        rest = self.targets[i].copy()
        if not self.welded:
            rest[:2] += self.offsets[i]
        rest[2] += bp.LIFT
        n = int(round(DESCEND / DESCEND_SPEED / bp.TIMESTEP))
        for k in range(n):
            p = rest.copy()
            p[2] += DESCEND * (1 - (k + 1) / n)
            self.step(hold=(qa, va, p))
        if self.welded:
            self.d.qpos[qa:qa + 3] = self.targets[i]
            self.d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
            self.d.qvel[va:va + 6] = 0
            self.m.eq_data[i, :] = 0
            self.m.eq_data[i, 3:6] = -self.targets[i]
            self.m.eq_data[i, 6:10] = [1, 0, 0, 0]
            self.m.eq_data[i, 10] = 1.0
            self.d.eq_active[i] = 1
            mujoco.mj_forward(self.m, self.d)
            self.run(SETTLE_WELD)
        else:
            self.d.qvel[va:va + 6] = 0
            self.run(bp.SETTLE)

    def play(self, tail=1.0):
        self.snap()
        self.run(0.15)
        for i in range(len(self.bricks)):
            self.place(i)
        self.run(tail)
        errs = []
        for i in range(len(self.bricks)):
            qa = self.m.jnt_qposadr[self.m.body(f"b{i}").jntadr[0]]
            errs.append((np.linalg.norm(self.d.qpos[qa:qa + 3] - self.targets[i]), bp._tilt(self.d.qpos[qa + 3:qa + 7])))
        self.stands = all(e <= bp.POS_TOL and t <= bp.TILT_TOL for e, t in errs)
        return self.frames


def panel(img, title, sub, status, status_ok):
    canvas = Image.new("RGB", (W, H + HEAD + FOOT), BG)
    canvas.paste(Image.fromarray(img), (0, HEAD))
    dr = ImageDraw.Draw(canvas)
    dr.text((24, 18), title, font=font(30, True), fill=INK)
    dr.text((24, 58), sub, font=font(21), fill=INK2)
    if status:
        col = (26, 127, 55) if status_ok else (192, 41, 41)
        mark = "✓" if status_ok else "✗"
        dr.text((24, H + HEAD + 6), mark, font=ImageFont.truetype(SYMBOLS, 25), fill=col)
        dr.text((54, H + HEAD + 8), status, font=font(23, True), fill=col)
    return canvas


def compose(panels, footer):
    w = sum(p.width for p in panels) + 16 * (len(panels) - 1)
    out = Image.new("RGB", (w, panels[0].height + 44), BG)
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + 16
    ImageDraw.Draw(out).text((24, panels[0].height + 8), footer, font=font(19), fill=MUTED)
    return out


def clip(sims, labels, footer, status_fn):
    frames = [s.play() for s in sims]
    n = max(len(f) for f in frames)
    out = []
    for k in range(n):
        ps = []
        for s, f, (title, sub) in zip(sims, frames, labels):
            img = f[min(k, len(f) - 1)]
            done = k >= len(f) - 1
            st, ok = status_fn(s) if done else ("", True)
            ps.append(panel(img, title, sub, st, ok))
        out.append(np.array(compose(ps, footer)))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    ro = json.loads((HERE / "rollouts.json").read_text())
    E, D = ro["episodes"], ro["designs"]
    pick = json.loads((HERE / "picks.json").read_text())
    meta = {"picks": pick, "clips": {}}

    # Clip 1: same design, two recipes, lattice rules (welded)
    di = pick["clip1_design"]
    tgt = D["small"][di]["occ"]
    a = E["small/baseline-s0"][di]
    b = E["small/sonnet-r0-t09"][di]
    s1 = Sim(a["bricks"], SPECS["small"], tgt, START, True, None)
    s2 = Sim(b["bricks"], SPECS["small"], tgt, SONNET, True, None)
    exact = {id(s1): a["exact"], id(s2): b["exact"]}
    c1 = clip([s1, s2], [("Starting recipe", "This agent: 29% exact on 1,024 hidden designs"),
                         ("Sonnet's winning recipe", "This agent: 99% exact on 1,024 hidden designs")],
              "MuJoCo replay of trained agents, lattice rules: each brick snaps where it lands. Grey ghost = target.",
              lambda s: ("Exact match" if exact[id(s)] else "Not an exact match", exact[id(s)]))
    Image.fromarray(s1.frames[-1]).save(OUT / "clip1_start_raw.png")
    Image.fromarray(s2.frames[-1]).save(OUT / "clip1_sonnet_raw.png")
    meta["clips"]["clip1"] = {"design": di, "frames": len(c1),
                              "agents": ["baseline-s0", "sonnet-r0-t09"], "exact": [a["exact"], b["exact"]]}

    # Clip 2: the same exact build, welded vs loose (check's seed-0 offsets)
    dj, agent = pick["clip2_design"], pick["clip2_agent"]
    e = E[f"small/{agent}"][dj]
    rng = np.random.default_rng(seed_of("small", agent, dj, 0))
    offs = rng.uniform(-bp.JITTER, bp.JITTER, size=(len(e["bricks"]), 2))
    tgt2 = D["small"][dj]["occ"]
    w = Sim(e["bricks"], SPECS["small"], tgt2, SONNET, True, None)
    lo = Sim(e["bricks"], SPECS["small"], tgt2, SONNET, False, offs)
    for s_ in (w, lo):
        s_.cam.distance *= 1.3
    c2 = clip([w, lo], [("Bricks snap in place", "What the lattice rules assume"),
                        ("Loose bricks", "Same build, real rigid bodies")],
              "Sonnet's winning recipe, one exact build. Loose bricks released with ±0.3 mm offsets, as in the physics check.",
              lambda s: ("Stands" if s.stands else "Falls", s.stands))
    Image.fromarray(w.frames[-1]).save(OUT / "clip2_welded_raw.png")
    Image.fromarray(lo.frames[-1]).save(OUT / "clip2_loose_raw.png")
    meta["clips"]["clip2"] = {"design": dj, "agent": agent, "frames": len(c2),
                              "welded_stands": w.stands, "loose_stands": lo.stands}

    # Clip 3: the large world, best transfer recipe, welded
    dk = pick["clip3_design"]
    tgt3 = D["large"][dk]["occ"]
    a3 = E["large/xfer-large-00-s0"][dk]
    b3 = E["large/xfer-large-01-s0"][dk]
    t1 = Sim(a3["bricks"], SPECS["large"], tgt3, START, True, None)
    t2 = Sim(b3["bricks"], SPECS["large"], tgt3, LARGE, True, None)
    ex3 = {id(t1): a3["exact"], id(t2): b3["exact"]}
    c3 = clip([t1, t2], [("Starting recipe, larger world", "This agent: 20% exact on 1,024 hidden designs"),
                         ("Best recipe, larger world", "This agent: 97% exact on 1,024 hidden designs")],
              "6×6×4 world with a policy twice as wide. Lattice rules: each brick snaps where it lands.",
              lambda s: ("Exact match" if ex3[id(s)] else "Not an exact match", ex3[id(s)]))
    meta["clips"]["clip3"] = {"design": dk, "frames": len(c3),
                              "agents": ["xfer-large-00-s0", "xfer-large-01-s0"], "exact": [a3["exact"], b3["exact"]]}

    hold = FPS * 2
    allf = []
    for c in (c1, c2, c3):
        allf += c + [c[-1]] * hold
    imageio.mimwrite(OUT / "brickworld_replay.mp4", allf, fps=FPS, codec="libx264", quality=8,
                     macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for name, c in (("clip1_two_recipes", c1), ("clip2_loose_bricks", c2), ("clip3_larger_world", c3)):
        imageio.mimwrite(OUT / f"{name}.mp4", c + [c[-1]] * hold, fps=FPS, codec="libx264", quality=8,
                         macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
        Image.fromarray(c[-1]).save(OUT / f"{name}_final.png")
        small = [Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2), Image.LANCZOS) for f in c[::3]]
        small += [small[-1]] * 20
        small[0].save(OUT / f"{name}.gif", save_all=True, append_images=small[1:], duration=100, loop=0,
                      optimize=True)
    (OUT / "render_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
