"""Loose-brick physics for brick-world builds: a static stability LP and MuJoCo replays.

A brick is {t, y, x, z, dy, dx} in lattice units (from rollout.py). Physical sizes follow
PLAN.md: 8 mm grid pitch, 9.6 mm tall, footprints 0.2 mm under the grid, ABS density,
friction 0.4.
"""
import numpy as np
import mujoco
from scipy.optimize import linprog

PITCH = 0.008          # m per lattice cell, x and y
HEIGHT = 0.0096        # m per layer
GAP = 0.0002           # footprint is this much smaller than the grid in x and y
DENSITY = 1050.0       # kg/m^3, ABS
FRICTION = 0.4
TIMESTEP = 0.0002
EPS = 0.01             # corner-force margin, in units of a 1x1 brick's weight

POS_TOL = 0.001        # m
TILT_TOL = np.deg2rad(5.0)
SETTLE = 0.25          # s after each brick
FINAL_SETTLE = 1.0     # s after the last brick
LIFT = 0.0005          # m above the lattice rest position at release
JITTER = 0.0003        # m, uniform +- in x and y


# ---------------------------------------------------------------- static analysis

def _patches(bricks):
    """Contact patches: (upper index, lower index or -1 for ground, x0, x1, y0, y1)."""
    out = []
    for i, b in enumerate(bricks):
        bx0, bx1, by0, by1 = b["x"], b["x"] + b["dx"], b["y"], b["y"] + b["dy"]
        if b["z"] == 0:
            out.append((i, -1, bx0, bx1, by0, by1))
            continue
        for j, c in enumerate(bricks[:i]):
            if c["z"] != b["z"] - 1:
                continue
            x0, x1 = max(bx0, c["x"]), min(bx1, c["x"] + c["dx"])
            y0, y1 = max(by0, c["y"]), min(by1, c["y"] + c["dy"])
            if x1 > x0 and y1 > y0:
                out.append((i, j, x0, x1, y0, y1))
    return out


def _feasible(bricks, patches, lb):
    n = len(bricks)
    corners = []
    for p in patches:
        i, j, x0, x1, y0, y1 = p
        for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            corners.append((i, j, cx, cy))
    A = np.zeros((3 * n, len(corners)))
    b = np.zeros(3 * n)
    for k, (i, j, cx, cy) in enumerate(corners):
        for who, sign in ((i, 1.0), (j, -1.0)):
            if who < 0:
                continue
            A[3 * who, k] += sign
            A[3 * who + 1, k] += sign * cx
            A[3 * who + 2, k] += sign * cy
    for i, br in enumerate(bricks):
        w = br["dx"] * br["dy"]
        b[3 * i] = w
        b[3 * i + 1] = w * (br["x"] + br["dx"] / 2)
        b[3 * i + 2] = w * (br["y"] + br["dy"] / 2)
    res = linprog(np.zeros(len(corners)), A_eq=A, b_eq=b, bounds=[(lb, None)] * len(corners),
                  method="highs")
    return res.status == 0


def classify(bricks):
    """'stable', 'marginal' or 'unstable' for loose rigid bricks under gravity."""
    if not bricks:
        return "stable"
    patches = _patches(bricks)
    if any(not any(p[0] == i for p in patches) for i in range(len(bricks))):
        return "unstable"  # a brick touching nothing below (cannot happen under lattice rules)
    if _feasible(bricks, patches, EPS):
        return "stable"
    if _feasible(bricks, patches, 0.0):
        return "marginal"
    return "unstable"


def classify_prefixes(bricks):
    return [classify(bricks[:k]) for k in range(1, len(bricks) + 1)]


def order_class(prefix_classes):
    """'stable' if every prefix is stable, 'unstable' if any prefix is, else 'marginal'."""
    if "unstable" in prefix_classes:
        return "unstable"
    if "marginal" in prefix_classes:
        return "marginal"
    return "stable"


# ---------------------------------------------------------------- MuJoCo

def lattice_pos(b, spec):
    """Centre of a brick in metres, grid centred on the origin."""
    cx = (b["x"] + b["dx"] / 2 - spec["X"] / 2) * PITCH
    cy = (b["y"] + b["dy"] / 2 - spec["Y"] / 2) * PITCH
    cz = (b["z"] + 0.5) * HEIGHT
    return np.array([cx, cy, cz])


def scene_xml(bricks, spec, colors=None, ghost=None, ghost_rgba="0.55 0.58 0.62 0.16",
              floor_rgba="0.94 0.94 0.95 1", welds=False):
    colors = colors or ["0.16 0.47 0.84 1"] * len(bricks)
    parts = [f"""<mujoco model="brickworld">
  <option timestep="{TIMESTEP}" integrator="implicitfast" cone="elliptic" gravity="0 0 -9.81"/>
  <visual><headlight ambient="0.42 0.42 0.42" diffuse="0.5 0.5 0.5" specular="0.05 0.05 0.05"/>
    <quality shadowsize="4096"/><global offwidth="1920" offheight="1920"/></visual>
  <default><geom friction="{FRICTION} 0.005 0.0001" solref="0.001 1" solimp="0.95 0.99 0.0005" condim="3"/></default>
  <worldbody>
    <light pos="-0.05 -0.08 0.25" dir="0.2 0.32 -1" diffuse="0.55 0.55 0.55" castshadow="true"/>
    <geom name="floor" type="plane" size="1 1 0.01" rgba="{floor_rgba}"/>"""]
    if ghost is not None:
        Z, Y, X = len(ghost), len(ghost[0]), len(ghost[0][0])
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    if ghost[z][y][x]:
                        c = lattice_pos({"x": x, "y": y, "z": z, "dx": 1, "dy": 1}, spec)
                        parts.append(f'    <geom type="box" pos="{c[0]} {c[1]} {c[2]}" size="{PITCH/2*0.96} {PITCH/2*0.96} {HEIGHT/2*0.96}" '
                                     f'rgba="{ghost_rgba}" contype="0" conaffinity="0" group="1"/>')
    for i, b in enumerate(bricks):
        park = np.array([0.5 + 0.05 * i, 0.5, HEIGHT / 2])
        hx = (b["dx"] * PITCH - GAP) / 2
        hy = (b["dy"] * PITCH - GAP) / 2
        parts.append(f'    <body name="b{i}" pos="{park[0]} {park[1]} {park[2]}"><freejoint/>'
                     f'<geom type="box" size="{hx} {hy} {HEIGHT/2}" density="{DENSITY}" rgba="{colors[i]}"/></body>')
    parts.append("  </worldbody>")
    if welds:
        parts.append("  <equality>" + "".join(f'<weld body1="b{i}" active="false"/>' for i in range(len(bricks))) + "</equality>")
    parts.append("</mujoco>")
    return "\n".join(parts)


def _tilt(quat):
    """Angle between the brick's z axis and world z."""
    w, x, y, z = quat
    zz = 1 - 2 * (x * x + y * y)
    return float(np.arccos(np.clip(zz, -1, 1)))


def replay(bricks, spec, seed=0, welded=False, frame_cb=None, frame_dt=1 / 30,
           settle=SETTLE, final_settle=FINAL_SETTLE, lift=LIFT, jitter=JITTER, offsets=None, **xml_kw):
    """Place bricks one by one in MuJoCo. Returns (stands, per-brick errors, offsets used)."""
    m = mujoco.MjModel.from_xml_string(scene_xml(bricks, spec, welds=welded, **xml_kw))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(seed)
    if offsets is None:
        offsets = rng.uniform(-jitter, jitter, size=(len(bricks), 2))
    targets = [lattice_pos(b, spec) for b in bricks]
    next_frame = [0.0]

    def run(seconds):
        n = int(round(seconds / TIMESTEP))
        for _ in range(n):
            mujoco.mj_step(m, d)
            if frame_cb is not None and d.time >= next_frame[0]:
                frame_cb(m, d)
                next_frame[0] += frame_dt

    for i, b in enumerate(bricks):
        jid = m.body(f"b{i}").jntadr[0]
        qa, va = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        p = targets[i].copy()
        p[:2] += offsets[i]
        p[2] += lift
        d.qpos[qa:qa + 3] = p
        d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        d.qvel[va:va + 6] = 0
        mujoco.mj_forward(m, d)
        run(settle)
        if welded:
            # snap: weld the brick to the world at its exact lattice pose
            eq = i
            m.eq_data[eq, :] = 0
            # body1 = brick, body2 = world: relpose is the world's pose in the brick frame
            m.eq_data[eq, 3:6] = -targets[i]
            m.eq_data[eq, 6:10] = [1, 0, 0, 0]
            m.eq_data[eq, 10] = 1.0
            d.eq_active[eq] = 1
            run(0.05)
    run(final_settle)
    errs = []
    for i in range(len(bricks)):
        jid = m.body(f"b{i}").jntadr[0]
        qa = m.jnt_qposadr[jid]
        errs.append((float(np.linalg.norm(d.qpos[qa:qa + 3] - targets[i])), _tilt(d.qpos[qa + 3:qa + 7])))
    stands = all(e <= POS_TOL and t <= TILT_TOL for e, t in errs)
    return stands, errs, offsets
