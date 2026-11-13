"""Replay trained agents on fresh designs and record every placement.

Runs in the experiment's own venv (read-only use of the frozen code and checkpoints):
  .venv/bin/python physics/rollout.py

Writes physics/rollouts.json:
  designs[scale]: fresh designs (seed 3003, none in dev or final), each with the occupancy
                  grid and the generator's own bricks (the reference way to build it)
  episodes[agent]: per design, the agent's legal placements in order and whether the
                   finished build matched the target exactly (lattice rules)
"""
import json
from pathlib import Path

import numpy as np
import torch

from brickworld.env import (BRICK_TYPES, SCALES, BrickWorld, design_hash, generate_designs,
                            rest_heights, zobrist_codes)
from brickworld.model import Policy, encode

ROOT = Path(__file__).resolve().parent.parent
TRIALS = ROOT / "campaigns" / "main" / "trials"
SEED, N = 3003, 512

AGENTS = {
    "small": {
        "baseline": [f"baseline-s{i}" for i in range(5)],
        "random_k32": ["random-r0-t01", "random-r1-t11", "random-r2-t13", "random-r3-t22", "random-r4-t22"],
        "sonnet_k32": ["sonnet-r0-t09", "sonnet-r1-t13", "sonnet-r2-t06", "sonnet-r3-t22", "sonnet-r4-t00"],
        "opus_k32": ["opus-r0-t05", "opus-r1-t17", "opus-r2-t04", "opus-r3-t06", "opus-r4-t17"],
    },
    "large": {
        "baseline": ["xfer-large-00-s0", "xfer-large-00-s1"],
        "best_transfer": ["xfer-large-01-s0", "xfer-large-01-s1"],  # random-r4-t22 retrained at 6x6x4
    },
}


def fresh_designs(spec):
    """N unique designs not in dev or final, with the generator's bricks."""
    held = set(torch.load(ROOT / "targets" / f"{spec.name}_heldout_hashes.pt").tolist())
    codes = zobrist_codes(spec)
    gen = torch.Generator().manual_seed(SEED)
    keep, seen = [], set()
    while len(keep) < N:
        occ, acts = generate_designs(spec, 4 * N, gen, return_actions=True)
        for i, h in enumerate(design_hash(occ, codes).tolist()):
            if h in held or h in seen:
                continue
            seen.add(h)
            keep.append((occ[i], acts[i]))
            if len(keep) == N:
                break
    out = []
    for occ, acts in keep:
        bricks = replay([a for a in acts.tolist() if a >= 0], spec)
        assert np.array_equal(cells_of(bricks, spec), occ.numpy()), "generator replay mismatch"
        out.append({"occ": occ.numpy().astype(int).tolist(), "bricks": bricks})
    return out


def decode(a, spec):
    yx = spec.Y * spec.X
    t, pos = divmod(a, yx)
    y, x = divmod(pos, spec.X)
    dy, dx = BRICK_TYPES[t]
    return t, y, x, dy, dx


def replay(actions, spec):
    """Lattice drop rule in NumPy: list of bricks {t,y,x,z,dy,dx}."""
    h = np.zeros((spec.Y, spec.X), dtype=int)
    bricks = []
    for a in actions:
        t, y, x, dy, dx = decode(a, spec)
        z = int(h[y:y + dy, x:x + dx].max())
        assert z < spec.Z and y + dy <= spec.Y and x + dx <= spec.X
        h[y:y + dy, x:x + dx] = z + 1
        bricks.append({"t": t, "y": y, "x": x, "z": z, "dy": dy, "dx": dx})
    return bricks


def cells_of(bricks, spec):
    occ = np.zeros((spec.Z, spec.Y, spec.X), dtype=bool)
    for b in bricks:
        occ[b["z"], b["y"]:b["y"] + b["dy"], b["x"]:b["x"] + b["dx"]] = True
    return occ


def run_agent(trial, spec, designs):
    ck = torch.load(TRIALS / trial / "ckpt.pt", map_location="cpu")
    assert ck["scale"] == spec.name
    policy = Policy(spec, ck["net_cfg"])
    policy.load_state_dict(ck["state_dict"])
    policy.eval()
    tg = torch.tensor(np.array([d["occ"] for d in designs]), dtype=torch.bool)
    world = BrickWorld(spec, len(designs), "cpu", targets=tg, auto_reset=False)
    obs_mode = ck["net_cfg"]["obs_mode"]
    placed = [[] for _ in designs]
    with torch.no_grad():
        for _ in range(spec.max_steps + 1):
            if bool(world.ended.all()):
                break
            logits, _ = policy(encode(world, obs_mode), world.valid_mask())
            a = logits.argmax(dim=1)
            z_all = rest_heights(world.height, spec)
            z = z_all.gather(1, a.clamp(max=spec.n_place - 1).unsqueeze(1)).squeeze(1)
            legal = (a != spec.done_action) & (z < spec.Z) & world.fits[a.clamp(max=spec.n_place - 1)] & ~world.ended
            for i in legal.nonzero().squeeze(1).tolist():
                placed[i].append(int(a[i]))
            world.step(a)
    build = world.occ.numpy()
    eps = []
    for i, d in enumerate(designs):
        bricks = replay(placed[i], spec)
        assert np.array_equal(cells_of(bricks, spec), build[i]), "agent replay mismatch"
        eps.append({"bricks": bricks, "exact": bool((build[i] == np.array(d["occ"], dtype=bool)).all())})
    return eps


def main():
    torch.set_num_threads(8)
    out = {"seed": SEED, "n": N, "designs": {}, "episodes": {}, "agents": AGENTS}
    for scale, groups in AGENTS.items():
        spec = SCALES[scale]
        designs = fresh_designs(spec)
        out["designs"][scale] = designs
        for group, trials in groups.items():
            for trial in trials:
                eps = run_agent(trial, spec, designs)
                out["episodes"][f"{scale}/{trial}"] = eps
                print(f"{scale:5s} {group:13s} {trial:18s} exact {np.mean([e['exact'] for e in eps]):.3f}")
    (Path(__file__).parent / "rollouts.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
