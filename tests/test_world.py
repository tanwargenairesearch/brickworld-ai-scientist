"""Verifier and world tests. Run: python tests/test_world.py (exits non-zero on failure)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brickworld.env import (SCALES, BRICK_TYPES, BrickWorld, StreamSource, design_hash,  # noqa: E402
                            generate_designs, has_hidden_cells, heights_of, zobrist_codes)
from evaluate import check, np_heights  # noqa: E402

SMALL = SCALES["small"]


def act(spec, t, y, x):
    return t * spec.Y * spec.X + y * spec.X + x


def run_actions(spec, target, actions):
    w = BrickWorld(spec, 1, "cpu", targets=target.unsqueeze(0), auto_reset=False)
    info = None
    for a in actions + [spec.done_action]:
        info = w.step(torch.tensor([a]))
        if bool(w.ended.all()):
            break
    return w, info


def test_replay_generated_designs_is_exact():
    for spec in SCALES.values():
        g = torch.Generator().manual_seed(7)
        designs, acts = generate_designs(spec, 300, g, return_actions=True)
        w = BrickWorld(spec, 300, "cpu", targets=designs, auto_reset=False)
        for i in range(spec.k_max):
            a = acts[:, i].clone()
            a[a < 0] = spec.done_action
            w.step(a)
        w.step(torch.full((300,), spec.done_action))
        assert bool(w.ended.all())
        sc = check(w.occ.numpy(), designs.numpy())
        assert sc["exact"].all(), f"{spec.name}: replaying the generating actions must be exact"


def test_equivalent_bricks_both_pass():
    s = SMALL
    tgt = torch.zeros(s.Z, s.Y, s.X, dtype=torch.bool)
    tgt[0, 0, 0:2] = True  # one 1x2 brick
    w1, _ = run_actions(s, tgt, [act(s, 1, 0, 0)])
    w2, _ = run_actions(s, tgt, [act(s, 0, 0, 0), act(s, 0, 0, 1)])
    assert check(w1.occ.numpy(), tgt[None].numpy())["exact"].all()
    assert check(w2.occ.numpy(), tgt[None].numpy())["exact"].all()


def test_bricks_drop_and_overhang():
    s = SMALL
    tgt = torch.zeros(s.Z, s.Y, s.X, dtype=torch.bool)
    w, _ = run_actions(s, tgt, [act(s, 0, 0, 0), act(s, 1, 0, 0)])
    occ = w.occ[0]
    assert occ[0, 0, 0] and occ[1, 0, 0] and occ[1, 0, 1]
    assert not occ[0, 0, 1], "the 1x2 brick must overhang, leaving an empty cell under it"
    assert int(occ.sum()) == 3
    assert bool(has_hidden_cells(w.occ)[0])


def test_planted_topview_flaw_is_real():
    s = SMALL
    overhang = torch.zeros(s.Z, s.Y, s.X, dtype=torch.bool)
    overhang[0, 0, 0] = overhang[1, 0, 0] = overhang[1, 0, 1] = True
    w, _ = run_actions(s, overhang, [act(s, 0, 0, 1), act(s, 0, 0, 0), act(s, 1, 0, 0)])
    sc = check(w.occ.numpy(), overhang[None].numpy())
    assert not sc["exact"][0], "filled-in build must fail the exact checker"
    assert sc["topview"][0], "but pass the top-view checker"


def test_extra_or_missing_cell_fails():
    s = SMALL
    tgt = torch.zeros(s.Z, s.Y, s.X, dtype=torch.bool)
    tgt[0, 1, 1] = True
    w, _ = run_actions(s, tgt, [act(s, 0, 1, 1), act(s, 0, 2, 2)])
    assert not check(w.occ.numpy(), tgt[None].numpy())["exact"][0]
    w, _ = run_actions(s, tgt, [])
    assert not check(w.occ.numpy(), tgt[None].numpy())["exact"][0]


def test_height_cap_masks_actions():
    s = SMALL
    tgt = torch.zeros(s.Z, s.Y, s.X, dtype=torch.bool)
    w = BrickWorld(s, 1, "cpu", targets=tgt[None], auto_reset=False)
    for _ in range(s.Z):
        w.step(torch.tensor([act(s, 0, 0, 0)]))
    m = w.valid_mask()[0]
    assert not m[act(s, 0, 0, 0)] and not m[act(s, 3, 0, 0)] and m[act(s, 0, 1, 1)]
    assert m[s.done_action]


def test_numpy_heights_match_env():
    for spec in SCALES.values():
        d = generate_designs(spec, 500, torch.Generator().manual_seed(3))
        assert np.array_equal(np_heights(d.numpy()), heights_of(d).numpy())


def test_heldout_disjoint_and_never_trained_on():
    for spec in SCALES.values():
        dev = torch.load(ROOT / "targets" / f"{spec.name}_dev.pt")
        fin = torch.load(ROOT / "holdout" / f"{spec.name}_final.pt")
        codes = zobrist_codes(spec)
        hd, hf = set(design_hash(dev, codes).tolist()), set(design_hash(fin, codes).tolist())
        assert len(hd) == len(dev) and len(hf) == len(fin), "held-out designs must be unique"
        assert not hd & hf, "dev and final must not overlap"
        held = torch.load(ROOT / "targets" / f"{spec.name}_heldout_hashes.pt")
        src = StreamSource(spec, 10_000_000 + 5, held, chunk=1 << 14)
        tr = src(20000)
        assert not set(design_hash(tr, codes).tolist()) & (hd | hf), "training stream leaked a held-out design"


def test_generator_deterministic():
    a = generate_designs(SMALL, 64, torch.Generator().manual_seed(11))
    b = generate_designs(SMALL, 64, torch.Generator().manual_seed(11))
    assert torch.equal(a, b)


def test_action_layout_matches_footprints():
    s = SMALL
    for t, (dy, dx) in enumerate(BRICK_TYPES):
        tgt = torch.zeros(s.Z, s.Y, s.X, dtype=torch.bool)
        w, _ = run_actions(s, tgt, [act(s, t, 1, 1)])
        assert int(w.occ.sum()) == dy * dx
        assert bool(w.occ[0, 0, 1:1 + dy, 1:1 + dx].all())


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
