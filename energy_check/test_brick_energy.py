"""Tests of the hand-set brick energy. Run: .venv/bin/python -I -B energy_check/test_brick_energy.py (non-zero exit on failure)."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import brick_energy as BE  # noqa: E402
from brickworld.env import BRICK_TYPES, SCALES, generate_designs, rest_heights  # noqa: E402
from evaluate import check  # noqa: E402

SMALL = SCALES["small"]
T = BE.T


def generator_boards(spec, n, seed):
    """n generator designs and, for each, the board of the generator's own bricks ([n, t, z, y, x] bool)."""
    occ, acts = generate_designs(spec, n, torch.Generator().manual_seed(seed), return_actions=True)
    YX = spec.Y * spec.X
    board = torch.zeros(n, T, spec.Z, spec.Y, spec.X, dtype=torch.bool)
    for k in range(n):
        height = torch.zeros(1, spec.Y, spec.X, dtype=torch.long)
        for a in acts[k].tolist():
            if a < 0:
                continue
            z = int(rest_heights(height, spec)[0, a])
            t, pos = divmod(a, YX)
            y, x = divmod(pos, spec.X)
            dy, dx = BRICK_TYPES[t]
            board[k, t, z, y, x] = True
            height[0, y:y + dy, x:x + dx] = z + 1
    return occ, board


def test_cover_and_sums_match_footprints():
    u = torch.zeros(1, T, 2, 4, 4)
    u[0, 3, 1, 1, 2] = 1          # a 2x2 on layer 1 at (1, 2)
    c = BE.cover(u)
    want = torch.zeros(2, 4, 4)
    want[1, 1:3, 2:4] = 1
    assert torch.equal(c[0], want)
    cells = torch.arange(32.0).view(1, 2, 4, 4)
    s = BE.brick_sums(cells)
    assert float(s[0, 3, 1, 1, 2]) == float(cells[0, 1, 1:3, 2:4].sum())
    assert float(s[0, 1, 0, 2, 0]) == float(cells[0, 0, 2, 0:2].sum())     # 1x2 = (dy 1, dx 2)


def test_sums_equal_an_integer_reference_in_a_large_batch():
    """`brick_sums` and `cover` against sums made with integer slices: no convolution and no rounding in the reference,
    so a convolution that rounded to the wrong whole number would fail here."""
    occ = generate_designs(SMALL, 200, torch.Generator().manual_seed(7))
    n, Z, Y, X = occ.shape
    want = torch.zeros(n, T, Z, Y, X, dtype=torch.long)
    for t, (dy, dx) in enumerate(BRICK_TYPES):
        for a in range(dy):
            for b in range(dx):
                want[:, t, :, :Y - a, :X - b] += occ[:, :, a:, b:].long()
    assert torch.equal(BE.brick_sums(occ.float()).long(), want), "a footprint sum differs from the integer reference"
    u = torch.rand(n, T, Z, Y, X, generator=torch.Generator().manual_seed(1)) < 0.2
    want = torch.zeros(n, Z, Y, X, dtype=torch.long)
    for t, (dy, dx) in enumerate(BRICK_TYPES):
        for a in range(dy):
            for b in range(dx):
                want[:, :, a:, b:] += u[:, t, :, :Y - a, :X - b].long()
    assert int(want.max()) > 1, "the reference never covers a cell twice"
    assert torch.equal(BE.cover(u.float()).long(), want), "a cover count differs from the integer reference"


def test_energy_equals_its_pair_form():
    gen = torch.Generator().manual_seed(1)
    target = generate_designs(SMALL, 1, gen)[0]
    K, B, const = BE.pair_form(target, mu=0.5)
    assert torch.equal(K, K.T) and float(K.diagonal().abs().sum()) == 0
    for _ in range(20):
        u = (torch.rand(1, T, SMALL.Z, SMALL.Y, SMALL.X, generator=gen) < 0.1).float()
        _, _, e = BE.energy(u, target.unsqueeze(0), mu=0.5)
        uf = u.flatten()
        assert abs(float(e) - float(0.5 * uf @ K @ uf + B @ uf + const)) < 1e-4


def test_generator_bricks_are_free_and_tile_the_target():
    for name in ("small", "large"):
        spec = SCALES[name]
        occ, board = generator_boards(spec, 200, seed=7)
        assert bool((board <= BE.allowed_bricks(occ)).all()), "a generator brick was clamped off"
        cov, count, _ = BE.energy(board.float(), occ, mu=0.0)
        assert float(cov.abs().sum()) == 0 and int(count.max()) <= spec.k_max


def test_layer_order_rebuilds_the_target_in_the_frozen_world():
    for name in ("small", "large"):
        spec = SCALES[name]
        occ, board = generator_boards(spec, 200, seed=11)
        built, illegal = BE.build(BE.bricks_to_actions(board, spec.done_action), occ, spec)
        assert illegal == 0
        assert bool(check(built.numpy(), occ.numpy())["exact"].all())


def test_unsupported_and_outside_bricks_are_clamped_off():
    target = torch.zeros(1, 2, 3, 3, dtype=torch.bool)
    target[0, 0, 0, 0] = True       # one ground cell
    target[0, 1, 0, 0] = True       # a cell on top of it
    target[0, 1, 0, 1] = True       # an overhang cell beside that
    free = BE.allowed_bricks(target)
    assert bool(free[0, 0, 0, 0, 0]) and bool(free[0, 0, 1, 0, 0])
    assert not bool(free[0, 0, 1, 0, 1]), "a 1x1 with nothing below it must be clamped off"
    assert bool(free[0, 1, 1, 0, 0]), "the 1x2 over the supported cell and the overhang is free"
    assert not bool(free[0, 1, 0, 0, 0]), "a brick that leaves the target must be clamped off"
    assert int(free.sum()) == 3


def test_search_solves_an_overhang_and_the_world_agrees():
    spec = SMALL
    target = torch.zeros(1, spec.Z, spec.Y, spec.X, dtype=torch.bool)
    target[0, 0, 2, 2:4] = True
    target[0, 1, 2, 1:5] = True     # four cells in a row on layer 1; only the middle two are supported
    out = BE.search(target, cap=spec.max_steps, mu=0.5, temp=0.5, chains=4, max_moves=2000, seed=0)
    assert bool(out["stopped"][0]) and int(out["cover_left"][0]) == 0
    layer1 = out["board"][0, :, 1]  # the only tiling: a 1x2 over each overhang cell and its supported neighbour
    assert bool(layer1[1, 2, 1]) and bool(layer1[1, 2, 3]) and int(layer1.sum()) == 2
    built, illegal = BE.build(BE.bricks_to_actions(out["board"], spec.done_action), target, spec)
    assert illegal == 0 and bool(check(built.numpy(), target.numpy())["exact"][0])


def test_search_respects_the_brick_cap():
    spec = SMALL
    target = torch.zeros(1, spec.Z, spec.Y, spec.X, dtype=torch.bool)
    target[0, 0, :2, :4] = True     # 8 ground cells: 8 bricks as 1x1s, 2 as 2x2s
    out = BE.search(target, cap=3, mu=0.5, temp=0.5, chains=4, max_moves=20000, seed=0)
    assert bool(out["stopped"][0]) and int(out["bricks"][0]) <= 3


def test_search_is_repeatable():
    occ = generate_designs(SMALL, 16, torch.Generator().manual_seed(3))
    a = BE.search(occ, SMALL.max_steps, 0.5, 0.5, 2, 500, seed=5)
    b = BE.search(occ, SMALL.max_steps, 0.5, 0.5, 2, 500, seed=5)
    assert torch.equal(a["board"], b["board"]) and torch.equal(a["moves"], b["moves"])


if __name__ == "__main__":
    torch.set_num_threads(4)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
