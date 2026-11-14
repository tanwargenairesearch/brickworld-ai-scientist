"""Tests of Stage H1's rules and fixes (PROTOCOL_H1.md, "Rule tests"). No design sets needed.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/test_rules_h.py   (prints PASS per test; exits 1 on failure)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import torch  # noqa: E402

from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import load_candidate  # noqa: E402
from arch_rsi.stage2 import w2b  # noqa: E402
from arch_rsi.worlds import RECIPE  # noqa: E402
from brickworld.env import BRICK_TYPES, BrickWorld, FixedSource, generate_designs, rest_heights  # noqa: E402
from brickworld.model import encode  # noqa: E402
from rules import (SHIFT, Rule, TorusWorld, booklet_channels, generate_designs_torus, mirror_perm,  # noqa: E402
                   torus_rest_heights)
from train import reward_fn  # noqa: E402

WS = common.register()
DEV = torch.device("mps")
FAILED = []


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except AssertionError as e:
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, e)
    return fn


def random_states(world, steps, gen):
    """Advance a world with random valid actions (DONE excluded), returning the observations along the way."""
    obs = []
    for _ in range(steps):
        m = world.valid_mask().cpu().float()
        m[:, -1] = 0
        m[m.sum(1) == 0, -1] = 1
        a = torch.multinomial(m, 1, generator=gen).squeeze(1).to(world.device)
        world.step(a)
        obs.append(encode(world, "voxels"))
    return obs


@test
def plain_hooks_are_identity():
    r = Rule("plain", WS, DEV)
    o, a, m, rew = torch.rand(3, 27, 8, 8), torch.arange(3), torch.ones(3, 257, dtype=torch.bool), torch.rand(3)
    assert r.obs(o) is o and r.act(a) is a and r.mask(m) is m and r.reward(rew, torch.zeros(3, dtype=torch.bool)) is rew


@test
def shift_moves_only_booklet_channels():
    g = torch.Generator().manual_seed(1)
    w = BrickWorld(WS, 64, DEV, target_source=FixedSource(generate_designs(WS, 256, g), 0))
    r = Rule("shift", WS, DEV)
    b = booklet_channels(WS.Z)
    rest = [c for c in range(4 * WS.Z + 3) if c not in b]
    for o in random_states(w, 10, g):
        s = r.obs(o)
        assert torch.equal(s[:, rest], o[:, rest])
        assert torch.equal(s[:, b], torch.roll(o[:, b], SHIFT, dims=3))
        assert torch.equal(s[:, b, :, (3 + SHIFT) % 8], o[:, b, :, 3])  # true column 3 is shown at column 7
        back = s.clone()
        back[:, b] = torch.roll(s[:, b], -SHIFT, dims=3)
        assert torch.equal(back, o)


@test
def mirror_lands_at_the_mirror_and_masks_match():
    p = mirror_perm(WS)
    T, Y, X = WS.n_types, WS.Y, WS.X
    assert p[-1] == WS.done_action
    for t, (dy, dx) in enumerate(BRICK_TYPES):
        for y in range(Y):
            for x in range(X):
                a = t * Y * X + y * X + x
                if x <= X - dx and y <= Y - dy:
                    assert p[a] == t * Y * X + y * X + (X - dx - x) and p[p[a]] == a
                else:
                    assert p[a] == a
    r = Rule("mirror", WS, DEV)
    w = BrickWorld(WS, 1, DEV, targets=torch.zeros(1, WS.Z, Y, X, dtype=torch.bool))
    w.step(r.act(torch.tensor([1 * Y * X + 3 * X + 0], device=DEV)))  # a 1x2 brick chosen at row 3, column 0
    assert w.occ[0, 0, 3].nonzero().flatten().tolist() == [6, 7]
    g = torch.Generator().manual_seed(2)
    w = BrickWorld(WS, 64, DEV, target_source=FixedSource(generate_designs(WS, 256, g), 0))
    random_states(w, 8, g)
    m = w.valid_mask()
    assert torch.equal(r.mask(m), m[:, p.to(DEV)])
    assert torch.equal(r.mask(torch.ones(1, 257, dtype=torch.bool, device=DEV)).sum(), torch.tensor(257, device=DEV))


@test
def torus_matches_plain_on_non_wrapping_placements():
    g = torch.Generator().manual_seed(3)
    designs = generate_designs(WS, 64, g)
    a_w = TorusWorld(WS, 64, DEV, targets=designs, auto_reset=False)
    b_w = BrickWorld(WS, 64, DEV, targets=designs, auto_reset=False)
    plain_fit = BrickWorld(WS, 1, DEV, targets=designs[:1]).fits
    for _ in range(25):
        m = (b_w.valid_mask() & torch.cat([plain_fit, torch.zeros(1, dtype=torch.bool, device=DEV)]).unsqueeze(0)).cpu().float()
        m[m.sum(1) == 0, -1] = 1
        a = torch.multinomial(m, 1, generator=g).squeeze(1).to(DEV)
        ia, ib = a_w.step(a), b_w.step(a)
        assert torch.equal(a_w.occ, b_w.occ) and torch.equal(a_w.height, b_w.height)
        for k in ("ended", "exact", "cells_in", "cells_out", "illegal", "ep_len"):
            assert torch.equal(ia[k], ib[k]), k
    h = torch.randint(0, WS.Z + 1, (32, WS.Y, WS.X))
    zt, zp = torus_rest_heights(h, WS), rest_heights(h, WS)
    fits = BrickWorld(WS, 1, "cpu", targets=designs[:1]).fits
    assert torch.equal(zt[:, fits], zp[:, fits])
    assert torch.equal(torus_rest_heights(h.to(DEV), WS).cpu(), zt)


@test
def torus_wraps_and_every_anchor_is_valid():
    Y, X = WS.Y, WS.X
    w = TorusWorld(WS, 1, DEV, targets=torch.zeros(1, WS.Z, Y, X, dtype=torch.bool))
    assert bool(w.valid_mask().all())
    w.step(torch.tensor([1 * Y * X + 2 * X + 7], device=DEV))  # 1x2 at row 2, column 7
    assert w.occ[0, 0, 2].nonzero().flatten().tolist() == [0, 7]
    w.step(torch.tensor([3 * Y * X + 7 * X + 7], device=DEV))  # 2x2 at row 7, column 7: the four corners, on the floor
    assert w.occ[0, 0].nonzero().tolist() == [[0, 0], [0, 7], [2, 0], [2, 7], [7, 0], [7, 7]], w.occ[0, 0].nonzero().tolist()
    w.step(torch.tensor([3 * Y * X + 1 * X + 7], device=DEV))  # 2x2 at row 1, column 7: covers (1,7),(1,0),(2,7),(2,0)
    assert w.occ[0, 1].nonzero().tolist() == [[1, 0], [1, 7], [2, 0], [2, 7]], w.occ[0, 1].nonzero().tolist()
    assert not bool(w.occ[0, 0, 1].any())  # rests on the row-2 brick, leaving hidden cells under row 1


@test
def torus_designs_rebuild_from_their_own_placements():
    g = torch.Generator().manual_seed(4)
    d, acts = generate_designs_torus(WS, 256, g, return_actions=True)
    w = TorusWorld(WS, 256, DEV, targets=d, auto_reset=False)
    for i in range(WS.k_max):
        a = acts[:, i].clone()
        a[a < 0] = WS.done_action
        w.step(a.to(DEV))
    assert torch.equal(w.occ.cpu(), d)
    Y, X = WS.Y, WS.X
    t, pos = acts // (Y * X), acts % (Y * X)
    y, x = pos // X, pos % X
    dy = torch.tensor([b[0] for b in BRICK_TYPES])[t.clamp(min=0)]
    dx = torch.tensor([b[1] for b in BRICK_TYPES])[t.clamp(min=0)]
    wraps = ((acts >= 0) & ((x > X - dx) | (y > Y - dy))).any(1).float().mean()
    assert wraps > 0.3, f"only {wraps:.2f} of designs use a wrapped brick"
    print(f"     {float(wraps):.0%} of 256 torus designs use at least one wrapped brick")


@test
def late_pays_the_same_total_at_the_end():
    g = torch.Generator().manual_seed(5)
    w = BrickWorld(WS, 128, DEV, target_source=FixedSource(generate_designs(WS, 512, g), 0))
    r = Rule("late", WS, DEV)
    acc_p, acc_l = torch.zeros(128, device=DEV), torch.zeros(128, device=DEV)
    ends = 0
    for _ in range(120):
        m = w.valid_mask().cpu().float()
        m[:, -1] = 0.05
        a = torch.multinomial(m, 1, generator=g).squeeze(1).to(DEV)
        info = w.step(a)
        rp = reward_fn(RECIPE, info, a != WS.done_action)
        rl = r.reward(rp, info["ended"])
        assert bool((rl[~info["ended"]] == 0).all())
        acc_p += rp
        acc_l += rl
        e = info["ended"]
        assert torch.allclose(acc_p[e], acc_l[e], atol=1e-5)
        ends += int(e.sum())
        acc_p[e] = 0
        acc_l[e] = 0
    assert ends > 200


def nets(fix, seed=7):
    a = arch_spec(WS)
    torch.manual_seed(seed)
    ref = CandidatePolicy(load_candidate(common.REF_A.read_text())(a), a).to(DEV).eval()
    torch.manual_seed(seed)
    fx = CandidatePolicy(load_candidate(common.FIXES[fix].read_text())(a), a).to(DEV).eval()
    return ref, fx


@test
def fix_shift_sees_exactly_the_plain_input():
    g = torch.Generator().manual_seed(6)
    w = BrickWorld(WS, 64, DEV, target_source=FixedSource(generate_designs(WS, 256, g), 0))
    ref, fx = nets("shift")
    r = Rule("shift", WS, DEV)
    m = torch.ones(64, 257, dtype=torch.bool, device=DEV)
    with torch.no_grad():
        for o in random_states(w, 6, g):
            la, va = ref(o, m)
            lb, vb = fx(r.obs(o), m)
            assert torch.equal(la, lb) and torch.equal(va, vb)


@test
def fix_mirror_is_reference_a_on_the_flipped_input():
    g = torch.Generator().manual_seed(8)
    w = BrickWorld(WS, 64, DEV, target_source=FixedSource(generate_designs(WS, 256, g), 0))
    ref, fx = nets("mirror")
    m = torch.ones(64, 257, dtype=torch.bool, device=DEV)
    with torch.no_grad():
        for o in random_states(w, 4, g):
            assert torch.equal(ref(o.flip(3), m)[0], fx(o, m)[0])


@test
def fix_late_is_reference_a_plus_recipe():
    s = common.FIXES["late"].read_text()
    assert w2b.strip_recipe(s) == w2b.strip_recipe(common.REF_A.read_text())
    assert w2b.parse_recipe(s) == {"gamma": 0.995, "gae_lambda": 0.99} and not w2b.validate(w2b.parse_recipe(s))


@test
def fix_torus_changes_only_padding():
    s, ref = common.FIXES["torus"].read_text(), common.REF_A.read_text()
    assert s.replace(', padding_mode="circular"', "").replace(', mode="circular"', "") == ref


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
