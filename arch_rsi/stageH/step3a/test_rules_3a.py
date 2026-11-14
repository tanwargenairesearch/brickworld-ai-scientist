"""Tests of Step 3a's rules and fixes (PROTOCOL_STEP3A.md, "Rule tests"), plus the derivation checks of the copies
and the fixes and the fixes' static legality. Small work only: a few greedy builds of 64 designs.
Runs on --device (default mps). On 14 November it was first run with --device cpu while Step 2 held the GPU
(decision 3); run_3a.py runs it again on the GPU before anything else.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3a/test_rules_3a.py [--device cpu]   (exits 1 on failure)
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common  # noqa: E402

import torch  # noqa: E402

import rules as R0  # noqa: E402
import rules3a as R3  # noqa: E402
from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import load_candidate  # noqa: E402
from brickworld.env import BRICK_TYPES, BrickWorld, FixedSource, StreamSource, footprints, generate_designs, rest_heights  # noqa: E402
from brickworld.model import encode  # noqa: E402
from eval_3a import greedy_build  # noqa: E402
from make_fixes_3a import FIXES  # noqa: E402

DEV = torch.device(sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "mps")
WS = common.register()
Y, X, Z = WS.Y, WS.X, WS.Z
B = R0.booklet_channels(Z)
REST = [c for c in range(4 * Z + 3) if c not in B]
CAL = common.S2C / "calib" / "b131072" / "refA_g_s2000" / "run" / "ckpt.pt"   # 2c's trained reference A, read only
PY = [sys.executable, "-I", "-B", "-W", "ignore"]
EMPTY = torch.zeros(2, Z, Y, X, dtype=torch.bool)
HELD = torch.load(common.TARGETS / "h_heldout_hashes.pt")
FAILED = []
# Where each booklet rule shows true cell (y, x), as the protocol's table states it.
SHOWN_AT = {"rowshift": lambda y, x: ((y + 4) % Y, x), "bookmirror": lambda y, x: (y, X - 1 - x),
            "transpose": lambda y, x: (x, y), "antitranspose": lambda y, x: (X - 1 - x, Y - 1 - y),
            "rot90": lambda y, x: (X - 1 - x, y), "rot270": lambda y, x: (x, Y - 1 - y)}


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except Exception as e:  # noqa: BLE001
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, type(e).__name__, e)
    return fn


def world(n, seed):
    g = torch.Generator().manual_seed(seed)
    return BrickWorld(WS, n, DEV, target_source=FixedSource(generate_designs(WS, 4 * n, g), 0)), g


def random_states(w, steps, g):
    """Advance a world with random valid actions (DONE excluded), returning the observations along the way."""
    obs = []
    for _ in range(steps):
        m = w.valid_mask().cpu().float()
        m[:, -1] = 0
        m[m.sum(1) == 0, -1] = 1
        w.step(torch.multinomial(m, 1, generator=g).squeeze(1).to(w.device))
        obs.append(encode(w, "voxels"))
    return obs


def policy(path, trained=True, seed=7):
    a = arch_spec(WS)
    torch.manual_seed(seed)
    p = CandidatePolicy(load_candidate(Path(path).read_text())(a), a)
    if trained:
        p.load_state_dict(torch.load(CAL, map_location="cpu")["state_dict"])
    return p.to(DEV).eval()


@test
def old_rules_run_rules_py():
    g = torch.Generator().manual_seed(11)
    o = torch.rand(5, 4 * Z + 3, Y, X, generator=g).to(DEV)
    a = torch.randint(0, WS.n_actions, (5,), generator=g).to(DEV)
    m = (torch.rand(5, WS.n_actions, generator=g) > 0.5).to(DEV)
    for name in R0.RULES:
        old, new = R0.Rule(name, WS, DEV), R3.Rule(name, WS, DEV)
        assert torch.equal(old.obs(o), new.obs(o)) and torch.equal(old.act(a), new.act(a)), name
        assert torch.equal(old.mask(m), new.mask(m)), name
        assert type(old.make_world(2, targets=EMPTY)) is type(new.make_world(2, targets=EMPTY)), name
        r, e = torch.rand(5, generator=g).to(DEV), torch.tensor([0, 1, 0, 0, 1], dtype=torch.bool, device=DEV)
        assert torch.equal(old.reward(r, e), new.reward(r, e)), name
    p = R3.Rule("plain", WS, DEV)
    assert p.obs(o) is o and p.act(a) is a and p.mask(m) is m


@test
def new_rules_keep_the_world_and_its_rewards():
    o, a = torch.rand(2, 4 * Z + 3, Y, X, device=DEV), torch.arange(2, device=DEV)
    m, r = torch.ones(2, WS.n_actions, dtype=torch.bool, device=DEV), torch.rand(2, device=DEV)
    for name in R3.NEW:
        rule = R3.Rule(name, WS, DEV)
        assert type(rule.make_world(2, targets=EMPTY)) is BrickWorld, name
        assert type(rule.make_stream(0, HELD)) is StreamSource, name
        assert rule.reward(r, torch.zeros(2, dtype=torch.bool, device=DEV)) is r, name
        if name in R3.BOOKLET:
            assert rule.act(a) is a and rule.mask(m) is m, name
        else:
            assert rule.obs(o) is o, name
    assert set(R3.NEW) == set(R3.BOOKLET) | set(R3.HANDS) and set(R3.FAMILY) == set(R3.PASSED_BEFORE) | set(R3.NEW)
    assert [R3.IDS[r] for r in R3.NEW] == [f"H{i}" for i in range(5, 13)]


@test
def booklet_rules_show_each_cell_where_the_protocol_says():
    w, g = world(64, 1)
    for o in random_states(w, 8, g):
        for name, at in SHOWN_AT.items():
            s = R3.Rule(name, WS, DEV).obs(o)
            assert torch.equal(s[:, REST], o[:, REST]), name
            for y in range(Y):
                for x in range(X):
                    yy, xx = at(y, x)
                    assert torch.equal(s[:, B, yy, xx], o[:, B, y, x]), (name, y, x)
    assert sorted(SHOWN_AT) == sorted(R3.BOOKLET)


@test
def booklet_fixes_see_exactly_the_plain_input():
    w, g = world(64, 2)
    ref = policy(common.REF_A, trained=False)
    m = torch.ones(64, WS.n_actions, dtype=torch.bool, device=DEV)
    states = random_states(w, 6, g)
    with torch.no_grad():
        for name in R3.BOOKLET:
            fx, rule = policy(FIXES[name], trained=False), R3.Rule(name, WS, DEV)
            for o in states:
                la, va = ref(o, m)
                lb, vb = fx(rule.obs(o), m)
                assert torch.equal(la, lb) and torch.equal(va, vb), name


@test
def handshift_lands_four_columns_over_and_masks_match():
    p = R3.hands_perm("handshift", WS)
    assert p[-1] == WS.done_action and torch.equal(p[p], torch.arange(WS.n_actions))  # 4 of 8 columns: its own inverse
    for t in range(WS.n_types):
        for y in range(Y):
            for x in range(X):
                assert p[t * Y * X + y * X + x] == t * Y * X + y * X + (x + 4) % X
    r = R3.Rule("handshift", WS, DEV)
    w = BrickWorld(WS, 1, DEV, targets=torch.zeros(1, Z, Y, X, dtype=torch.bool))
    w.step(r.act(torch.tensor([1 * Y * X + 3 * X + 5], device=DEV)))  # a 1x2 brick chosen at row 3, column 5
    assert w.occ[0, 0, 3].nonzero().flatten().tolist() == [1, 2]
    w, g = world(64, 3)
    random_states(w, 8, g)
    wm = w.valid_mask()
    am = r.mask(wm)
    assert torch.equal(am, wm[:, p.to(DEV)]) and torch.equal(am.sum(1), wm.sum(1))
    t12 = 1 * Y * X
    assert not bool(R3.Rule("handshift", WS, DEV).mask(torch.ones(1, WS.n_actions, dtype=torch.bool, device=DEV)
                                                       & BrickWorld(WS, 1, DEV, targets=torch.zeros(1, Z, Y, X, dtype=torch.bool)).valid_mask())[0, t12 + 3])
    # a 1x2 brick chosen at column 3 would land at column 7 and stick out, so the agent's mask hides it


@test
def vmirror_lands_at_the_mirror_and_masks_match():
    p = R3.hands_perm("vmirror", WS)
    assert p[-1] == WS.done_action and torch.equal(p[p], torch.arange(WS.n_actions))
    for t, (dy, dx) in enumerate(BRICK_TYPES):
        for y in range(Y):
            for x in range(X):
                a = t * Y * X + y * X + x
                want = t * Y * X + (Y - dy - y) * X + x if (y <= Y - dy and x <= X - dx) else a
                assert p[a] == want, (t, y, x)
    r = R3.Rule("vmirror", WS, DEV)
    w = BrickWorld(WS, 1, DEV, targets=torch.zeros(1, Z, Y, X, dtype=torch.bool))
    w.step(r.act(torch.tensor([2 * Y * X + 1 * X + 3], device=DEV)))  # a 2x1 brick chosen at row 1, column 3
    assert w.occ[0, 0, :, 3].nonzero().flatten().tolist() == [5, 6]
    w, g = world(64, 4)
    random_states(w, 8, g)
    wm = w.valid_mask()
    assert torch.equal(r.mask(wm), wm[:, p.to(DEV)])


@test
def the_design_generator_is_symmetric_top_to_bottom():
    """Each drop picks a type among those that fit somewhere, then an anchor uniformly among its valid anchors, so the
    designs are symmetric under the flip if the valid anchors and their landing layers map onto each other."""
    p = R3.hands_perm("vmirror", WS)[:-1]
    _, fits = footprints(WS)
    fits = fits.reshape(-1)
    assert torch.equal(fits[p], fits)
    h = torch.randint(0, Z + 1, (64, Y, X), generator=torch.Generator().manual_seed(5))
    z, zf = rest_heights(h, WS), rest_heights(h.flip(1), WS)
    assert torch.equal(zf[:, p][:, fits], z[:, fits])


@test
def every_fix_builds_what_reference_a_builds_in_the_plain_world():
    """Greedy builds with 2c's trained reference A weights: each fix in its rule world builds exactly what reference A
    builds in the plain world on the same designs (for "vmirror": the mirror image, on the mirrored designs)."""
    g = torch.Generator().manual_seed(9)
    d = generate_designs(WS, 64, g)
    ref = policy(common.REF_A)
    occ_p, done_p = greedy_build(ref, R3.Rule("plain", WS, DEV), d)
    exact = (occ_p == d).flatten(1).all(1).float().mean()
    assert exact > 0.3, f"the trained reference builds only {exact:.0%}, too little to test with"
    occ_pm, done_pm = greedy_build(ref, R3.Rule("plain", WS, DEV), d.flip(2))
    for name in R3.NEW:
        occ, done = greedy_build(policy(FIXES[name]), R3.Rule(name, WS, DEV), d)
        if name == "vmirror":
            assert torch.equal(occ, occ_pm.flip(2)) and torch.equal(done, done_pm), name
        else:
            assert torch.equal(occ, occ_p) and torch.equal(done, done_p), name
    print(f"     the trained reference builds {exact:.0%} of 64 designs; every fix matches it")


@test
def reference_a_fails_in_every_new_rule_world():
    """Not a pass rule, a sanity check of the setup: trained in the plain world, reference A should build much less
    under each rule. (The calibration measures reference A trained in each rule world instead.)"""
    g = torch.Generator().manual_seed(10)
    d = generate_designs(WS, 64, g)
    ref = policy(common.REF_A)
    base = float((greedy_build(ref, R3.Rule("plain", WS, DEV), d)[0] == d).flatten(1).all(1).float().mean())
    got = {}
    for name in R3.NEW:
        occ, _ = greedy_build(ref, R3.Rule(name, WS, DEV), d)
        got[name] = float((occ == d).flatten(1).all(1).float().mean())
        assert got[name] < base, (name, got[name], base)
    print(f"     plain {base:.0%}; under each rule: " + ", ".join(f"{R3.IDS[k]} {v:.0%}" for k, v in got.items()))


@test
def copies_and_fixes_match_their_derivations():
    for script in ("make_copies_3a.py", "make_fixes_3a.py"):
        p = subprocess.run(PY + [str(HERE / script), "--check"], capture_output=True, text=True, cwd=common.ROOT)
        assert p.returncode == 0, (script, p.stdout, p.stderr[-400:])


@test
def fixes_pass_the_static_rules_at_reference_a_cost():
    sp = importlib.util.spec_from_file_location("check_2b", common.ROOT / "arch_rsi" / "stage2" / "check_2b.py")
    C2 = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(C2)
    a = arch_spec(WS)
    ref = common.REF_A.read_text()
    f0 = C2.C.train_flops(ref, a)
    n0 = sum(q.numel() for q in policy(common.REF_A, trained=False).parameters())
    for name, path in FIXES.items():
        src = path.read_text()
        assert C2.static_check(src) == [], (name, C2.static_check(src))
        assert C2.C.train_flops(src, a) == f0, name
        assert sum(q.numel() for q in policy(path, trained=False).parameters()) == n0, name


if __name__ == "__main__":
    print(f"device: {DEV.type}")
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
