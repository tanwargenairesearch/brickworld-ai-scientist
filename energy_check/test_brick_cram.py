"""Tests of the learned brick energy and of its Metal kernel.
Run: .venv/bin/python -I -B energy_check/test_brick_cram.py (non-zero exit on failure; needs an Apple GPU)."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import brick_cram as BC  # noqa: E402
import brick_energy as BE  # noqa: E402
import metal_search as MS  # noqa: E402
import run_check as RC  # noqa: E402
from brickworld.env import SCALES, design_hash, zobrist_codes  # noqa: E402
from evaluate import check  # noqa: E402

SMALL, LARGE = SCALES["small"], SCALES["large"]
_HELD = {}


def held(spec) -> torch.Tensor:
    """Hashes of the dev, final and tuning designs of a world: no test draws one of them."""
    if spec.name not in _HELD:
        _, _, hashes = RC.worlds()[spec.name]
        tune = design_hash(RC.tuning_designs(spec, hashes), zobrist_codes(spec))
        _HELD[spec.name] = torch.cat([torch.load(hashes), tune])
    return _HELD[spec.name]


def data(spec, n, seed):
    target, bricks = BC.solved_designs(spec, n, seed, held(spec))
    assert not bool(torch.isin(design_hash(target, zobrist_codes(spec)), held(spec)).any())
    lab = BC.labels_from_bricks(target, bricks, torch.Generator().manual_seed(seed))
    return target, lab, BC.tokens(target)


def flat(tok, chains=1):
    return np.repeat(tok.reshape(len(tok), -1).numpy().astype(np.int32), chains, 0)


def dims(spec):
    return (spec.Z, spec.Y, spec.X)


def test_label_tables():
    seen = set()
    for t in range(4):
        for p in range(len(BC.CELLS[t])):
            for q in range(len(BC.CELLS[t]) if t else 1):
                lab = BC.label_of(t, p, q)
                assert (BC.TYPE[lab], BC.POS[lab], BC.ANCHOR[lab]) == (t, p, q)
                seen.add(lab)
    assert seen == set(range(1, BC.C))


def test_solution_labels_obey_the_rules_and_decode_to_the_generator_bricks():
    okK, okB = BC.true_rules()
    for spec in (SMALL, LARGE):
        target, bricks = BC.solved_designs(spec, 300, 5, held(spec))
        lab = BC.labels_from_bricks(target, bricks, torch.Generator().manual_seed(5))
        tok = BC.tokens(target)
        cK, cB = BC.class_counts(lab, tok)
        assert not bool((cK > 0)[~okK].any()) and not bool((cB > 0)[~okB].any()), "a solution holds a forbidden term"
        board = BC.decode(lab)
        assert int(board.sum()) == int((bricks[:, :, 0] >= 0).sum())
        built, illegal = BE.build(BE.bricks_to_actions(board, spec.done_action), target, spec)
        assert illegal == 0 and bool(check(built.numpy(), target.numpy())["exact"].all())


def test_energy_is_the_sum_of_its_terms_and_matches_local_energies():
    target, lab, tok = data(SMALL, 20, 1)
    g = torch.Generator().manual_seed(0)
    K, B = torch.randn(4, BC.C, BC.C, generator=g), torch.randn(2, BC.C, generator=g)
    e0 = BC.board_energy(K, B, lab, tok)
    L = BC.local_energies(K, B, lab, tok)
    n, z, y, x = [int(v[0]) for v in (tok >= 0).nonzero(as_tuple=True)]
    for new in (1, 7, 19):                      # changing one cell changes E by the difference of its local energies
        lab2 = lab.clone()
        lab2[n, z, y, x] = new
        want = float(L[n, z, y, x, new] - L[n, z, y, x, lab[n, z, y, x]])
        assert abs(float(BC.board_energy(K, B, lab2, tok)[n] - e0[n]) - want) < 1e-3


def test_violations_agree_with_the_rules():
    tabs = BC.rule_tables()
    target, lab, tok = data(SMALL, 50, 2)
    nbr = BC.neighbours(*dims(SMALL))
    y, t = lab.reshape(50, -1).numpy().astype(np.int32), flat(tok)
    assert not BC.violations(y, t, tabs, nbr).any(), "a solution has a violated term"
    y2 = y.copy()
    rows, cols = np.nonzero(t >= 0)
    y2[rows[0], cols[0]] = 1 + (y2[rows[0], cols[0]] % 25)        # any other label breaks a rule at that cell
    assert BC.violations(y2, t, tabs, nbr)[rows[0], cols[0]]


def test_kernel_equals_reference_move_for_move_at_temp_0():
    target, lab, tok = data(LARGE, 96, 3)
    tabs = BC.rule_tables()
    rng = np.random.default_rng(0)                # costs on a grid of 1/64: every sum is exact in any order
    tabs = {**tabs, "W8": np.concatenate([w := rng.integers(-64, 64, (4, BC.C, BC.C)).astype(np.float32) / 64,
                                          w.transpose(0, 2, 1)]),
            "B": rng.integers(-64, 64, (2, BC.C)).astype(np.float32) / 64}
    t = flat(tok)
    for moves in (0, 1, 37, 300):
        y_ref, f_ref = BC.search_reference(tabs, t, dims(LARGE), moves, 0.0, seed=11)
        y_gpu, f_gpu, _ = MS.search_gpu(tabs, t, dims(LARGE), moves, 0.0, seed=11, chunk=64)
        assert np.array_equal(f_ref, f_gpu), f"stops differ after {moves} moves"
        assert np.array_equal(y_ref, y_gpu), f"labels differ after {moves} moves"


def test_kernel_stops_where_the_reference_stops():
    """Chains that start at a solution or a few wrong labels away from one: the kernel reports the same stop move as
    the reference for every chain, over several move budgets and launch sizes, a budget of 0 included."""
    target, lab, tok = data(LARGE, 96, 6)
    tabs = BC.rule_tables()                       # costs 0 and 1: every sum is exact
    t = flat(tok)
    y0 = lab.reshape(96, -1).numpy().astype(np.int32).copy()
    rng = np.random.default_rng(0)
    for i in range(32, 96):                       # chains 0..31 start solved; the others get 1 to 3 other labels
        cells = rng.choice(np.nonzero(t[i] >= 0)[0], size=1 + i % 3, replace=False)
        y0[i, cells] = 1 + (y0[i, cells] - 1 + rng.integers(1, 25, len(cells))) % 25
    stops = set()
    for moves, chunk in ((0, 64), (1, 64), (5, 2), (40, 7), (300, 64)):
        y_ref, f_ref = BC.search_reference(tabs, t, dims(LARGE), moves, 0.0, seed=11, y0=y0)
        y_gpu, f_gpu, _ = MS.search_gpu(tabs, t, dims(LARGE), moves, 0.0, seed=11, chunk=chunk, y0=y0)
        assert np.array_equal(f_ref, f_gpu), f"stops differ with a budget of {moves} moves"
        assert np.array_equal(y_ref, y_gpu), f"labels differ with a budget of {moves} moves"
        assert (f_ref[:32] == 0).all() and (f_ref[32:] != 0).all(), "a solved chain must stop at move 0, and only those"
        stops |= set(f_ref.tolist())
    assert -1 in stops and len([v for v in stops if v > 0]) >= 5, "the test did not exercise stops at several moves"


def test_unsolvable_is_a_proof():
    """`unsolvable` on a case worked by hand, under the true rules, and against the search."""
    rules = BC.rule_tables()
    two = np.zeros((1, 2), np.int32)              # a 1 x 1 x 2 grid, both cells filled: one 1x2 brick or two 1x1
    no_1x1 = {**rules, "badB": rules["badB"].copy()}
    no_1x1["badB"][0, 1] = 1
    no_1x2 = {**no_1x1, "badB": no_1x1["badB"].copy()}
    no_1x2["badB"][0, 2:4] = 1                    # the left piece of a 1x2 brick, either anchor
    assert not BC.unsolvable(rules, two, (1, 1, 2))[0] and not BC.unsolvable(no_1x1, two, (1, 1, 2))[0]
    assert BC.unsolvable(no_1x2, two, (1, 1, 2))[0]
    target, _, tok = data(LARGE, 256, 12)
    t = flat(tok)
    assert not BC.unsolvable(rules, t, dims(LARGE)).any(), "a generator design is impossible under the true rules"
    _, lab_s, tok_s = data(SMALL, 300, 9)         # a fit on few designs flags allowed terms that some designs need
    K, B, _ = BC.count_fit(lab_s, tok_s)
    tabs = BC.search_tables(K, B, lab_s, tok_s)
    u = BC.unsolvable(tabs, t, dims(LARGE))
    _, found, _ = MS.search_gpu(tabs, np.repeat(t, 4, 0), dims(LARGE), 20000, 0.6, seed=2, chains=4)
    solved = (found.reshape(-1, 4) >= 0).any(1)
    assert u.any(), "the small fit was expected to make some design impossible"
    assert not (u & solved).any(), "the search stopped on a design proved to have no labelling"


def test_held_out_solutions_under_the_counted_energy():
    """Solutions the fit has not seen: most have energy 0, a few do not (an allowed term the fit never saw), and no
    term the fit clears is a forbidden one."""
    _, lab, tok = data(SMALL, 4000, 9)
    K, B, _ = BC.count_fit(lab, tok)
    okK, okB = BC.true_rules()
    assert not bool(((K == 0) & ~okK)[:, 1:, 1:].any()) and not bool(((B == 0) & ~okB)[:, 1:].any()), \
        "the fit clears a forbidden term"
    for spec, seed in ((SMALL, 31), (LARGE, 32)):
        _, lab_h, tok_h = data(spec, 2000, seed)
        e = BC.board_energy(K, B, lab_h, tok_h)
        share = float((e == 0).float().mean())
        assert 0.99 < share, f"only {share:.4f} of the held-out {spec.name} solutions have energy 0"
        assert float(e.max()) > 0, "expected a held-out solution with a term the fit never saw (4,000 designs)"


def test_kernel_flags_and_stops_are_true_at_temp_above_0():
    tabs = BC.rule_tables()
    for spec in (SMALL, LARGE):
        target, lab, tok = data(spec, 128, 4)
        t = flat(tok, chains=2)
        y, found, _ = MS.search_gpu(tabs, t, dims(spec), 20000, 0.6, seed=7, chains=2)
        nbr = BC.neighbours(*dims(spec))
        clean = ~BC.violations(y, t, tabs, nbr).any(1)
        assert np.array_equal(clean[found >= 0], np.ones(int((found >= 0).sum()), bool)), "a stopped chain has a violation"
        assert not clean[found == -1].any(), "a running chain has no violation left"
        assert ((y > 0) == (t >= 0)).all(), "a label moved onto an empty cell or off a filled one"
        first = np.where(found.reshape(-1, 2) >= 0, found.reshape(-1, 2), 1 << 30).argmin(1)
        solved = (found.reshape(-1, 2) >= 0).any(1)
        labs = torch.from_numpy(y.reshape(-1, 2, *target.shape[1:])[np.arange(len(target)), first]).long()
        built, illegal = BE.build(BE.bricks_to_actions(BC.decode(labs), spec.done_action), target, spec)
        exact = check(built.numpy(), target.numpy())["exact"]
        over = BC.decode(labs).flatten(1).sum(1).numpy() > spec.max_steps
        assert solved.mean() > 0.9, f"the hand-set rules solved only {solved.mean():.2f}"
        assert exact[solved & ~over].all(), "a labelling with no violated term did not build the target"


def test_kernel_matches_reference_in_distribution():
    """One free cell beside one clamped cell: the kernel's first label follows softmax(-L / temp), as the reference's."""
    tabs = BC.rule_tables()
    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, BC.C, BC.C)).astype(np.float32)
    tabs = {**tabs, "W8": np.concatenate([w, w.transpose(0, 2, 1)]), "B": rng.normal(size=(2, BC.C)).astype(np.float32),
            "bad8": np.ones_like(tabs["bad8"]), "badB": np.ones_like(tabs["badB"])}
    h = 40000
    tok = np.full((h, 4), -1, np.int32)           # a 1 x 2 x 2 grid with one filled cell
    tok[:, 0] = 0
    y0 = np.zeros((h, 4), np.int32)
    y0[:, 0] = 3
    L = tabs["B"][0] + tabs["W8"][:, :, 0].sum(0)  # every neighbour of the cell is empty or outside the grid
    p = np.exp(-L[1:] / 0.7 - (-L[1:] / 0.7).max())
    p /= p.sum()
    for name, y in (("kernel", MS.search_gpu(tabs, tok, (1, 2, 2), 1, 0.7, seed=3, y0=y0)[0]),
                    ("reference", BC.search_reference(tabs, tok, (1, 2, 2), 1, 0.7, seed=3, y0=y0)[0])):
        freq = np.bincount(y[:, 0], minlength=BC.C)[1:] / h
        assert np.abs(freq - p).max() < 0.01, f"{name}: label frequencies are off by {np.abs(freq - p).max():.4f}"


def test_count_fit_flags_every_forbidden_term_and_no_term_of_a_solution():
    target, lab, tok = data(SMALL, 1000, 9)
    K, B, _ = BC.count_fit(lab, tok)
    cmp = BC.compare_with_rules(K > 0, B > 0)
    assert cmp["forbidden_pair_terms_flagged"] == cmp["forbidden_pair_terms"] == 1196
    assert cmp["forbidden_input_terms_flagged"] == cmp["forbidden_input_terms"] == 9
    tabs = BC.search_tables(K, B, lab, tok)
    assert tabs["tau_pair"] == 0.0 and tabs["tau_input"] == 0.0
    assert float(BC.board_energy(K, B, lab, tok).abs().max()) == 0.0, "a training solution has energy above 0"
    y, t = lab.reshape(len(lab), -1).numpy().astype(np.int32), flat(tok)
    assert not BC.violations(y, t, tabs, BC.neighbours(*dims(SMALL))).any()


def test_counted_energy_from_the_small_world_solves_a_larger_one():
    _, lab, tok = data(SMALL, 4000, 9)
    K, B, _ = BC.count_fit(lab, tok)
    tabs = BC.search_tables(K, B, lab, tok)
    target, _, tok2 = data(LARGE, 128, 21)
    t = flat(tok2, chains=4)
    y, found, _ = MS.search_gpu(tabs, t, dims(LARGE), 20000, 0.6, seed=2, chains=4)
    f = found.reshape(-1, 4)
    solved = (f >= 0).any(1)
    first = np.where(f >= 0, f, 1 << 30).argmin(1)
    labs = torch.from_numpy(y.reshape(-1, 4, *target.shape[1:])[np.arange(len(target)), first]).long()
    built, _ = BE.build(BE.bricks_to_actions(BC.decode(labs), LARGE.done_action), target, LARGE)
    exact = check(built.numpy(), target.numpy())["exact"]
    assert solved.mean() > 0.95 and exact[solved].all()


def test_pseudolikelihood_fit_lowers_its_loss():
    _, lab, tok = data(SMALL, 200, 9)
    start = float(BC.pl_loss(torch.zeros(4, BC.C, BC.C), torch.zeros(2, BC.C), lab, tok))
    _, _, _, loss = BC.fit(lab, tok, lr=30.0, updates=40)
    assert abs(start - np.log(25)) < 1e-4 and loss < 0.2 * start


if __name__ == "__main__":
    torch.set_num_threads(4)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
