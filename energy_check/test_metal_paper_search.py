"""Tests of the Metal kernel of metal_paper_search.py against the NumPy search of paper_rules.py.
Run: .venv/bin/python -I -B energy_check/test_metal_paper_search.py (non-zero exit on failure; needs an Apple GPU)."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import brick_cram as BC  # noqa: E402
import metal_paper_search as MP  # noqa: E402
import paper_rules as PR  # noqa: E402
from test_brick_cram import LARGE, SMALL, data, dims, flat  # noqa: E402

GRID_ST = {"ebar": 0.31, "hbar": 1.0, "delta": 0.5, "tau": 0.0}       # thresholds that no sum of 1/64ths can equal


def grid_costs(seed):
    """Costs on a grid of 1/64: every sum is exact in float32 and in float64, in any order."""
    rng = np.random.default_rng(seed)
    K = torch.from_numpy(rng.integers(-64, 64, (4, BC.C, BC.C)).astype(np.float32) / 64)
    K[:, 0, 0] = 0.0
    return K, torch.from_numpy(rng.integers(-64, 64, (2, BC.C)).astype(np.float32) / 64)


def rule_costs():
    okK, okB = BC.true_rules()
    K, B = (~okK).float(), (~okB).float()
    K[:, 0, 0] = 0.0
    B[:, 0] = 0.0
    return K, B


def both(K, B, st, t, spec, rule, temperature, cap, chunk, tabs=None, y0=None, seed=11):
    ref = PR.search(K, B, st, t, dims(spec), rule, temperature, 1, cap, seed, tabs=tabs, hashed=True, y0=y0)
    y, found, energy, _ = MP.search_chains(K, B, st, t, dims(spec), rule, temperature, cap, seed, tabs=tabs,
                                           chunk=chunk, y0=y0)
    return ref, y, found, energy


def test_kernel_equals_reference_move_for_move_on_random_costs():
    """Rules tau, local and fixed at temperature 0 on random grid costs: the same labels, stop moves and energies for
    every chain, over several move budgets and launch sizes. tau is set so that about half of the chains stop."""
    K, B = grid_costs(0)
    _, _, tok = data(LARGE, 96, 3)
    t = flat(tok)
    probe = PR.search(K, B, {**GRID_ST, "tau": -np.inf}, t, dims(LARGE), "tau", 0.0, 1, 40, 11, hashed=True)
    st = {**GRID_ST, "tau": float(np.median(probe["last_energy"])) + 0.25}
    for rule in ("tau", "local", "fixed"):
        stops = set()
        for cap, chunk in ((0, 64), (1, 64), (7, 2), (40, 7), (300, 64)):
            ref, y, found, energy = both(K, B, st, t, LARGE, rule, 0.0, cap, chunk)
            assert np.array_equal(ref["stopped"], found >= 0), f"{rule}: stops differ with a budget of {cap}"
            assert np.array_equal(ref["moves"][ref["stopped"]], found[found >= 0]), f"{rule}: stop moves differ ({cap})"
            assert np.array_equal(ref["last"], y), f"{rule}: labels differ with a budget of {cap}"
            assert np.abs(ref["last_energy"] - energy).max() < 1e-3, f"{rule}: energies differ with a budget of {cap}"
            stops |= set(found.tolist())
        if rule != "local":                         # a margin of 0.5 at every cell is rare on random costs
            assert -1 in stops and len([v for v in stops if v > 0]) >= 5, f"{rule}: stops were not exercised"


def test_kernel_equals_reference_on_the_term_rule():
    """The term rule on the hand-set rule costs, from solutions with 0 to 3 wrong labels, at temperature 0."""
    K, B = rule_costs()
    _, lab_s, tok_s = data(SMALL, 300, 5)
    st = PR.statistics(K, B, lab_s, tok_s)
    tabs = PR.term_tables(K, B, st, lab_s, tok_s, "terms")
    _, lab, tok = data(LARGE, 96, 6)
    t = flat(tok)
    y0 = lab.reshape(96, -1).numpy().astype(np.int32).copy()
    rng = np.random.default_rng(0)
    for i in range(32, 96):
        cells = rng.choice(np.nonzero(t[i] >= 0)[0], size=1 + i % 3, replace=False)
        y0[i, cells] = 1 + (y0[i, cells] - 1 + rng.integers(1, 25, len(cells))) % 25
    stops = set()
    for cap, chunk in ((0, 64), (1, 64), (5, 2), (40, 7), (300, 64)):
        ref, y, found, _ = both(K, B, st, t, LARGE, "terms", 0.0, cap, chunk, tabs=tabs, y0=y0)
        assert np.array_equal(ref["stopped"], found >= 0) and np.array_equal(ref["last"], y), f"differ at budget {cap}"
        assert np.array_equal(ref["moves"][ref["stopped"]], found[found >= 0])
        assert (found[:32] == 0).all() and (found[32:] != 0).all(), "a solved chain must stop at move 0, and only those"
        stops |= set(found.tolist())
    assert len([v for v in stops if v > 0]) >= 5, "stops at several moves were not exercised"


def test_every_kernel_stop_holds_in_float64_at_temperature_above_0():
    """Each rule on the hand-set rule costs with sampling: every stop the kernel reports holds when the labels are
    judged again in float64, and under the energy and term rules it is a tiling the world allows."""
    K, B = rule_costs()
    _, lab_s, tok_s = data(SMALL, 300, 5)
    st = PR.statistics(K, B, lab_s, tok_s)
    _, _, tok = data(LARGE, 128, 7)
    t = flat(tok)
    for rule in PR.RULES:
        tabs = PR.term_tables(K, B, st, lab_s, tok_s, rule) if rule in ("terms", "terms1") else None
        out = MP.search(K, B, st, t, dims(LARGE), rule, 0.6, 4, 5000, seed=3, tabs=tabs)
        s = out["stopped"]
        assert s.mean() > 0.5, f"{rule}: only {s.mean():.2f} stopped"
        assert out["confirmed"][s].all(), f"{rule}: a stop does not hold in float64"
        assert ((out["labels"] > 0) == (t >= 0)).all(), "a label moved onto an empty cell or off a filled one"
        if rule in ("tau", "terms", "terms1"):
            bad = BC.violations(out["labels"][s], t[s], BC.rule_tables(), BC.neighbours(*dims(LARGE))).any(1)
            assert not bad.any(), f"{rule}: stopped on a labelling that breaks a rule"


def test_kernel_draws_labels_from_the_softmax():
    """One free cell: the kernel's first label follows softmax(-L / T), T = temperature * hbar."""
    rng = np.random.default_rng(1)
    K = torch.from_numpy(rng.normal(size=(4, BC.C, BC.C)).astype(np.float32))
    B = torch.from_numpy(rng.normal(size=(2, BC.C)).astype(np.float32))
    h = 40000
    tok = np.full((h, 4), -1, np.int32)           # a 1 x 2 x 2 grid with one filled cell
    tok[:, 0] = 0
    y0 = np.zeros((h, 4), np.int32)
    y0[:, 0] = 3
    st = {"tau": -np.inf, "ebar": 0.0, "hbar": 2.0, "delta": 0.0}
    Kn = K.numpy()
    L = B.numpy()[0] + np.concatenate([Kn, Kn.transpose(0, 2, 1)])[:, :, 0].sum(0)
    p = np.exp(-L[1:] / 0.7 - (-L[1:] / 0.7).max())
    p /= p.sum()
    y, _, _, _ = MP.search_chains(K, B, st, tok, (1, 2, 2), "tau", 0.35, 1, seed=3, y0=y0)
    freq = np.bincount(y[:, 0], minlength=BC.C)[1:] / h
    assert np.abs(freq - p).max() < 0.01, f"label frequencies are off by {np.abs(freq - p).max():.4f}"


if __name__ == "__main__":
    torch.set_num_threads(4)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
