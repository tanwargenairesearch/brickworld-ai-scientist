"""Tests of paper_rules.py (the paper's statistics, stopping rules and search on the brick labels).
Run: .venv/bin/python -I -B energy_check/test_paper_rules.py (non-zero exit on failure; CPU only)."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import brick_cram as BC  # noqa: E402
import paper_rules as PR  # noqa: E402
from test_brick_cram import LARGE, SMALL, data, dims, flat  # noqa: E402


def random_costs(seed):
    g = torch.Generator().manual_seed(seed)
    K, B = torch.randn(4, BC.C, BC.C, generator=g), torch.randn(2, BC.C, generator=g)
    K[:, 0, 0] = 0.0                               # a pair of two empty cells is in no cell's local energy
    return K, B


def test_fields_and_energy_equal_those_of_brick_cram():
    K, B = random_costs(0)
    _, lab, tok = data(LARGE, 20, 1)
    t, y = flat(tok), lab.reshape(20, -1).numpy().astype(np.int64)
    ch = PR.Chains(K, B, t, dims(LARGE), y)
    L = BC.local_energies(K, B, lab, tok).reshape(20, -1, BC.C).numpy()         # float32 against float64
    assert np.abs(ch.g - L)[t >= 0].max() < 1e-4, "local energies differ"
    assert np.abs(ch.E - BC.board_energy(K.double(), B.double(), lab, tok).numpy()).max() < 1e-9, "board energies differ"
    own = np.take_along_axis(L, y[..., None], 2)[..., 0]
    assert np.abs(ch.e - own)[t >= 0].max() < 1e-4 and np.isinf(ch.margin[t < 0]).all()


def test_margin_and_delta_on_a_case_worked_by_hand():
    """A 1 x 1 x 2 grid holding one 1x2 brick (labels 2 and 4). The only non-zero cost rewards that pair by 1, so each
    cell's own label has local energy -1, every other label 0, and the margin is 1."""
    K, B = torch.zeros(4, BC.C, BC.C), torch.zeros(2, BC.C)
    K[0, 2, 4] = -1.0                              # offset (0, 1): label 2 with label 4 to its right
    lab, tok = torch.tensor([[[[2, 4]]]]), torch.zeros(1, 1, 1, 2, dtype=torch.long)
    st = PR.statistics(K, B, lab, tok)
    assert st["tau"] == -1.0 and st["ebar"] == -1.0 and st["delta"] == 1.0 and np.isnan(st["hbar"])
    ch = PR.Chains(K, B, np.zeros((1, 2), np.int32), (1, 1, 2), np.array([[2, 4]]))
    assert ch.margin.tolist() == [[1.0, 1.0]] and ch.E.tolist() == [-1.0]
    ch.move(np.array([0]), np.array([1]), np.array([7]))               # break the brick: both margins turn
    assert ch.E.tolist() == [0.0] and ch.margin.tolist() == [[0.0, -1.0]]


def test_cached_values_follow_the_moves():
    """300 sampled moves on random costs with a rule that never stops: `search` itself compares every cached field
    and energy with a recomputation, and raises if they drifted."""
    K, B = random_costs(2)
    _, lab, tok = data(SMALL, 16, 3)
    st = {**PR.statistics(K, B, lab, tok), "tau": -np.inf, "hbar": 1.0}
    out = PR.search(K, B, st, flat(tok), dims(SMALL), "tau", 1.0, 2, 300, seed=4)
    assert not out["stopped"].any() and (out["moves"] == 300).all()
    labels = torch.from_numpy(out["labels"].reshape(16, *tok.shape[1:])).long()
    assert np.abs(BC.board_energy(K.double(), B.double(), labels, tok).numpy() - out["energy"]).max() < 1e-6


def test_statistics_of_the_counted_fit():
    _, lab, tok = data(SMALL, 1000, 9)
    K, B, _ = BC.count_fit(lab, tok)
    st = PR.statistics(K, B, lab, tok)
    assert st["tau"] == 0.0 and st["ebar"] == 0.0 and st["hbar"] == 1.0 and st["delta"] >= 0.0
    assert st["tK"] == [0.0] * 4 and st["tB"] == [0.0] * 2
    one = PR.term_tables(K, B, st, lab, tok, "terms1")
    per_table = PR.term_tables(K, B, st, lab, tok, "terms")
    assert np.array_equal(one["bad8"], per_table["bad8"]) and np.array_equal(one["badB"], per_table["badB"])


def test_every_rule_stops_only_where_the_rule_holds():
    """Each rule on the hand-set rule costs: a design reported as stopped passes the rule when its labels are judged
    afresh, and under the term rules it is a tiling the world allows."""
    okK, okB = BC.true_rules()
    K, B = (~okK).float(), (~okB).float()
    K[:, 0, 0] = 0.0
    B[:, 0] = 0.0
    _, lab, tok = data(SMALL, 300, 5)
    st = PR.statistics(K, B, lab, tok)
    target, _, tok_l = data(LARGE, 48, 6)
    t = flat(tok_l)
    for rule in PR.RULES:
        tabs = PR.term_tables(K, B, st, lab, tok, rule) if rule in ("terms", "terms1") else None
        out = PR.search(K, B, st, t, dims(LARGE), rule, 0.6, 4, 3000, seed=1, tabs=tabs)
        s = out["stopped"]
        assert s.any(), f"{rule}: nothing stopped"
        ok = PR.passes_rule(K, B, st, t[s], dims(LARGE), out["labels"][s], rule, tabs)
        assert ok.all(), f"{rule}: a stopped design does not pass the rule"
        if rule in ("tau", "terms", "terms1"):     # energy 0 under the true rules is a valid tiling
            bad = BC.violations(out["labels"][s], t[s], BC.rule_tables(), BC.neighbours(*dims(LARGE))).any(1)
            assert not bad.any(), f"{rule}: stopped on a labelling that breaks a rule"


if __name__ == "__main__":
    torch.set_num_threads(4)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
