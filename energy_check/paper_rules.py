"""The paper's own inference statistics, stopping rules and focused search, on the labels of brick_cram.py.

Written after reading the paper (arXiv 2611.17309, App. C) and its public code (github.com/Prefrontal-AI/CRAM,
Apache-2.0: cram/cram_graph_coloring.py and cram/cram_nqueens.py), whose definitions this follows. No code is copied.
The search here is NumPy on the CPU. It is separate from the Metal kernel, which knows one rule only.

Dimension names: r chains (n designs x `chains`), m cells, a and c labels (C = 26), k the 8 neighbours.

Statistics, fitted from the training labellings (`statistics`):
  tau    the highest board energy of a training solution
  ebar   the mean, over filled cells, of the local energy of the cell's own label
  hbar   the mean positive pair cost: the unit of temperature and of the focus threshold
  delta  the smallest margin of a filled cell: lowest local energy of another label, less that of its own label
  tK[4], tB[2]  per cost table (one per offset, one per input token), the highest cost paid by a training solution

Stopping rules (the paper uses one per task):
  tau     board energy <= tau + 1e-3                                   (Sudoku)
  terms   every pair term and input term <= its table's threshold      (graph coloring; one threshold per table)
  terms1  the same with one threshold over all pair terms and one over all input terms: `brick_cram.search_tables`,
          the rule of this folder's scored runs
  local   every filled cell has a margin of at least delta - 1e-3      (N-Queens)
  fixed   no single-cell change lowers the energy                      (maze; moves at temperature 0)

Moves.
  tau, terms, terms1: pick uniformly among the filled cells whose own local energy exceeds ebar + theta * hbar (among
          all filled cells when there is none); draw the label from softmax(-local energy / T), T = temperature * hbar.
  local:  pick uniformly among the filled cells whose label is not the one of lowest local energy (among all filled
          cells when there is none); draw the label the same way.
  fixed:  set a uniformly random cell that is not at its lowest-energy label to that label.
A design runs `chains` chains from uniformly random labels and stops at the first chain that passes the rule.
A design that never stops returns the lowest-energy board any of its chains saw.
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import brick_cram as BC  # noqa: E402

C = BC.C
RULES = ("tau", "terms", "terms1", "local", "fixed")


def statistics(K: torch.Tensor, B: torch.Tensor, lab: torch.Tensor, tok: torch.Tensor) -> dict:
    """The inference statistics of costs K [4, C, C], B [2, C] on training labellings lab, tok [n, z, y, x], in the
    search's own arithmetic (float64)."""
    n = len(lab)
    t = tok.reshape(n, -1).numpy().astype(np.int32)
    ch = Chains(K, B, t, tuple(lab.shape[1:]), lab.reshape(n, -1).numpy().astype(np.int64))
    filled = t >= 0
    cK, cB = BC.class_counts(lab, tok)
    Kd, Bd = K.double(), B.double()
    pos = Kd[Kd > 0]
    return {
        "tau": float(ch.E.max()),
        "ebar": float(ch.e[filled].mean()),
        "hbar": float(pos.mean()) if pos.numel() else float("nan"),
        "delta": float(ch.margin[filled].min()),
        "tK": _thresholds(Kd, cK),
        "tB": _thresholds(Bd, cB),
        "share_of_training_cells_at_their_lowest_label": float((ch.margin[filled] >= 0).mean()),
    }


def _thresholds(costs: torch.Tensor, counts: torch.Tensor) -> list:
    """Per cost table, the highest cost a training solution pays. A table with no term in a training solution takes
    the highest threshold of the other tables (the paper's code does the same for a state that never occurs)."""
    seen = [costs[i][counts[i] > 0] for i in range(len(costs))]
    top = max(float(v.max()) for v in seen if v.numel())
    return [float(v.max()) if v.numel() else top for v in seen]


def term_tables(K: torch.Tensor, B: torch.Tensor, st: dict, lab, tok, rule: str) -> dict:
    """bad8 [8, C, C] and badB [2, C] int32 for the two term rules (1 where a term is above its threshold)."""
    if rule == "terms1":
        tabs = BC.search_tables(K, B, lab, tok)
        return {"bad8": tabs["bad8"], "badB": tabs["badB"]}
    badK = torch.stack([K[d].double() > st["tK"][d] + 1e-6 for d in range(4)])
    badK[:, 0, 0] = False
    badB = torch.stack([B[s].double() > st["tB"][s] + 1e-6 for s in range(2)])
    badB[:, 0] = False
    return {"bad8": torch.cat([badK, badK.transpose(1, 2)]).numpy().astype(np.int32), "badB": badB.numpy().astype(np.int32)}


class Chains:
    """The state of r chains on one grid size: labels y, cached local energies g, board energy E, and per cell the
    own-label local energy e and the margin (inf at empty cells; e is -inf there)."""

    def __init__(self, K, B, tok: np.ndarray, dims, y: np.ndarray):
        Z, Y, X = dims
        self.nbr = BC.neighbours(Z, Y, X)                                   # [m, 8], -1 outside the grid
        Kn = K.double().numpy()
        self.W8 = np.concatenate([Kn, Kn.transpose(0, 2, 1)])               # [8, C, C]: label a here, c at neighbour k
        self.B = B.double().numpy()
        self.tok, self.filled, self.y = tok, tok >= 0, y
        self.g = self.fields(y)
        self.E = self.energy(y, self.g)
        r, m = y.shape
        self.e = np.full((r, m), -np.inf)
        self.margin = np.full((r, m), np.inf)
        cells = np.broadcast_to(np.arange(m), (r, m))
        self.refresh(np.arange(r), cells)

    def fields(self, y: np.ndarray) -> np.ndarray:
        """g [r, m, C]: the local energy of every label at every cell, the other cells held at y."""
        ynb = np.where(self.nbr[None] >= 0, y[:, np.clip(self.nbr, 0, None)], 0)        # [r, m, 8]
        g = self.B[np.clip(self.tok, 0, None)].copy()
        for k in range(8):
            g += self.W8[k][:, ynb[:, :, k]].transpose(1, 2, 0)
        return g

    def energy(self, y: np.ndarray, g: np.ndarray) -> np.ndarray:
        """E [r]: every filled cell's input cost, plus every pair cost once (a pair of two filled cells is seen from
        both ends, so each end counts half)."""
        own = np.take_along_axis(g, y[..., None], 2)[..., 0]                 # input cost + all 8 pair costs
        nb_filled = np.where(self.nbr[None] >= 0, self.filled[:, np.clip(self.nbr, 0, None)], False)   # [r, m, 8]
        ynb = np.where(self.nbr[None] >= 0, y[:, np.clip(self.nbr, 0, None)], 0)
        half = np.zeros(y.shape)
        for k in range(8):
            half += np.where(nb_filled[:, :, k], 0.5, 0.0) * self.W8[k][y, ynb[:, :, k]]
        return np.where(self.filled, own - half, 0.0).sum(1)

    def refresh(self, rows: np.ndarray, cells: np.ndarray):
        """Recompute e and margin at `cells` [len(rows), j] (-1 entries are skipped) from the cached fields."""
        ok = cells >= 0
        rr = np.broadcast_to(rows[:, None], cells.shape)[ok]
        cc = cells[ok]
        gt = self.g[rr, cc]                                                 # [q, C]
        yt = self.y[rr, cc]
        q = np.arange(len(rr))
        own = gt[q, yt]
        gt = gt.copy()
        gt[q, yt] = np.inf
        gt[:, 0] = np.inf
        f = self.filled[rr, cc]
        self.e[rr, cc] = np.where(f, own, -np.inf)
        self.margin[rr, cc] = np.where(f, gt.min(1) - own, np.inf)

    def move(self, rows: np.ndarray, v: np.ndarray, new: np.ndarray):
        """Set cell v [len(rows)] of each chain in `rows` to label `new`, and update every cached quantity."""
        old = self.y[rows, v]
        local = self.g[rows, v]
        q = np.arange(len(rows))
        self.E[rows] += local[q, new] - local[q, old]
        self.y[rows, v] = new
        nb = self.nbr[v]                                                    # [len(rows), 8]
        for k in range(8):
            j, kk = nb[:, k], (k + 4) % 8                                   # v is neighbour kk of its neighbour k
            ok = j >= 0
            self.g[rows[ok], j[ok]] += self.W8[kk][:, new[ok]].T - self.W8[kk][:, old[ok]].T
        self.refresh(rows, np.concatenate([v[:, None], nb], 1))


def search(K, B, st: dict, tok: np.ndarray, dims, rule: str, temperature: float, chains: int, cap: int, seed: int,
           tabs: dict = None, theta: float = 0.45) -> dict:
    """The focused search of the module docstring on designs tok [n, m] int32.
    Returns labels [n, m] int32 (the design's first chain to stop, else its lowest-energy board), stopped [n] bool,
    moves [n] (of the returned chain) and energy [n]. Every stop is checked again from recomputed fields."""
    assert rule in RULES and (rule not in ("terms", "terms1") or tabs is not None)
    n, m = tok.shape
    inst = np.repeat(np.arange(n), chains)
    t = tok[inst]
    rng = np.random.default_rng(seed)
    y = np.where(t >= 0, rng.integers(1, C, size=t.shape), 0).astype(np.int64)
    ch = Chains(K, B, t, dims, y)
    r = len(inst)
    T = temperature * st["hbar"] if rule != "fixed" else 0.0
    focus_at = st["ebar"] + theta * st["hbar"]
    nbr = ch.nbr

    def violated(rows, cells):
        """[len(rows), j] bool: has the cell a term above its threshold? (-1 entries give False.)"""
        j = np.clip(cells, 0, None)
        a, s = ch.y[rows[:, None], j], ch.tok[rows[:, None], j]
        nb = nbr[j]
        c = np.where(nb >= 0, ch.y[rows[:, None, None], np.clip(nb, 0, None)], 0)
        bad = tabs["badB"][np.clip(s, 0, None), a] != 0
        bad |= (tabs["bad8"][np.arange(8)[None, None, :], a[:, :, None], c] != 0).any(2)
        return bad & (s >= 0) & (cells >= 0)

    viol = violated(np.arange(r), np.broadcast_to(np.arange(m), (r, m))) if tabs is not None else None

    def passes(rows):
        if rule == "tau":
            return ch.E[rows] <= st["tau"] + 1e-3
        if rule in ("terms", "terms1"):
            return ~viol[rows].any(1)
        if rule == "local":
            return (ch.margin[rows] >= st["delta"] - 1e-3).all(1)
        return (ch.margin[rows] >= 0).all(1)

    stopped = passes(np.arange(r))
    alive = ~np.repeat(stopped.reshape(n, chains).any(1), chains)
    moves = np.zeros(r, np.int64)
    best, bestE = ch.y.copy(), ch.E.copy()
    for _ in range(cap):
        act = np.flatnonzero(alive)
        if len(act) == 0:
            break
        hot = (ch.e[act] > focus_at) if rule in ("tau", "terms", "terms1") else (ch.margin[act] < 0)
        cand = np.where(hot.any(1)[:, None], hot, ch.filled[act])
        pick = np.floor(rng.random(len(act)) * cand.sum(1)).astype(np.int64)
        v = (np.cumsum(cand, 1) > pick[:, None]).argmax(1)
        local = ch.g[act, v][:, 1:]
        if T > 0:
            u = rng.uniform(np.finfo(float).tiny, 1.0, size=local.shape)
            new = 1 + (-local / T - np.log(-np.log(u))).argmax(1)
        else:
            new = 1 + local.argmin(1)
        ch.move(act, v, new)
        if viol is not None:
            touched = np.concatenate([v[:, None], nbr[v]], 1)
            ok = touched >= 0
            viol[np.broadcast_to(act[:, None], touched.shape)[ok], touched[ok]] = violated(act, touched)[ok]
        moves[act] += 1
        better = act[ch.E[act] < bestE[act]]
        best[better], bestE[better] = ch.y[better], ch.E[better]
        stopped[act] = passes(act)
        alive[act] = ~np.repeat(stopped.reshape(n, chains).any(1), chains)[act]

    # every cached quantity against a recomputation, and every stop against the rule on recomputed values
    g2 = ch.fields(ch.y)
    assert np.abs(g2 - ch.g).max() < 1e-6 and np.abs(ch.energy(ch.y, g2) - ch.E).max() < 1e-6, "cached fields drifted"
    if viol is not None:
        assert np.array_equal(viol, violated(np.arange(r), np.broadcast_to(np.arange(m), (r, m))))
    s2 = stopped.reshape(n, chains)
    done = s2.any(1)
    first = s2.argmax(1)
    lowest = bestE.reshape(n, chains).argmin(1)
    chain = np.where(done, first, lowest)
    idx = np.arange(n) * chains + chain
    labels = np.where(done[:, None], ch.y[idx], best[idx]).astype(np.int32)
    return {"labels": labels, "stopped": done, "moves": moves[idx], "energy": np.where(done, ch.E[idx], bestE[idx])}


def passes_rule(K, B, st: dict, tok: np.ndarray, dims, y: np.ndarray, rule: str, tabs: dict = None) -> np.ndarray:
    """[n] bool: does the labelling y [n, m] of designs tok pass the stopping rule? No search."""
    ch = Chains(K, B, tok, dims, y.astype(np.int64))
    if rule == "tau":
        return ch.E <= st["tau"] + 1e-3
    if rule in ("terms", "terms1"):
        return ~BC.violations(y.astype(np.int32), tok, tabs, ch.nbr).any(1)
    if rule == "local":
        return (ch.margin >= st["delta"] - 1e-3).all(1)
    return (ch.margin >= 0).all(1)
