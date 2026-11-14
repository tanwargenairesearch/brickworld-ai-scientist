"""A learned pairwise energy for the brick-assembly world: fitted from solved designs, searched by single-cell moves.

The method follows "Classical Energies Learn to Reason 1,000x Faster than Looped Models" (arXiv 2611.17309): an energy
of pair and input costs, fitted at solved examples with no search inside the fit, then searched by focused single-cell
moves that stop on a test fitted from the training solutions. The paper fits by pseudolikelihood. The fit used here is
a count (`count_fit`); the paper's objective is kept as `fit`, for comparison. No code is copied from that paper's
repository.

Dimension names: n designs, h chains, z layers, y rows, x columns, m = z*y*x cells, a and c labels (C = 26),
d the 4 half offsets of the 3 x 3 window in a layer, k the 8 neighbours.

Labels. Every filled cell of a design carries one label that says which piece of which brick it is, and which cell of
that brick is the brick's anchor (a cell that rests on something). 0 is an empty cell (clamped, never moved).
    1            a 1x1 brick (it is its own anchor)
    2 + 2p + q   a 1x2 brick: p = 0 left, 1 right;  q = the position of the anchor
    6 + 2p + q   a 2x1 brick: p = 0 top, 1 bottom
    10 + 4p + q  a 2x2 brick: p = 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right
The anchor makes the world's support rule a rule about one cell (the anchor must rest on something) and about pairs of
cells (the cells of one brick name the same anchor).

Input. One token per cell: -1 empty, 0 filled and resting on the ground or on a filled cell, 1 filled over an empty cell.

Energy of a labelling y, with the labels of cells outside the grid read as 0:

    E(y) = sum over cells i and half offsets d of K[d, y_i, y_(i+d)]  +  sum over filled cells i of B[token_i, y_i]

K (4, C, C) and B (2, C) are the only parameters: 2,750 numbers, shared by every cell, layer and world size.

Supplied by hand, and fitted. Supplied: the label set (it names the four brick shapes, their pieces and an anchor), the
input tokens (they say whether a cell rests on something), locality (a 3 x 3 window in one layer) and translation
invariance. Much of the world's structure is in those choices. Fitted from solved designs: which pair terms and which
input terms may occur. No table of allowed terms is written into the model: `true_rules` writes one down for tests and
diagnostics only.

Two fits, both from solved designs alone.
- `count_fit`, the one used: a pair term or an input term costs 0 if it occurs in a training solution and 1 if it
  never does. This is one pass of counting: the Hebbian rule of an associative memory, which stores which labels occur
  side by side.
- `fit`, the paper's objective: pseudolikelihood at the labelled solutions (the mean over filled cells of
  -log softmax(-local energy) at the cell's label, the other cells held at their labels), by full-batch SGD with
  momentum and a cosine learning rate. The runs of it in `run_learned.py pl` and `pl_l1` reach a loss near 0 with no
  cost threshold that separates the forbidden terms from the allowed ones, with or without the paper's L1 penalty,
  under this module's stopping test (one threshold for all pair tables; not the paper's per-table rule). That is a
  finding about those runs and that test, not about the objective: the counting energy times 20 has loss 0 under the
  same objective and keeps the counting energy's stopping test. This fit was later searched with each of the paper's
  four stopping rules (`paper_rules.py`, `metal_paper_search.py`, `run_paper_rules_wide.py`); the README gives the
  results.
Neither fit runs a search or differentiates through one.

Stopping test (`search_tables`). A pair term is violated when its cost is above the highest pair cost that occurs in a
training solution, and an input term likewise. A chain stops at the first labelling with no violated term. Under
`count_fit` every training solution has energy 0, so the test is 'energy 0'.

What the count gets wrong. An allowed term that occurs in no training solution also costs 1. A design whose every
tiling needs such a term has no labelling of energy 0, and no search can stop on it. `unsolvable` proves that for a
given design by arc consistency; the README gives the measured rate.

Search. `search_reference` (NumPy) and the Metal kernel of `metal_search.py` run the same moves: pick a filled cell
that has a violated term, uniformly at random; redraw its label from softmax(-local energy / temp). Random numbers
come from a counter hash of (seed, chain, move), so the two implementations can be compared move for move.
"""
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from brickworld.env import BRICK_TYPES, design_hash, footprints, generate_designs, rest_heights, zobrist_codes  # noqa: E402

C = 26
HALF = ((0, 1), (1, -1), (1, 0), (1, 1))                         # (row, column) offsets, one per unordered pair
NB8 = HALF + tuple((-dr, -dc) for dr, dc in HALF)                 # the 8 neighbours, in the order of the search tables
CELLS = tuple(tuple((p // dx, p % dx) for p in range(dy * dx)) for dy, dx in BRICK_TYPES)   # cell offsets per type
BASE, MULT = (1, 2, 6, 10), (0, 2, 2, 4)
TYPE = np.array([-1, 0] + [1] * 4 + [2] * 4 + [3] * 16)
POS = np.array([0, 0] + [p for p in range(2) for _ in range(2)] * 2 + [p for p in range(4) for _ in range(4)])
ANCHOR = np.array([0, 0] + [0, 1] * 4 + [0, 1, 2, 3] * 4)


def label_of(t: int, p: int, q: int) -> int:
    return BASE[t] + MULT[t] * p + (q if t else 0)


# ---------------------------------------------------------------- solved designs and their labels

def replay(spec, acts: torch.Tensor) -> torch.Tensor:
    """Generator actions [n, k] (-1 unused) -> bricks [n, k, 4] long (type, layer, row, column), -1 rows unused."""
    n, kmax = acts.shape
    YX = spec.Y * spec.X
    M, _ = footprints(spec)
    height = torch.zeros(n, spec.Y, spec.X, dtype=torch.long)
    out = torch.full((n, kmax, 4), -1, dtype=torch.long)
    rows = torch.arange(n)
    for i in range(kmax):
        a = acts[:, i]
        act = a >= 0
        ac = a.clamp(min=0)
        z = rest_heights(height, spec)[rows, ac]
        pos = ac % YX
        brick = torch.stack([ac // YX, z, pos // spec.X, pos % spec.X], 1)
        height = torch.where(M[ac] & act.view(n, 1, 1), (z + 1).view(n, 1, 1), height)
        out[:, i] = torch.where(act.view(n, 1), brick, torch.full_like(brick, -1))
    return out


def solved_designs(spec, n: int, seed: int, exclude: torch.Tensor):
    """n designs from the frozen generator with the generator's own bricks, none of them in `exclude` (design hashes).
    Returns (target [n, z, y, x] bool, bricks [n, k, 4] long)."""
    gen = torch.Generator().manual_seed(seed)
    codes = zobrist_codes(spec)
    occs, bricks, have = [], [], 0
    while have < n:
        occ, acts = generate_designs(spec, 2048, gen, return_actions=True)
        keep = ~torch.isin(design_hash(occ, codes), exclude)
        occs.append(occ[keep])
        bricks.append(replay(spec, acts[keep]))
        have += int(keep.sum())
    return torch.cat(occs)[:n], torch.cat(bricks)[:n]


def tokens(target: torch.Tensor) -> torch.Tensor:
    """target [n, z, y, x] bool -> [n, z, y, x] long: -1 empty, 0 filled and resting on something, 1 filled over empty."""
    below = torch.ones_like(target)
    below[:, 1:] = target[:, :-1]
    return torch.where(target, (~below).long(), torch.full(target.shape, -1, dtype=torch.long))


def labels_from_bricks(target: torch.Tensor, bricks: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Labels [n, z, y, x] long of the solved designs. Each brick's anchor is drawn uniformly from its cells that rest
    on something."""
    n = target.shape[0]
    rests = tokens(target) == 0
    lab = torch.zeros(target.shape, dtype=torch.long)
    for i in range(bricks.shape[1]):
        t, z, y, x = bricks[:, i].unbind(1)
        for tt in range(len(BRICK_TYPES)):
            sel = (t == tt).nonzero().flatten()
            if len(sel) == 0:
                continue
            zz, yy, xx = z[sel], y[sel], x[sel]
            s = torch.stack([rests[sel, zz, yy + dy, xx + dx] for dy, dx in CELLS[tt]], 1)
            assert bool(s.any(1).all()), "a generator brick rests on nothing"
            q = torch.rand(s.shape, generator=gen).masked_fill(~s, -1.0).argmax(1)
            for p, (dy, dx) in enumerate(CELLS[tt]):
                lab[sel, zz, yy + dy, xx + dx] = BASE[tt] + MULT[tt] * p + (q if tt else 0)
    assert bool(((lab > 0) == target).all()) and n == lab.shape[0]
    return lab


# ---------------------------------------------------------------- the energy and its fit

def kernel(K: torch.Tensor) -> torch.Tensor:
    """K (4, C, C) -> W (C, C, 3, 3): W[a, c, 1 + dr, 1 + dc] is the cost of label a at a cell with label c at the cell
    (dr, dc) away, so that F.conv2d(one-hot labels, W) is every label's pair energy at every cell."""
    W = K.new_zeros(C, C, 3, 3)
    for d, (dr, dc) in enumerate(HALF):
        W[:, :, 1 + dr, 1 + dc] = K[d]
        W[:, :, 1 - dr, 1 - dc] = K[d].T
    return W


def local_energies(K, B, lab, tok):
    """[n, z, y, x, C]: the energy of each label at each cell, the other cells held at `lab` (less a constant)."""
    n, Z, Y, X = lab.shape
    onehot = F.one_hot(F.pad(lab.reshape(n * Z, Y, X), (1, 1, 1, 1)), C).permute(0, 3, 1, 2).float()
    field = F.conv2d(onehot, kernel(K)).permute(0, 2, 3, 1).reshape(n, Z, Y, X, C)
    return field + B[tok.clamp(min=0)]


def pl_loss(K, B, lab, tok):
    """Pseudolikelihood loss: the mean over filled cells of -log p(label | every other cell), over the labels 1..25."""
    free = (tok >= 0).reshape(-1)
    logits = -local_energies(K, B, lab, tok).reshape(-1, C)[:, 1:]
    nll = F.cross_entropy(logits, (lab.reshape(-1) - 1).clamp(min=0), reduction="none")
    return (nll * free).sum() / free.sum()


def board_energy(K, B, lab, tok):
    """E(y) [n] of labellings lab [n, z, y, x] (module docstring)."""
    n, Z, Y, X = lab.shape
    padded = F.pad(lab.reshape(n * Z, Y, X), (1, 1, 1, 1))
    pair = torch.zeros(n * Z)
    for d, (dr, dc) in enumerate(HALF):
        a, c = _pairs(padded, dr, dc)
        pair = pair + K[d][a, c].flatten(1).sum(1)
    own = B[tok.clamp(min=0)].gather(-1, lab.unsqueeze(-1)).squeeze(-1) * (tok >= 0)
    return pair.view(n, Z).sum(1) + own.flatten(1).sum(1)


def _pairs(padded, dr, dc):
    """Labels (a at i, c at i + (dr, dc)) of every cell pair of a padded label map [b, y + 2, x + 2]."""
    Hp, Wp = padded.shape[1:]
    c0, c1 = max(0, -dc), Wp - max(0, dc)
    return padded[:, :Hp - dr, c0:c1], padded[:, dr:, c0 + dc:c1 + dc]


def fit(lab, tok, lr: float, updates: int, momentum: float = 0.5, device: str = "cpu", nonneg: bool = False,
        l1: float = 0.0):
    """Fit K and B on labelled solutions by pseudolikelihood. nonneg: project every cost to >= 0 after each step, so
    the fit can only raise the cost of what it must rule out. l1: the paper's L1 penalty, a soft-threshold of
    (learning rate of the step) * l1 after each step. Returns (K, B on the CPU, seconds of the update loop, the loss
    before the last step)."""
    lab, tok = lab.to(device), tok.to(device)
    K = torch.zeros(4, C, C, device=device, requires_grad=True)
    B = torch.zeros(2, C, device=device, requires_grad=True)
    bufs = [torch.zeros_like(K), torch.zeros_like(B)]
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    loss = None
    for k in range(updates):
        loss = pl_loss(K, B, lab, tok)
        grads = torch.autograd.grad(loss, [K, B])
        lr_k = lr * 0.5 * (1.0 + math.cos(math.pi * k / updates))
        with torch.no_grad():
            for p, g, buf in zip((K, B), grads, bufs):
                buf.mul_(momentum).add_(g)
                p.sub_(lr_k * buf)
                if l1 > 0:
                    p.copy_(torch.sign(p) * (p.abs() - lr_k * l1).clamp(min=0.0))
                if nonneg:
                    p.clamp_(min=0.0)
    last = float(loss.detach())
    if device == "mps":
        torch.mps.synchronize()
    return K.detach().cpu(), B.detach().cpu(), time.perf_counter() - t0, last


# ---------------------------------------------------------------- the stopping test and the search tables

def class_counts(lab: torch.Tensor, tok: torch.Tensor):
    """How often each pair term and each input term occurs in labellings: (cK [4, C, C], cB [2, C]) long.
    Pairs with a cell outside the grid count with label 0 there; pairs of two empty cells do not count."""
    n, Z, Y, X = lab.shape
    padded = F.pad(lab.reshape(n * Z, Y, X), (1, 1, 1, 1))
    cK = torch.zeros(4, C, C, dtype=torch.long)
    for d, (dr, dc) in enumerate(HALF):
        a, c = _pairs(padded, dr, dc)
        cK[d] = torch.bincount((a * C + c).flatten(), minlength=C * C).view(C, C)
    cK[:, 0, 0] = 0
    free = tok >= 0
    cB = torch.bincount((tok[free] * C + lab[free]).flatten(), minlength=2 * C).view(2, C)
    return cK, cB


def search_tables(K, B, lab, tok, tol: float = 1e-3) -> dict:
    """The search's tables (NumPy), with the stopping thresholds fitted on the training labellings `lab`:
    W8 [8, C, C] float32 (W8[k, a, c]: cost of label a with label c at neighbour k), B [2, C] float32,
    bad8 [8, C, C] and badB [2, C] int32 (1 where a term is above its threshold), tau_pair, tau_input."""
    cK, cB = class_counts(lab, tok)
    tau_pair = float(K[cK > 0].max())
    tau_input = float(B[cB > 0].max())
    badK = K > tau_pair + tol
    badK[:, 0, 0] = False
    badB = B > tau_input + tol
    badB[:, 0] = False
    return {
        "W8": torch.cat([K, K.transpose(1, 2)]).numpy().astype(np.float32),
        "B": B.numpy().astype(np.float32),
        "bad8": torch.cat([badK, badK.transpose(1, 2)]).numpy().astype(np.int32),
        "badB": badB.numpy().astype(np.int32),
        "tau_pair": tau_pair, "tau_input": tau_input,
        "seen_pair_terms": int((cK > 0).sum()), "seen_input_terms": int((cB > 0).sum()),
    }


def count_fit(lab: torch.Tensor, tok: torch.Tensor):
    """The one-shot fit: a pair term or an input term costs 0 if it occurs in a training labelling and 1 if it never
    does. Returns (K [4, C, C], B [2, C], seconds). Every training solution then has energy 0, so the stopping test
    of `search_tables` (no term above the highest cost in a training solution) is 'energy 0'."""
    t0 = time.perf_counter()
    cK, cB = class_counts(lab, tok)
    K, B = (cK == 0).float(), (cB == 0).float()
    K[:, 0, 0] = 0.0
    B[:, 0] = 0.0
    return K, B, time.perf_counter() - t0


def compare_with_rules(K_bad: torch.Tensor, B_bad: torch.Tensor) -> dict:
    """How the terms a fit flags as violated (bool [4, C, C] and [2, C]) compare with `true_rules`. A diagnostic."""
    okK, okB = true_rules()
    okK[:, 0, 0] = True
    pair = torch.ones_like(okK)
    pair[:, 0, 0] = False
    inp = torch.ones_like(okB)
    inp[:, 0] = False
    return {
        "forbidden_pair_terms": int((~okK & pair).sum()), "forbidden_pair_terms_flagged": int((~okK & K_bad & pair).sum()),
        "allowed_pair_terms": int((okK & pair).sum()), "allowed_pair_terms_flagged": int((okK & K_bad & pair).sum()),
        "forbidden_input_terms": int((~okB & inp).sum()), "forbidden_input_terms_flagged": int((~okB & B_bad & inp).sum()),
        "allowed_input_terms": int((okB & inp).sum()), "allowed_input_terms_flagged": int((okB & B_bad & inp).sum()),
    }


def true_rules():
    """The world's rules as allowed terms: (okK [4, C, C] bool, okB [2, C] bool). For tests and diagnostics only:
    nothing in the model, the fit or the search reads this.
    A pair is allowed when neither cell needs a different label at the other one: every piece of a multi-cell brick
    needs the brick's other pieces, with the same anchor, at their offsets. An anchor cell must rest on something."""
    partner = [dict() for _ in range(C)]
    for lab in range(1, C):
        t, p, q = int(TYPE[lab]), int(POS[lab]), int(ANCHOR[lab])
        for p2, (dy, dx) in enumerate(CELLS[t]):
            if p2 != p:
                partner[lab][(dy - CELLS[t][p][0], dx - CELLS[t][p][1])] = label_of(t, p2, q)
    okK = torch.ones(4, C, C, dtype=torch.bool)
    for d, (dr, dc) in enumerate(HALF):
        for a in range(C):
            for c in range(C):
                okK[d, a, c] = partner[a].get((dr, dc), c) == c and partner[c].get((-dr, -dc), a) == a
    okB = torch.ones(2, C, dtype=torch.bool)
    okB[1, 1:] = torch.from_numpy(POS[1:] != ANCHOR[1:])
    okB[1, 1] = False
    return okK, okB


def rule_tables() -> dict:
    """Search tables set by hand from `true_rules` (cost 1 on every forbidden term): the hand-set energy of this label
    set, used to test the search and the kernel without a fit."""
    okK, okB = true_rules()
    K = (~okK).float()
    K[:, 0, 0] = 0.0
    B = (~okB).float()
    bad = (K > 0)
    return {"W8": torch.cat([K, K.transpose(1, 2)]).numpy().astype(np.float32), "B": B.numpy().astype(np.float32),
            "bad8": torch.cat([bad, bad.transpose(1, 2)]).numpy().astype(np.int32),
            "badB": (B > 0).numpy().astype(np.int32), "tau_pair": 0.0, "tau_input": 0.0}


# ---------------------------------------------------------------- the reference search (NumPy)

def hash32(x):
    """A 32-bit mixing function on uint32 arrays (the kernel's `hash32`)."""
    x = x ^ (x >> np.uint32(16))
    x = x * np.uint32(0x7FEB352D)
    x = x ^ (x >> np.uint32(15))
    x = x * np.uint32(0x846CA68B)
    return x ^ (x >> np.uint32(16))


def rnd(seed, pid, t, s):
    """The random 32-bit number of (seed, chain pid, move t, stream s); uint32 arrays that broadcast."""
    u = lambda v: np.asarray(v, dtype=np.uint32)  # noqa: E731
    with np.errstate(over="ignore"):
        return hash32(u(seed) ^ hash32(u(pid) * np.uint32(0x9E3779B9) + hash32(u(t) * np.uint32(0x85EBCA6B) + u(s))))


def neighbours(Z: int, Y: int, X: int) -> np.ndarray:
    """[m, 8] int32: the cell index of each neighbour (order NB8), -1 outside the grid."""
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    out = np.full((Z * Y * X, 8), -1, dtype=np.int32)
    for k, (dr, dc) in enumerate(NB8):
        y2, x2 = y + dr, x + dc
        ok = (y2 >= 0) & (y2 < Y) & (x2 >= 0) & (x2 < X)
        out[ok.ravel(), k] = ((z * Y + y2) * X + x2)[ok]
    return out


def init_labels(tok: np.ndarray, seed: int) -> np.ndarray:
    """A uniformly random label 1..25 at every filled cell, 0 elsewhere. tok [h, m] -> [h, m] int32."""
    h, m = tok.shape
    r = rnd(seed, np.arange(h)[:, None], np.arange(m)[None, :], 1000)
    return np.where(tok >= 0, 1 + (r % np.uint32(C - 1)).astype(np.int32), 0).astype(np.int32)


def violations(y: np.ndarray, tok: np.ndarray, tabs: dict, nbr: np.ndarray, cells=None) -> np.ndarray:
    """[h, j] bool: does each cell have a violated term? cells [h, j] (default: every cell; -1 entries give False)."""
    h = y.shape[0]
    if cells is None:
        cells = np.broadcast_to(np.arange(y.shape[1]), y.shape)
    rows = np.arange(h)[:, None]
    j = np.clip(cells, 0, None)
    a, s = y[rows, j], tok[rows, j]
    nb = nbr[j]                                                        # [h, j, 8]
    c = np.where(nb >= 0, y[rows[:, :, None], np.clip(nb, 0, None)], 0)
    bad = tabs["badB"][np.clip(s, 0, None), a] != 0
    bad |= (tabs["bad8"][np.arange(8)[None, None, :], a[:, :, None], c] != 0).any(2)
    return bad & (s >= 0) & (cells >= 0)


def unsolvable(tabs: dict, tok: np.ndarray, dims, sweeps: int = 1000) -> np.ndarray:
    """[h] bool: True where NO labelling of a design can pass the stopping test of `tabs`, so no search can stop on it.
    A proof by arc consistency. Every filled cell starts with the labels its input term allows; a label is struck out
    when some neighbour has no remaining label that the pair term allows beside it; a design whose cell runs out of
    labels has no labelling without a violated term. False proves nothing: arc consistency can miss an impossibility.
    tok [h, m] int32, one row per design. A diagnostic: nothing in the fit or the search calls this."""
    Z, Y, X = dims
    h, m = tok.shape
    nbr = neighbours(Z, Y, X)
    okB = np.concatenate([np.zeros((2, 1), bool), tabs["badB"][:, 1:] == 0], 1)             # [2, C], label 0 never
    dom = np.zeros((h, m + 1, C), np.float32)                                              # row m: outside the grid
    dom[:, :m] = np.where((tok >= 0)[:, :, None], okB[np.clip(tok, 0, None)], np.eye(C, dtype=bool)[0])
    dom[:, m, 0] = 1.0
    ok8 = (tabs["bad8"] == 0).astype(np.float32)                                           # [8, C, C]: a here, c there
    filled = tok >= 0
    nb = np.where(nbr >= 0, nbr, m)                                                        # [m, 8]
    for _ in range(sweeps):
        keep = np.ones((h, m, C), bool)
        for k in range(8):
            keep &= (dom[:, nb[:, k]] @ ok8[k].T) > 0                                      # some allowed c remains
        new = dom[:, :m] * np.where(filled[:, :, None], keep, True)
        if np.array_equal(new, dom[:, :m]):
            break
        dom[:, :m] = new
    return (dom[:, :m].sum(2) == 0).any(1)


def search_reference(tabs: dict, tok: np.ndarray, dims, max_moves: int, temp: float, seed: int, y0=None):
    """The search on the CPU, one row of `tok` [h, m] per chain. Returns (y [h, m] int32, found [h] int32): `found`
    is the number of moves a chain had made when it first had no violated term, or -1."""
    Z, Y, X = dims
    nbr = neighbours(Z, Y, X)
    y = init_labels(tok, seed) if y0 is None else y0.copy()
    h = y.shape[0]
    viol = violations(y, tok, tabs, nbr)
    found = np.full(h, -1, dtype=np.int32)
    W8, B = tabs["W8"], tabs["B"]
    streams = (1 + np.arange(1, C))[None, :]
    for t in range(max_moves + 1):
        nflag = viol.sum(1)
        found[(found < 0) & (nflag == 0)] = t
        idx = np.nonzero(found < 0)[0]
        if t == max_moves or len(idx) == 0:
            break
        r = (rnd(seed, idx, t, 0) % nflag[idx].astype(np.uint32)).astype(np.int64)
        cell = (np.cumsum(viol[idx], 1) > r[:, None]).argmax(1)
        nb = nbr[cell]
        ynb = np.where(nb >= 0, y[idx[:, None], np.clip(nb, 0, None)], 0)
        L = B[tok[idx, cell]].copy()
        for k in range(8):
            L += W8[k][:, ynb[:, k]].T
        if temp > 0:
            u = (rnd(seed, idx[:, None], t, streams) >> np.uint32(8)).astype(np.float32) * np.float32(1.0 / 16777216.0)
            g = -np.log(-np.log(np.maximum(u, np.float32(1e-12))))
            new = 1 + (-L[:, 1:] / np.float32(temp) + g).argmax(1)
        else:
            new = 1 + L[:, 1:].argmin(1)
        y[idx, cell] = new
        touched = np.concatenate([cell[:, None], nb], 1)
        v = violations(y[idx], tok[idx], tabs, nbr, touched)
        ok = touched >= 0
        viol[np.broadcast_to(idx[:, None], touched.shape)[ok], touched[ok]] = v[ok]
    return y, found


# ---------------------------------------------------------------- from labels to bricks

def decode(lab: torch.Tensor) -> torch.Tensor:
    """Labels [n, z, y, x] -> bricks [n, t, z, y, x] bool: one brick wherever a label is a brick's first piece."""
    first = torch.from_numpy((POS == 0) & (TYPE >= 0))
    kind = torch.from_numpy(TYPE.clip(0))
    return torch.stack([first[lab] & (kind[lab] == t) for t in range(len(BRICK_TYPES))], 1)
