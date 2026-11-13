"""A hand-set pairwise energy for the brick-assembly world, searched by single-brick moves.

No weight is learned and no policy is trained. The question this answers: can a few hand-set pairwise rules, with a
local search, build every design in the world? (The check of "Classical Energies Learn to Reason 1,000x Faster than
Looped Models", arXiv 2611.17309, section 4.3, applied to this world. No code is copied from that paper's repository.)

Dimension names: n boards (designs x chains), t brick types (4), z layers, y rows, x columns.

Variables. One on/off variable u[t, z, y, x] per brick: type t resting on layer z with its anchor at (y, x).

Energy of a board u for a target design g (1 at a filled cell):

    E(u) = sum over cells c of (cover_c(u) - g_c)^2  +  mu * (number of bricks on),

with cover_c(u) the number of 'on' bricks whose footprint holds cell c. Expanded, this is a pairwise energy:
the pair cost of two bricks is 2 per cell they share, and the input cost of a brick is mu + (1 - 2 g_c) summed over its
cells (`pair_form` returns both; tests check the identity). The first term is 0 exactly when the bricks tile the target.

Clamps (the input). A brick is clamped off unless it fits in the grid, lies inside the target and, above layer 0, has at
least one target cell directly below one of its cells (`allowed_bricks`). The last rule is the world's drop rule: with
the layers built from the bottom up, a brick placed over such a cell comes to rest on the intended layer.

Search (`search`). Chains start from the empty board. One move picks one brick uniformly at random among the free
bricks that touch a violated cell (cover != target); if there is none, among all free bricks. The brick is switched on
with probability sigmoid(-dE / temp), dE being the energy with it on minus the energy with it off (at temp 0: on when
dE < 0, unchanged when dE = 0). A chain stops at the first board whose cover term is 0 and whose brick count is at most
the world's step limit; a design stops with its first chain.

Build (`bricks_to_actions`, `build`). The 'on' bricks, lowest layer first, are placed in the frozen `BrickWorld`.
The caller scores the result with the frozen checker of `evaluate.py`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from brickworld.env import BRICK_TYPES, BrickWorld  # noqa: E402

T = len(BRICK_TYPES)


def brick_sums(cells: torch.Tensor) -> torch.Tensor:
    """cells [n, z, y, x] float -> [n, t, z, y, x]: the sum of `cells` over each brick's footprint.
    A footprint that leaves the grid sums over the part inside (such bricks are never free).
    `cells` must hold whole numbers: the result is rounded, because a batched convolution can return 0.99999994 for 1."""
    n, Z, Y, X = cells.shape
    m = cells.reshape(n * Z, 1, Y, X)
    out = []
    for dy, dx in BRICK_TYPES:
        s = F.conv2d(F.pad(m, (0, dx - 1, 0, dy - 1)), torch.ones(1, 1, dy, dx))
        out.append(s.view(n, Z, Y, X))
    return torch.stack(out, 1).round()


def cover(u: torch.Tensor) -> torch.Tensor:
    """u [n, t, z, y, x] float (1 = on) -> [n, z, y, x]: the number of 'on' bricks that hold each cell (rounded, as
    in `brick_sums`)."""
    n, _, Z, Y, X = u.shape
    c = torch.zeros(n * Z, 1, Y, X)
    for t, (dy, dx) in enumerate(BRICK_TYPES):
        c = c + F.conv_transpose2d(u[:, t].reshape(n * Z, 1, Y, X), torch.ones(1, 1, dy, dx))[:, :, :Y, :X]
    return c.view(n, Z, Y, X).round()


def brick_sizes() -> torch.Tensor:
    """[1, t, 1, 1, 1]: the number of cells of each brick type."""
    return torch.tensor([float(dy * dx) for dy, dx in BRICK_TYPES]).view(1, T, 1, 1, 1)


def allowed_bricks(target: torch.Tensor) -> torch.Tensor:
    """target [n, z, y, x] bool -> [n, t, z, y, x] bool: the free bricks (module docstring, 'Clamps')."""
    n, Z, Y, X = target.shape
    g = target.float()
    below = torch.ones_like(g)          # layer 0 rests on the ground
    below[:, 1:] = g[:, :-1]
    inside = brick_sums(g) == brick_sizes()
    supported = brick_sums(below) > 0
    fits = torch.zeros(1, T, 1, Y, X, dtype=torch.bool)
    for t, (dy, dx) in enumerate(BRICK_TYPES):
        fits[:, t, :, :Y - dy + 1, :X - dx + 1] = True
    return fits & inside & supported


def energy(u: torch.Tensor, target: torch.Tensor, mu: float):
    """(cover term [n], brick count [n], total energy [n]) of boards u [n, t, z, y, x] for targets [n, z, y, x]."""
    cov = ((cover(u) - target.float()) ** 2).flatten(1).sum(1)
    count = u.flatten(1).sum(1)
    return cov, count, cov + mu * count


def pair_form(target: torch.Tensor, mu: float):
    """The same energy as pair and input costs, for one target [z, y, x]: (K [v, v], B [v], const) with v = t*z*y*x and
    E(u) = 1/2 u^T K u + B . u + const. K[b, b'] = 2 x (cells shared by bricks b and b'), 0 on the diagonal;
    B[b] = mu + sum over the cells c of b of (1 - 2 g_c); const = the number of target cells."""
    Z, Y, X = target.shape
    v = T * Z * Y * X
    foot = cover(torch.eye(v).view(v, T, Z, Y, X)).flatten(1)   # [v, cells]: the cells of each brick
    K = 2.0 * foot @ foot.T
    K.fill_diagonal_(0.0)
    g = target.float().flatten()
    B = mu + foot @ (1.0 - 2.0 * g)
    return K, B, float(g.sum())


@torch.no_grad()
def search(target: torch.Tensor, cap: int, mu: float, temp: float, chains: int, max_moves: int, seed: int,
           compact_every: int = 32) -> dict:
    """Focused single-brick moves from the empty board (module docstring, 'Search').

    target [d, z, y, x] bool; cap: the most bricks a build may use.
    Returns: stopped [d] bool; moves [d] (moves of the first chain to stop, max_moves if none stopped);
    board [d, t, z, y, x] bool (the stopping board; if none stopped, the lowest-energy board any chain visited);
    cover_left [d] (cover term of `board`); bricks [d] (bricks on in `board`)."""
    d, Z, Y, X = target.shape
    gen = torch.Generator().manual_seed(seed)
    sizes = brick_sizes()
    design = torch.arange(d).repeat_interleave(chains)          # row -> design
    g = target[design].float()
    free = allowed_bricks(target)[design]
    u = torch.zeros(len(design), T, Z, Y, X)
    best_e = torch.full((len(design),), float("inf"))
    best_u = u.clone()
    stopped = torch.zeros(d, dtype=torch.bool)
    moves = torch.full((d,), max_moves, dtype=torch.long)
    board = torch.zeros(d, T, Z, Y, X, dtype=torch.bool)
    fb_e = torch.full((d,), float("inf"))                       # fallback: lowest energy seen, per design

    def retire(rows):
        """Keep the lowest-energy board of the rows that leave the search (designs that did not stop use it)."""
        for r in rows.tolist():
            k = int(design[r])
            if not stopped[k] and best_e[r] < fb_e[k]:
                fb_e[k] = best_e[r]
                board[k] = best_u[r] > 0

    for move in range(max_moves + 1):
        diff = cover(u) - g
        cov = (diff ** 2).flatten(1).sum(1)
        count = u.flatten(1).sum(1)
        e = cov + mu * count
        better = e < best_e
        best_e = torch.where(better, e, best_e)
        best_u[better] = u[better]
        for r in ((cov == 0) & (count <= cap)).nonzero().flatten().tolist():
            k = int(design[r])
            if not stopped[k]:
                stopped[k], moves[k], board[k] = True, move, u[r] > 0
        if move == max_moves or bool(stopped.all()):
            break
        if move % compact_every == 0:                           # drop the chains of the designs that stopped
            keep = ~stopped[design]
            if not bool(keep.all()):
                design, g, free, u, best_e, best_u = (a[keep] for a in (design, g, free, u, best_e, best_u))
                diff = cover(u) - g
        flagged = free & (brick_sums((diff != 0).float()) > 0)
        pool = torch.where(flagged.flatten(1).any(1).view(-1, 1, 1, 1, 1), flagged, free).flatten(1)
        score = torch.rand(pool.shape, generator=gen).masked_fill(~pool, -1.0)
        var = score.argmax(1, keepdim=True)
        has = pool.any(1)
        ub = u.flatten(1).gather(1, var).squeeze(1)
        size = sizes.expand(1, T, Z, Y, X).flatten(1).expand(len(design), -1).gather(1, var).squeeze(1)
        s_cov = brick_sums(diff + g).flatten(1).gather(1, var).squeeze(1)
        dE = 2.0 * s_cov - 2.0 * size * ub - size + mu          # energy with the brick on minus off (it lies in the target)
        if temp > 0:
            new = (torch.rand(len(design), generator=gen) < torch.sigmoid(-dE / temp)).float()
        else:
            new = torch.where(dE < 0, torch.ones_like(ub), torch.where(dE > 0, torch.zeros_like(ub), ub))
        uf = u.flatten(1)
        uf.scatter_(1, var, torch.where(has, new, ub).unsqueeze(1))
        u = uf.view(-1, T, Z, Y, X)
    retire(torch.arange(len(design)))
    cov, count, _ = energy(board.float(), target, mu)
    return {"stopped": stopped, "moves": moves, "board": board, "cover_left": cov.long(), "bricks": count.long()}


def bricks_to_actions(board: torch.Tensor, done_action: int) -> torch.Tensor:
    """board [d, t, z, y, x] bool -> [d, L + 1] long: each design's 'on' bricks as world actions (t * Y * X + y * X + x),
    lowest layer first, padded with DONE."""
    d, _, Z, Y, X = board.shape
    by_layer = board.permute(0, 2, 1, 3, 4).reshape(d, Z, T * Y * X)
    L = int(by_layer.flatten(1).sum(1).max()) if d else 0
    acts = torch.full((d, L + 1), done_action, dtype=torch.long)
    for k in range(d):
        a = by_layer[k].nonzero()[:, 1]                         # rows come sorted by layer, then by action
        acts[k, :len(a)] = a
    return acts


def build(actions: torch.Tensor, target: torch.Tensor, spec):
    """Step the frozen world through `actions` [d, L]. Returns (built occupancy [d, z, y, x] bool, illegal placements)."""
    world = BrickWorld(spec, len(target), "cpu", targets=target, auto_reset=False)
    illegal = 0
    for s in range(actions.shape[1]):
        if bool(world.ended.all()):
            break
        illegal += int(world.step(actions[:, s])["illegal"].sum())
    return world.occ.clone(), illegal
