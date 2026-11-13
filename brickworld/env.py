"""Batched brick-assembly world: lattice rules, no contact physics.

Each world is a Z x Y x X occupancy grid. A brick is a one-cell-high plate with a
rectangular footprint. Placing a brick drops it until it rests on the highest
filled cell under its footprint (or on the ground). Every brick is therefore
supported by at least one cell, and overhangs with empty cells under them can form.

Actions: (brick type, anchor y, anchor x) placements, plus DONE. An episode ends on
DONE or after `max_steps` placements. The task: make the occupancy grid equal the
target design exactly.

Everything is a batched tensor operation, so the same code runs on CPU, Apple MPS
or CUDA. Rewards are not computed here; the environment reports raw quantities and
the training recipe turns them into a reward (see train.py).
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# (dy, dx) footprints: 1x1, 1x2, 2x1, 2x2
BRICK_TYPES = ((1, 1), (1, 2), (2, 1), (2, 2))


@dataclass(frozen=True)
class WorldSpec:
    name: str
    X: int
    Y: int
    Z: int
    k_min: int  # bricks used to generate a target design
    k_max: int

    @property
    def n_types(self) -> int:
        return len(BRICK_TYPES)

    @property
    def n_place(self) -> int:
        return self.n_types * self.Y * self.X

    @property
    def n_actions(self) -> int:
        return self.n_place + 1

    @property
    def done_action(self) -> int:
        return self.n_place

    @property
    def max_steps(self) -> int:
        return 2 * self.k_max + 1

    @property
    def n_cells(self) -> int:
        return self.X * self.Y * self.Z


SCALES = {
    "small": WorldSpec("small", X=5, Y=5, Z=3, k_min=3, k_max=8),
    "large": WorldSpec("large", X=6, Y=6, Z=4, k_min=4, k_max=12),
}


def footprints(spec: WorldSpec):
    """Footprint mask per placement action, [n_place, Y, X], and whether it fits."""
    T, Y, X = spec.n_types, spec.Y, spec.X
    M = torch.zeros(T, Y, X, Y, X, dtype=torch.bool)
    fits = torch.zeros(T, Y, X, dtype=torch.bool)
    for t, (dy, dx) in enumerate(BRICK_TYPES):
        for y in range(Y - dy + 1):
            for x in range(X - dx + 1):
                M[t, y, x, y:y + dy, x:x + dx] = True
                fits[t, y, x] = True
    return M.view(T * Y * X, Y, X), fits.view(T * Y * X)


def rest_heights(height: torch.Tensor, spec: WorldSpec) -> torch.Tensor:
    """Layer each placement would land on, [B, n_place]; Z where it doesn't fit."""
    B = height.shape[0]
    h = height.float().unsqueeze(1)
    out = []
    for dy, dx in BRICK_TYPES:
        p = F.max_pool2d(h, kernel_size=(dy, dx), stride=1)  # [B,1,Y-dy+1,X-dx+1]
        p = F.pad(p, (0, dx - 1, 0, dy - 1), value=float(spec.Z))
        out.append(p)
    return torch.cat(out, dim=1).view(B, spec.n_place).long()


def heights_of(occ: torch.Tensor) -> torch.Tensor:
    """Top-surface height per column: 1 + highest filled layer, 0 if empty. occ [B,Z,Y,X]."""
    Z = occ.shape[1]
    levels = torch.arange(1, Z + 1, device=occ.device).view(1, Z, 1, 1)
    return (occ.long() * levels).amax(dim=1)


def zobrist_codes(spec: WorldSpec) -> torch.Tensor:
    g = torch.Generator().manual_seed(1234567 + spec.n_cells)
    return torch.randint(0, 2 ** 50, (spec.n_cells,), generator=g, dtype=torch.long)


def design_hash(occ: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Order-independent hash of each design, [B] int64 (sum of 50-bit cell codes)."""
    flat = occ.reshape(occ.shape[0], -1).long()
    return (flat * codes.to(occ.device)).sum(dim=1)


def generate_designs(spec: WorldSpec, n: int, gen: torch.Generator, return_actions: bool = False):
    """n random buildable designs [n,Z,Y,X] bool (CPU): k random drops, k ~ U[k_min, k_max].

    Each drop picks a brick type uniformly among types that still fit somewhere,
    then an anchor uniformly among that type's valid anchors. With return_actions, also
    returns the [n, k_max] placement actions used (-1 where unused), for tests.
    """
    acts = torch.full((n, spec.k_max), -1, dtype=torch.long)
    M, fits = footprints(spec)
    T, YX = spec.n_types, spec.Y * spec.X
    occ = torch.zeros(n, spec.Z, spec.Y, spec.X, dtype=torch.bool)
    height = torch.zeros(n, spec.Y, spec.X, dtype=torch.long)
    k = torch.randint(spec.k_min, spec.k_max + 1, (n,), generator=gen)
    rows = torch.arange(n)
    for i in range(spec.k_max):
        z_all = rest_heights(height, spec)
        valid = (z_all < spec.Z) & fits.unsqueeze(0)
        vt = valid.view(n, T, YX)
        type_ok = vt.any(-1)
        active = (k > i) & type_ok.any(-1)
        tw = torch.where(type_ok.any(-1, keepdim=True), type_ok.float(), torch.ones(n, T))
        t = torch.multinomial(tw, 1, generator=gen).squeeze(1)
        am = vt[rows, t].float()
        am = torch.where(am.sum(-1, keepdim=True) > 0, am, torch.ones_like(am))
        pos = torch.multinomial(am, 1, generator=gen).squeeze(1)
        a = t * YX + pos
        z = z_all[rows, a]
        fp = M[a] & active.view(n, 1, 1)
        zc = z.clamp(max=spec.Z - 1)
        occ[rows, zc] = occ[rows, zc] | fp
        height = torch.where(fp, (zc + 1).view(n, 1, 1), height)
        acts[:, i] = torch.where(active, a, torch.full_like(a, -1))
    return (occ, acts) if return_actions else occ


def has_hidden_cells(occ: torch.Tensor) -> torch.Tensor:
    """True where the design has an empty cell under a filled one (top view can't see it)."""
    h = heights_of(occ)
    Z = occ.shape[1]
    levels = torch.arange(Z, device=occ.device).view(1, Z, 1, 1)
    full = levels < h.unsqueeze(1)
    return (full & ~occ).flatten(1).any(1)


class BrickWorld:
    """B parallel worlds with auto-reset from a target source.

    target_source: a callable n -> [n,Z,Y,X] bool targets (on CPU), used on reset.
    If auto_reset is False (evaluation), worlds stay ended once they end.
    """

    def __init__(self, spec: WorldSpec, n: int, device, target_source=None,
                 targets: torch.Tensor = None, auto_reset: bool = True):
        self.spec, self.n, self.device = spec, n, torch.device(device)
        M, fits = footprints(spec)
        self.M, self.fits = M.to(self.device), fits.to(self.device)
        self.source = target_source
        self.auto_reset = auto_reset
        self.rows = torch.arange(n, device=self.device)
        self.occ = torch.zeros(n, spec.Z, spec.Y, spec.X, dtype=torch.bool, device=self.device)
        self.height = torch.zeros(n, spec.Y, spec.X, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.ended = torch.zeros(n, dtype=torch.bool, device=self.device)
        init = targets if targets is not None else self.source(n)
        self.target = init.to(self.device).clone()
        self.target_height = heights_of(self.target)
        self.target_cells = self.target.flatten(1).sum(1)

    def valid_mask(self) -> torch.Tensor:
        """[B, n_actions] bool; DONE is always valid."""
        z_all = rest_heights(self.height, self.spec)
        place = (z_all < self.spec.Z) & self.fits.unsqueeze(0)
        done = torch.ones(self.n, 1, dtype=torch.bool, device=self.device)
        return torch.cat([place, done], dim=1)

    def _metrics(self):
        inter = (self.occ & self.target).flatten(1).sum(1)
        union = (self.occ | self.target).flatten(1).sum(1)
        iou = inter.float() / union.clamp(min=1).float()
        hm_frac = (self.height == self.target_height).flatten(1).float().mean(1)
        return iou, hm_frac

    def step(self, actions: torch.Tensor) -> dict:
        spec = self.spec
        is_done = actions == spec.done_action
        a = actions.clamp(max=spec.n_place - 1)
        z_all = rest_heights(self.height, spec)
        z = z_all.gather(1, a.unsqueeze(1)).squeeze(1)
        legal = (~is_done) & (z < spec.Z) & self.fits[a] & (~self.ended)
        iou_prev, hm_prev = self._metrics()
        fp = self.M[a] & legal.view(-1, 1, 1)
        zc = z.clamp(max=spec.Z - 1)
        tgt_layer = self.target[self.rows, zc]
        cells_in = (fp & tgt_layer).flatten(1).sum(1)
        cells_out = fp.flatten(1).sum(1) - cells_in
        self.occ[self.rows, zc] = self.occ[self.rows, zc] | fp
        self.height = torch.where(fp, (zc + 1).view(-1, 1, 1), self.height)
        self.steps = self.steps + (~self.ended).long()
        iou_new, hm_new = self._metrics()
        newly_ended = (~self.ended) & (is_done | (self.steps >= spec.max_steps) | ~legal & ~is_done)
        exact = (self.occ == self.target).flatten(1).all(1)
        hm_match = (self.height == self.target_height).flatten(1).all(1)
        info = {
            "ended": newly_ended,
            "exact": exact & newly_ended,
            "hm_match": hm_match & newly_ended,
            "cells_in": cells_in, "cells_out": cells_out,
            "target_cells": self.target_cells.clone(),
            "iou_prev": iou_prev, "iou_new": iou_new,
            "hm_prev": hm_prev, "hm_new": hm_new,
            "illegal": (~legal) & (~is_done) & (~self.ended),
            "ep_len": self.steps.clone(),
        }
        if self.auto_reset:
            self._reset(newly_ended)
        else:
            self.ended = self.ended | newly_ended
        return info

    def _reset(self, mask: torch.Tensor):
        n = int(mask.sum())
        if n == 0:
            return
        idx = mask.nonzero().squeeze(1)
        new_t = self.source(n).to(self.device)
        self.target[idx] = new_t
        self.target_height[idx] = heights_of(new_t)
        self.target_cells[idx] = new_t.flatten(1).sum(1)
        self.occ[idx] = False
        self.height[idx] = 0
        self.steps[idx] = 0


class StreamSource:
    """Fresh designs in a fixed order from a seeded generator, rejecting held-out designs."""

    def __init__(self, spec: WorldSpec, seed: int, heldout_hashes: torch.Tensor, chunk: int = 1 << 16):
        self.spec, self.chunk = spec, chunk
        self.gen = torch.Generator().manual_seed(seed)
        self.codes = zobrist_codes(spec)
        self.heldout = heldout_hashes.clone()
        self.buf = torch.zeros(0, spec.Z, spec.Y, spec.X, dtype=torch.bool)
        self.pos = 0
        self.drawn = 0
        self.rejected = 0

    def _refill(self):
        d = generate_designs(self.spec, self.chunk, self.gen)
        keep = ~torch.isin(design_hash(d, self.codes), self.heldout)
        self.rejected += int((~keep).sum())
        self.drawn += d.shape[0]
        self.buf = torch.cat([self.buf[self.pos:], d[keep]])
        self.pos = 0

    def __call__(self, n: int) -> torch.Tensor:
        while self.buf.shape[0] - self.pos < n:
            self._refill()
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out


class FixedSource:
    """Uniform draws (with replacement) from a fixed set of N training designs."""

    def __init__(self, designs: torch.Tensor, seed: int):
        self.designs = designs
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, n: int) -> torch.Tensor:
        i = torch.randint(0, self.designs.shape[0], (n,), generator=self.gen)
        return self.designs[i]
