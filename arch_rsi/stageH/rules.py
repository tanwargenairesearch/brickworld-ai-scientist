"""Stage H1's hidden rules (PROTOCOL_H1.md), applied around the frozen world without editing brickworld/.

A Rule gives the hooks that train_h.py, eval_h.py and check_h.py call:
  make_world(n, **kw)          the world: BrickWorld, or TorusWorld for "torus"
  make_stream(seed, heldout)   the training design stream: StreamSource, or TorusStreamSource for "torus"
  obs(o)                       what the agent sees, from encode(world, "voxels")
  act(a)                       agent action -> world action
  mask(m)                      the world's valid mask -> the mask in the agent's action terms
  reward(r, ended)             the reward paid to the agent this step (stateful for "late")
For "plain" every hook returns its input unchanged (the same object), so train_h.py trains exactly as the
frozen Stage 2b trainer does.

Rules (W_s, 8x8x6):
  shift   the booklet channels (target, missing, extra, target height) are shown SHIFT columns to the right, wrapping
  mirror  a placement lands at the left-right mirror of its anchor: anchor x of a brick dx wide lands at X-dx-x
  torus   the board wraps both ways; every anchor is valid; designs are generated under the same rule
  late    all reward is held back and paid as one sum on the episode's last step
"""
import torch
import torch.nn.functional as F

from brickworld.env import BRICK_TYPES, BrickWorld, StreamSource, design_hash, heights_of

SHIFT = 4
RULES = ("plain", "shift", "mirror", "torus", "late")


def booklet_channels(Z: int) -> list:
    """encode(..., "voxels") channels: occ [0,Z), tgt [Z,2Z), missing [2Z,3Z), extra [3Z,4Z), h 4Z, th 4Z+1, frac 4Z+2."""
    return list(range(Z, 4 * Z)) + [4 * Z + 1]


def mirror_perm(spec) -> torch.Tensor:
    """[n_actions] agent action -> world action for "mirror". Anchors that don't fit and DONE map to themselves."""
    T, Y, X = spec.n_types, spec.Y, spec.X
    p = torch.arange(spec.n_actions)
    for t, (dy, dx) in enumerate(BRICK_TYPES):
        for y in range(Y - dy + 1):
            for x in range(X - dx + 1):
                p[t * Y * X + y * X + x] = t * Y * X + y * X + (X - dx - x)
    return p


# ---- wraparound board -------------------------------------------------------------------------------------------

def torus_footprints(spec):
    """Footprint per placement on the wrapped board, [n_place, Y, X], and fits (all True)."""
    T, Y, X = spec.n_types, spec.Y, spec.X
    M = torch.zeros(T, Y, X, Y, X, dtype=torch.bool)
    for t, (dy, dx) in enumerate(BRICK_TYPES):
        for y in range(Y):
            for x in range(X):
                for i in range(dy):
                    for j in range(dx):
                        M[t, y, x, (y + i) % Y, (x + j) % X] = True
    return M.view(T * Y * X, Y, X), torch.ones(T * Y * X, dtype=torch.bool)


def torus_rest_heights(height: torch.Tensor, spec) -> torch.Tensor:
    """Layer each placement would land on, [B, n_place], with footprints wrapping around both edges."""
    B = height.shape[0]
    h = height.float().unsqueeze(1)
    out = []
    for dy, dx in BRICK_TYPES:
        q = F.pad(h, (0, dx - 1, 0, dy - 1), mode="circular") if dy * dx > 1 else h
        out.append(F.max_pool2d(q, kernel_size=(dy, dx), stride=1))
    return torch.cat(out, dim=1).view(B, spec.n_place).long()


def generate_designs_torus(spec, n: int, gen: torch.Generator, return_actions: bool = False):
    """brickworld.env.generate_designs on the wrapped board: the same draws, torus footprints and landing layers."""
    acts = torch.full((n, spec.k_max), -1, dtype=torch.long)
    M, fits = torus_footprints(spec)
    T, YX = spec.n_types, spec.Y * spec.X
    occ = torch.zeros(n, spec.Z, spec.Y, spec.X, dtype=torch.bool)
    height = torch.zeros(n, spec.Y, spec.X, dtype=torch.long)
    k = torch.randint(spec.k_min, spec.k_max + 1, (n,), generator=gen)
    rows = torch.arange(n)
    for i in range(spec.k_max):
        z_all = torus_rest_heights(height, spec)
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


class TorusWorld(BrickWorld):
    """BrickWorld on the wrapped board. valid_mask and step are BrickWorld's with torus footprints and landing layers."""

    def __init__(self, spec, n, device, target_source=None, targets=None, auto_reset=True):
        super().__init__(spec, n, device, target_source=target_source, targets=targets, auto_reset=auto_reset)
        M, fits = torus_footprints(spec)
        self.M, self.fits = M.to(self.device), fits.to(self.device)

    def valid_mask(self) -> torch.Tensor:
        z_all = torus_rest_heights(self.height, self.spec)
        place = (z_all < self.spec.Z) & self.fits.unsqueeze(0)
        done = torch.ones(self.n, 1, dtype=torch.bool, device=self.device)
        return torch.cat([place, done], dim=1)

    def step(self, actions: torch.Tensor) -> dict:
        spec = self.spec
        is_done = actions == spec.done_action
        a = actions.clamp(max=spec.n_place - 1)
        z_all = torus_rest_heights(self.height, spec)
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


class TorusStreamSource(StreamSource):
    """StreamSource drawing wrapped-board designs."""

    def _refill(self):
        d = generate_designs_torus(self.spec, self.chunk, self.gen)
        keep = ~torch.isin(design_hash(d, self.codes), self.heldout)
        self.rejected += int((~keep).sum())
        self.drawn += d.shape[0]
        self.buf = torch.cat([self.buf[self.pos:], d[keep]])
        self.pos = 0


# ---- the hooks --------------------------------------------------------------------------------------------------

class Rule:
    def __init__(self, name: str, spec, device):
        if name not in RULES:
            raise ValueError(f"unknown rule {name!r}")
        self.name, self.spec, self.device = name, spec, torch.device(device)
        self.perm = mirror_perm(spec).to(self.device) if name == "mirror" else None
        self.ch = torch.tensor(booklet_channels(spec.Z), device=self.device) if name == "shift" else None
        self.held = None

    def make_world(self, n, **kw):
        return (TorusWorld if self.name == "torus" else BrickWorld)(self.spec, n, self.device, **kw)

    def make_stream(self, seed, heldout):
        return (TorusStreamSource if self.name == "torus" else StreamSource)(self.spec, seed, heldout)

    def obs(self, o):
        if self.name != "shift":
            return o
        o = o.clone()
        o[:, self.ch] = torch.roll(o[:, self.ch], SHIFT, dims=3)
        return o

    def act(self, a):
        return self.perm[a] if self.name == "mirror" else a

    def mask(self, m):
        return m[:, self.perm] if self.name == "mirror" else m

    def reward(self, r, ended):
        if self.name != "late":
            return r
        if self.held is None:
            self.held = torch.zeros_like(r)
        self.held = self.held + r
        out = torch.where(ended, self.held, torch.zeros_like(r))
        self.held = torch.where(ended, torch.zeros_like(r), self.held)
        return out


def designs_hidden(designs: torch.Tensor) -> torch.Tensor:
    """True where a design has an empty cell under a filled one (as brickworld.env.has_hidden_cells)."""
    h = heights_of(designs)
    levels = torch.arange(designs.shape[1], device=designs.device).view(1, -1, 1, 1)
    return ((levels < h.unsqueeze(1)) & ~designs).flatten(1).any(1)
