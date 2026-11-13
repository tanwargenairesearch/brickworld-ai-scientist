"""What a candidate architecture file must provide, and the facts it may rely on.

A candidate is one Python file that defines
    build_policy(spec: ArchSpec) -> torch.nn.Module
whose module's forward(obs, mask) returns (logits [B, spec.n_actions], value [B]).
The frozen wrapper (CandidatePolicy) masks invalid actions after forward, so a candidate
cannot make an illegal move more likely than a legal one.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from brickworld.env import BRICK_TYPES
from brickworld.model import obs_channels


@dataclass(frozen=True)
class ArchSpec:
    X: int
    Y: int
    Z: int
    footprints: tuple      # (dy, dx) per brick type, in action order
    in_channels: int       # 4*Z + 3
    n_types: int
    n_place: int           # n_types * Y * X
    n_actions: int         # n_place + 1 (the last action is DONE)


def arch_spec(world) -> ArchSpec:
    return ArchSpec(X=world.X, Y=world.Y, Z=world.Z, footprints=tuple(BRICK_TYPES),
                    in_channels=obs_channels(world, "voxels"), n_types=world.n_types,
                    n_place=world.n_place, n_actions=world.n_actions)


class CandidatePolicy(nn.Module):
    """Frozen wrapper: calls the candidate network, checks shapes, applies the action mask."""

    def __init__(self, net: nn.Module, spec: ArchSpec):
        super().__init__()
        self.net, self.spec = net, spec

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        logits, value = self.net(obs, mask)
        B = obs.shape[0]
        if tuple(logits.shape) != (B, self.spec.n_actions) or tuple(value.shape) != (B,):
            raise ValueError(f"forward returned logits {tuple(logits.shape)} and value {tuple(value.shape)}; "
                             f"expected ({B}, {self.spec.n_actions}) and ({B},)")
        return logits.float().masked_fill(~mask, -1e9), value.float()


INTERFACE_TEXT = """## The interface your architecture file must implement
Write one self-contained Python file. It may import only `torch`, `torch.nn`, `torch.nn.functional` and `math`
(for example `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`, `import math`).
It must define

    def build_policy(spec) -> torch.nn.Module

The returned module's `forward(obs, mask)` must return `(logits, value)`:
- `obs`: float tensor [B, C, Y, X] with C = spec.in_channels = 4*Z + 3. Channels, with layer z = 0 at the bottom:
  - 0 .. Z-1: the current build, one 0/1 channel per layer;
  - Z .. 2Z-1: the target design, one 0/1 channel per layer;
  - 2Z .. 3Z-1: cells the target has and the build lacks ("missing");
  - 3Z .. 4Z-1: cells the build has and the target lacks ("extra"; these can never be removed);
  - 4Z: the build's column heights divided by Z; 4Z+1: the target's column heights divided by Z;
  - 4Z+2: the fraction of the episode's placement limit used so far (same value in every cell).
- `mask`: bool tensor [B, spec.n_actions], True where an action is legal (it fits in the grid and lands below the height limit). DONE is always legal.
- `logits`: float tensor [B, spec.n_actions]. Action a < spec.n_place places brick type t = a // (Y*X) with its
  top-left corner at y = (a % (Y*X)) // X, x = a % X; type t covers spec.footprints[t] = (dy, dx) cells,
  rows y .. y+dy-1 and columns x .. x+dx-1. The brick lands one layer above the highest filled cell under its
  footprint. The last action, index spec.n_place, is DONE. Logits for illegal actions are replaced with -1e9
  after your forward, so you do not need to mask them yourself (you may use the mask as an input feature).
- `value`: float tensor [B], the critic's estimate of the return.
- `spec` fields: X, Y, Z, footprints, in_channels, n_types, n_place, n_actions. The same file must work for any
  grid size: never hard-code X, Y or Z.

Rules checked before any training (a file that breaks one is rejected without training):
- only the imports above; no file, network, process or system access; no `eval`, `exec`, `compile`, `open`,
  `getattr`, `setattr`, `globals`, `__import__`; no attribute whose name starts with `_` except `__init__`;
  nothing from `torch.load`, `torch.save`, `torch.hub`, `torch.jit`, `torch.compile`, `torch.utils`, `torch.ops`,
  `torch.backends`, `torch.cuda`, `torch.mps`, `torch.distributed`, `torch.multiprocessing`, `torch.autograd`,
  `torch.random`, or anything that sets seeds, threads or global state;
- top-level code may only import, define functions and classes, and assign constants;
- at most {max_params:,} parameters;
- training compute per sample (one forward and backward pass) at most {flops_ratio:g} times the starting
  architecture's on the same world, counted with torch.utils.flop_counter, which counts matrix multiplications
  and convolutions. This ratio also sets the number of training steps (see the budget). Operations the counter
  does not see still take time, and the trial's time limit applies;
- the policy must be learned, and so must its features:
  - every use of `obs` and `mask` must pass through a learned layer (for example nn.Conv2d, nn.Conv3d, nn.Linear,
    or attention with learned projections) before any pooling, nonlinearity, comparison, rounding, cumulative
    sum or arithmetic between channels. Before that first learned layer you may only select, slice, reshape,
    permute, pad, concatenate, and add constant coordinate or positional channels. After it, any operation is
    allowed, including pooling over brick footprints. So do not hand-compute features such as landing layers,
    the cells a placement would fill, or the cells it would seal: the network must learn them;
  - use standard initialisation (PyTorch's defaults, or standard schemes such as orthogonal, Xavier, Kaiming,
    or zeros or small values for output layers). Hand-set weight values that make the network rank placements
    well before training count as hand-writing the policy, even if they are trainable or sit behind a gate;
  - checks on {n_check} check designs (not the dev designs): the untrained network may build at most
    {max_untrained:.0%} of them exactly and fill at most {max_clean:.0%} of their target cells before its first
    wrong cell; after a short training run of {short_steps:,} steps it may build at most {max_short:.0%} exactly.
    The short run also catches files that crash during training;
  - after the campaign, an independent reviewer reads every selected architecture's code; one that breaks
    these rules is reported as a violation;
- outputs must have the shapes above and be finite;
- the file must differ from every earlier trial in the search."""
