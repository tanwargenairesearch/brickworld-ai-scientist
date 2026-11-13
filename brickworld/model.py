"""Observation encodings and the policy/value network. Both are chosen by the config."""
import torch
import torch.nn as nn

from .env import WorldSpec


def obs_channels(spec: WorldSpec, obs_mode: str) -> int:
    if obs_mode == "voxels":
        return 4 * spec.Z + 3
    if obs_mode == "heightmaps":
        return 4
    raise ValueError(obs_mode)


def encode(world, obs_mode: str) -> torch.Tensor:
    """[B, C, Y, X] float observation of current state and target."""
    spec = world.spec
    Z = float(spec.Z)
    h = world.height.float().unsqueeze(1) / Z
    th = world.target_height.float().unsqueeze(1) / Z
    frac = (world.steps.float() / spec.max_steps).view(-1, 1, 1, 1).expand_as(h)
    if obs_mode == "heightmaps":
        return torch.cat([h, th, th - h, frac], dim=1)
    occ, tgt = world.occ, world.target
    missing = tgt & ~occ
    extra = occ & ~tgt
    vox = torch.cat([occ, tgt, missing, extra], dim=1).float()
    return torch.cat([vox, h, th, frac], dim=1)


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[name]()


class Policy(nn.Module):
    """encoder 'conv': fully convolutional; one logit per (brick type, anchor) from a
    1x1 conv head, DONE logit and value from globally pooled features.
    encoder 'mlp': flattened grid -> MLP -> all logits and value."""

    def __init__(self, spec: WorldSpec, cfg: dict):
        super().__init__()
        self.spec = spec
        self.kind = cfg["encoder"]
        C = obs_channels(spec, cfg["obs_mode"])
        w, depth = cfg["width"], cfg["depth"]
        layernorm = cfg["norm"] == "layernorm"
        if self.kind == "conv":
            layers, c_in = [], C
            for _ in range(depth):
                layers.append(nn.Conv2d(c_in, w, 3, padding=1))
                if layernorm:
                    layers.append(nn.GroupNorm(1, w))
                layers.append(_act(cfg["activation"]))
                c_in = w
            self.body = nn.Sequential(*layers)
            self.place_head = nn.Conv2d(w, spec.n_types, 1)
            self.glob = nn.Sequential(nn.Linear(2 * w, w), _act(cfg["activation"]))
            self.done_head = nn.Linear(w, 1)
            self.value_head = nn.Linear(w, 1)
        elif self.kind == "mlp":
            layers, d_in = [], C * spec.Y * spec.X
            for _ in range(depth):
                layers.append(nn.Linear(d_in, w))
                if layernorm:
                    layers.append(nn.LayerNorm(w))
                layers.append(_act(cfg["activation"]))
                d_in = w
            self.body = nn.Sequential(*layers)
            self.logit_head = nn.Linear(w, spec.n_actions)
            self.value_head = nn.Linear(w, 1)
        else:
            raise ValueError(self.kind)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        if self.kind == "conv":
            f = self.body(obs)
            place = self.place_head(f).flatten(1)  # [B, T*Y*X], type-major like the env
            g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
            logits = torch.cat([place, self.done_head(g)], dim=1)
            value = self.value_head(g).squeeze(1)
        else:
            f = self.body(obs.flatten(1))
            logits = self.logit_head(f)
            value = self.value_head(f).squeeze(1)
        logits = logits.masked_fill(~mask, -1e9)
        return logits, value
