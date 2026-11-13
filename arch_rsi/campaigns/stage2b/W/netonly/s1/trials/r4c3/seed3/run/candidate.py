import torch
import torch.nn as nn
import torch.nn.functional as F

W = 28
G = 64


def layer_inputs(obs, Z):
    B, _, Y, X = obs.shape
    build = obs[:, 0:Z]
    tgt = obs[:, Z:2 * Z]
    miss = obs[:, 2 * Z:3 * Z]
    extra = obs[:, 3 * Z:4 * Z]
    build_below = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
    tgt_below = F.pad(tgt, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
    miss_below = F.pad(miss, (0, 0, 0, 0, 1, 0))[:, :Z]
    tgt_above = F.pad(tgt, (0, 0, 0, 0, 0, 1))[:, 1:]
    miss_above = F.pad(miss, (0, 0, 0, 0, 0, 1))[:, 1:]
    per = torch.stack([build, tgt, miss, extra, build_below, tgt_below,
                       miss_below, tgt_above, miss_above], dim=2)
    gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
    zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
    return torch.cat([per, gl, zc], dim=2).reshape(B * Z, 13, Y, X)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.l1 = nn.Conv2d(13, W, 1)
        self.l2 = nn.Conv2d(W, W, 3, padding=1)
        self.mix = nn.Conv2d(2 * W, G, 1)
        self.ctx = nn.Linear(G, G)
        self.place_head = nn.Conv2d(G, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * G, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.vnet = nn.Sequential(nn.Linear(2 * G, G), nn.ReLU(), nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))

    def forward(self, obs, mask):
        Z = self.Z
        B, _, Y, X = obs.shape
        x = layer_inputs(obs, Z)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = h.reshape(B, Z, W, Y, X)
        f = torch.cat([h.amax(1), h.mean(1)], dim=1)
        f = F.relu(self.mix(f))
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        fc = F.relu(f + self.ctx(g).view(B, G, 1, 1))
        place = self.place_head(fc).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        value = self.vnet(pooled.detach()).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
