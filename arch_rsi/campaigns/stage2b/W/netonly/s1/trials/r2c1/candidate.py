import torch
import torch.nn as nn
import torch.nn.functional as F

W = 28
G = 64
HID = 12


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
        self.T = spec.n_types
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.l1 = nn.Conv2d(13, W, 1)
        self.l2 = nn.Conv2d(W, W, 3, padding=1)
        self.heads1 = nn.ModuleList([nn.Conv2d(W, HID, (dy, dx)) for (dy, dx) in self.fps])
        self.heads2 = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.mix = nn.Conv2d(2 * W, G, 1)
        self.place_head = nn.Conv2d(G, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * G, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, _, Y, X = obs.shape
        x = layer_inputs(obs, Z)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        outs = []
        for (dy, dx), h1, h2 in zip(self.fps, self.heads1, self.heads2):
            p = F.pad(h, (0, dx - 1, 0, dy - 1))
            outs.append(h2(F.relu(h1(p))))
        o = torch.stack(outs, dim=1).reshape(B, Z, self.T, 2, Y, X)
        score = o[:, :, :, 0]
        gate = o[:, :, :, 1]
        wts = torch.softmax(gate, dim=1)
        zplace = (wts * score).sum(1)
        hz = h.reshape(B, Z, W, Y, X)
        f = torch.cat([hz.amax(1), hz.mean(1)], dim=1)
        f = F.relu(self.mix(f))
        place = (zplace + self.place_head(f)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
