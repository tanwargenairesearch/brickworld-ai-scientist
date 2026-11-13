import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2}

LW = 32
WIDTH = 64
HID = 32


class FootHead(nn.Module):
    def __init__(self, c, hidden, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c, hidden, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hidden, 1, 1) for _ in self.fps])

    def forward(self, f):
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.convs, self.outs):
            g = F.pad(f, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(conv(g))))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.layer_enc = nn.Sequential(
            nn.Conv2d(10, LW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(LW, LW, 1), nn.ReLU())
        self.mix = nn.Sequential(
            nn.Conv2d(self.Z * LW, WIDTH, 1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.place_head = FootHead(WIDTH, HID, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, C, Y, X = obs.shape
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        ground = obs.new_ones(B, 1, Y, X)
        build_below = torch.cat([ground, build[:, :Z - 1]], dim=1)
        tgt_below = torch.cat([ground, tgt[:, :Z - 1]], dim=1)
        shared = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zpos = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zpos = zpos.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        per = torch.stack([build, tgt, miss, extra, build_below, tgt_below], dim=2)
        x = torch.cat([per, shared, zpos], dim=2).reshape(B * Z, 10, Y, X)
        h = self.layer_enc(x).reshape(B, Z * LW, Y, X)
        f = self.mix(h)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
