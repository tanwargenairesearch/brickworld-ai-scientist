import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_minibatches": 32, "lr": 0.001, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.005}

COL = 48
W = 32
P = 24
H = 48
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, COL, 1), nn.ReLU())
        self.inp = nn.Conv2d(COL, W, 3, padding=1)
        self.gin = nn.Linear(2 * W, W)
        self.b1 = nn.Conv2d(W, W, 3, padding=1)
        self.b2 = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(COL + W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(P * dy * dx, H, 1) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        c = self.col(obs)
        f = F.relu(self.inp(c))
        g0 = self.gin(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = F.relu(f + g0[:, :, None, None])
        f = F.relu(f + self.b1(f))
        f = F.relu(f + self.b2(f))
        p = F.relu(self.proj(torch.cat([c, f], dim=1)))
        Y = p.shape[2]
        X = p.shape[3]
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                z = p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                z = torch.cat([q[:, :, oy:oy + Y, ox:ox + X] for oy in range(dy) for ox in range(dx)], dim=1)
            res.append(out(F.relu(mix(z))))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
