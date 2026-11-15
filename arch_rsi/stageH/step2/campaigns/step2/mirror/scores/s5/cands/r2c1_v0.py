import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "gamma": 0.97, "ent_coef": 0.005}

HID = 48
W = 32
PROJ = 16
MIX = 24
GH = 64


def pool_fp(q, dy, dx):
    if dy * dx == 1:
        return q, q, q
    r = F.pad(q, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(r, (dy, dx), stride=1)
    mn = -F.max_pool2d(-r, (dy, dx), stride=1)
    av = F.avg_pool2d(r, (dy, dx), stride=1)
    return mx, mn, av


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 5 + spec.in_channels + self.Z
        self.stem = nn.Sequential(nn.Conv2d(cin, HID, 1), nn.ReLU(), nn.Conv2d(HID, W, 1), nn.ReLU())
        self.conv = nn.Conv2d(W, W, 3, padding=1)
        self.ctx = nn.Linear(2 * W, W)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, MIX, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(MIX, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def layer_input(self, obs):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        below = F.pad(build[:, :Z - 1], (0, 0, 0, 0, 1, 0), value=1.0)
        own = torch.stack([build, tgt, miss, extra, below], dim=2)
        col = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype)
        pos = eye.view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([own, col, pos], dim=2)
        return x.reshape(B * Z, x.shape[2], Y, X)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        h = self.stem(self.layer_input(obs))
        hz = h.reshape(B, Z, W, Y, X)
        g = self.ctx(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        g = g.view(B, 1, W, 1, 1).expand(B, Z, W, 1, 1).reshape(B * Z, W, 1, 1)
        h = F.relu(h + self.conv(h) + g)
        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            mx, mn, av = pool_fp(p, dy, dx)
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        sc = torch.cat(res, dim=1).reshape(B, Z, len(self.fps), Y, X)
        place = torch.logsumexp(sc, dim=1).flatten(1)
        hz = h.reshape(B, Z, W, Y, X)
        gg = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
