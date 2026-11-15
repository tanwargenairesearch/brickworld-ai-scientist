import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.8}

E = 16
H = 24
T = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.layer_enc = nn.Sequential(
            nn.Conv2d(12, E, 1), nn.ReLU(),
            nn.Conv2d(E, E, 1), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Conv2d(spec.in_channels, T, 1), nn.ReLU(),
            nn.Conv2d(T, T, 3, padding=1), nn.ReLU(),
            nn.Conv2d(T, T, 3, padding=1), nn.ReLU())
        self.mix = nn.ModuleList([nn.Conv2d(3 * E, H, 1) for _ in self.fps])
        self.ctx = nn.ModuleList([nn.Conv2d(T, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * T + E, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        cur = torch.stack([b, t, m, e], dim=2)
        below = F.pad(cur, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([cur, below, gl, zc], dim=2).reshape(B * Z, 12, Y, X)
        h = self.layer_enc(x)
        tr = self.trunk(obs)
        res = []
        for (dy, dx), mix, ctx, out in zip(self.fps, self.mix, self.ctx, self.outs):
            if dy * dx == 1:
                mx, mn, av = h, h, h
            else:
                q = F.pad(h, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            c = ctx(tr).unsqueeze(1).expand(B, Z, H, Y, X).reshape(B * Z, H, Y, X)
            u = F.relu(mix(torch.cat([mx, mn, av], dim=1)) + c)
            s = out(u).view(B, Z, Y, X).sum(1)
            res.append(s)
        place = torch.stack(res, dim=1).flatten(1)
        hm = h.view(B, Z, E, Y, X).mean((1, 3, 4))
        g = self.glob(torch.cat([tr.mean((2, 3)), tr.amax((2, 3)), hm], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
