import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

E = 16
H = 20
T = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        C = int(spec.in_channels)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.enc_layer = nn.Conv2d(8 + self.Z, E, 1)
        self.enc_col = nn.Conv2d(C, E, 1)
        self.enc2 = nn.Conv2d(E, E, 1)
        self.trunk = nn.Sequential(
            nn.Conv2d(C, T, 1), nn.ReLU(),
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
        zoh = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        xl = torch.cat([cur, below, zoh], dim=2).reshape(B * Z, 8 + Z, Y, X)
        a = self.enc_layer(xl).view(B, Z, E, Y, X) + self.enc_col(obs).unsqueeze(1)
        h = F.relu(a).reshape(B * Z, E, Y, X)
        h = F.relu(self.enc2(h))
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
        hm = h.view(B, Z, E, Y, X).amax(1).mean((2, 3))
        g = self.glob(torch.cat([tr.mean((2, 3)), tr.amax((2, 3)), hm], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
