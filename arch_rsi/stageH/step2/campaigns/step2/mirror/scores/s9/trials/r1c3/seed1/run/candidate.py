import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 3, "ent_coef": 0.003}

TRUNK = 48
COL = 32
W = 32
P = 16
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.trunk = nn.Sequential(
            nn.Conv2d(spec.in_channels, TRUNK, 1), nn.ReLU(),
            nn.Conv2d(TRUNK, COL, 3, padding=1), nn.ReLU())
        self.c2 = nn.Conv2d(COL, COL, 3, padding=1)
        lin = 11 + self.Z + COL
        self.l1 = nn.Conv2d(lin, W, 1)
        self.l2 = nn.Conv2d(W, W, 1)
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * COL, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        f = self.trunk(obs)
        f = F.relu(f + self.c2(f))
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        ma = F.pad(m, (0, 0, 0, 0, 0, 1))[:, 1:]
        ba = F.pad(b, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([b, t, m, e, bb, tb, ma, ba], dim=2)
        sc = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        ff = f.unsqueeze(1).expand(B, Z, f.shape[1], Y, X)
        x = torch.cat([per, sc, eye, ff], dim=2).reshape(B * Z, -1, Y, X)
        h = F.relu(self.l1(x))
        h = F.relu(h + self.l2(h))
        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        s = torch.cat(res, dim=1).reshape(B, Z, -1, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
