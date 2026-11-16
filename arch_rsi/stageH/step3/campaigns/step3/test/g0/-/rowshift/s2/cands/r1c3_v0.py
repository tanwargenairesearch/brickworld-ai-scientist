import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 64, "rollout_len": 16, "n_minibatches": 8}

STEM = 64
COL = 48
WIDTH = 40
P = 16
H = 24
NLIN = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.lay_in = nn.Conv2d(NLIN, P, 1)
        self.lay_ctx = nn.Conv2d(WIDTH, P, 1)
        self.zemb = nn.Embedding(self.Z, P)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + P, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))

        bld = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        mis = obs[:, 2 * Z:3 * Z]
        ext = obs[:, 3 * Z:4 * Z]
        bld_below = F.pad(bld[:, :Z - 1], (0, 0, 0, 0, 1, 0), value=1.0)
        mis_below = F.pad(mis[:, :Z - 1], (0, 0, 0, 0, 1, 0), value=0.0)
        tgt_above = F.pad(tgt[:, 1:], (0, 0, 0, 0, 0, 1), value=0.0)
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1).expand(B, Z, Y, X) / float(max(Z - 1, 1))
        li = torch.stack([bld, tgt, mis, ext, bld_below, mis_below, tgt_above, zc], dim=2)
        li = li.reshape(B * Z, NLIN, Y, X)
        u = self.lay_in(li).view(B, Z, P, Y, X)
        u = u + self.lay_ctx(f).unsqueeze(1) + self.zemb.weight.view(1, Z, P, 1, 1)
        u = F.relu(u)
        ug = u.amax(dim=(1, 3, 4))
        u = u.reshape(B * Z, P, Y, X)

        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = u, u, u
            else:
                q = F.pad(u, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            sc = out(h).view(B, Z, Y * X)
            res.append(torch.logsumexp(sc, dim=1))
        place = torch.cat(res, dim=1)

        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), ug], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
