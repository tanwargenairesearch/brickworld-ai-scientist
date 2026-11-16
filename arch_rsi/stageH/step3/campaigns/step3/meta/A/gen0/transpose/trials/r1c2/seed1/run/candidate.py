import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

W = 24
PROJ = 12
HID = 16
NIN = 12


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.c0 = nn.Conv2d(NIN, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 1)
        self.colmix = nn.Conv2d(W, W, 1)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]

        def below(v):
            return F.pad(v, (0, 0, 0, 0, 1, 0))[:, :Z]

        def above(v):
            return F.pad(v, (0, 0, 0, 0, 0, 1))[:, 1:]

        per = torch.stack([b, t, m, e, below(b), below(t), above(b), above(t)], dim=2)
        ext = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        lvl = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, ext, lvl], dim=2).reshape(B * Z, NIN, Y, X)

        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        f = F.relu(h + self.c2(h))
        fz = f.view(B, Z, W, Y, X)
        cm = self.colmix(fz.amax(1)).unsqueeze(1)
        fz = F.relu(fz + cm)
        f = fz.reshape(B * Z, W, Y, X)

        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            hh = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(hh))
        s = torch.cat(res, dim=1)
        T = s.shape[1]
        s = s.view(B, Z, T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)

        g = self.glob(torch.cat([fz.mean((1, 3, 4)), fz.amax(4).amax(3).amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
