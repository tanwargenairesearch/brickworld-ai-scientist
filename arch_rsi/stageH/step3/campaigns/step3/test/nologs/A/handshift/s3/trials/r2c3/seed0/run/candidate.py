import torch
import torch.nn as nn
import torch.nn.functional as F

W = 24
P = 16
GL = 64
NIN = 16

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.95, "clip_eps": 0.2}


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
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
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.stem = nn.Conv2d(NIN, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.col = nn.Conv2d(2 * W, W, 1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.head = PoolHead(W, spec.footprints, P, P)
        self.glob = nn.Sequential(nn.Linear(2 * W, GL), nn.ReLU())
        self.done_head = nn.Linear(GL, 1)
        self.value_head = nn.Linear(GL, 1)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X, T = self.Z, self.Y, self.X, self.T
        L = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=2)
        ones = torch.ones(B, 1, 2, Y, X, device=obs.device, dtype=obs.dtype)
        zeros2 = torch.zeros(B, 1, 2, Y, X, device=obs.device, dtype=obs.dtype)
        zeros4 = torch.zeros(B, 1, 4, Y, X, device=obs.device, dtype=obs.dtype)
        ground = torch.cat([ones, zeros2], dim=2)
        below = torch.cat([ground, L[:, :Z - 1]], dim=1)
        above = torch.cat([L[:, 1:], zeros4], dim=1)
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([below, L, above, g3, zc], dim=2).reshape(B * Z, NIN, Y, X)
        h = F.relu(self.stem(x))
        h = F.relu(h + self.c1(h))
        h5 = h.reshape(B, Z, W, Y, X)
        colf = torch.cat([h5.mean(1), h5.amax(1)], dim=1)
        c = self.col(colf)
        h = F.relu(h5 + c.unsqueeze(1)).reshape(B * Z, W, Y, X)
        f = F.relu(h + self.c2(h))
        s = self.head(f).reshape(B, Z, T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        f5 = f.reshape(B, Z, W, Y, X)
        g = self.glob(torch.cat([f5.mean((1, 3, 4)), f5.amax((1, 3, 4))], dim=1))
        d = self.done_head(g)
        lp_place = F.log_softmax(place, dim=1) + F.logsigmoid(-d)
        lp_done = F.logsigmoid(d)
        logits = torch.cat([lp_place, lp_done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
