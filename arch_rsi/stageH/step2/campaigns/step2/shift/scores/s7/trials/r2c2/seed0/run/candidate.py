import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.002, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

TR = 32
CTX = 24
CU = 8
LH = 32
PJ = 16
MH = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        c = spec.in_channels
        self.trunk = nn.Sequential(
            nn.Conv2d(c, TR, 1), nn.ReLU(),
            nn.Conv2d(TR, TR, 3, padding=1), nn.ReLU(),
            nn.Conv2d(TR, CTX, 3, padding=1), nn.ReLU())
        self.lin0 = nn.Conv2d(6, CU, 1)
        self.lay = nn.Conv2d(10 + CTX + 2 * CU, LH, 1)
        self.lay2 = nn.Conv2d(LH, LH, 1)
        self.proj = nn.Conv2d(LH, PJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PJ, MH, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(MH, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * CTX + 2 * LH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        ctx = self.trunk(obs)
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bp = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tp = F.pad(t, (0, 0, 0, 0, 1, 0))[:, :Z]
        per = torch.stack([b, t, m, e, bp, tp], dim=2)
        u = self.lin0(per.reshape(B * Z, 6, Y, X)).view(B, Z, CU, Y, X)
        cum = torch.cumsum(u, dim=1)
        below = cum - u
        above = cum[:, Z - 1:Z] - cum
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        ce = ctx.unsqueeze(1).expand(B, Z, CTX, Y, X)
        tok = torch.cat([per, gl, zc, ce, below, above], dim=2).reshape(B * Z, 10 + CTX + 2 * CU, Y, X)
        h = F.relu(self.lay(tok))
        h = F.relu(h + self.lay2(h))
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
            s = out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))).view(B, Z, Y, X)
            res.append(torch.logsumexp(s, dim=1).flatten(1))
        place = torch.cat(res, dim=1)
        hz = h.view(B, Z, LH, Y, X)
        g = self.glob(torch.cat([ctx.mean((2, 3)), ctx.amax((2, 3)), hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
