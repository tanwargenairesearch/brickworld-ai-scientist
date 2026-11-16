import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "lr": 0.001, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.9}

STEM = 48
COLF = 32
TW = 32
LH = 48
LF = 16
MH = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        C = int(spec.in_channels)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(C, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COLF, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COLF, TW, 3, padding=1)
        self.c2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.lin_in = nn.Conv2d(C + Z + 8, LH, 1)
        self.ctx = nn.Conv2d(TW, LH, 1)
        self.lin2 = nn.Conv2d(LH, LF, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * LF, MH, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(MH, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TW + 2 * LF, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))

        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = torch.cat([torch.zeros_like(lay[:, :, :1]), lay[:, :, :-1]], dim=2)
        cur = lay.permute(0, 2, 1, 3, 4)
        bel = below.permute(0, 2, 1, 3, 4)
        col = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([col, eye, cur, bel], dim=2).reshape(B * Z, C + Z + 8, Y, X)

        ctx = self.ctx(f)
        h = self.lin_in(inp).view(B, Z, LH, Y, X) + ctx.unsqueeze(1)
        h = F.relu(h).reshape(B * Z, LH, Y, X)
        p = self.lin2(h)

        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            res.append(o.view(B, Z, Y, X))
        sc = torch.stack(res, dim=1)
        place = torch.logsumexp(sc, dim=2).flatten(1)

        pz = p.view(B, Z, LF, Y, X)
        gin = torch.cat([f.mean((2, 3)), f.amax((2, 3)),
                         pz.mean((1, 3, 4)), pz.amax(dim=(1, 3, 4))], dim=1)
        g = self.glob(gin)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
