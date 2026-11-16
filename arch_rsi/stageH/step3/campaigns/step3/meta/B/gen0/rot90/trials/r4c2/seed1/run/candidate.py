import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95}

CX = 32
D = 32
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        C = int(spec.in_channels)
        Y = int(spec.Y)
        X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        ys = torch.linspace(-1.0, 1.0, Y).view(1, 1, Y, 1).expand(1, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X).view(1, 1, 1, X).expand(1, 1, Y, X)
        self.register_buffer('coords', torch.cat([ys, xs], 1).contiguous())
        self.register_buffer('zpos', torch.eye(Z))
        self.ctx_in = nn.Conv2d(C + 2, CX, 1)
        self.ctx1 = nn.Conv2d(CX, CX, 3, padding=1)
        self.ctx2 = nn.Conv2d(CX, CX, 3, padding=1)
        self.l1 = nn.Conv2d(C, D, 1)
        self.l1c = nn.Conv2d(CX, D, 1)
        self.zl = nn.Linear(Z, D)
        self.l2 = nn.Conv2d(D, D, 1)
        self.l3 = nn.Conv2d(D, D, 1)
        self.mix = nn.ModuleList([nn.Conv2d(2 * D, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * CX + 2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64 + 2, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        co = self.coords.expand(B, 2, Y, X)
        t = F.relu(self.ctx_in(torch.cat([obs, co], dim=1)))
        t = F.relu(t + self.ctx1(t))
        t = F.relu(t + self.ctx2(t))
        base = self.l1(obs) + self.l1c(t)
        ze = self.zl(self.zpos).view(1, Z, D, 1, 1)
        u = F.relu(base.unsqueeze(1) + ze).reshape(B * Z, D, Y, X)
        u = F.relu(u + self.l2(u))
        u = F.relu(u + self.l3(u))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn = u, u
            else:
                q = F.pad(u, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            s = out(F.relu(mix(torch.cat([mx, mn], dim=1))))
            s = s.reshape(B, Z, Y * X)
            res.append(torch.logsumexp(s, dim=1))
        place = torch.cat(res, dim=1)
        ul = u.reshape(B, Z, D, Y, X)
        g = torch.cat([t.mean((2, 3)), t.amax((2, 3)), ul.mean((1, 3, 4)), ul.amax((1, 3, 4))], dim=1)
        g = self.glob(g)
        pm = place.detach()
        dfeat = torch.cat([pm.amax(1, keepdim=True), pm.mean(1, keepdim=True)], dim=1)
        done = self.done_head(torch.cat([g, dfeat], dim=1))
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
