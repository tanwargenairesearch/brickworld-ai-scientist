import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "lr": 0.001, "ent_coef": 0.01, "gae_lambda": 0.95}

T2 = 32
LH = 48
LP = 16
HID = 24
G = 64


def pool_fp(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        R = 2 * Z - 1
        nin = 4 * R + 3 + Z
        self.nin = nin
        self.t_in = nn.Conv2d(spec.in_channels, T2, 1)
        self.t_c = nn.Conv2d(T2, T2, 3, padding=1)
        self.l_in = nn.Conv2d(nin, LH, 1)
        self.l_tr = nn.Conv2d(T2, LH, 1)
        self.l2 = nn.Conv2d(LH, LH, 1)
        self.l_proj = nn.Conv2d(LH, LP, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * LP, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * LH + 2 * T2, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def rel(self, v, lo):
        Z = self.Z
        R = 2 * Z - 1
        vp = F.pad(v, (0, 0, 0, 0, Z - 1, 0), value=lo)
        vp = F.pad(vp, (0, 0, 0, 0, 0, Z - 1), value=0.0)
        return torch.stack([vp[:, z:z + R] for z in range(Z)], dim=1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        t = F.relu(self.t_in(obs))
        t = F.relu(t + self.t_c(t))
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        relc = torch.cat([self.rel(build, 1.0), self.rel(target, 1.0),
                          self.rel(miss, 0.0), self.rel(extra, 0.0)], dim=2)
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        lin = torch.cat([relc, g3, eye], dim=2).reshape(B * Z, self.nin, Y, X)
        tr = self.l_tr(t).unsqueeze(1).expand(B, Z, LH, Y, X).reshape(B * Z, LH, Y, X)
        h = F.relu(self.l_in(lin) + tr)
        h = F.relu(h + self.l2(h))
        p = self.l_proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            mx, mn, av = pool_fp(p, dy, dx)
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))).reshape(B, Z, Y, X)
            res.append(torch.logsumexp(o, dim=1))
        place = torch.stack(res, dim=1).flatten(1)
        hz = h.reshape(B, Z, LH, Y, X)
        gf = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4)), t.mean((2, 3)), t.amax((2, 3))], dim=1)
        g = self.glob(gf)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
