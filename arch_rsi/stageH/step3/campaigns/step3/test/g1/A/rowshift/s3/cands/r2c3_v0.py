import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

CD = 32
E = 32
NH = 4
W = 32
PROJ = 32
HID = 32
G = 64
RG = 1.4142135623730951


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def norm(c):
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return [p, p, p]
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return [mx, mn, av]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.c1 = init_(nn.Conv2d(cin, 48, 1), RG)
        self.c2 = init_(nn.Conv2d(48, CD, 1), RG)
        self.lk = init_(nn.Linear(4, E), RG)
        self.lc = init_(nn.Linear(CD, E), RG)
        self.zpos = nn.Parameter(torch.zeros(self.Z, E))
        nn.init.normal_(self.zpos, std=0.02)
        self.ln_k = nn.LayerNorm(E)
        self.lh = init_(nn.Linear(E, E), RG)
        self.score = init_(nn.Linear(E, NH), 1.0)
        self.val = init_(nn.Linear(E, E), 1.0)
        self.t1 = init_(nn.Conv2d(CD + E, W, 3, padding=1), RG)
        self.n1 = norm(W)
        self.t2 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n2 = norm(W)
        hc = W + CD + E
        self.proj = init_(nn.Conv2d(hc, PROJ, 1), 1.0)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * PROJ, HID, 1), RG) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(hc, 1, 1), 0.01)
        gin = 2 * W + 2 * E
        self.gnorm = nn.LayerNorm(gin)
        self.glob = init_(nn.Linear(gin, G), RG)
        self.type_head = init_(nn.Linear(G, nt), 0.01)
        self.done_head = init_(nn.Linear(G, 1), 0.01)
        self.value_head = init_(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        c = F.relu(self.c2(F.relu(self.c1(obs))))
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 3, 4, 2, 1)
        k = self.lk(per) + self.zpos + self.lc(c.permute(0, 2, 3, 1)).unsqueeze(3)
        k = F.relu(self.lh(F.relu(self.ln_k(k))))
        a = torch.softmax(self.score(k), dim=3)
        vv = self.val(k).view(B, Y, X, Z, NH, E // NH)
        v = (a.unsqueeze(-1) * vv).sum(3).reshape(B, Y, X, E).permute(0, 3, 1, 2)
        x = F.relu(self.n1(self.t1(torch.cat([c, v], dim=1))))
        f = F.relu(x + self.n2(self.t2(x)))
        hc = torch.cat([f, c, v], dim=1)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3)), v.mean((2, 3)), v.amax((2, 3))], dim=1)
        g = F.relu(self.glob(self.gnorm(g)))
        tb = self.type_head(g)
        anc = self.anchor(hc)
        p = self.proj(hc)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            z = torch.cat(fp_pool(p, dy, dx), dim=1)
            z = F.relu(self.mix[i](z))
            res.append(self.outs[i](z) + anc + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
