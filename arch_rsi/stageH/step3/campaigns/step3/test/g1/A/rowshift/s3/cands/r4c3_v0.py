import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95, "clip_eps": 0.2}

W = 32
CTX = 8
K = 8
HH = 2
PROJ = 16
HID = 32
CNT = 16
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
        self.stem1 = init_(nn.Conv2d(cin, 48, 1), RG)
        self.ns = norm(48)
        self.stem2 = init_(nn.Conv2d(48, W, 1), RG)
        self.c1 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n1 = norm(W)
        self.c2 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n2 = norm(W)
        self.ctx = init_(nn.Conv2d(W, CTX, 1), 1.0)
        self.lk = init_(nn.Linear(4, K), RG)
        self.lc = init_(nn.Linear(CTX, K), RG)
        self.zpos = nn.Parameter(torch.zeros(self.Z, K))
        nn.init.normal_(self.zpos, std=0.1)
        self.lnk = nn.LayerNorm(K)
        self.lk2 = init_(nn.Linear(K, K), 1.0)
        self.hgt = init_(nn.Conv2d(cin, HH, 1), 1.0)
        self.tau = nn.Parameter(torch.zeros(HH))
        self.cnt = init_(nn.Conv2d(cin, CNT, 1), RG)
        self.proj = init_(nn.Conv2d(W, PROJ, 1), 1.0)
        gd = HH * 6 * K + 2 * HH + 3 * PROJ
        self.mix = nn.ModuleList([init_(nn.Conv2d(gd, HID, 1), RG) for _ in self.fps])
        self.mix2 = nn.ModuleList([init_(nn.Conv2d(HID, HID, 1), RG) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(W, 1, 1), 0.01)
        gin = 2 * W + 2 * K + CNT
        self.gnorm = nn.LayerNorm(gin)
        self.glob = init_(nn.Linear(gin, G), RG)
        self.type_head = init_(nn.Linear(G, nt), 0.01)
        self.done_head = init_(nn.Linear(G, 1), 0.01)
        self.value_head = init_(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = F.relu(self.stem2(F.relu(self.ns(self.stem1(obs)))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        ctx = self.ctx(f).permute(0, 2, 3, 1)
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 3, 4, 2, 1)
        k = self.lk(per) + self.zpos + self.lc(ctx).unsqueeze(3)
        k = self.lk2(F.relu(self.lnk(k)))
        kf = k.permute(0, 3, 4, 1, 2).reshape(B, Z * K, Y, X)
        hl = self.hgt(obs) * Z
        sharp = (4.0 * F.softplus(self.tau)).view(1, HH, 1, 1, 1)
        zs = torch.arange(Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1)
        p = self.proj(f)
        cnt = torch.log1p(F.relu(self.cnt(obs)).sum((2, 3)))
        k5 = kf.view(B, Z, K, Y, X)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3)), k5.amax((1, 3, 4)), k5.mean((1, 3, 4)), cnt], dim=1)
        g = F.relu(self.glob(self.gnorm(g)))
        tb = self.type_head(g)
        anc = self.anchor(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            lmx, lmn, _ = fp_pool(hl, dy, dx)
            d = lmx.unsqueeze(2) - zs
            w_at = torch.softmax(-sharp * d * d, dim=2).unsqueeze(3)
            w_bl = torch.sigmoid(sharp * (d - 0.5)).unsqueeze(3)
            pk = torch.stack(fp_pool(kf, dy, dx), 1).view(B, 3, Z, K, Y, X)
            pk = pk.permute(0, 2, 1, 3, 4, 5).reshape(B, 1, Z, 3 * K, Y, X)
            at = (w_at * pk).sum(2).reshape(B, HH * 3 * K, Y, X)
            bl = (w_bl * pk).sum(2).reshape(B, HH * 3 * K, Y, X) / Z
            h = torch.cat([at, bl, lmx / Z, (lmx - lmn) / Z] + fp_pool(p, dy, dx), dim=1)
            h = F.relu(self.mix[i](h))
            h = F.relu(self.mix2[i](h))
            res.append(self.outs[i](h) + anc + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
