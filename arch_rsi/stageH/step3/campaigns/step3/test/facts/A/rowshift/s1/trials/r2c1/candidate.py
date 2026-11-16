import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 256, "rollout_len": 8, "n_minibatches": 4, "ppo_epochs": 1, "gae_lambda": 0.85, "ent_coef": 0.005, "clip_eps": 0.25}

CW = 32
D = 16
HID = 24
G = 64


def fp_pool(t, dy, dx):
    if dy * dx == 1:
        return t, t, t
    q = F.pad(t, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, 48, 1), nn.ReLU(),
            nn.Conv2d(48, CW, 3, padding=1), nn.ReLU())
        self.hlog = nn.Conv2d(CW, Z + 1, 1)
        self.enc1 = nn.Conv2d(4 + Z + CW, D, 1)
        self.enc2 = nn.Conv2d(D, D, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * D, HID, 1) for _ in self.fps])
        self.ctx = nn.ModuleList([nn.Conv2d(CW, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * CW, G), nn.ReLU())
        self.gproj = nn.Linear(G, HID)
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        for o in self.outs:
            nn.init.orthogonal_(o.weight, gain=0.01)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        c = self.col(obs)
        g = self.glob(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1))
        layers = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=2)
        zid = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        cz = c.unsqueeze(1).expand(B, Z, CW, Y, X)
        e = torch.cat([layers, zid, cz], dim=2).reshape(B * Z, 4 + Z + CW, Y, X)
        e = F.relu(self.enc1(e))
        e = F.relu(e + self.enc2(e))
        hp = torch.softmax(self.hlog(c), dim=1)
        logc = torch.log(torch.cumsum(hp, dim=1).clamp(min=1e-6, max=1.0))[:, :Z]
        gp = self.gproj(g)[:, :, None, None]
        res = []
        for (dy, dx), mix, ctx, out in zip(self.fps, self.mix, self.ctx, self.outs):
            mx, mn, av = fp_pool(e, dy, dx)
            hz = F.relu(mix(torch.cat([mx, mn, av], dim=1))).reshape(B, Z, HID, Y, X)
            if dy * dx == 1:
                lc = logc
                cav = c
            else:
                lc = F.avg_pool2d(F.pad(logc, (0, dx - 1, 0, dy - 1)), (dy, dx), stride=1) * float(dy * dx)
                cav = F.avg_pool2d(F.pad(c, (0, dx - 1, 0, dy - 1)), (dy, dx), stride=1)
            cdf = torch.exp(lc)
            prev = torch.cat([torch.zeros_like(cdf[:, :1]), cdf[:, :-1]], dim=1)
            pmf = (cdf - prev).clamp(min=0.0)
            agg = (pmf.unsqueeze(2) * hz).sum(1)
            h = F.relu(agg + ctx(cav) + gp)
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
