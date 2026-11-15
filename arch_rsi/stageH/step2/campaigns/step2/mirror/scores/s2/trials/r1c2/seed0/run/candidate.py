import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.001}

TW = 32
LH = 32
P = 16
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.trunk_in = nn.Conv2d(spec.in_channels, TW, 1)
        self.t1 = nn.Conv2d(TW, TW, 3, padding=1)
        self.t2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.ctx = nn.Conv2d(TW, LH, 1)
        self.lin = nn.Conv2d(16, LH, 1)
        self.l2 = nn.Conv2d(LH, LH, 1)
        self.proj = nn.Conv2d(LH, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TW + LH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        t = F.relu(self.trunk_in(obs))
        t = F.relu(t + self.t1(t))
        t = F.relu(t + self.t2(t))
        L = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        Lp = F.pad(L, (0, 0, 0, 0, 1, 1))
        cols = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        inp = torch.cat([Lp[:, :, 0:Z], Lp[:, :, 1:Z + 1], Lp[:, :, 2:Z + 2], cols, zc], dim=1)
        inp = inp.permute(0, 2, 1, 3, 4).reshape(B * Z, 16, Y, X)
        c = self.ctx(t).unsqueeze(1).expand(B, Z, LH, Y, X).reshape(B * Z, LH, Y, X)
        h = F.relu(self.lin(inp) + c)
        h = F.relu(h + self.l2(h))
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
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            o = o.reshape(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 0], dim=1)
            res.append((w * o[:, :, 1]).sum(1))
        place = torch.stack(res, dim=1).flatten(1)
        hz = h.reshape(B, Z, LH, Y, X).amax(dim=(1, 3, 4))
        g = self.glob(torch.cat([t.mean((2, 3)), t.amax((2, 3)), hz], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
