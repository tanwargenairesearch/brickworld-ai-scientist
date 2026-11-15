import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.98, "gae_lambda": 0.95}

TR = 32
LW = 24
PROJ = 12
HID = 20
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        C = int(spec.in_channels)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Conv2d(C, TR, 1)
        self.c1 = nn.Conv2d(TR, TR, 3, padding=1)
        self.c2 = nn.Conv2d(TR, TR, 3, padding=1)
        self.colproj = nn.Conv2d(TR, LW, 1)
        self.lin = nn.Conv2d(6 + C + Z, LW, 1)
        self.l2 = nn.Conv2d(LW, LW, 1)
        self.proj = nn.Conv2d(LW, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TR, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = F.relu(self.stem(obs))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0))[:, :Z]
        mb = F.pad(m, (0, 0, 0, 0, 1, 0))[:, :Z]
        local = torch.stack([b, t, m, e, bb, mb], dim=2)
        colraw = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([local, colraw, eye], dim=2).reshape(B * Z, 6 + C + Z, Y, X)
        cp = self.colproj(f).unsqueeze(1).expand(B, Z, LW, Y, X).reshape(B * Z, LW, Y, X)
        u = F.relu(self.lin(inp) + cp)
        u = F.relu(u + self.l2(u))
        p = self.proj(u)
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
            o = out(h).view(B, Z, 2, Y, X)
            sc = o[:, :, 0]
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * sc).sum(1).flatten(1))
        place = torch.cat(res, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
