import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ent_coef": 0.01, "gamma": 0.97}

STEM = 64
WIDTH = 40
PROJ = 24
HID = 24
GSCALE = 4.0


def pooled(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        cin = int(spec.in_channels) + 2 * self.T
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU())
        self.c1 = nn.Conv2d(STEM, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in range(2 * self.T)])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in range(2 * self.T)])
        self.gate = nn.Parameter(torch.zeros(self.T, 4))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = self.Y
        X = self.X
        T = self.T
        m = mask[:, :T * Y * X].float()
        m1 = m.reshape(B, T, Y, X)
        m2 = m.reshape(B, T, X, Y).transpose(2, 3)
        h = self.stem(torch.cat([obs, m1, m2], dim=1))
        x = F.relu(self.c1(h))
        x = F.relu(x + self.c2(x))
        gctx = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        f = F.relu(x + self.ctx(gctx)[:, :, None, None])
        p = self.proj(f)
        w = torch.softmax(GSCALE * self.gate, dim=1)
        res = []
        for t, (dy, dx) in enumerate(self.fps):
            sa = self.outs[2 * t](F.relu(self.mix[2 * t](pooled(p, dy, dx))))[:, 0]
            sb = self.outs[2 * t + 1](F.relu(self.mix[2 * t + 1](pooled(p, dx, dy))))[:, 0]
            v0 = sa.reshape(B, -1)
            v1 = sb.reshape(B, -1)
            v2 = sa.transpose(1, 2).reshape(B, -1)
            v3 = sb.transpose(1, 2).reshape(B, -1)
            vs = torch.stack([v0, v1, v2, v3], dim=2)
            res.append((vs * w[t]).sum(2))
        place = torch.cat(res, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
