import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.98, "gae_lambda": 0.95, "lr": 0.0015, "ent_coef": 0.01}

W = 32
P = 32
H = 32
HID = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
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
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        cin = spec.in_channels + self.T
        self.stem = nn.Sequential(
            nn.Conv2d(cin, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.b1 = Block(W)
        self.glob1 = nn.Linear(2 * W, W)
        self.b2 = Block(W)
        self.head = PoolHead(W, spec.footprints, P, H)
        self.g = nn.Sequential(nn.Linear(2 * W, HID), nn.ReLU())
        self.done = nn.Linear(HID, 1)
        self.value = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].reshape(B, self.T, self.Y, self.X).to(obs.dtype)
        x = torch.cat([obs, m], dim=1)
        h = self.stem(x)
        h = self.b1(h)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        h = F.relu(h + self.glob1(pooled)[:, :, None, None])
        h = self.b2(h)
        place = self.head(h)
        g = self.g(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
