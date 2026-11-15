import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.02, "gamma": 0.97, "gae_lambda": 0.92}

COLH = 96
COL = 64
WIDTH = 48
PROJ = 32
HID = 32
CW = 32
DONE_SCALE = 0.2


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


class Critic(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.a = nn.Conv2d(cin, CW, 1)
        self.b = nn.Conv2d(CW, CW, 3, padding=1)
        self.fc = nn.Sequential(nn.Linear(2 * CW, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs):
        h = F.relu(self.a(obs))
        h = F.relu(h + self.b(h))
        g = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        return self.fc(g).squeeze(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, COLH, 1), nn.ReLU(),
            nn.Conv2d(COLH, COLH, 1), nn.ReLU(),
            nn.Conv2d(COLH, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH + COL, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * (WIDTH + COL), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.zeros_(self.done_head.bias)
        self.critic = Critic(spec.in_channels)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        h = torch.cat([s, f], dim=1)
        place = self.head(h)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        done = self.done_head(g) * DONE_SCALE
        logits = torch.cat([place, done], dim=1)
        return logits, self.critic(obs)


def build_policy(spec):
    return Net(spec)
