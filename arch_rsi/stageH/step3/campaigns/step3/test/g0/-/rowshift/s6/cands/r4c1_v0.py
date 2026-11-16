import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.001, "ent_coef": 0.003}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 24
USE_MASK = True


def pool3(p, kh, kw, br):
    if kh * kw == 1:
        return torch.cat([p, p, p], dim=1)
    if br:
        q = F.pad(p, (kw - 1, 0, kh - 1, 0))
    else:
        q = F.pad(p, (0, kw - 1, 0, kh - 1))
    mx = F.max_pool2d(q, (kh, kw), stride=1)
    mn = -F.max_pool2d(-q, (kh, kw), stride=1)
    av = F.avg_pool2d(q, (kh, kw), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.T = T
        self.n_place = int(spec.n_place)
        cin = int(spec.in_channels) + (2 * T if USE_MASK else 0)
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.variants = [
            [(dy, dx, False) for dy, dx in self.fps],
            [(dx, dy, False) for dy, dx in self.fps],
            [(dy, dx, True) for dy, dx in self.fps],
        ]
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in range(3 * T)])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in range(3 * T)])
        self.n_orders = 3 if Y == X else 2
        self.gain = nn.Parameter(torch.full((3, self.n_orders), 0.3))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        T = self.T
        if USE_MASK:
            m = mask[:, :self.n_place].to(obs.dtype)
            m1 = m.reshape(B, T, Y, X)
            m2 = m.reshape(B, Y, X, T).permute(0, 3, 1, 2)
            x = torch.cat([obs, m1, m2], dim=1)
        else:
            x = obs
        s = self.stem(x)
        h1 = F.relu(self.c1(s))
        f = F.relu(h1 + self.c2(h1))
        p = self.proj(f)
        place = None
        for v in range(3):
            maps = []
            for t in range(T):
                kh, kw, br = self.variants[v][t]
                i = v * T + t
                h = F.relu(self.mix[i](pool3(p, kh, kw, br)))
                maps.append(self.outs[i](h))
            S = torch.cat(maps, dim=1)
            orders = [S.flatten(1), S.permute(0, 2, 3, 1).flatten(1)]
            if self.n_orders == 3:
                orders.append(S.transpose(2, 3).flatten(1))
            for o in range(self.n_orders):
                term = self.gain[v, o] * orders[o]
                place = term if place is None else place + term
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
