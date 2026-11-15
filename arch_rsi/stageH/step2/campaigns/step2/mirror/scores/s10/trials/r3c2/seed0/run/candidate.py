import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 32, "n_minibatches": 8, "lr": 0.001, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.003}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
RED = 32


def orient_map(z, k):
    if k >= 4:
        z = z.transpose(2, 3)
    j = k % 4
    if j == 1:
        z = torch.flip(z, [3])
    elif j == 2:
        z = torch.flip(z, [2])
    elif j == 3:
        z = torch.flip(z, [2, 3])
    return z


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.nt = int(spec.n_types)
        self.np = int(spec.n_place)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.no = 8 if self.Y == self.X else 4
        cin = int(spec.in_channels) + self.nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.red = nn.Conv2d(HID * len(self.fps), RED, 1)
        self.orient = nn.ModuleList([nn.Conv2d(RED + self.nt, self.nt, 3, padding=1) for _ in range(self.no)])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        mp = mask[:, :self.np].reshape(B, self.nt, self.Y, self.X).to(obs.dtype)
        s = self.stem(torch.cat([obs, mp], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        p = self.proj(f)
        hs = []
        for (dy, dx), mix in zip(self.fps, self.mix):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            hs.append(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
        r = F.relu(self.red(torch.cat(hs, dim=1)))
        place = None
        for k in range(self.no):
            o = self.orient[k](torch.cat([orient_map(r, k), mp], dim=1))
            place = o if place is None else place + o
        place = place.reshape(B, -1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
