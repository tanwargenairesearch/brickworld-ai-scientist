import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
K = 16
FC_C = 8

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "lr": 0.0015, "gamma": 0.97}


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


def all_views(p, square):
    v = [p, p.flip(2), p.flip(3), p.flip(2, 3)]
    if square:
        v = v + [t.transpose(2, 3) for t in v]
    return torch.cat(v, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.square = self.Y == self.X
        nv = 8 if self.square else 4
        cells = self.Y * self.X
        self.stem = nn.Sequential(
            nn.Conv2d(int(spec.in_channels) + self.nt, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.pre = nn.Conv2d(WIDTH, K, 1)
        self.post = nn.Sequential(
            nn.Conv2d(nv * K, HID, 1), nn.ReLU(),
            nn.Conv2d(HID, self.nt, 1))
        self.fcp = nn.Conv2d(WIDTH, FC_C, 1)
        self.fc = nn.Linear(FC_C * cells, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        m = mask[:, :self.n_place].reshape(b, self.nt, self.Y, self.X).to(obs.dtype)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(self.c1(s))
        x = F.relu(x + self.c2(x))
        f = F.relu(x + self.c3(x))
        v = all_views(F.relu(self.pre(f)), self.square)
        place = self.head(f) + self.post(v).flatten(1) + self.fc(F.relu(self.fcp(f)).flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
