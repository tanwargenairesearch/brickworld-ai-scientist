import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 16
HID = 16
GATE_SCALE = 5.0


def make_ops(Y, X):
    ops = []
    trs = [False, True] if Y == X else [False]
    for tr in trs:
        for fy in (False, True):
            for fx in (False, True):
                ops.append((tr, fy, fx))
    return ops


def apply_op(t, op):
    tr, fy, fx = op
    if tr:
        t = t.transpose(2, 3)
    if fy:
        t = t.flip(2)
    if fx:
        t = t.flip(3)
    return t


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.ops = make_ops(self.Y, self.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.gate = nn.Parameter(torch.zeros(len(self.ops)))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        G = len(self.ops)
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        p = self.proj(f)
        q = torch.cat([apply_op(p, op) for op in self.ops], dim=0).contiguous()
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = q, q, q
            else:
                qq = F.pad(q, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(qq, (dy, dx), stride=1)
                mn = -F.max_pool2d(-qq, (dy, dx), stride=1)
                av = F.avg_pool2d(qq, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        lg = torch.cat(res, dim=1).flatten(1).reshape(G, B, -1)
        w = torch.softmax(self.gate * GATE_SCALE, dim=0).view(G, 1, 1)
        place = (w * lg).sum(0)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
