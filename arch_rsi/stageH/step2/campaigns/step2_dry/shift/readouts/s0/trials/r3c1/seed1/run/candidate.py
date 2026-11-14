import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "ent_coef": 0.02, "n_minibatches": 16}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32


def align(obs):
    s = obs.shape[-1] // 2
    r = torch.cat([obs[..., s:], obs[..., :s]], dim=-1)
    return torch.cat([obs, r], dim=1)


class FootHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.convs = nn.ModuleList([nn.Conv2d(proj, hid, (dy, dx)) for dy, dx in self.fps])
        self.mids = nn.ModuleList([nn.Conv2d(hid, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = F.relu(self.proj(f))
        res = []
        for (dy, dx), conv, mid, out in zip(self.fps, self.convs, self.mids, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            h = F.relu(conv(q))
            h = F.relu(mid(h))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2 * spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = FootHead(WIDTH + COL, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(align(obs))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(torch.cat([f, s], dim=1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
