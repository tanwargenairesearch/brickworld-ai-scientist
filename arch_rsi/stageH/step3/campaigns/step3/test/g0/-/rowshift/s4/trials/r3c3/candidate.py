import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "lr": 0.0007, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
WIDTH = 40
PROJ = 32
HID = 32
DILS = [1, 2, 1, 2]


class Block(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=d, dilation=d)
        self.ctx = nn.Linear(2 * w, w)

    def forward(self, x):
        c = self.ctx(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        return F.relu(x + self.conv(x) + c.unsqueeze(2).unsqueeze(3))


class FootHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.convs = nn.ModuleList([nn.Conv2d(proj, hid, (dy, dx)) for dy, dx in self.fps])
        self.mm = nn.ModuleList([nn.Conv2d(2 * proj, hid, 1, bias=False) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), conv, mm, out in zip(self.fps, self.convs, self.mm, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            a = conv(q)
            if dy * dx == 1:
                mx, mn = p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(a + mm(torch.cat([mx, mn], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + self.T, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.blocks = nn.ModuleList([Block(WIDTH, d) for d in DILS])
        self.head = FootHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, Y, X)
        f = self.stem(torch.cat([obs, m], dim=1))
        for blk in self.blocks:
            f = blk(f)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
