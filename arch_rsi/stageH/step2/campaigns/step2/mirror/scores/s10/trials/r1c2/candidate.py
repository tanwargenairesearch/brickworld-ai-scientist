import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.001}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32


class NestHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.ga = nn.ModuleList([nn.Conv2d(proj, hid, 1) for _ in self.fps])
        self.gb = nn.ModuleList([nn.Conv2d(proj, hid, 1) for _ in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(2 * hid + proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        Y = p.shape[2]
        X = p.shape[3]
        res = []
        for (dy, dx), ga, gb, mix, out in zip(self.fps, self.ga, self.gb, self.mix, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(q, (dy, dx), stride=1)
            a = ga(mx)
            b = F.pad(gb(p), (0, dx - 1, 0, dy - 1))
            hs = []
            for oy in range(dy):
                for ox in range(dx):
                    hs.append(F.relu(a + b[:, :, oy:oy + Y, ox:ox + X]))
            h = torch.stack(hs, 0)
            agg = torch.cat([h.amin(0), h.mean(0), mx], dim=1)
            z = F.relu(mix(agg))
            res.append(out(z))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = NestHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
