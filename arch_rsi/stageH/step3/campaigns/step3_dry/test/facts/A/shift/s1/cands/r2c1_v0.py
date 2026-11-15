import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
PROJ = 24
HID = 32
N_LAYOUTS = 3


def window_stats(p, kh, kw, top_left):
    if kh * kw == 1:
        return [p, p]
    if top_left:
        q = F.pad(p, (0, kw - 1, 0, kh - 1))
    else:
        q = F.pad(p, (kw - 1, 0, kh - 1, 0))
    mx = F.max_pool2d(q, (kh, kw), stride=1)
    av = F.avg_pool2d(q, (kh, kw), stride=1)
    return [mx, av]


class HedgeHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(6 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, N_LAYOUTS, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            feats = (window_stats(p, dy, dx, True) + window_stats(p, dy, dx, False)
                     + window_stats(p, dx, dy, True))
            h = F.relu(mix(torch.cat(feats, dim=1)))
            res.append(out(h))
        o = torch.stack(res, dim=1)
        a = o[:, :, 0].flatten(1)
        b = o[:, :, 1].transpose(2, 3).flatten(1)
        c = o[:, :, 2].permute(0, 2, 3, 1).flatten(1)
        return a + b + c


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = HedgeHead(WIDTH, spec.footprints, PROJ, HID)
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
