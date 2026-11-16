import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "gae_lambda": 0.85}

W = 40
HID = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, 96, 1), nn.ReLU(),
            nn.Conv2d(96, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.n1 = nn.GroupNorm(4, W)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.n2 = nn.GroupNorm(4, W)
        self.gctx = nn.Linear(2 * W, W)
        self.mix = nn.ModuleList([nn.Conv2d(W * dy * dx, HID, 1) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        Y = obs.shape[2]
        X = obs.shape[3]
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        g0 = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        x = x + self.gctx(g0)[:, :, None, None]
        f = F.relu(x + self.n2(self.c2(x)))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                z = f
            else:
                q = F.pad(f, (0, dx - 1, 0, dy - 1))
                z = torch.cat([q[:, :, i:i + Y, j:j + X] for i in range(dy) for j in range(dx)], dim=1)
            res.append(out(F.relu(mix(z))))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
