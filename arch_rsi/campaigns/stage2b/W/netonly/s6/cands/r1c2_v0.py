import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 56
K = 8
HID = 16


def fp_pool(u, dy, dx):
    if dy == 1 and dx == 1:
        return u, u
    p = F.pad(u, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(p, (dy, dx), stride=1)
    mn = -F.max_pool2d(-p, (dy, dx), stride=1)
    return mx, mn


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.cell = nn.Conv2d(WIDTH + spec.in_channels, self.Z * K, 1)
        self.hid = nn.ModuleList([nn.Linear(2 * K, HID) for _ in self.fps])
        self.out = nn.ModuleList([nn.Linear(HID, 1) for _ in self.fps])
        self.zbias = nn.Parameter(torch.zeros(len(self.fps), self.Z))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        u = self.cell(torch.cat([f, obs], dim=1))
        B, _, Y, X = u.shape
        Z = self.Z
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            mx, mn = fp_pool(u, dy, dx)
            h = torch.cat([mx.reshape(B, Z, K, Y, X), mn.reshape(B, Z, K, Y, X)], dim=2)
            h = h.permute(0, 3, 4, 1, 2)
            s = self.out[t](F.relu(self.hid[t](h))).squeeze(-1) + self.zbias[t]
            outs.append(torch.logsumexp(s, dim=-1).flatten(1))
        place = torch.cat(outs, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
