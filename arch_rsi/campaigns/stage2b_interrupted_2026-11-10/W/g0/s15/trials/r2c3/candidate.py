import torch
import torch.nn as nn
import torch.nn.functional as F

W = 32
HID = 16

RECIPE = {"n_envs": 128, "n_minibatches": 16}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.glob_conv = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.slice_conv = nn.Conv2d(8, W, 3, padding=1)
        self.zemb = nn.Embedding(self.Z, W)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.footprints = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * W, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.footprints
        ])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        cur = torch.stack([build, tgt, miss, extra], dim=2)
        below = F.pad(cur, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        sl = torch.cat([cur, below], dim=2).reshape(B * Z, 8, Y, X)
        a = self.glob_conv(obs)
        s = self.slice_conv(sl).reshape(B, Z, W, Y, X)
        zi = torch.arange(Z, device=obs.device)
        e = self.zemb(zi).view(1, Z, W, 1, 1)
        h = F.relu(a.unsqueeze(1) + s + e).reshape(B * Z, W, Y, X)
        h = h + F.relu(self.conv2(h))
        outs = []
        for (dy, dx), head in zip(self.footprints, self.heads):
            if dy == 1 and dx == 1:
                mx = h
                mn = h
            else:
                p = F.pad(h, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(p, kernel_size=(dy, dx), stride=1)
                mn = -F.max_pool2d(-p, kernel_size=(dy, dx), stride=1)
            o = head(torch.cat([mx, mn], dim=1)).reshape(B, Z, Y, X)
            outs.append(torch.logsumexp(o, dim=1))
        place = torch.stack(outs, dim=1).flatten(1)
        hz = h.reshape(B, Z, W, Y, X)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
