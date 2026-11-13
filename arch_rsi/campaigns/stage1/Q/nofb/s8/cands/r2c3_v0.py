import torch
import torch.nn as nn
import torch.nn.functional as F

W = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.c1 = nn.Conv2d(10, W, 3, padding=1)
        self.col = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.ctx = nn.Linear(2 * W, W)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.heads = nn.ModuleList([nn.Conv2d(4 * W, 2, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        build = obs[:, :Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        bb = F.pad(build, (0, 0, 0, 0, 1, 0))[:, :Z]
        tb = F.pad(target, (0, 0, 0, 0, 1, 0))[:, :Z]
        per = torch.stack([build, target, miss, extra, bb, tb], dim=2)
        planes = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        v = torch.cat([per, planes, zc], dim=2).reshape(B * Z, 10, Y, X)
        h = self.c1(v).view(B, Z, W, Y, X) + self.col(obs).unsqueeze(1)
        h = F.relu(h)
        c = self.ctx(torch.cat([h.mean((1, 3, 4)), h.amax((1, 3, 4))], dim=1))
        h = F.relu(h + c.view(B, 1, W, 1, 1)).reshape(B * Z, W, Y, X)
        h = h + F.relu(self.c2(h))
        outs = []
        for (dy, dx), head in zip(self.footprints, self.heads):
            fp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(fp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-fp, (dy, dx), stride=1)
            av = F.avg_pool2d(fp, (dy, dx), stride=1)
            o = head(torch.cat([h, mx, mn, av], dim=1)).view(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 0], dim=1)
            outs.append((w * o[:, :, 1]).sum(1))
        place = torch.stack(outs, dim=1).flatten(1)
        hz = h.view(B, Z, W, Y, X)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
