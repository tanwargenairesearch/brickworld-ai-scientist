import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.conv1 = nn.Conv3d(8, WIDTH, 3, padding=1)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.conv2 = nn.Conv3d(WIDTH, WIDTH, (1, 3, 3), padding=(0, 1, 1))
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.heads = nn.ModuleList([nn.Conv3d(3 * WIDTH, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, C, Y, X = obs.shape
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        vox = torch.stack([build, target, miss, extra], dim=1)
        g2 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([vox, g2, zc], dim=1)
        h = F.relu(self.conv1(x))
        g = F.relu(self.ctx(torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], dim=1)))
        h = h + g[:, :, None, None, None]
        f = h + F.relu(self.conv2(h))
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool3d(p, (1, dy, dx), stride=1)
            mn = -F.max_pool3d(-p, (1, dy, dx), stride=1)
            av = F.avg_pool3d(p, (1, dy, dx), stride=1)
            s = head(torch.cat([mx, mn, av], dim=1))
            s = torch.logsumexp(s, dim=2)
            outs.append(s.flatten(1))
        place = torch.cat(outs, dim=1)
        z = self.glob(torch.cat([f.mean((2, 3, 4)), f.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(z)], dim=1)
        return logits, self.value_head(z).squeeze(1)


def build_policy(spec):
    return Net(spec)
