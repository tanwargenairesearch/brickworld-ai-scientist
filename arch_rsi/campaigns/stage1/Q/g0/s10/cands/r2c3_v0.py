import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HEAD = 16
C3 = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.z3 = nn.Conv3d(5, C3, 3, padding=1)
        self.conv1 = nn.Conv2d(C3 * self.Z + 3, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.pre = nn.Conv2d(WIDTH, HEAD, 1)
        self.fp_convs = nn.ModuleList()
        self.pool_heads = nn.ModuleList()
        for (dy, dx) in self.fps:
            self.fp_convs.append(nn.Conv2d(WIDTH, 1, (dy, dx)))
            self.pool_heads.append(nn.Sequential(
                nn.Conv2d(2 * HEAD, HEAD, 1), nn.ReLU(), nn.Conv2d(HEAD, 1, 1)))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        v = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype)
        zc = zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        v = torch.cat([v, zc], dim=1)
        h = F.relu(self.z3(v)).reshape(B, C3 * Z, Y, X)
        x = torch.cat([h, obs[:, 4 * Z:]], dim=1)
        f = F.relu(self.conv2(F.relu(self.conv1(x))))
        p = self.pre(f)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            pad = (0, dx - 1, 0, dy - 1)
            a = self.fp_convs[i](F.pad(f, pad))
            pp = F.pad(p, pad)
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            b = self.pool_heads[i](torch.cat([mx, mn], dim=1))
            outs.append((a + b).flatten(1))
        place = torch.cat(outs, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
