import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
K = 12
HID = 12


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.T = spec.n_types
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        Z, T = self.Z, self.T
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.layer_proj = nn.Conv2d(WIDTH, Z * K, 1)
        self.vox_proj = nn.Conv3d(9, K, 1)
        self.type_hidden = nn.Conv3d(T * 3 * K, T * HID, 1, groups=T)
        self.type_out = nn.Conv3d(T * HID, 2 * T, 1, groups=T)
        nn.init.zeros_(self.type_out.weight)
        nn.init.zeros_(self.type_out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)
        if Z > 1:
            zc = torch.linspace(0.0, 1.0, Z)
        else:
            zc = torch.zeros(1)
        self.register_buffer('zc', zc.view(1, 1, Z, 1, 1))

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z, T = self.Z, self.T
        f = self.body(obs)
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        missing = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        build_below = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        missing_below = F.pad(missing, (0, 0, 0, 0, 1, 0), value=0.0)[:, :Z]
        height = obs[:, 4 * Z:4 * Z + 1].unsqueeze(2).expand(B, 1, Z, Y, X)
        theight = obs[:, 4 * Z + 1:4 * Z + 2].unsqueeze(2).expand(B, 1, Z, Y, X)
        zc = self.zc.expand(B, 1, Z, Y, X)
        vox = torch.stack([build, target, missing, extra, build_below, missing_below], dim=1)
        vox = torch.cat([vox, height, theight, zc], dim=1)
        p = F.relu(self.layer_proj(f).reshape(B, K, Z, Y, X) + self.vox_proj(vox))
        pooled = []
        for (dy, dx) in self.footprints:
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool3d(q, (1, dy, dx), stride=1)
            mn = -F.max_pool3d(-q, (1, dy, dx), stride=1)
            av = F.avg_pool3d(q, (1, dy, dx), stride=1)
            pooled += [mx, mn, av]
        h = torch.cat(pooled, dim=1)
        o = self.type_out(F.relu(self.type_hidden(h))).reshape(B, T, 2, Z, Y, X)
        w = torch.softmax(o[:, :, 1], dim=2)
        place = (w * o[:, :, 0]).sum(2).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
