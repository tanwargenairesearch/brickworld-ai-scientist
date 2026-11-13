import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 56
LH = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.n_types = spec.n_types
        self.Z = spec.Z
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.f_proj = nn.Conv2d(WIDTH, LH, 1)
        self.l_proj = nn.Conv2d(5 + spec.Z, LH, 1)
        self.layer_mid = nn.Conv2d(LH, LH, 1)
        self.type_out = nn.ModuleList([nn.Conv2d(3 * LH, 2, 1) for _ in range(self.n_types)])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        f = self.body(obs)
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        build = obs[:, :Z]
        below = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z].unsqueeze(2)
        zc = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        xl = torch.cat([lay, below, zc], dim=2).reshape(B * Z, 5 + Z, Y, X)
        a = self.f_proj(f)
        b = self.l_proj(xl).view(B, Z, LH, Y, X)
        h = F.relu(b + a.unsqueeze(1)).reshape(B * Z, LH, Y, X)
        h = F.relu(self.layer_mid(h))
        outs = []
        for t in range(self.n_types):
            dy, dx = self.footprints[t]
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            mean = F.avg_pool2d(hp, (dy, dx), stride=1)
            o = self.type_out[t](torch.cat([mx, mn, mean], dim=1)).view(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            outs.append((w * o[:, :, 0]).sum(1))
        place = torch.stack(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
