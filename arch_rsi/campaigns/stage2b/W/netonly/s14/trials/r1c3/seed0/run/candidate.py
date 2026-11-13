import torch
import torch.nn as nn
import torch.nn.functional as F

LW = 28
TW = 52


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.register_buffer("zcoord", torch.arange(spec.Z, dtype=torch.float32) / spec.Z)
        self.lconv1 = nn.Conv2d(12, LW, 3, padding=1)
        self.lconv2 = nn.Conv2d(LW, LW, 1)
        self.trunk = nn.Conv2d(2 * LW, TW, 3, padding=1)
        self.place_head = nn.Conv2d(TW, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * TW, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, _, Y, X = obs.shape
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        missing = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        below_b = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        below_t = F.pad(target, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        above_b = F.pad(build, (0, 0, 0, 0, 0, 1))[:, 1:]
        above_t = F.pad(target, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([build, target, missing, extra, below_b, below_t, above_b, above_t], dim=2)
        glob_ch = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = self.zcoord.to(obs.dtype).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, glob_ch, zc], dim=2).reshape(B * Z, 12, Y, X)
        h = F.relu(self.lconv1(x))
        h = F.relu(self.lconv2(h))
        h = h.reshape(B, Z, LW, Y, X)
        agg = torch.cat([h.amax(1), h.mean(1)], dim=1)
        f = F.relu(self.trunk(agg))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
