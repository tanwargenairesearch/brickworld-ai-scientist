import torch
import torch.nn as nn
import torch.nn.functional as F

LW = 24
TW = 48
HEAD_CH = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.n_types = spec.n_types
        self.register_buffer("zcoord", torch.arange(spec.Z, dtype=torch.float32) / spec.Z)
        self.lconv1 = nn.Conv2d(12, LW, 3, padding=1)
        self.lconv2 = nn.Conv2d(LW, LW, 1)
        self.trunk = nn.Conv2d(2 * LW, TW, 3, padding=1)
        self.back = nn.Conv2d(TW, LW, 1)
        self.cell_head = nn.Conv2d(LW, spec.n_types * HEAD_CH, 1)
        self.combine = nn.Conv2d(spec.n_types * 3 * HEAD_CH, spec.n_types * 2, 1, groups=spec.n_types)
        self.glob = nn.Sequential(nn.Linear(2 * TW, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        T = self.n_types
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
        h5 = h.reshape(B, Z, LW, Y, X)
        agg = torch.cat([h5.amax(1), h5.mean(1)], dim=1)
        f = F.relu(self.trunk(agg))
        bk = self.back(f).unsqueeze(1)
        h2 = F.relu(h5 + bk).reshape(B * Z, LW, Y, X)
        c = self.cell_head(h2)
        outs = []
        for t, (dy, dx) in enumerate(self.footprints):
            ct = c[:, t * HEAD_CH:(t + 1) * HEAD_CH]
            cp = F.pad(ct, (0, dx - 1, 0, dy - 1))
            mean = F.avg_pool2d(cp, (dy, dx), stride=1)
            mx = F.max_pool2d(cp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-cp, (dy, dx), stride=1)
            outs += [mean, mn, mx]
        sg = self.combine(torch.cat(outs, dim=1)).reshape(B, Z, T, 2, Y, X)
        score = sg[:, :, :, 0]
        gate = torch.softmax(sg[:, :, :, 1], dim=1)
        place = (gate * score).sum(1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
