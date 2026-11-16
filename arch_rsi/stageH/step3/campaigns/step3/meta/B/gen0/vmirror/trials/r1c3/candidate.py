import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "ent_coef": 0.005}

H = 32
PROJ = 16
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        C = spec.in_channels
        self.col = nn.Sequential(
            nn.Conv2d(C, 48, 1), nn.ReLU(),
            nn.Conv2d(48, H, 1), nn.ReLU())
        self.ctx = nn.Conv2d(H, H, 3, padding=1)
        self.local = nn.Conv2d(8 + self.Z, H, 1)
        self.lay2 = nn.Conv2d(H, H, 1)
        self.proj = nn.Conv2d(H, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(4 * H, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        b_below = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        t_below = F.pad(tgt, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        m_below = F.pad(miss, (0, 0, 0, 0, 1, 0))[:, :Z]
        b_above = F.pad(build, (0, 0, 0, 0, 0, 1))[:, 1:]
        loc = torch.stack([build, tgt, miss, extra, b_below, t_below, m_below, b_above], dim=2)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        lay = torch.cat([loc, eye], dim=2).reshape(B * Z, 8 + Z, Y, X)

        col = self.col(obs)
        col = F.relu(col + self.ctx(col))
        colr = col.unsqueeze(1).expand(B, Z, H, Y, X).reshape(B * Z, H, Y, X)

        h = F.relu(self.local(lay) + colr)
        h = F.relu(h + self.lay2(h))
        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            u = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(u))
        r = torch.stack(res, dim=1).view(B, Z, self.T, 2, Y, X)
        score = r[:, :, :, 0]
        gate = r[:, :, :, 1]
        w = torch.softmax(gate, dim=1)
        place = (w * score).sum(1).flatten(1)

        hz = h.view(B, Z, H, Y, X)
        g = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4)),
                       col.mean((2, 3)), col.amax((2, 3))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
