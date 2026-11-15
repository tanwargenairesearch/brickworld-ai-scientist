import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}

TW = 32
LW = 24
PROJ = 16
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(C, 48, 1), nn.ReLU(),
            nn.Conv2d(48, TW, 1), nn.ReLU())
        self.c1 = nn.Conv2d(TW, TW, 3, padding=1)
        self.c2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.lin = nn.Conv2d(9 + 3 + self.Z, LW, 1)
        self.ltr = nn.Conv2d(TW, LW, 1)
        self.lc = nn.Conv2d(LW, LW, 3, padding=1)
        self.proj = nn.Conv2d(LW, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TW, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        # shared 2D trunk
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        # per-layer inputs (only slicing / padding / stacking before learned layer)
        b = F.pad(obs[:, 0:Z], (0, 0, 0, 0, 1, 1))
        t = F.pad(obs[:, Z:2 * Z], (0, 0, 0, 0, 1, 1))
        m = F.pad(obs[:, 2 * Z:3 * Z], (0, 0, 0, 0, 1, 1))
        e = obs[:, 3 * Z:4 * Z]
        bl = torch.stack([b[:, 0:Z], b[:, 1:Z + 1], b[:, 2:Z + 2],
                          t[:, 0:Z], t[:, 1:Z + 1], t[:, 2:Z + 2],
                          m[:, 0:Z], m[:, 1:Z + 1], e], dim=2)
        ex = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        oh = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([bl, ex, oh], dim=2).reshape(B * Z, 12 + Z, Y, X)
        tr = self.ltr(f).unsqueeze(1).expand(B, Z, LW, Y, X).reshape(B * Z, LW, Y, X)
        h = F.relu(self.lin(inp) + tr)
        h = F.relu(h + self.lc(h))
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
            hh = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            o = out(hh).reshape(B, Z, 2, Y, X)
            score = o[:, :, 0]
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * score).sum(1).flatten(1))
        place = torch.cat(res, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
