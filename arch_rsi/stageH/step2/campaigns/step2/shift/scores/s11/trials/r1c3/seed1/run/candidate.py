import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8}

W = 24
PROJ = 16
HID = 16
NLOC = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.gin = nn.Conv2d(spec.in_channels, W, 1)
        self.lin = nn.Conv2d(NLOC + self.Z, W, 1)
        self.c3 = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        ba = F.pad(b, (0, 0, 0, 0, 0, 1), value=0.0)[:, 1:]
        ta = F.pad(t, (0, 0, 0, 0, 0, 1), value=0.0)[:, 1:]
        loc = torch.stack([b, t, m, e, bb, tb, ba, ta], dim=2)
        oh = torch.eye(Z, device=obs.device, dtype=obs.dtype)
        oh = oh.view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        loc = torch.cat([loc, oh], dim=2).reshape(B * Z, NLOC + Z, Y, X)
        g = self.gin(obs).unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h = F.relu(self.lin(loc) + g)
        h = F.relu(h + self.c3(h))
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
        s = torch.cat(res, dim=1)
        T = s.shape[1]
        s = s.view(B, Z, T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        hv = h.view(B, Z, W, Y, X)
        pooled = torch.cat([hv.amax(dim=(1, 3, 4)), hv.mean(dim=(1, 3, 4))], dim=1)
        gg = self.glob(pooled)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
