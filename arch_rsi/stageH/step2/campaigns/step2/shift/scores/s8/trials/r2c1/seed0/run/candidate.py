import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16}

H1 = 48
H2 = 32
CTX = 16
PROJ = 24
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = spec.in_channels + 5 + 1 + self.Z
        self.cell = nn.Sequential(nn.Conv2d(cin, H1, 1), nn.ReLU(), nn.Conv2d(H1, H2, 1), nn.ReLU())
        self.ctx = nn.Conv2d(H2, CTX, 3, padding=1)
        self.proj = nn.Conv2d(H2 + CTX, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * (H2 + CTX), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def layer_input(self, obs):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        below = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        per = torch.stack([build, target, miss, extra, below], dim=2)
        full = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        zs = torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z
        zc = zs.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        oh = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([per, full, zc, oh], dim=2)
        return x.reshape(B * Z, -1, Y, X)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        h = self.cell(self.layer_input(obs))
        h = torch.cat([h, F.relu(self.ctx(h))], dim=1)
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
            res.append(out(hh))
        lg = torch.cat(res, dim=1)
        T = lg.shape[1]
        lg = torch.logsumexp(lg.view(B, Z, T, Y, X), dim=1)
        place = lg.reshape(B, T * Y * X)
        hv = h.view(B, Z, h.shape[1], Y, X)
        g = self.glob(torch.cat([hv.mean((1, 3, 4)), hv.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
