import torch
import torch.nn as nn
import torch.nn.functional as F

SC = 48
SF = 32
PROJ = 24
HID = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "lr": 0.001}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = spec.in_channels + 8 + self.Z
        self.stem = nn.Sequential(
            nn.Conv2d(cin, SC, 1), nn.ReLU(),
            nn.Conv2d(SC, SF, 1), nn.ReLU())
        self.gb = nn.Linear(2 * SF, SF)
        self.proj = nn.Conv2d(SF, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * SF, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        lay = torch.stack([build, tgt, miss, extra], dim=2)
        below = F.pad(lay, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        full = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        zid = torch.eye(Z, device=obs.device, dtype=obs.dtype)[None, :, :, None, None].expand(B, Z, Z, Y, X)
        s = torch.cat([full, lay, below, zid], dim=2).reshape(B * Z, C + 8 + Z, Y, X)
        h = self.stem(s)
        h = torch.flip(h, dims=[3])
        hz = h.view(B, Z, SF, Y, X)
        g0 = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1)
        hz = F.relu(hz + self.gb(g0)[:, None, :, None, None])
        h = hz.reshape(B * Z, SF, Y, X)
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
            u = u.view(B, Z, HID, Y, X).amax(1)
            res.append(out(u))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
