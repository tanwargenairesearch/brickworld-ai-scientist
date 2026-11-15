import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1.5e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 48
FEAT = 32
PROJ = 16
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 7 + int(spec.in_channels) + self.Z
        self.register_buffer('zpos', torch.eye(self.Z).view(1, self.Z, self.Z, 1, 1))
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, FEAT, 1), nn.ReLU())
        self.proj = nn.Conv2d(FEAT, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(FEAT * (2 + self.Z), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        mb = F.pad(m, (0, 0, 0, 0, 1, 0))[:, :Z]
        lay = torch.stack([b, t, m, e, bb, tb, mb], dim=2)
        col = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        zp = self.zpos.expand(B, Z, Z, Y, X)
        h = torch.cat([lay, col, zp], dim=2).reshape(B * Z, -1, Y, X)
        h = self.stem(h)
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
        T = len(self.fps)
        o = torch.stack(res, dim=1).reshape(B, Z, T, 2, Y, X)
        s = o[:, :, :, 0]
        g = o[:, :, :, 1]
        w = torch.softmax(g, dim=1)
        place = (w * s).sum(1).reshape(B, -1)
        hf = h.reshape(B, Z, FEAT, Y, X)
        per_layer = hf.amax((3, 4)).reshape(B, Z * FEAT)
        gfeat = torch.cat([hf.mean((1, 3, 4)), hf.amax((1, 3, 4)), per_layer], dim=1)
        gg = self.glob(gfeat)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
