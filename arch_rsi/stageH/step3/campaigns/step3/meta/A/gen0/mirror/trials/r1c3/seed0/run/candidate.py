import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.005}

HID = 40
TR = 24
PROJ = 16
MIX = 16
GH = 64


def foot_pool(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = int(spec.in_channels)
        Z = self.Z
        self.register_buffer("zemb", torch.eye(Z))
        self.trunk = nn.Sequential(
            nn.Conv2d(C, TR, 1), nn.ReLU(),
            nn.Conv2d(TR, TR, 3, padding=1), nn.ReLU())
        cin = 12 + C + Z + TR
        self.s1 = nn.Conv2d(cin, HID, 1)
        self.s2 = nn.Conv2d(HID, HID, 1)
        self.proj = nn.Conv2d(HID, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, MIX, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(MIX, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * HID + 2 * TR, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        t = self.trunk(obs)
        local = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=2)
        padded = F.pad(local, (0, 0, 0, 0, 0, 0, 1, 1))
        below = padded[:, 0:Z]
        above = padded[:, 2:Z + 2]
        g = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        zc = self.zemb.view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        tc = t.unsqueeze(1).expand(B, Z, t.shape[1], Y, X)
        x = torch.cat([local, below, above, g, zc, tc], dim=2).reshape(B * Z, -1, Y, X)
        h = F.relu(self.s1(x))
        h = F.relu(h + self.s2(h))
        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            u = F.relu(mix(foot_pool(p, dy, dx)))
            res.append(out(u))
        s = torch.cat(res, dim=1).view(B, Z, len(self.fps), Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        hz = h.view(B, Z, HID, Y, X)
        feat = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4)), t.mean((2, 3)), t.amax((2, 3))], dim=1)
        gg = self.glob(feat)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
