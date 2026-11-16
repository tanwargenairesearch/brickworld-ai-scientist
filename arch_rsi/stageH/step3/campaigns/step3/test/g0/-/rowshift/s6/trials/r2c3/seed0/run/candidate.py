import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "lr": 0.002, "ent_coef": 0.003}

S1 = 32
W = 16
PROJ = 12
HID = 16


class LayerNet(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        self.R = 2 * Z - 1
        cin = 5 * self.R + 3
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        valid = torch.zeros(Z, self.R)
        for k in range(Z):
            for j in range(self.R):
                if 0 <= k + j - (Z - 1) < Z:
                    valid[k, j] = 1.0
        self.register_buffer("valid", valid)
        self.stem = nn.Sequential(
            nn.Conv2d(cin, S1, 1), nn.ReLU(),
            nn.Conv2d(S1, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        R = self.R
        parts = []
        for g in range(4):
            G = obs[:, g * Z:(g + 1) * Z]
            Gp = F.pad(G, (0, 0, 0, 0, Z - 1, Z - 1))
            parts.append(torch.stack([Gp[:, k:k + R] for k in range(Z)], dim=1))
        v = self.valid.view(1, Z, R, 1, 1).expand(B, Z, R, Y, X)
        misc = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        x = torch.cat(parts + [v, misc], dim=2).reshape(B * Z, 5 * R + 3, Y, X)
        s = self.stem(x)
        h1 = F.relu(s + self.c1(s))
        f = F.relu(h1 + self.c2(h1))
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        place = torch.cat(res, dim=1).reshape(B, Z, -1)
        place = torch.logsumexp(place, dim=1)
        fz = f.reshape(B, Z, W, Y, X)
        g = self.glob(torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return LayerNet(spec)
