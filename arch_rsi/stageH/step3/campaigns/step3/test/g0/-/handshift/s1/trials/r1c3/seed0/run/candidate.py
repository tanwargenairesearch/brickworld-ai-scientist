import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.97}

HID = 64
CTX = 32
K = 8
MIXH = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Z = self.Z
        C = spec.in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(C, HID, 1), nn.ReLU(),
            nn.Conv2d(HID, HID, 1), nn.ReLU())
        self.ctx = nn.Conv2d(HID, CTX, 3, padding=1)
        self.layer_feat = nn.Conv2d(HID + CTX, Z * K, 1)
        self.mix = nn.ModuleList([nn.Conv2d(Z * 3 * K, Z * MIXH, 1, groups=Z) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(Z * MIXH, Z, 1, groups=Z) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * (HID + CTX), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        h = self.stem(obs)
        c = F.relu(self.ctx(h))
        hc = torch.cat([h, c], dim=1)
        q = self.layer_feat(hc)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = q, q, q
            else:
                p = F.pad(q, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(p, (dy, dx), stride=1)
                mn = -F.max_pool2d(-p, (dy, dx), stride=1)
                av = F.avg_pool2d(p, (dy, dx), stride=1)
            u = torch.stack([mx, mn, av], dim=1)
            u = u.reshape(B, 3, Z, K, Y, X).permute(0, 2, 1, 3, 4, 5).reshape(B, Z * 3 * K, Y, X)
            s = out(F.relu(mix(u)))
            res.append(torch.logsumexp(s, dim=1))
        place = torch.stack(res, dim=1).flatten(1)
        g = self.glob(torch.cat([hc.mean((2, 3)), hc.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
