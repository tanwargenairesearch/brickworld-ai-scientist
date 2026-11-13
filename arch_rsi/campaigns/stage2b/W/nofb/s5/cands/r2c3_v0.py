import torch
import torch.nn as nn
import torch.nn.functional as F

W = 32
K = 8
HID = 8

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.005, "gae_lambda": 0.95}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        n = len(self.fps)
        self.n = n
        self.glob_in = nn.Conv2d(spec.in_channels, W, 1)
        self.layer_in = nn.Conv2d(12, W, 1)
        self.zemb = nn.Parameter(torch.randn(spec.Z, W) * 0.1)
        self.conv = nn.Conv2d(W, W, 3, padding=1)
        self.pw = nn.Conv2d(W, W, 1)
        self.proj = nn.Conv2d(W, K * n, 1)
        self.anchor = nn.Conv2d(W, HID * n, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * K, HID, 1) for _ in self.fps])
        self.out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(3 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], 2)
        padded = F.pad(per, (0, 0, 0, 0, 0, 0, 1, 1))
        below = padded[:, 0:Z]
        above = padded[:, 2:Z + 2]
        lay = torch.cat([per, below, above], 2).reshape(B * Z, 12, Y, X)
        g = self.glob_in(obs)
        h = self.layer_in(lay).view(B, Z, W, Y, X) + g.unsqueeze(1) + self.zemb.view(1, Z, W, 1, 1)
        h = F.relu(h).reshape(B * Z, W, Y, X)
        h = F.relu(self.conv(h))
        h = F.relu(self.pw(h) + h)
        p = F.relu(self.proj(h))
        a = self.anchor(h)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            pt = p[:, t * K:(t + 1) * K]
            if dy > 1 or dx > 1:
                pt = F.pad(pt, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(pt, (dy, dx), stride=1)
                mn = -F.max_pool2d(-pt, (dy, dx), stride=1)
                av = F.avg_pool2d(pt, (dy, dx), stride=1)
            else:
                mx = pt
                mn = pt
                av = pt
            q = F.relu(self.mix[t](torch.cat([mx, mn, av], 1)) + a[:, t * HID:(t + 1) * HID])
            outs.append(self.out[t](q))
        s = torch.cat(outs, 1).view(B, Z, self.n, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        hz = h.view(B, Z, W, Y, X)
        pooled = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4)), F.relu(g).mean((2, 3))], dim=1)
        gg = self.glob(pooled)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
