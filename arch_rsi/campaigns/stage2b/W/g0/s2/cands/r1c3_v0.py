import torch
import torch.nn as nn
import torch.nn.functional as F

W = 32
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.nt = spec.n_types
        c_l = 14
        self.ctx = nn.Conv2d(spec.in_channels, W, 1)
        self.conv1 = nn.Conv2d(c_l, W, 3, padding=1)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.score = nn.Conv2d(W, 2 * spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(4 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        g = obs[:, 4 * Z:4 * Z + 3]
        cur = torch.stack([b, t, m, e], dim=2)
        nb = torch.stack([b, t, m], dim=2)
        zero = torch.zeros_like(nb[:, :1])
        below = torch.cat([zero, nb[:, :-1]], dim=1)
        above = torch.cat([nb[:, 1:], zero], dim=1)
        gl = g.unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([cur, below, above, gl, zc], dim=2).reshape(B * Z, 14, Y, X)
        ctx = self.ctx(obs)
        h = self.conv1(x).view(B, Z, W, Y, X) + ctx.unsqueeze(1)
        h = F.relu(h).reshape(B * Z, W, Y, X)
        h = F.relu(self.conv2(h))
        s = self.score(h).view(B, Z, 2, self.nt, Y, X)
        w = torch.softmax(s[:, :, 0], dim=1)
        place = (w * s[:, :, 1]).sum(1).flatten(1)
        h5 = h.view(B, Z, W, Y, X)
        c = F.relu(ctx)
        pooled = torch.cat([h5.mean((1, 3, 4)), h5.amax((1, 3, 4)), c.mean((2, 3)), c.amax((2, 3))], dim=1)
        gf = self.glob(pooled)
        logits = torch.cat([place, self.done_head(gf)], dim=1)
        return logits, self.value_head(gf).squeeze(1)


def build_policy(spec):
    return Net(spec)
