import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1.5e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005}

W = 32
P = 16
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 12 + 3 + self.Z
        self.stem = nn.Conv2d(cin, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.ctx = nn.Conv2d(W, W, 1)
        self.c2 = nn.Conv2d(W, W, 1)
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        L = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        Lp = F.pad(L, (0, 0, 0, 0, 1, 1))
        below = Lp[:, :, 0:Z]
        above = Lp[:, :, 2:Z + 2]
        g = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zpos = torch.eye(Z, device=obs.device, dtype=obs.dtype).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([L, below, above, g, zpos], dim=1)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * Z, -1, Y, X)
        h = F.relu(self.stem(x))
        h = F.relu(h + self.c1(h))
        hz = h.reshape(B, Z, W, Y, X)
        c = self.ctx(hz.mean(1))
        h = F.relu((hz + c.unsqueeze(1)).reshape(B * Z, W, Y, X))
        h = F.relu(h + self.c2(h))
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
            t = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(t))
        scores = torch.cat(res, dim=1).reshape(B, Z, len(self.fps), Y, X)
        place = torch.logsumexp(scores, dim=1).flatten(1)
        hz = h.reshape(B, Z, W, Y, X)
        gf = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(gf)], dim=1)
        return logits, self.value_head(gf).squeeze(1)


def build_policy(spec):
    return Net(spec)
