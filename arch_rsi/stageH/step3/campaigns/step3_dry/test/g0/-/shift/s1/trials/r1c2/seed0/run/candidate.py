import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97}

C = 24
COLF = 32
P = 16
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, 48, 1), nn.ReLU(),
            nn.Conv2d(48, COLF, 1), nn.ReLU())
        self.cc1 = nn.Conv2d(COLF, COLF, 3, padding=1)
        self.cc2 = nn.Conv2d(COLF, COLF, 3, padding=1)
        self.lin = nn.Conv2d(15 + Z, C, 1)
        self.colproj = nn.Conv2d(COLF, C, 1)
        self.sp = nn.Conv2d(C, C, 3, padding=1)
        self.mixz = nn.Conv2d(3 * C, C, 1)
        self.proj = nn.Conv2d(C, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * C + 2 * COLF, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        pz = F.pad(per, (0, 0, 0, 0, 1, 1))
        layers = torch.cat([pz[:, :, 0:Z], pz[:, :, 1:Z + 1], pz[:, :, 2:Z + 2]], dim=1)
        gin = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        onehot = torch.eye(Z, device=obs.device, dtype=obs.dtype).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        L = torch.cat([layers, gin, onehot], dim=1)
        L = L.permute(0, 2, 1, 3, 4).reshape(B * Z, 15 + Z, Y, X)

        c = self.col(obs)
        c = F.relu(c + self.cc1(c))
        c = F.relu(c + self.cc2(c))
        cp = self.colproj(c).unsqueeze(1).expand(B, Z, C, Y, X).reshape(B * Z, C, Y, X)

        h = F.relu(self.lin(L) + cp)
        h = F.relu(h + self.sp(h))
        hz = h.reshape(B, Z, C, Y, X)
        z0 = torch.zeros_like(hz[:, :1])
        below = torch.cat([z0, hz[:, :Z - 1]], dim=1)
        above = torch.cat([hz[:, 1:], z0], dim=1)
        nb = torch.cat([below, hz, above], dim=2).reshape(B * Z, 3 * C, Y, X)
        h = F.relu(h + self.mixz(nb))

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
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            o = o.reshape(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * o[:, :, 0]).sum(1).flatten(1))
        place = torch.cat(res, dim=1)

        hf = h.reshape(B, Z, C, Y, X)
        g = torch.cat([hf.mean((1, 3, 4)), hf.amax((1, 3, 4)), c.mean((2, 3)), c.amax((2, 3))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
