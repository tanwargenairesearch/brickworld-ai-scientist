import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_minibatches": 64, "gamma": 0.97, "ent_coef": 0.005}

COL = 48
CTX = 24
E = 24
K = 6
HID = 24
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, COL, 1), nn.ReLU())
        self.ctx = nn.Conv2d(COL, CTX, 3, padding=1)
        self.layer_in = nn.Linear(8 + Z, E)
        self.layer_ctx = nn.Conv2d(COL + CTX, Z * E, 1)
        self.scan = nn.Linear(E, K)
        self.layer_mid = nn.Linear(E + 2 * K, E)
        self.mix = nn.ModuleList([nn.Linear(2 * E, HID) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Linear(HID, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * (COL + CTX) + Z * E, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        c = self.col(obs)
        k = F.relu(self.ctx(c))
        ck = torch.cat([c, k], dim=1)
        bits = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(bits, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        lay = torch.cat([bits, below], dim=1).permute(0, 2, 3, 4, 1)
        pos = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1, Z).expand(B, Z, Y, X, Z)
        u = self.layer_in(torch.cat([lay, pos], dim=-1))
        lc = self.layer_ctx(ck).view(B, Z, E, Y, X).permute(0, 1, 3, 4, 2)
        u = F.relu(u + lc)
        s = F.relu(self.scan(u))
        zero = torch.zeros_like(s[:, :1])
        lo_full = torch.cummax(s, dim=1)[0]
        lo = torch.cat([zero, lo_full[:, :-1]], dim=1)
        hi_full = torch.flip(torch.cummax(torch.flip(s, [1]), dim=1)[0], [1])
        hi = torch.cat([hi_full[:, 1:], zero], dim=1)
        u = F.relu(u + self.layer_mid(torch.cat([u, lo, hi], dim=-1)))
        h = u.permute(0, 1, 4, 2, 3).reshape(B * Z, E, Y, X)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn = h, h
            else:
                q = F.pad(h, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            zf = torch.cat([mx, mn], dim=1).view(B, Z, 2 * E, Y, X).permute(0, 1, 3, 4, 2)
            sc = out(F.relu(mix(zf))).squeeze(-1)
            res.append(torch.logsumexp(sc, dim=1))
        place = torch.stack(res, dim=1).flatten(1)
        g = self.glob(torch.cat([c.mean((2, 3)), c.amax((2, 3)), k.mean((2, 3)), k.amax((2, 3)),
                                 u.amax((2, 3)).flatten(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
