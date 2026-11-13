"""r2c2 plus per-layer footprint scoring: shared conv over each layer's slices, logsumexp over layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HEAD = 32
LCH = 9
RECIPE = {"n_envs": 128, "ppo_epochs": 2, "n_minibatches": 16}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.Z = int(spec.Z)
        self.body = nn.Sequential(
            nn.Conv2d(c, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.foot = [(int(a), int(b)) for a, b in spec.footprints]
        self.heads_f = nn.ModuleList()
        self.heads_l = nn.ModuleList()
        self.gproj = nn.ModuleList()
        self.outs = nn.ModuleList()
        for dy, dx in self.foot:
            self.heads_f.append(nn.Conv2d(WIDTH + c, HEAD, (dy, dx)))
            self.heads_l.append(nn.Conv2d(LCH, HEAD, (dy, dx), bias=False))
            self.gproj.append(nn.Linear(WIDTH, HEAD))
            self.outs.append(nn.Conv2d(HEAD, 1, 1))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, _, Y, X = obs.shape
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bp = F.pad(b, (0, 0, 0, 0, 1, 0))[:, :Z]
        tp = F.pad(t, (0, 0, 0, 0, 1, 0))[:, :Z]
        hb = obs[:, 4 * Z:4 * Z + 1].expand(B, Z, Y, X)
        ht = obs[:, 4 * Z + 1:4 * Z + 2].expand(B, Z, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1).expand(B, Z, Y, X)
        L = torch.stack([b, t, m, e, bp, tp, hb, ht, zc], dim=2).reshape(B * Z, LCH, Y, X)

        f = self.body(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = torch.cat([f, obs], dim=1)
        parts = []
        for (dy, dx), cf, cl, gp, out in zip(self.foot, self.heads_f, self.heads_l, self.gproj, self.outs):
            pf = F.pad(h, (0, dx - 1, 0, dy - 1))
            pl = F.pad(L, (0, dx - 1, 0, dy - 1))
            zf = cf(pf) + gp(g)[:, :, None, None]
            zl = cl(pl).view(B, Z, HEAD, Y, X)
            hid = F.relu(zl + zf[:, None])
            s = out(hid.reshape(B * Z, HEAD, Y, X)).view(B, Z, Y, X)
            parts.append(torch.logsumexp(s, dim=1).flatten(1))
        logits = torch.cat(parts + [self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
