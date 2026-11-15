import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.0006, 'n_envs': 128, 'rollout_len': 16, 'n_minibatches': 8, 'ppo_epochs': 2, 'ent_coef': 0.02}

CTX = 16
LW = 24
PROJ = 16
HID = 16
GH = 64


class LayerNet(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.ctx = nn.Sequential(
            nn.Conv2d(spec.in_channels, 32, 1), nn.ReLU(),
            nn.Conv2d(32, CTX, 1), nn.ReLU())
        nin = 8 + self.Z + CTX
        self.inp = nn.Conv2d(nin, LW, 1)
        self.c1 = nn.Conv2d(LW, LW, 3, padding=1)
        self.proj = nn.Conv2d(LW, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * LW + 2 * CTX, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        occ = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        mis = obs[:, 2 * Z:3 * Z]
        ext = obs[:, 3 * Z:4 * Z]
        one = torch.ones(B, 1, Y, X, device=obs.device, dtype=obs.dtype)
        zero = torch.zeros(B, 1, Y, X, device=obs.device, dtype=obs.dtype)
        occ_p = torch.cat([one, occ, zero], 1)
        tgt_p = torch.cat([one, tgt, zero], 1)
        per = torch.stack([occ, tgt, mis, ext,
                           occ_p[:, 0:Z], tgt_p[:, 0:Z],
                           occ_p[:, 2:Z + 2], tgt_p[:, 2:Z + 2]], dim=2)
        ctx = self.ctx(obs)
        ctxz = ctx.unsqueeze(1).expand(B, Z, CTX, Y, X)
        lid = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([per, lid, ctxz], 2).reshape(B * Z, 8 + Z + CTX, Y, X)
        h = F.relu(self.inp(inp))
        h = F.relu(h + self.c1(h))
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
            hh = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(hh))
        T = len(self.fps)
        s = torch.cat(res, dim=1).view(B, Z, T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        hz = h.view(B, Z, LW, Y, X)
        g = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4)),
                       ctx.mean((2, 3)), ctx.amax((2, 3))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return LayerNet(spec)
