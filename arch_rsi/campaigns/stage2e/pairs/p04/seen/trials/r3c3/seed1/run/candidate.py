import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
PROJ = 32
HID = 32
LK = 8

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16}


def fp_pool(t, dy, dx):
    if dy > 1 or dx > 1:
        p = F.pad(t, (0, dx - 1, 0, dy - 1))
        mx = F.max_pool2d(p, (dy, dx), stride=1)
        mn = -F.max_pool2d(-p, (dy, dx), stride=1)
        return mx, mn
    return t, t


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.skip = nn.Conv2d(spec.in_channels, PROJ, 1)
        self.layer_enc = nn.Sequential(
            nn.Conv2d(7, LK, 1), nn.ReLU(), nn.Conv2d(LK, LK, 1), nn.ReLU()
        )
        n_in = 6 * PROJ + 3 * self.Z * LK
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Conv2d(n_in, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        f = self.body(obs)
        g = F.relu(torch.cat([self.proj(f), self.skip(obs)], dim=1))
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        hts = obs[:, 4 * Z:4 * Z + 2].unsqueeze(1).expand(B, Z, 2, Y, X)
        zpos = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zpos = zpos.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        lay = torch.cat([per, hts, zpos], dim=2).reshape(B * Z, 7, Y, X)
        h = self.layer_enc(lay)
        outs = []
        for (dy, dx), mlp in zip(self.fps, self.mlps):
            gmx, gmn = fp_pool(g, dy, dx)
            hmx, hmn = fp_pool(h, dy, dx)
            lf = torch.cat([h, hmx, hmn], dim=1).reshape(B, Z * 3 * LK, Y, X)
            outs.append(mlp(torch.cat([g, gmx, gmn, lf], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
