import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "n_minibatches": 16}

COLW = 48
COLO = 32
W = 24
P = 16
H = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col1 = nn.Conv2d(spec.in_channels, COLW, 1)
        self.col2 = nn.Conv2d(COLW, COLO, 3, padding=1)
        self.col_to_layer = nn.Conv2d(COLO, W, 1)
        self.layer_in = nn.Conv2d(13 + Z, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.zconv = nn.Conv2d(W, W, (3, 1), padding=(1, 0))
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W + 2 * COLO, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        self.register_buffer("eye", torch.eye(Z))

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        g3 = obs[:, 4 * Z:4 * Z + 3]
        z0 = torch.zeros_like(b[:, :1])
        bp = torch.cat([z0, b, z0], dim=1)
        tp = torch.cat([z0, t, z0], dim=1)
        mp = torch.cat([z0, m, z0], dim=1)
        feats = torch.stack([b, t, m, e,
                             bp[:, :Z], tp[:, :Z], mp[:, :Z],
                             bp[:, 2:], tp[:, 2:], mp[:, 2:]], dim=2)
        gl = g3.unsqueeze(1).expand(B, Z, 3, Y, X)
        oh = self.eye.view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        li = torch.cat([feats, gl, oh], dim=2).reshape(B * Z, 13 + Z, Y, X)

        col = F.relu(self.col1(obs))
        col = F.relu(self.col2(col))
        cl = self.col_to_layer(col)
        cl = cl.unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)

        h = F.relu(self.layer_in(li) + cl)
        h = F.relu(h + self.c1(h))
        hz = h.reshape(B, Z, W, Y * X).permute(0, 2, 1, 3)
        hz = F.relu(hz + self.zconv(hz))
        h = hz.permute(0, 2, 1, 3).reshape(B * Z, W, Y, X)

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
            s = o[:, :, 0]
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * s).sum(1).flatten(1))
        place = torch.cat(res, dim=1)

        h5 = h.reshape(B, Z, W, Y, X)
        gv = torch.cat([h5.mean((1, 3, 4)), h5.amax((1, 3, 4)),
                        col.mean((2, 3)), col.amax((2, 3))], dim=1)
        g = self.glob(gv)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
