import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gae_lambda": 0.95}

TR = 48
W = 32
P = 24
Q = 16
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        k = 8 + 3 + self.Z
        self.stem = nn.Sequential(nn.Conv2d(spec.in_channels, TR, 1), nn.ReLU())
        self.c1 = nn.Conv2d(TR, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.lin = nn.Conv2d(k, P, 1)
        self.ctx = nn.Conv2d(W, P, 1)
        self.lmix = nn.Conv2d(P, Q, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * Q, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        one = torch.ones_like(b[:, :1])
        zero = torch.zeros_like(b[:, :1])
        b_dn = torch.cat([one, b[:, :Z - 1]], 1)
        t_dn = torch.cat([one, t[:, :Z - 1]], 1)
        m_dn = torch.cat([zero, m[:, :Z - 1]], 1)
        t_up = torch.cat([t[:, 1:], zero], 1)
        per = torch.stack([b, t, m, e, b_dn, t_dn, m_dn, t_up], 2)
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype)
        zo = eye.view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([per, g3, zo], 2).reshape(B * Z, 8 + 3 + Z, Y, X)

        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))

        h = self.lin(inp).view(B, Z, P, Y, X) + self.ctx(f).unsqueeze(1)
        h = F.relu(h).reshape(B * Z, P, Y, X)
        p = self.lmix(h)
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
            sc = out(hh).view(B, Z, Y, X)
            res.append(torch.logsumexp(sc, dim=1))
        place = torch.stack(res, 1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
