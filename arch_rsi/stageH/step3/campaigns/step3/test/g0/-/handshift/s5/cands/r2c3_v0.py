import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16,
          "n_minibatches": 16, "ent_coef": 0.005}

W = 40
CW = 32
ZC = 16
P = 16
H = 16
G = 64


def footprint_pool(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = int(spec.in_channels)
        self.nrest = C - 4 * Z
        cin = 4 * (2 * Z - 1) + self.nrest + Z
        self.inp = nn.Conv2d(cin, W, 1)
        self.ctx1 = nn.Conv2d(C, CW, 1)
        self.ctx2 = nn.Conv2d(CW, CW, 3, padding=1)
        self.ctx_to = nn.Conv2d(CW, W, 1)
        self.l2 = nn.Conv2d(W, W, 1)
        self.zc_in = nn.Conv2d(W, ZC, 1)
        self.zc_sp = nn.Conv2d(ZC, ZC, 3, padding=1)
        self.zc_out = nn.Conv2d(ZC, W, 1)
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W + 2 * CW, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        padz = F.pad(lay, (0, 0, 0, 0, Z - 1, Z - 1))
        rel = torch.stack([padz[:, :, z:z + 2 * Z - 1] for z in range(Z)], dim=1)
        rel = rel.reshape(B * Z, 4 * (2 * Z - 1), Y, X)
        rest = obs[:, 4 * Z:].unsqueeze(1).expand(B, Z, self.nrest, Y, X)
        rest = rest.reshape(B * Z, self.nrest, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype)
        zoh = eye.view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X).reshape(B * Z, Z, Y, X)
        lin = torch.cat([rel, rest, zoh], dim=1)

        c = F.relu(self.ctx1(obs))
        c = F.relu(self.ctx2(c))
        cz = self.ctx_to(c).unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h = F.relu(self.inp(lin) + cz)
        h = F.relu(h + self.l2(h))
        q = F.relu(self.zc_sp(F.relu(self.zc_in(h))))
        h = F.relu(h + self.zc_out(q))

        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            s = out(F.relu(mix(footprint_pool(p, dy, dx))))
            res.append(torch.logsumexp(s.reshape(B, Z, Y * X), dim=1))
        place = torch.cat(res, dim=1)

        hz = h.reshape(B, Z, W, Y * X)
        gin = torch.cat([hz.mean((1, 3)), hz.amax(3).amax(1),
                         c.mean((2, 3)), c.amax((2, 3))], dim=1)
        g = self.glob(gin)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
