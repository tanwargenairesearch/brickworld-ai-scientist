import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.9}

TR = 32
LH = 32
P = 12
HID = 32
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.trunk1 = nn.Conv2d(spec.in_channels, TR, 1)
        self.trunk2 = nn.Conv2d(TR, TR, 3, padding=1)
        self.l1 = nn.Conv2d(7 + Z + TR, LH, 1)
        self.l2 = nn.Conv2d(LH, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P * Z, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * TR + 2 * P * Z, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)
        self.register_buffer("lpos", torch.eye(Z))

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        t = F.relu(self.trunk1(obs))
        t = F.relu(t + self.trunk2(t))
        per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=2)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        pos = self.lpos.view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        tt = t.unsqueeze(1).expand(B, Z, TR, Y, X)
        h = torch.cat([per, gl, pos, tt], dim=2).reshape(B * Z, 7 + Z + TR, Y, X)
        h = F.relu(self.l1(h))
        q = self.l2(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = q, q, q
            else:
                qp = F.pad(q, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(qp, (dy, dx), stride=1)
                mn = -F.max_pool2d(-qp, (dy, dx), stride=1)
                av = F.avg_pool2d(qp, (dy, dx), stride=1)
            u = torch.cat([mx, mn, av], dim=1).reshape(B, Z * 3 * P, Y, X)
            res.append(out(F.relu(mix(u))))
        place = torch.cat(res, dim=1).flatten(1)
        ql = q.reshape(B, Z * P, Y, X)
        g = torch.cat([t.mean((2, 3)), t.amax((2, 3)), ql.mean((2, 3)), ql.amax((2, 3))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
