import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95}

F1 = 24
F2 = 24
PROJ = 16
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Z = self.Z
        cin = 8 * Z + 3 + self.T + Z
        self.l1 = nn.Conv2d(cin, F1, 1)
        self.l2 = nn.Conv2d(F1, F2, 3, padding=1)
        self.l3 = nn.Conv2d(F2, F2, 1)
        self.proj = nn.Conv2d(F2, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * F2, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        self.register_buffer("zoh", torch.eye(Z).view(1, Z, Z, 1, 1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = self.Y
        X = self.X
        Z = self.Z
        T = self.T
        wins = []
        for gi in range(4):
            g = obs[:, gi * Z:(gi + 1) * Z]
            v = 1.0 if gi < 2 else 0.0
            gp = F.pad(g, (0, 0, 0, 0, Z, Z - 1), value=v)
            wins.append(torch.stack([gp[:, z:z + 2 * Z] for z in range(Z)], dim=1))
        m = mask[:, :T * Y * X].float().reshape(B, 1, T, Y, X).expand(B, Z, T, Y, X)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zo = self.zoh.expand(B, Z, Z, Y, X)
        h = torch.cat(wins + [gl, m, zo], dim=2).reshape(B * Z, -1, Y, X)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = F.relu(h + self.l3(h))
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
        s = torch.cat(res, dim=1).reshape(B, Z, T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        hb = h.reshape(B, Z, F2, Y, X)
        g = self.glob(torch.cat([hb.mean((1, 3, 4)), hb.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
