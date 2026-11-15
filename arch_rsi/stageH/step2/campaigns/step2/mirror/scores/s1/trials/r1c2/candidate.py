import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "lr": 0.0015, "ent_coef": 0.003}

TRUNK = 48
K = 16
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        self.stem = nn.Conv2d(C, TRUNK, 1)
        self.c1 = nn.Conv2d(TRUNK, TRUNK, 3, padding=1)
        self.layer_in = nn.Conv2d(C, self.Z * K, 1)
        self.ctx = nn.Conv2d(TRUNK, K, 1)
        self.layer_mix = nn.Conv2d(K, K, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * K, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TRUNK + 2 * K, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        s = F.relu(self.stem(obs))
        t = F.relu(s + self.c1(s))
        lz = self.layer_in(obs).view(B, Z, K, Y, X) + self.ctx(t).unsqueeze(1)
        lz = F.relu(lz).reshape(B * Z, K, Y, X)
        lz = F.relu(lz + self.layer_mix(lz))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = lz, lz, lz
            else:
                q = F.pad(lz, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            sc = out(h).view(B, Z, Y, X)
            res.append(torch.logsumexp(sc, dim=1))
        place = torch.stack(res, dim=1).flatten(1)
        l5 = lz.view(B, Z, K, Y, X)
        g = torch.cat([t.mean((2, 3)), t.amax((2, 3)),
                       l5.mean((1, 3, 4)), l5.amax(4).amax(3).amax(1)], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
