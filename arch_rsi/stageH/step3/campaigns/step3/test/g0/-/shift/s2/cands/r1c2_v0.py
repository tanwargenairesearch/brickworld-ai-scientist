import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.003}

W = 32
PROJ = 16
HID = 24
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 4 * 3 + 3 + 1
        self.cin = cin
        self.stem = nn.Sequential(nn.Conv2d(cin, W, 1), nn.ReLU(), nn.Conv2d(W, W, 1), nn.ReLU())
        self.conv = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def windows(self, p, below):
        Z = self.Z
        q = F.pad(p, (0, 0, 0, 0, 1, 0), value=below)
        q = F.pad(q, (0, 0, 0, 0, 0, 1), value=0.0)
        return torch.stack([q[:, 0:Z], q[:, 1:Z + 1], q[:, 2:Z + 2]], dim=2)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        parts = [self.windows(build, 1.0), self.windows(target, 1.0),
                 self.windows(miss, 0.0), self.windows(extra, 0.0)]
        gch = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        h = torch.cat(parts + [gch, zc], dim=2).reshape(B * Z, self.cin, Y, X)
        s = self.stem(h)
        f = F.relu(s + self.conv(s))
        p = self.proj(f)
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
        sc = torch.cat(res, dim=1)
        T = sc.shape[1]
        sc = sc.reshape(B, Z, T, Y, X)
        place = torch.logsumexp(sc, dim=1).flatten(1)
        fz = f.reshape(B, Z, W, Y, X)
        g = self.glob(torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
