import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 64, "rollout_len": 16, "n_minibatches": 4, "lr": 0.001, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

S1 = 48
W = 32
PROJ = 16
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 8 + spec.in_channels + self.Z
        self.stem = nn.Sequential(
            nn.Conv2d(cin, S1, 1), nn.ReLU(),
            nn.Conv2d(S1, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        bld = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        mis = obs[:, 2 * Z:3 * Z]
        ext = obs[:, 3 * Z:4 * Z]
        cur = torch.stack([bld, tgt, mis, ext], dim=2)
        bb = F.pad(bld, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(tgt, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        mb = F.pad(mis, (0, 0, 0, 0, 1, 0), value=0.0)[:, :Z]
        eb = F.pad(ext, (0, 0, 0, 0, 1, 0), value=0.0)[:, :Z]
        below = torch.stack([bb, tb, mb, eb], dim=2)
        full = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        zoh = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([cur, below, full, zoh], dim=2).reshape(B * Z, -1, Y, X)
        s = self.stem(inp)
        f = F.relu(s + self.c1(s))
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
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        T = len(self.fps)
        sc = torch.cat(res, dim=1).view(B, Z, T, Y, X)
        place = torch.logsumexp(sc, dim=1).flatten(1)
        fz = f.view(B, Z, W, Y, X)
        g = self.glob(torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
