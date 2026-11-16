import torch
import torch.nn as nn
import torch.nn.functional as F

W = 32
PROJ = 16
HID = 16
GLOB = 64

RECIPE = {"lr": 0.001, "n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 13
        self.stem1 = nn.Conv2d(cin, W, 1)
        self.stem2 = nn.Conv2d(W, W, 1)
        self.cross = nn.Conv2d(W, W, 1)
        self.conv = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ + W, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        cur = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        mis = obs[:, 2 * Z:3 * Z]
        ext = obs[:, 3 * Z:4 * Z]

        def below(t):
            return F.pad(t, (0, 0, 0, 0, 1, 0))[:, :Z]

        def above(t):
            return F.pad(t, (0, 0, 0, 0, 0, 1))[:, 1:]

        per = torch.stack([cur, tgt, mis, ext, below(cur), below(tgt), below(mis), above(cur), above(tgt)], dim=2)
        g = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        lz = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, g, lz], dim=2).reshape(B * Z, 13, Y, X)
        h = F.relu(self.stem1(x))
        h = self.stem2(h)
        c = h.view(B, Z, W, Y, X).amax(1)
        h = h + self.cross(c).unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h = h + self.conv(F.relu(h))
        f = F.relu(h)
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
            hh = F.relu(mix(torch.cat([mx, mn, av, f], dim=1)))
            res.append(out(hh))
        s = torch.cat(res, dim=1)
        T = s.shape[1]
        s = s.view(B, Z, T, Y, X).logsumexp(1)
        place = s.flatten(1)
        fz = f.view(B, Z, W, Y, X)
        pooled = torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1)
        gg = self.glob(pooled)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
