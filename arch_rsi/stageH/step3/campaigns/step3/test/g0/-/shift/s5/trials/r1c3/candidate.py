import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2}

TR = 32
LH = 16
MH = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, 48, 1), nn.ReLU(),
            nn.Conv2d(48, TR, 1), nn.ReLU())
        self.c1 = nn.Conv2d(TR, TR, 3, padding=1)
        self.c2 = nn.Conv2d(TR, TR, 3, padding=1)
        lin = 12 + self.Z
        self.lay_in = nn.Conv2d(lin, LH, 1)
        self.ctx = nn.Conv2d(TR, LH, 1)
        self.lay2 = nn.Conv2d(LH, LH, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * LH, MH, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(MH, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TR, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        self.register_buffer("onehot", torch.eye(self.Z).view(1, self.Z, self.Z, 1, 1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))

        L = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        cur = L.permute(0, 2, 1, 3, 4)
        L3 = L[:, :3]
        zero = torch.zeros_like(L3[:, :, :1])
        below = torch.cat([zero, L3[:, :, :-1]], dim=2).permute(0, 2, 1, 3, 4)
        above = torch.cat([L3[:, :, 1:], zero], dim=2).permute(0, 2, 1, 3, 4)
        hts = obs[:, 4 * Z:4 * Z + 2].unsqueeze(1).expand(B, Z, 2, Y, X)
        oh = self.onehot.expand(B, Z, Z, Y, X)
        li = torch.cat([cur, below, above, hts, oh], dim=2).reshape(B * Z, 12 + Z, Y, X)

        c = self.ctx(f).unsqueeze(1).expand(B, Z, LH, Y, X).reshape(B * Z, LH, Y, X)
        h = F.relu(self.lay_in(li) + c)
        h = F.relu(h + self.lay2(h))

        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = h, h, h
            else:
                q = F.pad(h, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            u = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(u))
        sc = torch.cat(res, dim=1).reshape(B, Z, len(self.fps) * Y * X)
        place = torch.logsumexp(sc, dim=1)

        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
