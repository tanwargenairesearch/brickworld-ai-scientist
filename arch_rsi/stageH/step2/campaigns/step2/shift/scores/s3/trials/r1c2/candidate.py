import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.003}

LC = 24
PROJ = 16
HID = 24
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Conv2d(spec.in_channels, LC, 1)
        self.lay = nn.Conv2d(12 + self.Z, LC, 1)
        self.register_buffer("onehot", torch.eye(self.Z).reshape(1, self.Z, self.Z, 1, 1))
        self.c1 = nn.Conv2d(LC, LC, 3, padding=1)
        self.c2 = nn.Conv2d(LC, LC, 1)
        self.proj = nn.Conv2d(LC, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * LC, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        g4 = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        p = F.pad(g4, (0, 0, 0, 0, 0, 0, 1, 1))
        nb = torch.cat([p[:, :-2], p[:, 1:-1], p[:, 2:]], dim=2)
        oh = self.onehot.expand(B, Z, Z, Y, X)
        x = torch.cat([nb, oh], dim=2).reshape(B * Z, 12 + Z, Y, X)
        h = self.lay(x).reshape(B, Z, LC, Y, X) + self.col(obs).unsqueeze(1)
        h = F.relu(h).reshape(B * Z, LC, Y, X)
        h = h + F.relu(self.c1(h))
        h = h + F.relu(self.c2(h))
        pr = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = pr, pr, pr
            else:
                q = F.pad(pr, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            u = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(u))
        T = len(res)
        o = torch.stack(res, dim=1).reshape(B, Z, T, 2, Y, X)
        score = o[:, :, :, 0]
        gate = torch.softmax(o[:, :, :, 1], dim=1)
        place = (gate * score).sum(1).flatten(1)
        hh = h.reshape(B, Z, LC, Y, X)
        g = self.glob(torch.cat([hh.mean((1, 3, 4)), hh.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
