import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 64, "rollout_len": 16, "n_minibatches": 4, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
LH = 32
KEY = 16
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fproj = nn.Conv2d(WIDTH, LH, 1)
        self.lin = nn.Conv2d(15 + Z, LH, 1)
        self.l2 = nn.Conv2d(LH, LH, 1)
        self.key = nn.Conv2d(LH, KEY, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * KEY, HID, 1) for _ in self.fps])
        self.gate = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.o1 = nn.ModuleList([nn.Conv2d(2 * HID, HID, 1) for _ in self.fps])
        self.o2 = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + LH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        L = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(L, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        above = F.pad(L, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        ext = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        pos = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([L, below, above, ext, pos], dim=1)
        inp = inp.permute(0, 2, 1, 3, 4).reshape(B * Z, 15 + Z, Y, X)
        fz = self.fproj(f).unsqueeze(1).expand(B, Z, LH, Y, X).reshape(B * Z, LH, Y, X)
        e = F.relu(self.lin(inp) + fz)
        e = F.relu(e + self.l2(e))
        k = self.key(e)
        res = []
        for (dy, dx), mix, gate, o1, o2 in zip(self.fps, self.mix, self.gate, self.o1, self.o2):
            if dy * dx == 1:
                mx, mn, av = k, k, k
            else:
                q = F.pad(k, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            w = torch.softmax(gate(h).view(B, Z, 1, Y, X), dim=1)
            hv = h.view(B, Z, HID, Y, X)
            agg = torch.cat([(w * hv).sum(1), hv.amax(1)], dim=1)
            res.append(o2(F.relu(o1(agg))))
        place = torch.cat(res, dim=1).flatten(1)
        eg = e.view(B, Z, LH, Y, X).amax(dim=(1, 3, 4))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), eg], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
