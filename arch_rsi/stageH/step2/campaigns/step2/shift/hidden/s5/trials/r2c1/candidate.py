import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16}

STEM = 64
COL = 48
WIDTH = 40
LD = 16
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.cproj = nn.Conv2d(WIDTH, LD, 1)
        self.lay = nn.Conv2d(5, LD, 1)
        self.lay2 = nn.Conv2d(LD, LD, 1)
        self.ground = nn.Parameter(torch.zeros(LD))
        self.mix = nn.ModuleList([nn.Conv2d(4 * LD, HID, 1) for _ in self.fps])
        self.gate = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.score = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        col = self.cproj(f)
        layers = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        inp = torch.cat([layers, zpos], 2).reshape(B * Z, 5, Y, X)
        e = F.relu(self.lay(inp).view(B, Z, LD, Y, X) + col.unsqueeze(1))
        e = F.relu(e + self.lay2(e.reshape(B * Z, LD, Y, X)).view(B, Z, LD, Y, X))
        gr = self.ground.view(1, 1, LD, 1, 1).expand(B, 1, LD, Y, X)
        below = torch.cat([gr, e[:, :-1]], 1)
        h = torch.cat([e, below], 2).reshape(B * Z, 2 * LD, Y, X)
        res = []
        for (dy, dx), mix, gt, sc in zip(self.fps, self.mix, self.gate, self.score):
            if dy * dx == 1:
                mx, mn = h, h
            else:
                q = F.pad(h, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            u = F.relu(mix(torch.cat([mx, mn], dim=1)))
            g = gt(u).view(B, Z, Y * X)
            v = sc(u).view(B, Z, Y * X)
            w = torch.softmax(g, dim=1)
            res.append((w * v).sum(1))
        place = torch.cat(res, dim=1)
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
