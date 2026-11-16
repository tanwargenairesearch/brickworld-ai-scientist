import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "lr": 0.002, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.005}

D = 24
WIDTH = 32
P = 16
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        vin = 15 + self.Z
        self.l1 = nn.Linear(vin, D)
        self.l2 = nn.Linear(3 * D, D)
        self.c1 = nn.Conv2d(2 * D + spec.in_channels, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.eh = nn.Linear(D, P)
        self.ef = nn.Linear(WIDTH, P)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        v = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(v, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        above = F.pad(v, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        g = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zpos = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        vin = torch.cat([v, below, above, g, zpos], dim=1).permute(0, 2, 3, 4, 1)
        h1 = F.relu(self.l1(vin))
        up = torch.cummax(h1, dim=1).values
        dn = torch.flip(torch.cummax(torch.flip(h1, [1]), dim=1).values, [1])
        h = F.relu(self.l2(torch.cat([h1, up, dn], dim=-1)))
        col = torch.cat([h.amax(1), h.mean(1)], dim=-1).permute(0, 3, 1, 2)
        x = F.relu(self.c1(torch.cat([col, obs], dim=1)))
        f = F.relu(x + self.c2(x))
        e = F.relu(self.eh(h) + self.ef(f.permute(0, 2, 3, 1)).unsqueeze(1))
        e = e.permute(0, 1, 4, 2, 3).reshape(B * Z, P, Y, X)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = e, e, e
            else:
                q = F.pad(e, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            s = out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))).reshape(B, Z, Y * X)
            res.append(torch.logsumexp(s, dim=1))
        place = torch.cat(res, dim=1)
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), h.amax((1, 2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
