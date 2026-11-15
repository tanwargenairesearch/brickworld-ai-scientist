import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2}

LH = 32
COL = 48
WIDTH = 40
PROJ = 32
HID = 32


class LayerEncoder(nn.Module):
    def __init__(self, Z, h, out):
        super().__init__()
        self.Z = Z
        self.h = h
        self.lin = nn.Conv2d(4 + Z, h, 1)
        self.gin = nn.Conv2d(3, h, 1)
        self.l2 = nn.Conv2d(3 * h, h, 1)
        self.o = nn.Conv2d(2 * h, out, 1)

    def forward(self, obs):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        pos = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([per, pos], dim=2).reshape(B * Z, 4 + Z, Y, X)
        a = self.lin(inp).view(B, Z, self.h, Y, X)
        gq = self.gin(obs[:, 4 * Z:4 * Z + 3]).unsqueeze(1)
        h = F.relu(a + gq)
        up = torch.cummax(h, dim=1)[0]
        dn = torch.flip(torch.cummax(torch.flip(h, [1]), dim=1)[0], [1])
        h2 = torch.cat([h, up, dn], dim=2).reshape(B * Z, 3 * self.h, Y, X)
        h2 = F.relu(self.l2(h2)).view(B, Z, self.h, Y, X)
        col = torch.cat([h2.amax(1), h2.mean(1)], dim=1)
        return F.relu(self.o(col))


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
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
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.enc = LayerEncoder(int(spec.Z), LH, COL)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.enc(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
