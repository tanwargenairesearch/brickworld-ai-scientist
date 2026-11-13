import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 24
PROJ = 12
HID = 16
GLOB = 64
RECIPE = {"ppo_epochs": 3}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        k_in = 12 + 3 + 1
        self.c1 = nn.Conv2d(k_in, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.mix = nn.Conv2d(3 * WIDTH, WIDTH, 1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.type_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PROJ, HID, 1), nn.ReLU(), nn.Conv2d(HID, 2, 1))
            for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        sl = []
        for gi in range(4):
            g = obs[:, gi * Z:(gi + 1) * Z]
            below = F.pad(g, (0, 0, 0, 0, 1, 0))[:, :Z]
            above = F.pad(g, (0, 0, 0, 0, 0, 1))[:, 1:]
            sl += [below, g, above]
        st = torch.stack(sl, 2)
        gch = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([st, gch, zpos], 2).reshape(B * Z, 16, Y, X)
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h5 = h.view(B, Z, WIDTH, Y, X)
        hb = F.pad(h5, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        ha = F.pad(h5, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:]
        m = torch.cat([hb, h5, ha], 2).reshape(B * Z, 3 * WIDTH, Y, X)
        f = F.relu(h + self.mix(m))
        p = self.proj(f)
        outs = []
        for fp, head in zip(self.footprints, self.type_heads):
            dy, dx = fp
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            av = F.avg_pool2d(pp, (dy, dx), stride=1)
            o = head(torch.cat([mx, mn, av], dim=1)).view(B, Z, 2, Y, X)
            score = o[:, :, 0]
            gate = torch.softmax(o[:, :, 1], dim=1)
            outs.append((score * gate).sum(1))
        place = torch.stack(outs, 1).flatten(1)
        fz = f.view(B, Z, WIDTH, Y, X)
        g = self.glob(torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
