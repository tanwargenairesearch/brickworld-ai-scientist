import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 20
PROJ = 12
HID = 12
GLOB = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.register_buffer('eye', torch.eye(Z))
        c_layer = 6 + 3 + Z
        self.conv1 = nn.Conv2d(c_layer, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.conv3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=2, dilation=2)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PROJ, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * Z * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        rest = obs[:, 4 * Z:4 * Z + 3]
        below = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        above = F.pad(target, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([below, build, target, above, miss, extra], dim=2)
        rest_e = rest.unsqueeze(1).expand(B, Z, 3, Y, X)
        lay = self.eye.to(obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([per, rest_e, lay], dim=2).reshape(B * Z, 9 + Z, Y, X)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h + F.relu(self.conv3(h))
        p = self.proj(h)
        outs = []
        for (dy, dx), head in zip(self.footprints, self.heads):
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            av = F.avg_pool2d(pp, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn, av], dim=1)))
        s = torch.cat(outs, dim=1).view(B, Z, len(self.footprints), Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        hm = h.view(B, Z, WIDTH, Y, X)
        gin = torch.cat([hm.mean((3, 4)).flatten(1), hm.amax((3, 4)).flatten(1)], dim=1)
        g = self.glob(gin)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
