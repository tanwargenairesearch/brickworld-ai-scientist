import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 24
HEAD = 12
HID = 8
GLOB = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        c_layer = 3 + 3 + 1 + 3 + self.Z
        self.conv1 = nn.Conv2d(c_layer, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, HEAD, 1)
        self.type_hidden = nn.ModuleList([nn.Conv2d(3 * HEAD, HID, 1) for _ in self.footprints])
        self.type_out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.footprints])
        for m in self.type_out:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        rest = obs[:, 4 * Z:4 * Z + 3]
        bp = F.pad(build, (0, 0, 0, 0, 1, 1))
        mp = F.pad(miss, (0, 0, 0, 0, 1, 1))
        bz = torch.stack([bp[:, z:z + 3] for z in range(Z)], dim=1)
        mz = torch.stack([mp[:, z:z + 3] for z in range(Z)], dim=1)
        ez = extra.unsqueeze(2)
        rz = rest.unsqueeze(1).expand(B, Z, 3, Y, X)
        oh = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([bz, mz, ez, rz, oh], dim=2).reshape(B * Z, -1, Y, X)
        f = F.relu(self.conv1(x))
        f = F.relu(self.conv2(f))
        p = F.relu(self.proj(f))
        outs = []
        for i, (dy, dx) in enumerate(self.footprints):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(q, (dy, dx), stride=1)
            av = F.avg_pool2d(q, (dy, dx), stride=1)
            mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(self.type_hidden[i](torch.cat([mx, av, mn], dim=1)))
            outs.append(self.type_out[i](h))
        s = torch.cat(outs, dim=1).view(B, Z, len(self.footprints), Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        fz = f.view(B, Z, WIDTH, Y, X)
        g = self.glob(torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
