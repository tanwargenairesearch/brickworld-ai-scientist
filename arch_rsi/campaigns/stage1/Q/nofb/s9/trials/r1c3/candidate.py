import torch
import torch.nn as nn
import torch.nn.functional as F

C3 = 16
WIDTH = 64
HID = 32


class FootprintHead(nn.Module):
    def __init__(self, c, footprints, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c, hid, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.convs, self.outs):
            h = F.relu(conv(F.pad(f, (0, dx - 1, 0, dy - 1))))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        zc = torch.arange(self.Z, dtype=torch.float32) / max(self.Z - 1, 1)
        self.register_buffer("zc", zc.view(1, 1, self.Z, 1, 1))
        self.c3a = nn.Conv3d(8, C3, 3, padding=1)
        self.c3b = nn.Conv3d(C3, C3, 3, padding=1)
        self.fold = nn.Conv2d(C3 * self.Z, WIDTH, 1)
        self.mid_glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH))
        self.head = FootprintHead(WIDTH, spec.footprints, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        planes = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = self.zc.expand(B, 1, Z, Y, X)
        v = torch.cat([vox, planes, zc], dim=1)
        h = F.relu(self.c3a(v))
        h = F.relu(self.c3b(h))
        f = F.relu(self.fold(h.reshape(B, C3 * Z, Y, X)))
        gm = self.mid_glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = F.relu(f + gm[:, :, None, None])
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
