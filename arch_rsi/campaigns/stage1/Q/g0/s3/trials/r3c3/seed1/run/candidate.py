import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
C3 = 8


def cln(ln, x):
    return ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.c3 = nn.Conv3d(8, C3, 3, padding=1)
        self.n3 = nn.LayerNorm(C3)
        self.c1 = nn.Conv2d(C3 * self.Z, WIDTH, 3, padding=1)
        self.n1 = nn.LayerNorm(WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.LayerNorm(WIDTH)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.heads = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for dy, dx in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.LayerNorm(WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        b, _, ny, nx = obs.shape
        z = self.Z
        vol = obs[:, :4 * z].reshape(b, 4, z, ny, nx)
        planes = obs[:, 4 * z:4 * z + 3].unsqueeze(2).expand(-1, -1, z, -1, -1)
        zc = torch.arange(z, device=obs.device, dtype=obs.dtype) / max(z - 1, 1)
        zc = zc.view(1, 1, z, 1, 1).expand(b, 1, z, ny, nx)
        v = torch.cat([vol, planes, zc], dim=1)
        h = self.c3(v)
        h = F.relu(self.n3(h.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3))
        h = h.reshape(b, C3 * z, ny, nx)
        f = F.relu(cln(self.n1, self.c1(h)))
        f = F.relu(cln(self.n2, self.c2(f)))
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            outs.append(head(p).flatten(1))
        place = torch.cat(outs, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
