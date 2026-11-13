import torch
import torch.nn as nn
import torch.nn.functional as F

W3 = 12
WIDTH = 48
HEAD = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.c3a = nn.Conv3d(8, W3, 3, padding=1)
        self.c3b = nn.Conv3d(W3, W3, 3, padding=1)
        self.c2 = nn.Conv2d(W3 * spec.Z, WIDTH, 3, padding=1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.fp_convs = nn.ModuleList([nn.Conv2d(WIDTH, HEAD, (dy, dx)) for dy, dx in self.fps])
        self.fp_out = nn.ModuleList([nn.Conv2d(HEAD, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        vol = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).reshape(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        v = torch.cat([vol, gl, zc], dim=1)
        v = F.relu(self.c3a(v))
        v = F.relu(self.c3b(v))
        f = F.relu(self.c2(v.reshape(B, W3 * Z, Y, X)))
        outs = []
        for (dy, dx), conv, out in zip(self.fps, self.fp_convs, self.fp_out):
            h = F.relu(conv(F.pad(f, (0, dx - 1, 0, dy - 1))))
            outs.append(out(h))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
