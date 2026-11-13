import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
C3 = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.register_buffer("zc", torch.linspace(0.0, 1.0, Z).view(1, 1, Z, 1, 1))
        self.conv3 = nn.Conv3d(8, C3, 3, padding=1)
        self.mix = nn.Conv2d(C3 * Z, WIDTH, 1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.place_heads = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for (dy, dx) in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        vol = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        per = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = self.zc.to(obs.dtype).expand(B, 1, Z, Y, X)
        v = torch.cat([vol, per, zc], dim=1)
        h = F.relu(self.conv3(v))
        h = h.reshape(B, C3 * Z, Y, X)
        f = F.relu(self.mix(h))
        f = F.relu(self.conv2(f))
        outs = []
        for (dy, dx), head in zip(self.fps, self.place_heads):
            fp = F.pad(f, (0, dx - 1, 0, dy - 1))
            outs.append(head(fp))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
