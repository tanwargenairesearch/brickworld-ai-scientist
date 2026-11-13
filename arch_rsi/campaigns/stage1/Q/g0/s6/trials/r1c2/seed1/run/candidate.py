import torch
import torch.nn as nn
import torch.nn.functional as F

VW = 16
WIDTH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = spec.Z
        self.Z = Z
        self.c1 = nn.Conv3d(5, VW, 3, padding=1)
        self.g1 = nn.Conv2d(3, VW, 1)
        self.c2 = nn.Conv3d(VW, VW, 3, padding=1)
        self.flat = nn.Conv2d(VW * Z, WIDTH, 1)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)
        if Z > 1:
            zc = torch.linspace(0.0, 1.0, Z)
        else:
            zc = torch.zeros(1)
        self.register_buffer('zc', zc.view(1, 1, Z, 1, 1))

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        zc = self.zc.expand(B, 1, Z, Y, X)
        vox = torch.cat([vox, zc], dim=1)
        h = self.c1(vox) + self.g1(obs[:, 4 * Z:]).unsqueeze(2)
        h = F.relu(h)
        h = F.relu(self.c2(h))
        h = h.reshape(B, VW * Z, Y, X)
        f = F.relu(self.flat(h))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
