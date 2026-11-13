import torch
import torch.nn as nn
import torch.nn.functional as F

C3 = 16
WIDTH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        self.c3a = nn.Conv3d(8, C3, 3, padding=1)
        self.c3b = nn.Conv3d(C3, C3, 3, padding=1)
        self.fuse = nn.Conv2d(C3 * Z, WIDTH, 1)
        if Z > 1:
            zc = torch.linspace(0.0, 1.0, Z)
        else:
            zc = torch.zeros(1)
        self.register_buffer('zpos', zc.view(1, 1, Z, 1, 1))
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        planes = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zp = self.zpos.expand(B, 1, Z, Y, X)
        x = torch.cat([vox, planes, zp], dim=1)
        h = F.relu(self.c3a(x))
        h = F.relu(self.c3b(h))
        f = F.relu(self.fuse(h.reshape(B, C3 * Z, Y, X)))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
