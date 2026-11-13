import torch
import torch.nn as nn
import torch.nn.functional as F

C3 = 16
WIDTH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.c1 = nn.Conv3d(8, C3, 3, padding=1)
        self.c2 = nn.Conv3d(C3, C3, 3, padding=1)
        self.mix = nn.Conv2d(C3 * spec.Z, WIDTH, 1)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, C, Y, X = obs.shape
        vox = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=1)
        bc = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        v = torch.cat([vox, bc, zc], dim=1)
        h = F.relu(self.c1(v))
        h = F.relu(self.c2(h))
        f = F.relu(self.mix(h.reshape(B, C3 * Z, Y, X)))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
