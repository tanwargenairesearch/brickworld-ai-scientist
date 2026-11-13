import torch
import torch.nn as nn

WIDTH = 64
C3 = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.conv3 = nn.Conv3d(8, C3, 3, padding=1)
        self.mix = nn.Conv2d(C3 * self.Z, WIDTH, 1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        vol = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        planes = obs[:, 4 * Z:4 * Z + 3].reshape(B, 3, 1, Y, X).expand(B, 3, Z, Y, X)
        zc = torch.arange(Z, dtype=obs.dtype, device=obs.device) / max(Z - 1, 1)
        zc = zc.reshape(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        v = torch.cat([vol, planes, zc], dim=1)
        h = torch.relu(self.conv3(v))
        h = h.reshape(B, C3 * Z, Y, X)
        f = torch.relu(self.mix(h))
        f = torch.relu(self.conv2(f))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
