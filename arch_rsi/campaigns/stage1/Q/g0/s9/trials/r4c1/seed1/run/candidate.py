import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
N_RES = 3
C3 = 12


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.kh = max(int(fp[0]) for fp in spec.footprints)
        self.kw = max(int(fp[1]) for fp in spec.footprints)
        self.conv3 = nn.Conv3d(8, C3, 3, padding=1)
        self.stem = nn.Conv2d(C3 * self.Z + spec.in_channels, WIDTH, 1)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_RES)])
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, (self.kh, self.kw))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, C, Y, X = obs.shape
        vox = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=1)
        scal = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        v = torch.cat([vox, scal, zc], dim=1)
        h3 = F.relu(self.conv3(v)).reshape(B, C3 * Z, Y, X)
        f = F.relu(self.stem(torch.cat([h3, obs], dim=1)))
        for conv in self.res:
            f = F.relu(f + conv(f))
        fp = F.pad(f, (0, self.kw - 1, 0, self.kh - 1))
        place = self.place_head(fp).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
