import torch
import torch.nn as nn
import torch.nn.functional as F

C3 = 16
WIDTH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.conv3 = nn.Conv3d(5, C3, 3, padding=1)
        self.conv_g = nn.Conv2d(3, C3, 3, padding=1)
        self.conv2 = nn.Conv2d(C3 * spec.Z, WIDTH, 3, padding=1)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, _, Y, X = obs.shape
        vox = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=1)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        vox = torch.cat([vox, zc], dim=1)
        h3 = self.conv3(vox)
        g2 = self.conv_g(obs[:, 4 * Z:4 * Z + 3])
        h3 = F.relu(h3 + g2.unsqueeze(2))
        h = h3.reshape(B, -1, Y, X)
        f = F.relu(self.conv2(h))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
