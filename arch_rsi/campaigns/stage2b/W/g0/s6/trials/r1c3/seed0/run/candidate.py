"""Per-layer shared 2D conv encoder (layer z sees layers z-1..z+1), pooled over layers, footprint head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

W = 24
H = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        cin = 12 + 3 + spec.Z
        self.conv1 = nn.Conv2d(cin, W, 3, padding=1)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.mix = nn.Conv2d(2 * W, H, 1)
        self.place_head = nn.Conv2d(H, spec.n_types, 2)
        self.glob = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU())
        self.done_head = nn.Linear(H, 1)
        self.value_head = nn.Linear(H, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], 1)
        pad = F.pad(per, (0, 0, 0, 0, 1, 1))
        nb = torch.cat([pad[:, :, 0:Z], pad[:, :, 1:Z + 1], pad[:, :, 2:Z + 2]], 1)
        g = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zid = torch.eye(Z, device=obs.device, dtype=obs.dtype).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([nb, g, zid], 1)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * Z, 15 + Z, Y, X)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.reshape(B, Z, W, Y, X)
        f = torch.cat([h.amax(1), h.mean(1)], 1)
        f = F.relu(self.mix(f))
        place = self.place_head(F.pad(f, (0, 1, 0, 1))).flatten(1)
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
