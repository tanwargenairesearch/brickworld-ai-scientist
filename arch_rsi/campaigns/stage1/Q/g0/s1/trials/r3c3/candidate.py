import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 16
CW = 8
GW = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.in3 = nn.Conv3d(5, WIDTH, 3, padding=1)
        self.in2 = nn.Conv2d(3, WIDTH, 1)
        self.c2 = nn.Conv3d(WIDTH, WIDTH, 3, padding=1)
        self.place_head = nn.Conv3d(WIDTH, spec.n_types, 1)
        self.gate_head = nn.Conv3d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.v_in3 = nn.Conv3d(5, CW, 3, padding=1)
        self.v_in2 = nn.Conv2d(3, CW, 1)
        self.v_c2 = nn.Conv3d(CW, CW, 3, padding=1)
        self.v_glob = nn.Sequential(nn.Linear(2 * CW, GW), nn.ReLU())
        self.value_head = nn.Linear(GW, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype)
        zc = zc.reshape(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([vox, zc], dim=1)
        flat = obs[:, 4 * Z:4 * Z + 3]
        h = F.relu(self.in3(x) + self.in2(flat).unsqueeze(2))
        h = F.relu(self.c2(h))
        q = self.place_head(h)
        w = torch.softmax(self.gate_head(h), dim=2)
        place = (w * q).sum(2).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        v = F.relu(self.v_in3(x) + self.v_in2(flat).unsqueeze(2))
        v = F.relu(self.v_c2(v))
        vg = self.v_glob(torch.cat([v.mean((2, 3, 4)), v.amax((2, 3, 4))], dim=1))
        return logits, self.value_head(vg).squeeze(1)


def build_policy(spec):
    return Net(spec)
