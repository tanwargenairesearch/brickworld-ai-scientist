import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.T = spec.n_types
        c_in = 12 + 3 + 1
        self.conv1 = nn.Conv2d(c_in, WIDTH, 3, padding=1)
        self.layer_emb = nn.Parameter(torch.zeros(spec.Z, WIDTH))
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(4 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        W = WIDTH
        L = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        Lp = F.pad(L, (0, 0, 0, 0, 1, 1))
        x = torch.cat([Lp[:, :, 0:Z], Lp[:, :, 1:Z + 1], Lp[:, :, 2:Z + 2]], dim=1)
        g = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype)
        zc = zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([x, g, zc], dim=1)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * Z, 16, Y, X)
        h = self.conv1(x).view(B, Z, W, Y, X) + self.layer_emb.view(1, Z, W, 1, 1)
        h = torch.relu(h).reshape(B * Z, W, Y, X)
        h = torch.relu(self.conv2(h))
        s = self.place_head(h).view(B, Z, self.T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        hz = h.view(B, Z, W, Y, X)
        f = torch.cat([hz.amax(1), hz.mean(1)], dim=1)
        q = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(q)], dim=1)
        return logits, self.value_head(q).squeeze(1)


def build_policy(spec):
    return Net(spec)
