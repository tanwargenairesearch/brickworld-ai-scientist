import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
K = 6
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.n_types = spec.n_types
        self.Z = spec.Z
        self.stem = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.conv1 = nn.Conv2d(W, W, 3, padding=1)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.film = nn.Linear(W, W)
        self.proj = nn.Conv2d(W, self.n_types * self.Z * K, 1)
        self.hid = nn.ModuleList([nn.Linear(2 * K, HID) for _ in range(self.n_types)])
        self.zemb = nn.Parameter(torch.zeros(self.n_types, self.Z, HID))
        outs = []
        for _ in range(self.n_types):
            o = nn.Linear(HID, 1)
            nn.init.orthogonal_(o.weight, gain=0.01)
            nn.init.zeros_(o.bias)
            outs.append(o)
        self.out = nn.ModuleList(outs)
        self.done_head = nn.Linear(W, 1)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        h = F.relu(self.stem(obs))
        h = h + F.relu(self.conv1(h))
        h = h + F.relu(self.conv2(h))
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        h2 = F.relu(h + self.film(g)[:, :, None, None])
        p = self.proj(h2)
        outs = []
        for t in range(self.n_types):
            dy, dx = self.fps[t]
            pt = p[:, t * Z * K:(t + 1) * Z * K]
            pp = F.pad(pt, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1).reshape(B, Z, K, Y, X)
            av = F.avg_pool2d(pp, (dy, dx), stride=1).reshape(B, Z, K, Y, X)
            q = torch.cat([mx, av], dim=2).permute(0, 3, 4, 1, 2)
            z = F.relu(self.hid[t](q) + self.zemb[t])
            s = self.out[t](z).squeeze(-1)
            outs.append(torch.logsumexp(s, dim=-1).reshape(B, Y * X))
        place = torch.cat(outs, dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
