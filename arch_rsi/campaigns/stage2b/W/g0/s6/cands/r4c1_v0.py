"""r2c3 residual+broadcast net with the legal-placement mask fed as input channels to the stem."""
import torch
import torch.nn as nn
import torch.nn.functional as F

W = 34


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, h):
        return F.relu(h + self.c2(F.relu(self.c1(h))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = spec.n_types
        self.npl = spec.n_place
        self.stem = nn.Conv2d(spec.in_channels + spec.n_types, W, 3, padding=1)
        self.block1 = Block(W)
        self.bcast = nn.Linear(2 * W, W)
        self.block2 = Block(W)
        self.place_head = nn.Conv2d(W, spec.n_types, 2)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        m = mask[:, :self.npl].to(obs.dtype).reshape(B, self.nt, Y, X)
        h = F.relu(self.stem(torch.cat([obs, m], 1)))
        h = self.block1(h)
        g0 = self.bcast(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        h = F.relu(h + g0[:, :, None, None])
        h = self.block2(h)
        place = self.place_head(F.pad(h, (0, 1, 0, 1))).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
