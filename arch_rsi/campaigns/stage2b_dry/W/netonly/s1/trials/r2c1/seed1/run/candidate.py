import torch
import torch.nn as nn
import torch.nn.functional as F

W = 24
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.T = spec.n_types
        cin = 16
        self.stem = nn.Conv2d(cin, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 1)
        self.head = nn.Conv2d(W, spec.n_types, 2)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)
        nn.init.orthogonal_(self.head.weight, 0.01)
        nn.init.zeros_(self.head.bias)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        per = torch.stack([build, target, miss, extra], 2)  # B,Z,4,Y,X
        zero = torch.zeros_like(per[:, :1])
        below = torch.cat([zero, per[:, :-1]], 1)
        above = torch.cat([per[:, 1:], zero], 1)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, below, above, gl, zc], 2).reshape(B * Z, 16, Y, X)
        h = F.relu(self.stem(x))
        h = F.relu(self.c1(h)) + h
        h = F.relu(self.c2(h)) + h
        p = self.head(F.pad(h, (0, 1, 0, 1)))
        p = p.view(B, Z, self.T, Y, X).logsumexp(1).flatten(1)
        hz = h.view(B, Z, W, Y, X)
        pooled = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], 1)
        g = self.glob(pooled)
        logits = torch.cat([p, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
