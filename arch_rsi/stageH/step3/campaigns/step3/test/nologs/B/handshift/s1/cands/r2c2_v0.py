import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

E = 48
W = 32
H = 32
G = 64
K = 14


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.T = int(spec.n_types)
        self.loc = nn.Conv2d(K, E, 1)
        self.full = nn.Conv2d(spec.in_channels, E, 1, bias=False)
        self.mix = nn.Conv2d(E, W, 1)
        self.res = nn.Conv2d(W, W, 3, padding=1)
        self.hconv = nn.Conv2d(W, H, 3, padding=1)
        self.out = nn.Conv2d(H, self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.done = nn.Sequential(nn.Linear(2 * W + H, G), nn.ReLU(), nn.Linear(G, 1))
        self.value = nn.Sequential(nn.Linear(2 * W + H, G), nn.ReLU(), nn.Linear(G, 1))
        nn.init.normal_(self.done[2].weight, std=0.01)
        nn.init.zeros_(self.done[2].bias)

    def per_layer(self, obs):
        Z = self.Z
        B, C, Y, X = obs.shape
        cur = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        mis = obs[:, 2 * Z:3 * Z]
        ext = obs[:, 3 * Z:4 * Z]
        rest = obs[:, 4 * Z:4 * Z + 3]

        def below(v):
            return F.pad(v, (0, 0, 0, 0, 1, 0))[:, :Z]

        def above(v):
            return F.pad(v, (0, 0, 0, 0, 0, 1))[:, 1:]

        per = torch.stack([cur, tgt, mis, ext, below(cur), below(tgt), below(mis),
                           above(cur), above(tgt), above(mis)], dim=2)
        rest5 = rest.unsqueeze(1).expand(B, Z, 3, Y, X)
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        return torch.cat([per, rest5, zpos], dim=2).reshape(B * Z, K, Y, X)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        f = self.per_layer(obs)
        g = self.full(obs).unsqueeze(1).expand(B, Z, E, Y, X).reshape(B * Z, E, Y, X)
        h = F.relu(self.loc(f) + g)
        h = F.relu(self.mix(h))
        h = F.relu(h + self.res(h))
        u = F.relu(self.hconv(h))
        s = self.out(u).view(B, Z, self.T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        hv = h.view(B, Z, W, Y * X)
        uv = u.view(B, Z, H, Y * X)
        gl = torch.cat([hv.mean((1, 3)), hv.amax((1, 3)), uv.amax((1, 3))], dim=1)
        logits = torch.cat([place, self.done(gl)], dim=1)
        return logits, self.value(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
