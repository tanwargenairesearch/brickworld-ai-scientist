import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.0015, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

W = 24
HH = 32
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.T = int(spec.n_types)
        cin = 4 * 3 + 3 + 1
        self.cin = cin
        self.stem = nn.Sequential(nn.Conv2d(cin, W, 1), nn.ReLU(), nn.Conv2d(W, W, 1), nn.ReLU())
        self.conv = nn.Conv2d(W, W, 3, padding=1)
        self.h1 = nn.Conv2d(W, HH, 3, padding=1)
        self.h2 = nn.Conv2d(HH, self.T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def windows(self, p, below):
        Z = self.Z
        q = F.pad(p, (0, 0, 0, 0, 1, 0), value=below)
        q = F.pad(q, (0, 0, 0, 0, 0, 1), value=0.0)
        return torch.stack([q[:, 0:Z], q[:, 1:Z + 1], q[:, 2:Z + 2]], dim=2)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        parts = [self.windows(build, 1.0), self.windows(target, 1.0),
                 self.windows(miss, 0.0), self.windows(extra, 0.0)]
        gch = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        h = torch.cat(parts + [gch, zc], dim=2).reshape(B * Z, self.cin, Y, X)
        s = self.stem(h)
        f = F.relu(s + self.conv(s))
        sc = self.h2(F.relu(self.h1(f)))
        sc = sc.reshape(B, Z, self.T, Y, X)
        place = torch.logsumexp(sc, dim=1).flatten(1)
        fz = f.reshape(B, Z, W, Y, X)
        g = self.glob(torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
