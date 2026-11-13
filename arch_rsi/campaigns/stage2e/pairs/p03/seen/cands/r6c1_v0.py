import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.98, "gae_lambda": 0.95}

W = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.T = spec.n_types
        cin = 12
        self.conv1 = nn.Conv2d(cin, W, 3, padding=1)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.mix = nn.Conv2d(W, W, 1)
        self.place_head = nn.Conv2d(W, 2 * spec.n_types, 2)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        T = self.T
        B, C, Y, X = obs.shape
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        hb = obs[:, 4 * Z:4 * Z + 1]
        ht = obs[:, 4 * Z + 1:4 * Z + 2]
        tt = obs[:, 4 * Z + 2:4 * Z + 3]
        bb = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(target, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        ba = F.pad(build, (0, 0, 0, 0, 0, 1))[:, 1:]
        ta = F.pad(target, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([build, target, miss, extra, bb, tb, ba, ta], dim=2)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        sh = torch.stack([hb, ht, tt], dim=2).reshape(B, 1, 3, Y, X).expand(B, Z, 3, Y, X)
        x = torch.cat([per, sh, zc], dim=2).reshape(B * Z, 12, Y, X)
        h1 = F.relu(self.conv1(x))
        zm = h1.view(B, Z, W, Y, X).amax(1)
        mb = self.mix(zm).unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h2 = F.relu(self.conv2(h1) + mb)
        out = self.place_head(F.pad(h2, (0, 1, 0, 1))).view(B, Z, 2 * T, Y, X)
        q = out[:, :, :T]
        gate = F.log_softmax(out[:, :, T:], dim=1)
        place = torch.logsumexp(q + gate, dim=1).flatten(1)
        h5 = h2.view(B, Z, W, Y, X)
        g = self.glob(torch.cat([h5.mean((1, 3, 4)), h5.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
