import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.99, "gae_lambda": 0.95}

WIDTH = 16
K = 8
HID = 16
GW = 64
CIN = 12


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.n_types = spec.n_types
        self.Z = spec.Z
        self.stem = nn.Conv2d(CIN, W, 3, padding=1)
        self.conv1 = nn.Conv2d(W, W, 3, padding=1)
        self.mix = nn.Conv2d(3 * W, W, 1)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(self.Z * W + W, GW), nn.ReLU())
        self.film = nn.Linear(GW, W)
        self.proj = nn.Conv2d(W, self.n_types * K, 1)
        heads = []
        for _ in range(self.n_types):
            out = nn.Conv2d(HID, 1, 1)
            nn.init.orthogonal_(out.weight, gain=0.01)
            nn.init.zeros_(out.bias)
            heads.append(nn.Sequential(nn.Conv2d(2 * K, HID, 1), nn.ReLU(), out))
        self.type_heads = nn.ModuleList(heads)
        self.done_head = nn.Linear(GW, 1)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(GW, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        W = WIDTH
        bld = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        mis = obs[:, 2 * Z:3 * Z]
        ext = obs[:, 3 * Z:4 * Z]
        bb = F.pad(bld, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(tgt, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        ba = F.pad(bld, (0, 0, 0, 0, 0, 1))[:, 1:]
        ta = F.pad(tgt, (0, 0, 0, 0, 0, 1))[:, 1:]
        hb = obs[:, 4 * Z:4 * Z + 1].expand(B, Z, Y, X)
        ht = obs[:, 4 * Z + 1:4 * Z + 2].expand(B, Z, Y, X)
        tm = obs[:, 4 * Z + 2:4 * Z + 3].expand(B, Z, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1).expand(B, Z, Y, X)
        x = torch.stack([bld, tgt, mis, ext, bb, tb, ba, ta, hb, ht, tm, zc], dim=2)
        x = x.reshape(B * Z, CIN, Y, X)
        h = F.relu(self.stem(x))
        h = h + F.relu(self.conv1(h))
        hz = h.reshape(B, Z, W, Y, X)
        below = torch.cummax(hz, dim=1).values
        above = torch.flip(torch.cummax(torch.flip(hz, [1]), dim=1).values, [1])
        m = torch.cat([hz, below, above], dim=2).reshape(B * Z, 3 * W, Y, X)
        h = h + F.relu(self.mix(m))
        h = h + F.relu(self.conv2(h))
        hz = h.reshape(B, Z, W, Y, X)
        pooled = torch.cat([hz.mean((3, 4)).reshape(B, Z * W), hz.amax((1, 3, 4))], dim=1)
        g = self.glob(pooled)
        h2 = F.relu(hz + self.film(g)[:, None, :, None, None]).reshape(B * Z, W, Y, X)
        p = self.proj(h2)
        outs = []
        for t in range(self.n_types):
            dy, dx = self.fps[t]
            pt = p[:, t * K:(t + 1) * K]
            pp = F.pad(pt, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            av = F.avg_pool2d(pp, (dy, dx), stride=1)
            s = self.type_heads[t](torch.cat([mx, av], dim=1)).reshape(B, Z, Y * X)
            outs.append(torch.logsumexp(s, dim=1))
        place = torch.cat(outs, dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
