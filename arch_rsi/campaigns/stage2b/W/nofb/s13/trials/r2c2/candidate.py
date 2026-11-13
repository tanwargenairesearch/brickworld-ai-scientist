import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2}

W = 32
P = 16
H = 32
GW = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        c_in = 11 + Z
        self.conv1 = nn.Conv2d(c_in, W, 3, padding=1)
        self.mix = nn.Conv2d(3 * W, W, 1)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, P, 1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        heads = []
        for _ in self.fps:
            h = nn.Sequential(nn.Conv2d(3 * P, H, 1), nn.ReLU(), nn.Conv2d(H, 2, 1))
            nn.init.normal_(h[2].weight, std=0.01)
            nn.init.zeros_(h[2].bias)
            heads.append(h)
        self.heads = nn.ModuleList(heads)
        self.glob = nn.Sequential(nn.Linear(2 * W, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(GW, 1)
        self.register_buffer("zid", torch.eye(Z))

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=2)
        below = F.pad(per[:, :, 0:2], (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        above = F.pad(per[:, :, 0:2], (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:]
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = self.zid.reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([per, below, above, g3, zc], dim=2).reshape(B * Z, 11 + Z, Y, X)
        f = F.relu(self.conv1(x))
        f5 = f.reshape(B, Z, W, Y, X)
        fb = F.pad(f5, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        fa = F.pad(f5, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:]
        m = torch.cat([f5, fb, fa], dim=2).reshape(B * Z, 3 * W, Y, X)
        f = f + F.relu(self.mix(m))
        f = f + F.relu(self.conv2(f))
        p = F.relu(self.proj(f))
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            h = head(torch.cat([p, mx, mn], dim=1)).reshape(B, Z, 2, Y, X)
            gate = torch.softmax(h[:, :, 0], dim=1)
            score = h[:, :, 1]
            outs.append((gate * score).sum(1).unsqueeze(1))
        place = torch.cat(outs, dim=1).flatten(1)
        f5 = f.reshape(B, Z, W, Y, X)
        pooled = torch.cat([f5.mean((1, 3, 4)), f5.amax((1, 3, 4))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
