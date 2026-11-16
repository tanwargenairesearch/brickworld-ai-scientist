import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.002, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 2, 'ent_coef': 0.02, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.25}

STEM = 48
W1 = 32
W2 = 48
W3 = 64
SKIP = 24
PROJ = 32
HID = 32
G = 64


def orth(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def coord_channels(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([ys, xs], dim=1)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(4, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(4, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(nn.Conv2d(cin + 2, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, W1, 1))
        self.b1 = Block(W1)
        self.nd1 = nn.GroupNorm(4, W1)
        self.down1 = nn.Conv2d(W1, W2, 3, stride=2, padding=1)
        self.b2 = Block(W2)
        self.nd2 = nn.GroupNorm(4, W2)
        self.down2 = nn.Conv2d(W2, W3, 3, stride=2, padding=1)
        self.up2 = nn.Conv2d(W3 + W2, W2, 1)
        self.up1 = nn.Conv2d(W2 + W1, W1, 1)
        self.fine = nn.Conv2d(W1, W1, 3, padding=1)
        self.nf = nn.GroupNorm(4, W1)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.proj = nn.Conv2d(W1 + SKIP, PROJ, 1)
        self.anchor = nn.Conv2d(W1 + SKIP, 1, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W1 + SKIP + W3, G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.type_bias = nn.Linear(G, len(self.fps))
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Sequential(nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                orth(m, math.sqrt(2.0))
        for m in self.outs:
            orth(m, 0.01)
        orth(self.anchor, 0.01)
        orth(self.type_bias, 0.01)
        orth(self.done_head, 0.01)
        orth(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        x0 = self.stem(torch.cat([obs, coord_channels(obs)], dim=1))
        e1 = self.b1(x0)
        d1 = self.down1(F.relu(self.nd1(e1)))
        e2 = self.b2(d1)
        z = F.relu(self.down2(F.relu(self.nd2(e2))))
        zg = z.mean((2, 3))
        u2 = F.interpolate(z, size=(e2.shape[2], e2.shape[3]), mode='nearest')
        u2 = F.relu(self.up2(torch.cat([u2, F.relu(e2)], dim=1)))
        u1 = F.interpolate(u2, size=(e1.shape[2], e1.shape[3]), mode='nearest')
        u1 = F.relu(self.up1(torch.cat([u1, F.relu(e1)], dim=1)))
        f = F.relu(self.nf(u1 + self.fine(u1)))
        sk = F.relu(self.skip(obs))
        fs = torch.cat([f, sk], dim=1)
        p = self.proj(fs)
        anc = self.anchor(fs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), sk.amax((2, 3)), zg], dim=1))
        tb = self.type_bias(g)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(self.mix[i](torch.cat([mx, mn, av], dim=1)))
            res.append(self.outs[i](h) + anc + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
