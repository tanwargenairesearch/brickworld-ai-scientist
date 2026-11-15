import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 4, "ppo_epochs": 1, "lr": 0.002, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 16
HID = 48
G = 64
DP = 8


def pool_stats(p, dy, dx):
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return [mx, mn, av]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        sizes = set()
        for a, b in spec.footprints:
            a = int(a)
            b = int(b)
            if a * b > 1:
                sizes.add((a, b))
                sizes.add((b, a))
        self.sizes = sorted(sizes)
        cin = int(spec.in_channels) + self.T
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        fc = COL + WIDTH
        self.proj = nn.Conv2d(fc, PROJ, 1)
        nin = fc + PROJ * (1 + 3 * len(self.sizes))
        self.mix = nn.Conv2d(nin, HID, 1)
        self.out = nn.Conv2d(HID, self.T, 3, padding=1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.dproj = nn.Conv2d(fc, DP, 1)
        self.dense = nn.Linear(DP * self.Y * self.X, int(spec.n_actions))
        nn.init.normal_(self.dense.weight, std=0.01)
        nn.init.zeros_(self.dense.bias)
        self.glob = nn.Sequential(nn.Linear(2 * fc, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        n = self.T * self.Y * self.X
        m = mask[:, :n].to(obs.dtype).reshape(B, self.T, self.Y, self.X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        fc = torch.cat([s, f], dim=1)
        p = self.proj(fc)
        feats = [fc, p]
        for dy, dx in self.sizes:
            feats = feats + pool_stats(p, dy, dx)
        h = F.relu(self.mix(torch.cat(feats, dim=1)))
        place = self.out(h).flatten(1)
        g = self.glob(torch.cat([fc.mean((2, 3)), fc.amax((2, 3))], dim=1))
        conv_logits = torch.cat([place, self.done_head(g)], dim=1)
        d = F.relu(self.dproj(fc)).flatten(1)
        logits = conv_logits + self.dense(d)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
