import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01, "lr": 0.001}

W = 40
P = 8
HH = 32
GC = 16


def orient(o, k):
    if k >= 4:
        o = o.transpose(2, 3)
    j = k % 4
    if j == 1:
        o = o.flip(2)
    elif j == 2:
        o = o.flip(3)
    elif j == 3:
        o = o.flip(2, 3)
    return o


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        shapes = []
        for s in fps:
            if s[0] * s[1] > 1 and s not in shapes:
                shapes.append(s)
        self.shapes = shapes
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + self.nt, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(3)])
        self.gctx = nn.Linear(2 * W, GC)
        self.proj = nn.Conv2d(W, P, 1)
        self.hconv = nn.Conv2d(W + GC + 2 * P * len(shapes), HH, 3, padding=1)
        self.hmid = nn.Conv2d(HH, HH, 1)
        self.n_br = 8 if self.Y == self.X else 4
        self.outs = nn.Conv2d(HH, self.nt * self.n_br, 1)
        self.bgate = nn.Parameter(torch.zeros(self.n_br))
        self.glob = nn.Sequential(nn.Linear(2 * W + 2 * HH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.nt, self.Y, self.X)
        f = self.stem(torch.cat([obs, m], dim=1))
        for blk in self.blocks:
            f = F.relu(f + blk(f))
        fpool = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        gc = F.relu(self.gctx(fpool)).view(B, GC, 1, 1).expand(B, GC, self.Y, self.X)
        p = self.proj(f)
        feats = [f, gc]
        for dy, dx in self.shapes:
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            feats.append(F.max_pool2d(q, (dy, dx), stride=1))
            feats.append(-F.max_pool2d(-q, (dy, dx), stride=1))
        h = F.relu(self.hconv(torch.cat(feats, dim=1)))
        h = F.relu(h + self.hmid(h))
        o = self.outs(h).reshape(B, self.n_br, self.nt, self.Y, self.X)
        w = torch.softmax(self.bgate, dim=0) * self.n_br
        place = w[0] * o[:, 0]
        for k in range(1, self.n_br):
            place = place + w[k] * orient(o[:, k], k)
        place = place.reshape(B, -1)
        g = self.glob(torch.cat([fpool, h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
