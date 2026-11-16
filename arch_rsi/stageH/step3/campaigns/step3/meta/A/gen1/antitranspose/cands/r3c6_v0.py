import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 1e-3, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 1, 'ent_coef': 0.01, 'gamma': 0.995, 'gae_lambda': 0.95, 'clip_eps': 0.2}

COLH = 128
D = 64
HT = 128
HC = 192
BLOCKS = 2
W = 48
PROJ = 48
FHID = 24
WHID = 12
G = 4


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def add_coords(obs):
    b = obs.shape[0]
    y = obs.shape[2]
    x = obs.shape[3]
    yy = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xx = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([obs, yy, xx], dim=1)


class MixBlock(nn.Module):
    def __init__(self, n, d, ht, hc):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.t1 = nn.Linear(n, ht)
        self.t2 = nn.Linear(ht, n)
        self.n2 = nn.LayerNorm(d)
        self.c1 = nn.Linear(d, hc)
        self.c2 = nn.Linear(hc, d)

    def forward(self, x):
        y = self.n1(x).transpose(1, 2)
        y = self.t2(F.gelu(self.t1(y))).transpose(1, 2)
        x = x + y
        x = x + self.c2(F.gelu(self.c1(self.n2(x))))
        return x


class FootHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.foot = nn.ModuleList([nn.Conv2d(proj, hid, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f, gb):
        p = F.relu(self.proj(f) + gb[:, :, None, None])
        res = []
        for (dy, dx), foot, out in zip(self.fps, self.foot, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(foot(q))))
        return torch.cat(res, dim=1)


class WinHead(nn.Module):
    def __init__(self, c, n, proj, hid):
        super().__init__()
        self.proj = nn.Conv2d(c, proj, 1)
        self.win = nn.Conv2d(proj, n * hid, 3, padding=1)
        self.out = nn.Conv2d(n * hid, n, 1, groups=n)

    def forward(self, f):
        p = F.relu(self.proj(f))
        return self.out(F.relu(self.win(p)))


class DenseHead(nn.Module):
    def __init__(self, c, y, x, n_out, ch, hid):
        super().__init__()
        self.proj = nn.Conv2d(c, ch, 1)
        self.l1 = nn.Linear(ch * y * x, hid)
        self.l2 = nn.Linear(hid, n_out)

    def forward(self, f):
        return self.l2(F.relu(self.l1(F.relu(self.proj(f)).flatten(1))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels + 2
        n = len(spec.footprints)
        N = spec.Y * spec.X
        self.col = nn.Sequential(
            nn.Conv2d(cin, COLH, 1), nn.GroupNorm(G, COLH), nn.ReLU(),
            nn.Conv2d(COLH, D, 1), nn.ReLU())
        self.pos = nn.Parameter(torch.zeros(1, N, D))
        self.blocks = nn.ModuleList([MixBlock(N, D, HT, HC) for _ in range(BLOCKS)])
        self.nf = nn.LayerNorm(D)
        self.cv = nn.Conv2d(D, W, 3, padding=1)
        self.cn = nn.GroupNorm(G, W)
        c = 2 * D + W
        self.glob = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU())
        self.gproj = nn.Linear(64, PROJ)
        self.fhead = FootHead(c, spec.footprints, PROJ, FHID)
        self.whead = WinHead(c, n, PROJ, WHID)
        self.inter = nn.Conv2d(c, n, 1)
        self.anchor = nn.Conv2d(c, 1, 1)
        self.tbias = nn.Parameter(torch.zeros(n))
        self.dense = DenseHead(c, spec.Y, spec.X, spec.n_actions, 8, 256)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_layer(m, math.sqrt(2))
        for m in self.blocks.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos, std=0.02)
        for m in self.fhead.outs:
            init_layer(m, 0.01)
        init_layer(self.whead.out, 0.01)
        init_layer(self.inter, 0.01)
        init_layer(self.anchor, 0.01)
        init_layer(self.dense.l2, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        c0 = self.col(add_coords(obs))
        t = c0.flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            t = blk(t)
        a = self.nf(t).transpose(1, 2).reshape(B, D, Y, X)
        f = F.relu(self.cn(self.cv(c0)))
        h = torch.cat([c0, a, f], dim=1)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        s = self.fhead(h, self.gproj(g)) + self.whead(h) + self.inter(h) + self.anchor(h) + self.tbias.view(1, -1, 1, 1)
        logits = torch.cat([s.flatten(1), self.done_head(g)], dim=1) + self.dense(h)
        return logits, self.value_head(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
