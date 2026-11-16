import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "ppo_epochs": 3, "n_minibatches": 16, "lr": 0.001, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
RAW = 16
PROJ = 16
HID = 16
G = 1.4142135623730951


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def pool_stats(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


def frames(t, square):
    out = [t, t.flip(-1), t.flip(-2), t.flip(-2, -1)]
    if square:
        u = t.transpose(-1, -2)
        out = out + [u, u.flip(-1), u.flip(-2), u.flip(-2, -1)]
    return out


def coords(obs):
    B, _, Y, X = obs.shape
    ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([obs, ys, xs], 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.square = int(spec.Y) == int(spec.X)
        self.nf = 8 if self.square else 4
        C = spec.in_channels + 2
        self.s1 = init_(nn.Conv2d(C, STEM, 1), G)
        self.s2 = init_(nn.Conv2d(STEM, COL, 1), G)
        self.c1 = init_(nn.Conv2d(COL, WIDTH, 3, padding=1), G)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = init_(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), G)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.raw = init_(nn.Conv2d(C, RAW, 1), G)
        self.proj = init_(nn.Conv2d(WIDTH + RAW, PROJ, 1), 1.0)
        self.anc = init_(nn.Conv2d(WIDTH + RAW, self.nf, 1), 0.01)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * PROJ, HID, 1), G) for _ in range(self.nf * T)])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in range(self.nf * T)])
        self.fw = nn.Parameter(torch.zeros(self.nf))
        self.glob = init_(nn.Linear(2 * WIDTH, 64), G)
        self.type_bias = init_(nn.Linear(64, T), 0.01)
        self.done_head = init_(nn.Linear(64, 1), 0.01)
        self.value_head = init_(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        z = coords(obs)
        B = z.shape[0]
        T = len(self.fps)
        s = F.relu(self.s2(F.relu(self.s1(z))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        fr = torch.cat([f, F.relu(self.raw(z))], 1)
        pf = frames(self.proj(fr), self.square)
        af = frames(self.anc(fr), self.square)
        w = torch.softmax(self.fw, 0)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)))
        place = 0.0
        for k in range(self.nf):
            res = []
            for i, (dy, dx) in enumerate(self.fps):
                mx, mn, av = pool_stats(pf[k], dy, dx)
                j = k * T + i
                h = F.relu(self.mix[j](torch.cat([mx, mn, av], 1)))
                res.append(self.outs[j](h))
            place = place + w[k] * (torch.cat(res, 1) + af[k][:, k:k + 1])
        place = place + self.type_bias(g).view(B, T, 1, 1)
        logits = torch.cat([place.flatten(1), self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
