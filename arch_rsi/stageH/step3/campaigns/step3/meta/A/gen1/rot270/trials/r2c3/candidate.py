import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 32, "ppo_epochs": 2, "n_minibatches": 8, "lr": 0.0015, "clip_eps": 0.3, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
RAW = 16
PROJ = 32
HID = 32
G = 1.4142135623730951


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def coords(obs):
    B, _, Y, X = obs.shape
    ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([obs, ys, xs], 1)


def pool_stats(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class AxialMix(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.row = init_(nn.Conv2d(2 * w, w, 1), G)
        self.col = init_(nn.Conv2d(2 * w, w, 1), G)
        self.pw = init_(nn.Conv2d(w, w, 1), G)
        self.n = nn.GroupNorm(4, w)

    def forward(self, x):
        r = self.row(torch.cat([x.mean(3, keepdim=True), x.amax(3, keepdim=True)], 1))
        c = self.col(torch.cat([x.mean(2, keepdim=True), x.amax(2, keepdim=True)], 1))
        return F.relu(x + self.n(self.pw(F.relu(r + c))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        C = spec.in_channels + 2
        self.s1 = init_(nn.Conv2d(C, STEM, 1), G)
        self.s2 = init_(nn.Conv2d(STEM, COL, 1), G)
        self.c1 = init_(nn.Conv2d(COL, WIDTH, 3, padding=1), G)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.ax = AxialMix(WIDTH)
        self.c2 = init_(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), G)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.raw = init_(nn.Conv2d(C, RAW, 1), G)
        self.proj = init_(nn.Conv2d(WIDTH + RAW, PROJ, 1), 1.0)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * PROJ, HID, 1), G) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(WIDTH + RAW, 1, 1), 0.01)
        self.vcell = init_(nn.Conv2d(WIDTH + RAW, 1, 1), 1.0)
        self.glob = init_(nn.Linear(2 * WIDTH, 64), G)
        self.type_bias = init_(nn.Linear(64, T), 0.01)
        self.done_head = init_(nn.Linear(64, 1), 0.01)
        self.value_head = init_(nn.Linear(64, 1), 1.0)
        self.dl = nn.Parameter(torch.zeros(1))
        self.temp = nn.Parameter(torch.zeros(1))

    def forward(self, obs, mask):
        z = coords(obs)
        B = z.shape[0]
        T = len(self.fps)
        s = F.relu(self.s2(F.relu(self.s1(z))))
        x = F.relu(self.n1(self.c1(s)))
        x = self.ax(x)
        f = F.relu(x + self.n2(self.c2(x)))
        fr = torch.cat([f, F.relu(self.raw(z))], 1)
        p = self.proj(fr)
        a = self.anchor(fr)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = pool_stats(p, dy, dx)
            h = F.relu(self.mix[i](torch.cat([mx, mn, av], 1)))
            res.append(self.outs[i](h))
        place = (torch.cat(res, 1) + a + self.type_bias(g).view(B, T, 1, 1)).flatten(1)
        done = self.done_head(g) + self.dl * torch.logsumexp(place, 1, keepdim=True)
        logits = torch.cat([place, done], 1) * torch.exp(self.temp.clamp(-2.0, 2.0))
        value = self.value_head(g).squeeze(1) + self.vcell(fr).mean((1, 2, 3))
        return logits, value


def build_policy(spec):
    return Net(spec)
