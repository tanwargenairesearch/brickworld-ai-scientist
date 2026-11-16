import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 32, "ppo_epochs": 2, "n_minibatches": 8, "lr": 0.002, "clip_eps": 0.3, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.003}

STEM = 64
COL = 48
WIDTH = 40
RAW = 16
K = 24
P = 16
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


def land_feats(hc, mc, p, dy, dx):
    Y, X = hc.shape[2], hc.shape[3]
    if dy * dx == 1:
        pr = hc * mc
        return torch.cat([hc, pr, pr, p, p], 1)
    hq = F.pad(hc, (0, dx - 1, 0, dy - 1))
    hp = F.max_pool2d(hq, (dy, dx), stride=1)
    mq = F.pad(mc, (0, dx - 1, 0, dy - 1))
    prods = torch.stack([hp * mq[:, :, i:i + Y, j:j + X] for i in range(dy) for j in range(dx)], 0)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    return torch.cat([hp, prods.amin(0), prods.mean(0), mx, mn], 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        Y, X = int(spec.Y), int(spec.X)
        C = spec.in_channels + 2
        self.s1 = init_(nn.Conv2d(C, STEM, 1), G)
        self.s2 = init_(nn.Conv2d(STEM, COL, 1), G)
        self.c1 = init_(nn.Conv2d(COL, WIDTH, 3, padding=1), G)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.dw = nn.Conv2d(WIDTH, WIDTH, (2 * Y - 1, 2 * X - 1), padding=(Y - 1, X - 1), groups=WIDTH)
        self.pw = init_(nn.Conv2d(WIDTH, WIDTH, 1), G)
        self.ndw = nn.GroupNorm(4, WIDTH)
        self.c2 = init_(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), G)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.raw = init_(nn.Conv2d(C, RAW, 1), G)
        self.hc = init_(nn.Conv2d(WIDTH + RAW, K, 1), 1.0)
        self.mc = init_(nn.Conv2d(WIDTH + RAW, K, 1), 1.0)
        self.proj = init_(nn.Conv2d(WIDTH + RAW, P, 1), 1.0)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * K + 2 * P, HID, 1), G) for _ in self.fps])
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
        x = F.relu(x + self.ndw(self.pw(self.dw(x))))
        f = F.relu(x + self.n2(self.c2(x)))
        fr = torch.cat([f, F.relu(self.raw(z))], 1)
        hc = torch.sigmoid(self.hc(fr))
        mc = self.mc(fr)
        p = self.proj(fr)
        a = self.anchor(fr)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            h = F.relu(self.mix[i](land_feats(hc, mc, p, dy, dx)))
            res.append(self.outs[i](h))
        place = (torch.cat(res, 1) + a + self.type_bias(g).view(B, T, 1, 1)).flatten(1)
        done = self.done_head(g) + self.dl * torch.logsumexp(place, 1, keepdim=True)
        logits = torch.cat([place, done], 1) * torch.exp(self.temp.clamp(-2.0, 2.0))
        value = self.value_head(g).squeeze(1) + self.vcell(fr).mean((1, 2, 3))
        return logits, value


def build_policy(spec):
    return Net(spec)
