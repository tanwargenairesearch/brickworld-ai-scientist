import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0025}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
GHID = 64
PROJ = 16
HID = 12
MIX_SCALE = 5.0
RG = 1.4142135623730951


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.s1 = init_layer(nn.Conv2d(cin, STEM, 1), RG)
        self.s2 = init_layer(nn.Conv2d(STEM, COL, 1), RG)
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1), RG)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), RG)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = init_layer(nn.Conv2d(cin, SKIP, 1), RG)
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return torch.cat([x, k], 1)


def frame_stack(p):
    a = torch.cat([p, p.flip([3]), p.flip([2]), p.flip([2, 3])], 0)
    q = p.transpose(2, 3)
    b = torch.cat([q, q.flip([3]), q.flip([2]), q.flip([2, 3])], 0)
    return a, b


class FramePool(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = init_layer(nn.Conv2d(c, proj, 1), 1.0)
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(3 * proj, hid, 1), RG) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(hid, 1, 1), 0.01) for _ in self.fps])

    def maps(self, p):
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1)

    def forward(self, f):
        B = f.shape[0]
        p = self.proj(f)
        a, b = frame_stack(p)
        la = self.maps(a).reshape(4, B, -1)
        lb = self.maps(b).reshape(4, B, -1)
        return torch.cat([la, lb], 0)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.head = FramePool(c, spec.footprints, PROJ, HID)
        self.alpha = nn.Parameter(torch.zeros(8))
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * c, GHID), RG), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GHID, 1), 0.01)
        self.value_head = init_layer(nn.Linear(GHID, 1), 1.0)
        self.type_bias = init_layer(nn.Linear(GHID, self.T), 0.01)

    def forward(self, obs, mask):
        B = obs.shape[0]
        f = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        L = self.head(f)
        w = torch.softmax(MIX_SCALE * self.alpha, 0)
        place = torch.einsum('g,gbn->bn', w, L)
        place = (place.reshape(B, self.T, -1) + self.type_bias(g).unsqueeze(2)).reshape(B, -1)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
