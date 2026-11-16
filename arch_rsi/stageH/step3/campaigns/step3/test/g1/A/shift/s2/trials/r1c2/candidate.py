import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 64, "rollout_len": 64, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.3}

STEM = 64
WIDTH = 40
SKIP = 16
PROJ = 32
HID = 32
AK = 16
GLOB = 64
DIL = (1, 2, 1)
RG = 1.4142135623730951
LOG2 = 0.6931471805599453


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def add_coords(obs):
    b = obs.shape[0]
    y = obs.shape[2]
    x = obs.shape[3]
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([obs, ys, xs], dim=1)


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = ortho(nn.Conv2d(c, proj, 1), 1.0)
        self.mix = nn.ModuleList([ortho(nn.Conv2d(3 * proj, hid, 1), RG) for fp in self.fps])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(hid, 1, 1), 0.01) for fp in self.fps])

    def forward(self, f):
        p = self.proj(f)
        logits = []
        hids = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            h = F.relu(mix(fp_pool(p, dy, dx)))
            hids.append(h)
            logits.append(out(h))
        return torch.cat(logits, dim=1), hids


class AndHead(nn.Module):
    def __init__(self, c, footprints, hid, k):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.u = ortho(nn.Conv2d(c, k, 1), 1.0)
        self.v = nn.ModuleList([ortho(nn.Conv2d(hid, k, 1), 0.05) for fp in self.fps])

    def forward(self, f, hids):
        u = self.u(f)
        yy = u.shape[2]
        xx = u.shape[3]
        res = []
        for (dy, dx), vl, h in zip(self.fps, self.v, hids):
            v = vl(h)
            q = F.pad(u, (0, dx - 1, 0, dy - 1))
            acc = torch.zeros_like(v[:, :1])
            for i in range(dy):
                for j in range(dx):
                    c = (q[:, :, i:i + yy, j:j + xx] * v).sum(1, keepdim=True)
                    acc = acc + F.logsigmoid(c) + LOG2
            res.append(acc)
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        t = spec.n_types
        self.stem1 = ortho(nn.Conv2d(c + 2, STEM, 1), RG)
        self.stem2 = ortho(nn.Conv2d(STEM, WIDTH, 1), RG)
        self.convs = nn.ModuleList([ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d), RG) for d in DIL])
        self.norms = nn.ModuleList([nn.GroupNorm(8, WIDTH) for d in DIL])
        self.skip = ortho(nn.Conv2d(c, SKIP, 1), RG)
        self.head = PoolHead(WIDTH + SKIP, spec.footprints, PROJ, HID)
        self.andh = AndHead(WIDTH + SKIP, spec.footprints, HID, AK)
        self.dense = ortho(nn.Conv2d(WIDTH + SKIP, t, 3, padding=1), 0.01)
        self.anchor = ortho(nn.Conv2d(WIDTH + SKIP, 1, 1), 0.01)
        self.glob = ortho(nn.Linear(2 * WIDTH + SKIP, GLOB), RG)
        self.type_term = ortho(nn.Linear(GLOB, t), 0.01)
        self.gdense = ortho(nn.Linear(GLOB, spec.n_place), 0.01)
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        b = obs.shape[0]
        yy = obs.shape[2]
        xx = obs.shape[3]
        x = F.relu(self.stem2(F.relu(self.stem1(add_coords(obs)))))
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(x + norm(conv(x)))
        k = F.relu(self.skip(obs))
        hf = torch.cat([x, k], dim=1)
        g = F.relu(self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3)), k.amax((2, 3))], dim=1)))
        place, hids = self.head(hf)
        place = place + self.andh(hf, hids) + self.dense(hf) + self.anchor(hf)
        place = place + self.type_term(g).view(b, -1, 1, 1) + self.gdense(g).view(b, -1, yy, xx)
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
