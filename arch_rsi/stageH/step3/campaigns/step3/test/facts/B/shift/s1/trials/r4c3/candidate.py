import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 40
GLOB = 64
NL = 16
P = 8
HID = 12
FRAMES = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]


def layouts(m, T, Y, X):
    B = m.shape[0]
    bases = [m.reshape(B, T, Y, X),
             m.reshape(B, T, X, Y).transpose(2, 3),
             m.reshape(B, Y, X, T).permute(0, 3, 1, 2),
             m.reshape(B, X, Y, T).permute(0, 3, 2, 1)]
    out = []
    for b in bases:
        out += [b, torch.flip(b, dims=[2]), torch.flip(b, dims=[3]), torch.flip(b, dims=[2, 3])]
    return out


def flip_k(f, k):
    if k == 1:
        return torch.flip(f, dims=[2])
    if k == 2:
        return torch.flip(f, dims=[3])
    if k == 3:
        return torch.flip(f, dims=[2, 3])
    return f


def fp_pool(g, dy, dx):
    if dy * dx == 1:
        return torch.cat([g, g], dim=1)
    q = F.pad(g, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, av], dim=1)


class FrameHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        T = len(self.fps)
        self.T = T
        self.proj = nn.Conv2d(c, P, 1)
        self.mix = nn.Conv2d(T * 4 * P, T * HID, 1, groups=T)
        self.outs = nn.ModuleList([nn.Conv2d(T * HID, 2 * T, 1, groups=T) for _ in FRAMES])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)

    def forward(self, f):
        p = self.proj(f)
        total = None
        for (tr, fl), out in zip(FRAMES, self.outs):
            g = flip_k(p, fl)
            if tr == 1:
                g = g.transpose(2, 3)
            cache = {}
            parts = []
            for dy, dx in self.fps:
                for key in [(dy, dx), (dx, dy)]:
                    if key not in cache:
                        cache[key] = fp_pool(g, key[0], key[1])
                    parts.append(cache[key])
            h = F.relu(self.mix(torch.cat(parts, dim=1)))
            o = out(h)
            a = o[:, 0::2].flatten(1)
            b = o[:, 1::2].permute(0, 2, 3, 1).flatten(1)
            v = a + b
            total = v if total is None else total + v
        return total


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        cin = spec.in_channels + NL * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = FrameHead(WIDTH, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype)
        ms = layouts(m, self.T, self.Y, self.X)
        ys = torch.linspace(-1.0, 1.0, self.Y, device=obs.device, dtype=obs.dtype).view(1, 1, self.Y, 1).expand(B, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, self.X).expand(B, 1, self.Y, self.X)
        h = torch.cat([obs] + ms + [ys, xs], dim=1)
        s = self.stem(h)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
