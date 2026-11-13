import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
HEADS = 4
PROJ = 32
HID = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2}


class AttnBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(c)
        self.mlp = nn.Sequential(nn.Linear(c, 2 * c), nn.ReLU(), nn.Linear(2 * c, c))

    def forward(self, x):
        B, C, Y, X = x.shape
        t = x.flatten(2).transpose(1, 2)
        h = self.ln1(t)
        a, _w = self.attn(h, h, h, need_weights=False)
        t = t + a
        t = t + self.mlp(self.ln2(t))
        return t.transpose(1, 2).reshape(B, C, Y, X)


class FootprintHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in footprints]
        self.proj = nn.Conv2d(c, PROJ, 1)
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PROJ, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps
        ])

    def forward(self, f):
        g = F.relu(self.proj(f))
        outs = []
        for (dy, dx), mlp in zip(self.fps, self.mlps):
            if dy > 1 or dx > 1:
                p = F.pad(g, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(p, (dy, dx), stride=1)
                mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            else:
                mx = g
                mn = g
            outs.append(mlp(torch.cat([g, mx, mn], dim=1)))
        return torch.cat(outs, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels + 2
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.attn = AttnBlock(WIDTH)
        self.place_head = FootprintHead(WIDTH, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype)
        yc = ys.view(1, 1, Y, 1).expand(B, 1, Y, X)
        xc = xs.view(1, 1, 1, X).expand(B, 1, Y, X)
        x = torch.cat([obs, yc, xc], dim=1)
        f = self.body(x)
        f = self.attn(f)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
