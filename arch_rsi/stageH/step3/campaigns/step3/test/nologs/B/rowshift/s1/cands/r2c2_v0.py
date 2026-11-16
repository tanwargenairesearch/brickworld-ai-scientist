import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16}

STEM = 48
W = 40
HID = 32
G = 64
DENSE_BUDGET = 400000


class CellNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = CellNorm(c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))


class FootHead(nn.Module):
    def __init__(self, c, footprints, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c, hid, (dy, dx)) for dy, dx in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(hid, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)

    def forward(self, f):
        res = []
        for (dy, dx), conv, mix, out in zip(self.fps, self.convs, self.mix, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1)) if dy * dx > 1 else f
            h = F.relu(conv(q))
            h = F.relu(mix(h))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cells = int(spec.Y) * int(spec.X)
        n_place = int(spec.n_place)
        k = DENSE_BUDGET // max(1, cells * n_place)
        k = max(4, min(24, k))
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.blocks = nn.Sequential(Block(W), Block(W), Block(W))
        self.fnorm = CellNorm(W)
        self.spatial = FootHead(W, spec.footprints, HID)
        self.cell = nn.Conv2d(W, k, 1)
        self.place = nn.Linear(k * cells, n_place)
        nn.init.orthogonal_(self.place.weight, 0.01)
        nn.init.zeros_(self.place.bias)
        self.done_mlp = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU(), nn.Linear(G, 1))
        self.value_mlp = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU(), nn.Linear(G, 1))
        nn.init.orthogonal_(self.done_mlp[2].weight, 0.01)
        nn.init.zeros_(self.done_mlp[2].bias)
        nn.init.orthogonal_(self.value_mlp[2].weight, 1.0)
        nn.init.zeros_(self.value_mlp[2].bias)

    def forward(self, obs, mask):
        x = self.stem(obs)
        x = self.blocks(x)
        f = F.relu(self.fnorm(x))
        c = F.relu(self.cell(f))
        place = self.place(c.flatten(1)) + self.spatial(f)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        logits = torch.cat([place, self.done_mlp(g)], dim=1)
        return logits, self.value_mlp(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
