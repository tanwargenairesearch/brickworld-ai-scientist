import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "n_minibatches": 16}

STEM = 64
WIDTH = 40
GLOB = 64
NL = 16


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


def unlayout(maps):
    total = None
    for i, mp in enumerate(maps):
        base = i // 4
        f = i % 4
        if f == 1:
            mp = torch.flip(mp, dims=[2])
        elif f == 2:
            mp = torch.flip(mp, dims=[3])
        elif f == 3:
            mp = torch.flip(mp, dims=[2, 3])
        if base == 0:
            v = mp.flatten(1)
        elif base == 1:
            v = mp.transpose(2, 3).flatten(1)
        elif base == 2:
            v = mp.permute(0, 2, 3, 1).flatten(1)
        else:
            v = mp.permute(0, 3, 2, 1).flatten(1)
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
        self.heads = nn.Conv2d(WIDTH, NL * self.T, 1)
        nn.init.normal_(self.heads.weight, std=0.01)
        nn.init.zeros_(self.heads.bias)
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
        hm = self.heads(f)
        maps = list(torch.split(hm, self.T, dim=1))
        place = unlayout(maps)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
