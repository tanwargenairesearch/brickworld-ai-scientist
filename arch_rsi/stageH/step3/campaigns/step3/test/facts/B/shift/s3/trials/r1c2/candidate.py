import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2}

COL = 64
WIDTH = 48
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.sq = self.Y == self.X
        self.nl = 4 if self.sq else 3
        cin = spec.in_channels + self.nl * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, COL, 1), nn.ReLU(),
            nn.Conv2d(COL, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out = nn.Conv2d(WIDTH, self.nl * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).contiguous())

    def layouts(self, m):
        outs = [m]
        if self.sq:
            outs.append(m.transpose(2, 3))
        outs.append(m.flip(2))
        outs.append(m.flip(3))
        return outs

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, self.Y, self.X)
        coords = self.coords.expand(B, 2, self.Y, self.X)
        inp = torch.cat([obs] + self.layouts(m) + [coords], dim=1)
        s = self.stem(inp)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        o = self.out(f)
        parts = torch.split(o, self.T, dim=1)
        place = parts[0]
        i = 1
        if self.sq:
            place = place + parts[i].transpose(2, 3)
            i += 1
        place = place + parts[i].flip(2)
        place = place + parts[i + 1].flip(3)
        place = place.reshape(B, self.n_place)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
