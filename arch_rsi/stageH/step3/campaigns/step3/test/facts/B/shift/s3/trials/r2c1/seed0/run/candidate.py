import torch
import torch.nn as nn
import torch.nn.functional as F

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
        sq = self.Y == self.X
        ops = []
        for t in ([0, 1] if sq else [0]):
            for fy in [0, 1]:
                for fx in [0, 1]:
                    ops.append((t, fy, fx))
        self.ops = ops
        self.nl = len(ops)
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

    def apply_op(self, m, op):
        t, fy, fx = op
        if t:
            m = m.transpose(2, 3)
        if fy:
            m = m.flip(2)
        if fx:
            m = m.flip(3)
        return m

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, self.Y, self.X)
        coords = self.coords.expand(B, 2, self.Y, self.X)
        lay = [self.apply_op(m, op) for op in self.ops]
        inp = torch.cat([obs] + lay + [coords], dim=1)
        s = self.stem(inp)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        o = self.out(f)
        parts = torch.split(o, self.T, dim=1)
        place = None
        for k in range(self.nl):
            q = self.apply_op(parts[k], self.ops[k]).reshape(B, self.n_place)
            place = q if place is None else place + q
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
