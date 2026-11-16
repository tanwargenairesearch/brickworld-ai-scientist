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
        geo = [(0, 0, 0)]
        if sq:
            geo.append((1, 0, 0))
        geo.append((0, 1, 0))
        geo.append((0, 0, 1))
        ops = []
        for order in [0, 1]:
            for g in geo:
                ops.append((order, g[0], g[1], g[2]))
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

    def geo_op(self, m, t, fy, fx):
        if t:
            m = m.transpose(2, 3)
        if fy:
            m = m.flip(2)
        if fx:
            m = m.flip(3)
        return m

    def forward(self, obs, mask):
        B = obs.shape[0]
        mf = mask[:, :self.n_place].to(obs.dtype)
        m_major = mf.reshape(B, self.T, self.Y, self.X)
        m_minor = mf.reshape(B, self.Y, self.X, self.T).permute(0, 3, 1, 2)
        lay = []
        for op in self.ops:
            base = m_minor if op[0] else m_major
            lay.append(self.geo_op(base, op[1], op[2], op[3]))
        coords = self.coords.expand(B, 2, self.Y, self.X)
        inp = torch.cat([obs] + lay + [coords], dim=1)
        s = self.stem(inp)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        o = self.out(f)
        parts = torch.split(o, self.T, dim=1)
        place = None
        for k in range(self.nl):
            op = self.ops[k]
            q = self.geo_op(parts[k], op[1], op[2], op[3])
            if op[0]:
                q = q.permute(0, 2, 3, 1).reshape(B, self.n_place)
            else:
                q = q.reshape(B, self.n_place)
            place = q if place is None else place + q
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
