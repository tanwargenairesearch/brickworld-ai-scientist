import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 48
GLOB = 64
ORDERS = ("TYX", "TXY", "YXT", "XYT", "YTX", "XTY")
FLIPS = ((),)
OFFSETS = (0,)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.dims = {"T": self.T, "Y": self.Y, "X": self.X}
        self.layouts = [(o, f, s) for s in OFFSETS for o in ORDERS for f in FLIPS]
        nl = len(self.layouts)
        cin = spec.in_channels + nl * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out = nn.Conv2d(WIDTH, nl * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, len(OFFSETS))
        self.value_head = nn.Linear(GLOB, 1)

    def lay(self, m, order, fl):
        b = m.shape[0]
        v = m.reshape(b, self.dims[order[0]], self.dims[order[1]], self.dims[order[2]])
        v = v.permute(0, 1 + order.index("T"), 1 + order.index("Y"), 1 + order.index("X"))
        if len(fl) > 0:
            v = v.flip(fl)
        return v

    def unlay(self, o, order, fl):
        b = o.shape[0]
        if len(fl) > 0:
            o = o.flip(fl)
        pos = {"T": 1, "Y": 2, "X": 3}
        o = o.permute(0, pos[order[0]], pos[order[1]], pos[order[2]])
        return o.reshape(b, -1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        mf = mask.to(obs.dtype)
        chans = [obs]
        for order, fl, s in self.layouts:
            chans.append(self.lay(mf[:, s:s + self.n_place], order, fl))
        ys = torch.linspace(-1.0, 1.0, self.Y, device=obs.device, dtype=obs.dtype)
        xs = torch.linspace(-1.0, 1.0, self.X, device=obs.device, dtype=obs.dtype)
        chans.append(ys.view(1, 1, self.Y, 1).expand(b, 1, self.Y, self.X))
        chans.append(xs.view(1, 1, 1, self.X).expand(b, 1, self.Y, self.X))
        h = self.stem(torch.cat(chans, dim=1))
        x = F.relu(h + self.c1(h))
        f = F.relu(x + self.c2(x))
        o = self.out(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        d = self.done_head(g)
        sums = [None for _ in OFFSETS]
        for k, (order, fl, s) in enumerate(self.layouts):
            p = self.unlay(o[:, k * self.T:(k + 1) * self.T], order, fl)
            j = OFFSETS.index(s)
            sums[j] = p if sums[j] is None else sums[j] + p
        logits = None
        for j, s in enumerate(OFFSETS):
            if s == 0:
                vec = torch.cat([sums[j], d[:, j:j + 1]], dim=1)
            else:
                vec = torch.cat([d[:, j:j + 1], sums[j]], dim=1)
            logits = vec if logits is None else logits + vec
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
