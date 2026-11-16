import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 48
GLOB = 64
LAYOUTS = ((False, ()), (True, ()), (False, (2,)), (False, (3,)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        nl = len(LAYOUTS)
        cin = 3 * self.C + nl * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out = nn.Conv2d(WIDTH, nl * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def lay(self, m, tr, fl):
        b = m.shape[0]
        if tr:
            v = m.reshape(b, self.T, self.X, self.Y).transpose(2, 3)
        else:
            v = m.reshape(b, self.T, self.Y, self.X)
        if len(fl) > 0:
            v = v.flip(fl)
        return v

    def unlay(self, o, tr, fl):
        b = o.shape[0]
        if len(fl) > 0:
            o = o.flip(fl)
        if tr:
            o = o.transpose(2, 3)
        return o.reshape(b, -1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype)
        o1 = obs.reshape(b, self.Y, self.X, self.C).permute(0, 3, 1, 2)
        o2 = obs.reshape(b, self.X, self.Y, self.C).permute(0, 3, 2, 1)
        chans = [obs, o1, o2]
        for tr, fl in LAYOUTS:
            chans.append(self.lay(m, tr, fl))
        ys = torch.linspace(-1.0, 1.0, self.Y, device=obs.device, dtype=obs.dtype)
        xs = torch.linspace(-1.0, 1.0, self.X, device=obs.device, dtype=obs.dtype)
        chans.append(ys.view(1, 1, self.Y, 1).expand(b, 1, self.Y, self.X))
        chans.append(xs.view(1, 1, 1, self.X).expand(b, 1, self.Y, self.X))
        h = self.stem(torch.cat(chans, dim=1))
        x = F.relu(h + self.c1(h))
        f = F.relu(x + self.c2(x))
        o = self.out(f)
        place = None
        for k, (tr, fl) in enumerate(LAYOUTS):
            p = self.unlay(o[:, k * self.T:(k + 1) * self.T], tr, fl)
            place = p if place is None else place + p
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
