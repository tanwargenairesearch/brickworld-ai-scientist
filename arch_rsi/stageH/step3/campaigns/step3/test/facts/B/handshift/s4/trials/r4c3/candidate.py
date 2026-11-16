import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 8, "n_envs": 64, "rollout_len": 16, "n_minibatches": 16}

STEM = 64
COL = 48
WIDTH = 40
LAYOUTS = [(0, None), (1, None), (0, 2), (0, 3)]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.L = len(LAYOUTS)
        cin = spec.in_channels + self.L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.gctx = nn.Linear(2 * WIDTH, WIDTH)
        self.out = nn.Conv2d(WIDTH, self.L * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def to_layout(self, m, base, fl):
        T, Y, X = self.T, self.Y, self.X
        if base == 0:
            a = m.reshape(-1, T, Y, X)
        else:
            a = m.reshape(-1, T, X, Y).transpose(2, 3)
        if fl is not None:
            a = torch.flip(a, dims=[fl])
        return a

    def from_layout(self, o, base, fl):
        B = o.shape[0]
        if fl is not None:
            o = torch.flip(o, dims=[fl])
        if base == 1:
            o = o.transpose(2, 3)
        return o.reshape(B, -1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        m = mask[:, :T * Y * X].float()
        lays = [self.to_layout(m, b, fl) for b, fl in LAYOUTS]
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = torch.cat([obs] + lays + [yy, xx], dim=1)
        s = self.stem(h)
        x = F.relu(self.c1(s))
        gx = self.gctx(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        hmid = F.relu(self.c2(x) + gx[:, :, None, None])
        f = F.relu(x + self.c3(hmid))
        o = self.out(f).view(B, self.L, T, Y, X)
        place = None
        for i, (b, fl) in enumerate(LAYOUTS):
            p = self.from_layout(o[:, i], b, fl)
            place = p if place is None else place + p
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
