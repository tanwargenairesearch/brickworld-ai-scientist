import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0005, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "ppo_epochs": 3, "ent_coef": 0.01, "gamma": 0.98, "gae_lambda": 0.95, "clip_eps": 0.15}

STEM = 64
COL = 48
W = 40
PROJ = 24
HID = 24
DH = 256
GH = 64


def anchor_pools(p, dy, dx):
    outs = [p]
    for a, b in ((dy, dx), (dx, dy)):
        for pad in ((0, b - 1, 0, a - 1), (b - 1, 0, a - 1, 0)):
            q = F.pad(p, pad)
            outs.append(F.max_pool2d(q, (a, b), stride=1))
            outs.append(F.avg_pool2d(q, (a, b), stride=1))
    return torch.cat(outs, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.nt = int(spec.n_types)
        self.nplace = int(spec.n_place)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = int(spec.in_channels) + self.nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.c3 = nn.Conv2d(W, W, 3, padding=2, dilation=2)
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.bcast = nn.Linear(GH, W)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(9 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        dc = max(2, min(16, 1024 // (Y * X)))
        self.dproj = nn.Conv2d(W, dc, 1)
        self.dfc = nn.Linear(dc * Y * X + GH, DH)
        self.dout = nn.Linear(DH, self.nplace)
        nn.init.orthogonal_(self.dout.weight, 0.01)
        nn.init.zeros_(self.dout.bias)
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.nplace].to(obs.dtype).reshape(B, self.nt, Y, X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(self.c1(s))
        x = F.relu(x + self.c2(x))
        x = F.relu(x + self.c3(x))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        f = F.relu(x + self.bcast(g).unsqueeze(-1).unsqueeze(-1))
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            res.append(out(F.relu(mix(anchor_pools(p, dy, dx)))))
        place = torch.cat(res, dim=1).flatten(1)
        d = torch.cat([F.relu(self.dproj(f)).flatten(1), g], dim=1)
        place = place + self.dout(F.relu(self.dfc(d)))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
