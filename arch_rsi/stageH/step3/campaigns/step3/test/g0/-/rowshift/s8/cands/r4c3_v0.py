import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
W = 40
TH = 64
NMIX = 2
HID = 32
GH = 64

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32}


class TokenMix(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.t1 = nn.Linear(n, TH)
        self.t2 = nn.Linear(TH, n)
        self.ln2 = nn.LayerNorm(d)
        self.c1 = nn.Linear(d, 2 * d)
        self.c2 = nn.Linear(2 * d, d)

    def forward(self, h):
        u = self.ln1(h).transpose(1, 2)
        h = h + self.t2(F.gelu(self.t1(u))).transpose(1, 2)
        h = h + self.c2(F.gelu(self.c1(self.ln2(h))))
        return h


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        n = int(spec.Y) * int(spec.X)
        cin = int(spec.in_channels) + self.T
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(2)])
        self.mixers = nn.ModuleList([TokenMix(n, W) for _ in range(NMIX)])
        self.lnf = nn.LayerNorm(W)
        self.head = nn.Sequential(nn.Linear(W, HID), nn.ReLU(), nn.Linear(HID, self.T))
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, Y, X)
        h = self.stem(torch.cat([obs, m], dim=1))
        for c in self.convs:
            h = F.relu(h + c(h))
        t = h.flatten(2).transpose(1, 2)
        for mx in self.mixers:
            t = mx(t)
        t = self.lnf(t)
        place = self.head(t).transpose(1, 2).reshape(B, -1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
