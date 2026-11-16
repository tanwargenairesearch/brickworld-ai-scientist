import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "lr": 0.0005, "gamma": 0.95, "gae_lambda": 0.9, "ent_coef": 0.005, "n_minibatches": 16}

STEM = 64
W = 32
H = 64
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nT = int(spec.n_types)
        cin = int(spec.in_channels) + self.nT
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.g1 = nn.Linear(2 * W, W)
        self.glob = nn.Sequential(nn.Linear(2 * W + 2 * W, G), nn.ReLU())
        self.hconv = nn.Conv2d(2 * W, H, 3, padding=1)
        self.gh = nn.Linear(G, H)
        self.out = nn.Conv2d(H, self.nT, 1)
        nn.init.orthogonal_(self.out.weight, 0.01)
        nn.init.zeros_(self.out.bias)
        self.done_head = nn.Linear(G, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.nT * Y * X].to(obs.dtype).reshape(B, self.nT, Y, X)
        s = self.stem(torch.cat([obs, m], dim=1))
        h = F.relu(self.c1(s))
        gc = self.g1(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        f = F.relu(h + self.c2(h) + gc[:, :, None, None])
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), s.mean((2, 3)), s.amax((2, 3))], dim=1))
        u = F.relu(self.hconv(torch.cat([s, f], dim=1)) + self.gh(g)[:, :, None, None])
        place = self.out(u).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
