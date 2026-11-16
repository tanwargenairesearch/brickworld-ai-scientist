import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0015, "ent_coef": 0.01}

W = 48
K = 8
H = 32
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        T = int(spec.n_types)
        cin = int(spec.in_channels)
        self.col = nn.Sequential(
            nn.Conv2d(cin, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.sq = nn.Conv2d(W, K, 1)
        self.amap = nn.Linear(K * Y * X, int(spec.n_place))
        self.loc = nn.Conv2d(W, H, 3, padding=1)
        self.loc_out = nn.Conv2d(H, T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        c = self.col(obs)
        x = F.relu(c + self.c1(c))
        f = F.relu(x + self.c2(x))
        s = self.sq(f).flatten(1)
        local = self.loc_out(F.relu(self.loc(f))).flatten(1)
        place = self.amap(s) + local
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
