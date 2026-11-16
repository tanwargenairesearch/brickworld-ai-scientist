import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.002, "ent_coef": 0.01}

C1 = 128
C2 = 128
C3 = 64
K = 8
G = 128


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        cin = int(spec.in_channels)
        self.col = nn.Sequential(
            nn.Conv2d(cin, C1, 1), nn.ReLU(),
            nn.Conv2d(C1, C2, 1), nn.ReLU(),
            nn.Conv2d(C2, C3, 1), nn.ReLU())
        self.sq = nn.Conv2d(C3, K, 1)
        self.amap = nn.Linear(K * Y * X, int(spec.n_place))
        self.glob = nn.Sequential(nn.Linear(2 * C3, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        c = self.col(obs)
        s = self.sq(c).flatten(1)
        place = self.amap(s)
        g = self.glob(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
