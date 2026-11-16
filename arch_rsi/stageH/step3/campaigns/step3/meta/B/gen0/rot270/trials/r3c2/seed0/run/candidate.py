import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.9}

STEM = 64
COL = 48
WIDTH = 40
D = 16
G = 64
LOC_SCALE = 8.0


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        A = int(spec.n_actions)
        N = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.vproj = nn.Conv2d(WIDTH, D, 1)
        self.loc = nn.Parameter(torch.zeros(N, A))
        self.wout = nn.Parameter(torch.zeros(D, A))
        nn.init.normal_(self.wout, std=0.1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, G), nn.ReLU())
        self.gout = nn.Linear(G, A)
        nn.init.orthogonal_(self.gout.weight, 0.01)
        nn.init.zeros_(self.gout.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        v = F.relu(self.vproj(f)).flatten(2)
        att = torch.softmax(self.loc * LOC_SCALE, dim=0)
        r = torch.matmul(v, att)
        logits = (r * self.wout.unsqueeze(0)).sum(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = logits + self.gout(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
