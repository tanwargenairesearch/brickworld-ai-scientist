import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
FC = 16
HIDF = 256

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.95, "clip_eps": 0.2}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        cin = int(spec.in_channels) + self.T
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fc = nn.Conv2d(WIDTH, FC, 1)
        self.fc1 = nn.Linear(FC * self.Y * self.X, HIDF)
        self.out = nn.Linear(HIDF, self.n_place)
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + HIDF, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, self.Y, self.X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.fc1(F.relu(self.fc(f)).flatten(1)))
        place = self.out(h)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), h], dim=1))
        d = self.done_head(g)
        lp_place = F.log_softmax(place, dim=1) + F.logsigmoid(-d)
        lp_done = F.logsigmoid(d)
        logits = torch.cat([lp_place, lp_done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
