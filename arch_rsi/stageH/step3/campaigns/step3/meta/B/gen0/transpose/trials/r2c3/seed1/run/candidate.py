import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.005, "gamma": 0.95, "gae_lambda": 0.9}

COLH = 64
W = 40
HH = 48
H2 = 32
DP = 8
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        cin = int(spec.in_channels) + self.T + 2
        self.col1 = nn.Conv2d(cin, COLH, 1)
        self.col2 = nn.Conv2d(COLH, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.anc = nn.Conv2d(W, HH, 2)
        self.anc_skip = nn.Conv2d(W, HH, 1)
        self.h2 = nn.Conv2d(HH, H2, 1)
        self.out = nn.Conv2d(H2, self.T, 1)
        self.dproj = nn.Conv2d(W, DP, 1)
        self.dense = nn.Linear(DP * self.Y * self.X, int(spec.n_actions))
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        mk = mask[:, :self.n_place].to(obs.dtype).reshape(B, T, Y, X)
        yy = torch.linspace(0.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(0.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        inp = torch.cat([obs, mk, yy, xx], dim=1)
        s = F.relu(self.col1(inp))
        s = F.relu(self.col2(s))
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        q = F.pad(f, (0, 1, 0, 1))
        h = F.relu(self.anc(q) + self.anc_skip(f))
        h = F.relu(self.h2(h))
        local = self.out(h).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([local, self.done_head(g)], dim=1)
        logits = logits + self.dense(F.relu(self.dproj(f)).flatten(1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
