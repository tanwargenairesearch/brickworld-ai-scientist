import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.95, "gae_lambda": 0.95}

STEM = 48
WIDTH = 32
D = 32
M = 2
H = 64
GH = 64
TEMP = 10.0


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = int(spec.in_channels)
        self.N = int(spec.Y) * int(spec.X)
        self.P = int(spec.n_place)
        self.stem = nn.Sequential(nn.Conv2d(C, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, STEM, 1), nn.ReLU())
        self.c1 = nn.Conv2d(STEM, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.cell = nn.Conv2d(C + WIDTH, D, 1)
        self.att = nn.Parameter(torch.zeros(M, self.P, self.N))
        self.lin = nn.Linear(M * D, H)
        self.gb = nn.Linear(GH, H)
        self.v = nn.Parameter(torch.zeros(self.P, H))
        nn.init.normal_(self.v, std=0.01)
        self.vb = nn.Parameter(torch.zeros(self.P))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(self.cell(torch.cat([obs, f], dim=1))).flatten(2)
        a = torch.softmax(self.att * TEMP, dim=-1)
        gat = torch.einsum('mpn,bdn->bpmd', a, h).reshape(B, self.P, M * D)
        z = F.relu(self.lin(gat) + self.gb(gl)[:, None, :])
        place = (z * self.v).sum(-1) + self.vb
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
