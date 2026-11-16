import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 64, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 4, "ent_coef": 0.005, "gamma": 0.97}

STEM = 64
COL = 48
WIDTH = 40
SPH = 32
GH = 64
NV = 7
GATE_SCALE = 5.0


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.sp = nn.Conv2d(WIDTH, SPH, 3, padding=1)
        self.out = nn.Conv2d(SPH, self.T, 1)
        self.gate = nn.Parameter(torch.zeros(NV))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        P = self.out(F.relu(self.sp(f)))
        views = [
            P.flatten(1),
            P.flip(2).flatten(1),
            P.flip(3).flatten(1),
            P.flip(2).flip(3).flatten(1),
            P.transpose(2, 3).flatten(1),
            P.permute(0, 2, 3, 1).flatten(1),
            P.permute(0, 3, 2, 1).flatten(1),
        ]
        V = torch.stack(views, dim=1)
        w = torch.softmax(self.gate * GATE_SCALE, dim=0)
        place = (V * w[None, :, None]).sum(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
