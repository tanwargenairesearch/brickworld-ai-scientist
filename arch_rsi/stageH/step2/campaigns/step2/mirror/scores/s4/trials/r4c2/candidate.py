import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 8, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
SH = 32
GATE_SCALE = 5.0
ORDERS = [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.n_actions = int(spec.n_actions)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.sc = nn.Sequential(
            nn.Conv2d(WIDTH, SH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(SH, self.T, 1))
        self.g_order = nn.Parameter(torch.zeros(len(ORDERS)))
        self.g_fy = nn.Parameter(torch.zeros(2))
        self.g_fx = nn.Parameter(torch.zeros(2))
        self.g_done = nn.Parameter(torch.zeros(2))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.act_head = nn.Linear(64, self.n_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        sm = self.sc(f)
        wy = torch.softmax(self.g_fy * GATE_SCALE, dim=0)
        wx = torch.softmax(self.g_fx * GATE_SCALE, dim=0)
        sy = wy[0] * sm + wy[1] * sm.flip(2)
        sxy = wx[0] * sy + wx[1] * sy.flip(3)
        wo = torch.softmax(self.g_order * GATE_SCALE, dim=0)
        place = 0.0
        for i, o in enumerate(ORDERS):
            place = place + wo[i] * sxy.permute(0, o[0], o[1], o[2]).reshape(B, -1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        d = self.done_head(g)
        wd = torch.softmax(self.g_done * GATE_SCALE, dim=0)
        logits = wd[0] * torch.cat([place, d], dim=1) + wd[1] * torch.cat([d, place], dim=1)
        logits = logits + self.act_head(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
