import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0015, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 48
MC = 4
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        n = self.Y * self.X
        A = int(spec.n_actions)
        self.memb = nn.Linear(A, MC * n)
        self.stem = nn.Sequential(
            nn.Conv2d(int(spec.in_channels) + MC, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.gproj = nn.Linear(2 * WIDTH, WIDTH)
        self.tproj = nn.Linear(WIDTH, HID)
        self.attn = nn.Parameter(torch.randn(A, n + 1))
        self.aemb = nn.Parameter(torch.zeros(A, HID))
        self.abias = nn.Parameter(torch.zeros(A))
        self.mid = nn.Linear(HID, HID)
        self.hout = nn.Linear(HID, 1)
        self.vh = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = self.memb(mask.to(obs.dtype)).reshape(B, MC, self.Y, self.X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        g = F.relu(self.gproj(pooled))
        tok = torch.cat([f.flatten(2).transpose(1, 2), g.unsqueeze(1)], dim=1)
        tok = self.tproj(tok)
        al = torch.softmax(self.attn, dim=1)
        ga = torch.matmul(al, tok)
        u = F.relu(ga + self.aemb)
        u = F.relu(self.mid(u))
        logits = self.hout(u).squeeze(2) + self.abias
        value = self.vh(pooled).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
