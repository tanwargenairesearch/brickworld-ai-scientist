import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16}

STEM = 64
WIDTH = 40
D = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_cells = int(spec.Y) * int(spec.X)
        self.n_actions = int(spec.n_actions)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.pos = nn.Embedding(self.n_cells, D)
        self.tok = nn.Linear(WIDTH, D)
        self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D)
        self.act = nn.Embedding(self.n_actions, D)
        self.mask_in = nn.Linear(1, D)
        self.q = nn.Linear(D, D)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.gq = nn.Linear(64, D)
        self.out1 = nn.Linear(2 * D, D)
        self.out2 = nn.Linear(D, 1)
        nn.init.orthogonal_(self.out2.weight, 0.01)
        nn.init.zeros_(self.out2.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        x = F.relu(x + self.c1(x))
        x = F.relu(x + self.c2(x))
        t = x.flatten(2).transpose(1, 2)
        cidx = torch.arange(self.n_cells, device=obs.device)
        h = self.tok(t) + self.pos(cidx).unsqueeze(0)
        kk = self.k(h)
        vv = self.v(h)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        aidx = torch.arange(self.n_actions, device=obs.device)
        a = self.act(aidx).unsqueeze(0) + self.mask_in(mask.to(obs.dtype).unsqueeze(-1))
        qq = self.q(a)
        att = torch.softmax(torch.matmul(qq, kk.transpose(1, 2)) / math.sqrt(D), dim=-1)
        o = torch.matmul(att, vv)
        z = F.relu(self.out1(torch.cat([o, a + self.gq(g).unsqueeze(1)], dim=-1)))
        logits = self.out2(z).squeeze(-1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
