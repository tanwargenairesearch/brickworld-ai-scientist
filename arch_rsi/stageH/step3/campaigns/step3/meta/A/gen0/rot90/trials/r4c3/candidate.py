import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

TR = 24
CTX = 8
HW = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = int(spec.in_channels)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.ky = max(a for a, b in fps) + 2
        self.kx = max(b for a, b in fps) + 2
        self.trunk = nn.Sequential(
            nn.Conv2d(c, 32, 1), nn.ReLU(),
            nn.Conv2d(32, TR, 3, padding=1), nn.ReLU(),
            nn.Conv2d(TR, TR, 3, padding=1), nn.ReLU())
        self.ctx = nn.Conv2d(TR, CTX, 1)
        self.win = nn.Conv2d(c + CTX, HW, (self.ky, self.kx))
        self.mid = nn.Conv2d(HW, HW, 1)
        self.out = nn.Conv2d(HW, int(spec.n_types), 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * TR, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        t = self.trunk(obs)
        u = torch.cat([obs, F.relu(self.ctx(t))], dim=1)
        q = F.pad(u, (1, self.kx - 2, 1, self.ky - 2))
        h = F.relu(self.win(q))
        h = F.relu(self.mid(h))
        place = self.out(h).flatten(1)
        g = self.glob(torch.cat([t.mean((2, 3)), t.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
