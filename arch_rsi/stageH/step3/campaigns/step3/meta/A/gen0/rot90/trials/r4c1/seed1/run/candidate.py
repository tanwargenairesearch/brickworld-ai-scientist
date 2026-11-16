import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

TR = 32
CTX = 16
H1 = 96
H2 = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = int(spec.in_channels)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.trunk = nn.Sequential(
            nn.Conv2d(c, 48, 1), nn.ReLU(),
            nn.Conv2d(48, TR, 3, padding=1), nn.ReLU(),
            nn.Conv2d(TR, TR, 3, padding=1), nn.ReLU())
        self.ctx = nn.Conv2d(TR, CTX, 1)
        cu = c + CTX
        self.l1 = nn.ModuleList([nn.Conv2d(dy * dx * cu, H1, 1) for dy, dx in self.fps])
        self.l2 = nn.ModuleList([nn.Conv2d(H1, H2, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H2, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * TR, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        t = self.trunk(obs)
        u = torch.cat([obs, F.relu(self.ctx(t))], dim=1)
        res = []
        for (dy, dx), l1, l2, out in zip(self.fps, self.l1, self.l2, self.outs):
            q = F.pad(u, (0, dx - 1, 0, dy - 1))
            cells = [q[:, :, i:i + Y, j:j + X] for i in range(dy) for j in range(dx)]
            h = F.relu(l1(torch.cat(cells, dim=1)))
            h = F.relu(l2(h))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([t.mean((2, 3)), t.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
