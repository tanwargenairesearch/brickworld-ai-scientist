import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "lr": 0.002}

WIDTH = 64
PW = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        c = spec.in_channels
        self.stem = nn.Sequential(nn.Conv2d(c, WIDTH, 1), nn.ReLU(), nn.Conv2d(WIDTH, WIDTH, 1), nn.ReLU())
        self.ctx = nn.Sequential(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.proj = nn.Conv2d(WIDTH, PW, 1)
        self.type_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PW, PW, 1), nn.ReLU(), nn.Conv2d(PW, 1, 1))
            for _ in self.footprints
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        h = self.stem(obs)
        g = self.ctx(h) + h
        q = self.proj(g)
        outs = []
        for (dy, dx), head in zip(self.footprints, self.type_heads):
            qp = F.pad(q, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(qp, kernel_size=(dy, dx), stride=1)
            mn = -F.max_pool2d(-qp, kernel_size=(dy, dx), stride=1)
            z = torch.cat([q, mx, mn], dim=1)
            outs.append(head(z).flatten(1))
        place = torch.cat(outs, dim=1)
        gl = self.glob(torch.cat([g.mean((2, 3)), g.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
