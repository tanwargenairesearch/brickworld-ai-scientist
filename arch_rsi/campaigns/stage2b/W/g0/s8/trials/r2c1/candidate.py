import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 8}

WIDTH = 40
N_RES = 5
CTX_AT = (1, 3)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.stem = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(N_RES)])
        self.ctx = nn.ModuleList([nn.Linear(2 * W, W) for _ in CTX_AT])
        self.place_head = nn.Conv2d(W, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def pool(self, h):
        return torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        j = 0
        for i, conv in enumerate(self.convs):
            if i in CTX_AT:
                h = h + self.ctx[j](self.pool(h))[:, :, None, None]
                j += 1
            h = F.relu(h + conv(h))
        place = self.place_head(h).flatten(1)
        g = self.glob(self.pool(h))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
