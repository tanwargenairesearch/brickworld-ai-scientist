import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
HEAD = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.foot = [(int(dy), int(dx)) for (dy, dx) in spec.footprints]
        self.foot_convs = nn.ModuleList([nn.Conv2d(WIDTH, HEAD, (dy, dx)) for (dy, dx) in self.foot])
        self.foot_out = nn.ModuleList([nn.Conv2d(HEAD, 1, 1) for _ in self.foot])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        outs = []
        for (dy, dx), conv, out in zip(self.foot, self.foot_convs, self.foot_out):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            outs.append(out(F.relu(conv(p))))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
