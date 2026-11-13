import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
N_BLOCKS = 2
HEAD = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16}


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=1)
        self.glob = nn.Linear(2 * w, w)

    def forward(self, x):
        h = F.relu(x)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        return x + self.conv(h) + g[:, :, None, None]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem1 = nn.Conv2d(spec.in_channels, WIDTH, 1)
        self.stem2 = nn.Conv2d(WIDTH, WIDTH, 1)
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(N_BLOCKS)])
        self.foot = [(int(dy), int(dx)) for (dy, dx) in spec.footprints]
        self.foot_convs = nn.ModuleList([nn.Conv2d(WIDTH, HEAD, (dy, dx)) for (dy, dx) in self.foot])
        self.foot_out = nn.ModuleList([nn.Conv2d(HEAD, 1, 1) for _ in self.foot])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        x = self.stem2(F.relu(self.stem1(obs)))
        for blk in self.blocks:
            x = blk(x)
        f = F.relu(x)
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
