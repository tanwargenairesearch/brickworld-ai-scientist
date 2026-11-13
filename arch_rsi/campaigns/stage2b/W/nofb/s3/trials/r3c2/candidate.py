import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 44
DEPTH = 3
HEAD = 16

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "gae_lambda": 0.95, "gamma": 0.99}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(DEPTH - 1)])
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.type_hidden = nn.ModuleList([nn.Conv2d(2 * WIDTH, HEAD, 1) for _ in self.fps])
        self.type_out = nn.ModuleList([nn.Conv2d(HEAD, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for conv in self.blocks:
            f = F.relu(f + conv(f))
        outs = []
        for fp, hid, out in zip(self.fps, self.type_hidden, self.type_out):
            dy, dx = fp
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            av = F.avg_pool2d(p, (dy, dx), stride=1)
            h = F.relu(hid(torch.cat([mx, av], dim=1)))
            outs.append(out(h))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
