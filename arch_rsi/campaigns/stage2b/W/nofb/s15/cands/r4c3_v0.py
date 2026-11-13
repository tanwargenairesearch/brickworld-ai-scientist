import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
DILATIONS = [1, 2, 1, 2]

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "gamma": 0.99, "gae_lambda": 0.95}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in DILATIONS])
        self.ctxs = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in DILATIONS])
        self.heads = nn.ModuleList([nn.Conv2d(3 * WIDTH, 1, 1) for _ in range(spec.n_types)])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for conv, ctx in zip(self.blocks, self.ctxs):
            c = ctx(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
            f = f + F.relu(conv(f + c[:, :, None, None]))
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            av = F.avg_pool2d(p, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
