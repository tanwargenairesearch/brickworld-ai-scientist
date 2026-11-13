import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
N_BLOCKS = 3
HEAD = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 4, "lr": 0.001, "gamma": 0.99, "gae_lambda": 0.95}


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=1)
        self.ctx = nn.Linear(2 * w, w)

    def forward(self, f):
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        h = self.conv(f) + self.ctx(g)[:, :, None, None]
        return f + F.relu(h)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [tuple(fp) for fp in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(N_BLOCKS)])
        self.proj = nn.Conv2d(WIDTH, HEAD, 1)
        self.type_hidden = nn.ModuleList([nn.Conv2d(3 * HEAD, HEAD, 1) for _ in self.footprints])
        self.type_out = nn.ModuleList([nn.Conv2d(HEAD, 1, 1) for _ in self.footprints])
        for m in self.type_out:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for blk in self.blocks:
            f = blk(f)
        p = F.relu(self.proj(f))
        outs = []
        for i, (dy, dx) in enumerate(self.footprints):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(q, (dy, dx), stride=1)
            av = F.avg_pool2d(q, (dy, dx), stride=1)
            mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(self.type_hidden[i](torch.cat([mx, av, mn], dim=1)))
            outs.append(self.type_out[i](h))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
