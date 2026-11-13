import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "n_minibatches": 16}

W = 64
H = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, W, 1), nn.ReLU(),
            nn.Conv2d(W, W, 1), nn.ReLU())
        self.ctx = nn.Conv2d(W, W, 3, padding=1)
        heads = []
        for _ in self.fps:
            out = nn.Conv2d(H, 1, 1)
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            heads.append(nn.Sequential(nn.Conv2d(2 * W, H, 1), nn.ReLU(), out))
        self.heads = nn.ModuleList(heads)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.done_head = nn.Linear(W, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        h = self.stem(obs)
        f = F.relu(h + self.ctx(h))
        outs = []
        for fp, head in zip(self.fps, self.heads):
            dy, dx = fp
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn], dim=1)).flatten(1))
        place = torch.cat(outs, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
