import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1}

WIDTH = 48
COLW = 32
POOL = 32
HEAD = 64


def fp_max(x, dy, dx):
    if dy == 1 and dx == 1:
        return x
    x = F.pad(x, (0, dx - 1, 0, dy - 1))
    return F.max_pool2d(x, (dy, dx), stride=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        T = spec.n_types
        self.conv1 = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.conv3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.col = nn.Conv2d(spec.in_channels, COLW, 1)
        fd = WIDTH + COLW
        self.pre = nn.Conv2d(fd, 2 * POOL, 1)
        self.head1 = nn.Conv2d(T * 2 * POOL, T * HEAD, 1, groups=T)
        self.head2 = nn.Conv2d(T * HEAD, T, 1, groups=T)
        self.glob = nn.Sequential(nn.Linear(2 * fd, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.conv1(obs))
        f = f + F.relu(self.conv2(f))
        f = f + F.relu(self.conv3(f))
        c = F.relu(self.col(obs))
        f = torch.cat([f, c], dim=1)
        p = self.pre(f)
        a = p[:, :POOL]
        b = p[:, POOL:]
        feats = []
        for (dy, dx) in self.fps:
            feats.append(fp_max(a, dy, dx))
            feats.append(-fp_max(-b, dy, dx))
        z = torch.cat(feats, dim=1)
        place = self.head2(F.relu(self.head1(z))).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
