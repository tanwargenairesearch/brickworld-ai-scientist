import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.003}

W = 32
DEPTH = 4


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.glob = nn.Linear(c, c)

    def forward(self, x):
        g = self.glob(x.amax((2, 3)))
        return F.relu(x + self.conv(x) + g[:, :, None, None])


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.npl = int(spec.n_place)
        cin = spec.in_channels + self.nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.blocks = nn.ModuleList([Block(W) for _ in range(DEPTH)])
        self.head = nn.Sequential(
            nn.Conv2d(W, W, 3, padding=1), nn.ReLU(),
            nn.Conv2d(W, self.nt, 1))
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        m = mask[:, :self.npl].float().view(B, self.nt, Y, X)
        x = self.stem(torch.cat([obs, m], dim=1))
        for blk in self.blocks:
            x = blk(x)
        place = self.head(x).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
