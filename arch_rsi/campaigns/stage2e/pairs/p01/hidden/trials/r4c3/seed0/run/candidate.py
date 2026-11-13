import torch
import torch.nn as nn
import torch.nn.functional as F

W = 48
H = 32
RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.97, "lr": 0.001}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.foots = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.cell = nn.Sequential(nn.Conv2d(spec.in_channels, W, 1), nn.ReLU(), nn.Conv2d(W, W, 1), nn.ReLU())
        self.sp1 = nn.Conv2d(W, W, 3, padding=1)
        self.sp2 = nn.Conv2d(W, W, 3, padding=2, dilation=2)
        self.sp3 = nn.Conv2d(W, W, 3, padding=4, dilation=4)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.film = nn.Linear(W, W)
        self.pre = nn.Conv2d(W, W, 1)
        heads = []
        for _ in self.foots:
            last = nn.Conv2d(H, 1, 1)
            nn.init.orthogonal_(last.weight, 0.01)
            nn.init.zeros_(last.bias)
            heads.append(nn.Sequential(nn.Conv2d(3 * W, H, 1), nn.ReLU(), last))
        self.heads = nn.ModuleList(heads)
        self.done_head = nn.Linear(W, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        c = self.cell(obs)
        h = F.relu(self.sp1(c))
        h = F.relu(self.sp2(h) + c)
        h = F.relu(self.sp3(h) + h)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        h2 = h + self.film(g)[:, :, None, None]
        p = self.pre(h2)
        outs = []
        for i, (dy, dx) in enumerate(self.foots):
            if dy == 1 and dx == 1:
                mx = p
                mn = p
            else:
                pp = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(pp, (dy, dx), stride=1)
                mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            outs.append(self.heads[i](torch.cat([mx, mn, h2], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
