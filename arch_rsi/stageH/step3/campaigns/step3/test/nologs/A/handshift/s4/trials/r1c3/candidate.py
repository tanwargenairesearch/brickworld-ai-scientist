import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.97, "gae_lambda": 0.95, "lr": 0.001, "ent_coef": 0.01, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32}

W = 48
NBLK = 3
HID = 32
GH = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.ctx = nn.Linear(c, c)

    def forward(self, h):
        g = self.ctx(h.mean((2, 3)))
        return F.relu(h + self.conv(h) + g[:, :, None, None])


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, W, 1)
        self.blocks = nn.ModuleList([Block(W) for _ in range(NBLK)])
        self.fconv = nn.ModuleList([nn.Conv2d(W, HID, (dy, dx)) for dy, dx in self.fps])
        self.fout = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        for b in self.blocks:
            h = b(h)
        res = []
        for (dy, dx), fc, fo in zip(self.fps, self.fconv, self.fout):
            q = F.pad(h, (0, dx - 1, 0, dy - 1))
            res.append(fo(F.relu(fc(q))))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
