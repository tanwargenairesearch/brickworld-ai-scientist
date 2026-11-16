import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 512, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95, "clip_eps": 0.2}

STEM = 64
WIDTH = 32
BLOCKS = 3
HHID = 32
HID = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.glob = nn.Linear(2 * c, c)

    def forward(self, x):
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        return F.relu(x + self.conv(x) + g[:, :, None, None])


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = spec.in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(BLOCKS)])
        self.fconv = nn.ModuleList([nn.Conv2d(cin + WIDTH, HHID, (dy, dx)) for dy, dx in self.fps])
        self.fout = nn.ModuleList([nn.Conv2d(HHID, 1, 1) for _ in self.fps])
        for o in self.fout:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, HID), nn.ReLU())
        self.done_head = nn.Linear(HID, 1)
        self.value_head = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for b in self.blocks:
            x = b(x)
        z = torch.cat([obs, x], dim=1)
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.fconv, self.fout):
            q = F.pad(z, (0, dx - 1, 0, dy - 1))
            h = F.relu(conv(q))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
