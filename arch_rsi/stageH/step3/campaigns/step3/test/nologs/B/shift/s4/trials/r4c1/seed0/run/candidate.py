import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.003}

STEM = 96
WIDTH = 48
HID = 32
NB = 2


def small_init(m):
    nn.init.normal_(m.weight, std=0.01)
    nn.init.zeros_(m.bias)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1))
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(NB)])
        self.norm = nn.GroupNorm(1, WIDTH)
        self.mask_in = nn.Conv2d(self.T, HID, 1)
        self.convs = nn.ModuleList([nn.Conv2d(WIDTH, HID, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            small_init(o)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        small_init(self.done_head)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        x = self.blocks(self.stem(obs))
        f = F.relu(self.norm(x))
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, Y, X)
        mh = self.mask_in(m)
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.convs, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(conv(q) + mh)))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
