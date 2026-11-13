import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
PROJ = 32
HID = 32

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 6}


class FootprintHead(nn.Module):
    def __init__(self, c, c_obs, c_glob, footprints):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in footprints]
        self.proj = nn.Conv2d(c, PROJ, 1)
        self.skip = nn.Conv2d(c_obs, PROJ, 1)
        self.first = nn.ModuleList([nn.Conv2d(6 * PROJ, HID, 1) for _ in self.fps])
        self.ctx = nn.ModuleList([nn.Linear(c_glob, HID) for _ in self.fps])
        self.last = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])

    def forward(self, f, obs, gvec):
        g = F.relu(torch.cat([self.proj(f), self.skip(obs)], dim=1))
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            if dy > 1 or dx > 1:
                p = F.pad(g, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(p, (dy, dx), stride=1)
                mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            else:
                mx = g
                mn = g
            h = self.first[i](torch.cat([g, mx, mn], dim=1))
            h = F.relu(h + self.ctx[i](gvec)[:, :, None, None])
            outs.append(self.last[i](h))
        return torch.cat(outs, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.place_head = FootprintHead(WIDTH, spec.in_channels, WIDTH, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.place_head(f, obs, g).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
