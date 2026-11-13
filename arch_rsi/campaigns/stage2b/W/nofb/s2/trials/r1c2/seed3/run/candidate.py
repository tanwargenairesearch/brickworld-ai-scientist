import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3}

WIDTH = 48
HID = 16
BLOCKS = 3


class FootHead(nn.Module):
    def __init__(self, c, hidden, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c, hidden, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hidden, 1, 1) for _ in self.fps])

    def forward(self, f):
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.convs, self.outs):
            g = F.pad(f, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(conv(g))))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(BLOCKS)])
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.place_head = FootHead(WIDTH, HID, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        x = F.relu(self.stem(obs))
        for i, conv in enumerate(self.blocks):
            x = x + F.relu(conv(x))
            if i == 0:
                c = self.ctx(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
                x = x + F.relu(c)[:, :, None, None]
        place = self.place_head(x).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
