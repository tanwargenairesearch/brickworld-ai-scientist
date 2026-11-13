import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"gamma": 0.99, "gae_lambda": 0.95}

WIDTH = 64
HEAD = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        c_in = int(spec.in_channels)
        self.conv1 = nn.Conv2d(c_in, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fp_hid = nn.ModuleList([nn.Conv2d(WIDTH + c_in, HEAD, (dy, dx)) for dy, dx in self.fps])
        self.fp_out = nn.ModuleList([nn.Conv2d(HEAD, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.conv2(F.relu(self.conv1(obs))))
        fr = torch.cat([f, obs], dim=1)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            h = F.relu(self.fp_hid[i](F.pad(fr, (0, dx - 1, 0, dy - 1))))
            outs.append(self.fp_out[i](h))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
