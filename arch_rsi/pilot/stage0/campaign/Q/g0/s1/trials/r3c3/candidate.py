import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
HEAD = 24
CW = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.fp_convs = nn.ModuleList([nn.Conv2d(WIDTH, HEAD, (dy, dx)) for dy, dx in self.fps])
        self.fp_out = nn.ModuleList([nn.Conv2d(HEAD, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.critic_body = nn.Sequential(
            nn.Conv2d(spec.in_channels, CW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(CW, CW, 3, padding=1), nn.ReLU())
        self.critic_mlp = nn.Sequential(nn.Linear(2 * CW, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        outs = []
        for (dy, dx), conv, out in zip(self.fps, self.fp_convs, self.fp_out):
            h = F.relu(conv(F.pad(f, (0, dx - 1, 0, dy - 1))))
            outs.append(out(h))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        c = self.critic_body(obs)
        cg = self.critic_mlp(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1))
        return logits, self.value_head(cg).squeeze(1)


def build_policy(spec):
    return Net(spec)
