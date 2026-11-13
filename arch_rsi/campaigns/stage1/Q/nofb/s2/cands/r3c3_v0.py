import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
W2 = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.down = nn.Conv2d(WIDTH, W2, 3, stride=2, padding=1)
        self.mid_glob = nn.Linear(2 * W2, W2)
        self.mid = nn.Conv2d(W2, W2, 3, padding=1)
        self.up = nn.Conv2d(W2, WIDTH, 1)
        self.out_conv = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.fp_conv = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for dy, dx in self.fps])
        self.fp_max = nn.ModuleList([nn.Conv2d(WIDTH, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2 * W2, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        s = F.relu(self.stem(obs))
        d = F.relu(self.down(s))
        gd = self.mid_glob(torch.cat([d.mean((2, 3)), d.amax((2, 3))], dim=1))
        m = F.relu(self.mid(F.relu(d + gd[:, :, None, None])))
        m = m + d
        u = F.interpolate(m, size=(Y, X), mode='nearest')
        h = F.relu(s + self.up(u))
        h = h + F.relu(self.out_conv(h))
        outs = []
        for (dy, dx), conv, mx in zip(self.fps, self.fp_conv, self.fp_max):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            pooled = F.max_pool2d(hp, (dy, dx), stride=1)
            outs.append(conv(hp) + mx(pooled))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3)), m.mean((2, 3)), m.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
