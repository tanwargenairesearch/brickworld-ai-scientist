import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_types = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.conv1 = nn.Conv2d(spec.in_channels + self.n_types, WIDTH, 3, padding=1)
        self.bcast = nn.Linear(2 * WIDTH, WIDTH)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.cell = nn.Conv2d(WIDTH, HID, 1)
        self.type_mix = nn.ModuleList([nn.Conv2d(3 * HID, HID, 1) for _ in self.fps])
        self.type_out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.n_types, Y, X)
        x = torch.cat([obs, m], dim=1)
        f = F.relu(self.conv1(x))
        g0 = self.bcast(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = F.relu(f + g0[:, :, None, None])
        f = F.relu(self.conv2(f))
        h = self.cell(f)
        outs = []
        for (dy, dx), mix, out in zip(self.fps, self.type_mix, self.type_out):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            av = F.avg_pool2d(hp, (dy, dx), stride=1)
            z = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            outs.append(out(z).flatten(1))
        place = torch.cat(outs, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
