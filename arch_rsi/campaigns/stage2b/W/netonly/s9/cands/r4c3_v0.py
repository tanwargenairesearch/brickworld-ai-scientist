import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 52
PROJ = 32
HID = 32


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        g = 2.0 ** 0.5
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        T = len(self.footprints)
        self.body = nn.Sequential(
            init_layer(nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), g), nn.ReLU(),
            init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), g), nn.ReLU())
        self.proj = nn.Sequential(init_layer(nn.Conv2d(WIDTH, PROJ, 1), g), nn.ReLU())
        hids, outs = [], []
        for _ in self.footprints:
            hids.append(nn.Sequential(init_layer(nn.Conv2d(3 * PROJ, HID, 1), g), nn.ReLU()))
            outs.append(init_layer(nn.Conv2d(HID, 1, 1), 0.01))
        self.type_hid = nn.ModuleList(hids)
        self.type_out = nn.ModuleList(outs)
        self.mix = init_layer(nn.Conv2d(T * HID, T, 3, padding=1), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, 64), g), nn.ReLU())
        self.done_head = init_layer(nn.Linear(64, 1), 0.01)
        self.value_head = init_layer(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        f = self.body(obs)
        p = self.proj(f)
        hs, outs = [], []
        for fp, hid, out in zip(self.footprints, self.type_hid, self.type_out):
            dy, dx = fp
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(q, (dy, dx), stride=1)
            av = F.avg_pool2d(q, (dy, dx), stride=1)
            mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = hid(torch.cat([mx, av, mn], dim=1))
            hs.append(h)
            outs.append(out(h))
        place = torch.cat(outs, dim=1) + self.mix(torch.cat(hs, dim=1))
        place = place.flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
