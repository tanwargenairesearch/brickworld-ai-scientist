import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.conv1 = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.fp_convs = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for (dy, dx) in self.fps])
        self.max_convs = nn.ModuleList([nn.Conv2d(WIDTH, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        x = F.relu(self.conv1(obs))
        c = self.ctx(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        x = F.relu(x + c[:, :, None, None])
        f = F.relu(self.conv2(x))
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            fp = F.pad(f, (0, dx - 1, 0, dy - 1))
            a = self.fp_convs[i](fp)
            m = F.max_pool2d(fp, (dy, dx), stride=1)
            outs.append(a + self.max_convs[i](m))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
