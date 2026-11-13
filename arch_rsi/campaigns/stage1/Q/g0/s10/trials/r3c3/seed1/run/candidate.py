import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HEAD = 16
LAYERS = 2


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.conv1 = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(LAYERS)])
        self.ctxs = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in range(LAYERS)])
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.pre = nn.Conv2d(WIDTH, HEAD, 1)
        self.fp_convs = nn.ModuleList()
        self.pool_heads = nn.ModuleList()
        for (dy, dx) in self.fps:
            self.fp_convs.append(nn.Conv2d(WIDTH, 1, (dy, dx)))
            self.pool_heads.append(nn.Sequential(
                nn.Conv2d(2 * HEAD, HEAD, 1), nn.ReLU(), nn.Conv2d(HEAD, 1, 1)))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        h = F.relu(self.conv1(obs))
        for conv, ctx in zip(self.convs, self.ctxs):
            c = ctx(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
            h = F.relu(conv(h) + c[:, :, None, None])
        f = h
        p = self.pre(f)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            pad = (0, dx - 1, 0, dy - 1)
            a = self.fp_convs[i](F.pad(f, pad))
            pp = F.pad(p, pad)
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            b = self.pool_heads[i](torch.cat([mx, mn], dim=1))
            outs.append((a + b).flatten(1))
        place = torch.cat(outs, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
