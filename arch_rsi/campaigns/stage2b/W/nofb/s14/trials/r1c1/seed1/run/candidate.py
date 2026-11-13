import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 44
HEAD = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.n_types = spec.n_types
        self.stem = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.conv1 = nn.Conv2d(W, W, 3, padding=1)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.film = nn.Linear(W, W)
        self.proj = nn.Conv2d(W, self.n_types * HEAD, 1)
        heads = []
        for _ in range(self.n_types):
            out = nn.Conv2d(HEAD, 1, 1)
            nn.init.orthogonal_(out.weight, gain=0.01)
            nn.init.zeros_(out.bias)
            heads.append(nn.Sequential(nn.Conv2d(2 * HEAD, HEAD, 1), nn.ReLU(), out))
        self.type_heads = nn.ModuleList(heads)
        self.done_head = nn.Linear(W, 1)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        h = h + F.relu(self.conv1(h))
        h = h + F.relu(self.conv2(h))
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        h2 = F.relu(h + self.film(g)[:, :, None, None])
        p = self.proj(h2)
        outs = []
        for t in range(self.n_types):
            dy, dx = self.fps[t]
            pt = p[:, t * HEAD:(t + 1) * HEAD]
            pp = F.pad(pt, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            av = F.avg_pool2d(pp, (dy, dx), stride=1)
            outs.append(self.type_heads[t](torch.cat([mx, av], dim=1)).flatten(1))
        place = torch.cat(outs, dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
