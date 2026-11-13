import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
STEM = 32


def foot_pool(f, footprints):
    Y, X = f.shape[2], f.shape[3]
    mdy = max(a for a, b in footprints)
    mdx = max(b for a, b in footprints)
    p = F.pad(f, (0, mdx - 1, 0, mdy - 1))
    res = []
    for dy, dx in footprints:
        views = [p[:, :, i:i + Y, j:j + X] for i in range(dy) for j in range(dx)]
        mx = views[0]
        mn = views[0]
        sm = views[0]
        for v in views[1:]:
            mx = torch.maximum(mx, v)
            mn = torch.minimum(mn, v)
            sm = sm + v
        res.append(torch.cat([mx, sm / len(views), mn], dim=1))
    return res


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(a), int(b)) for a, b in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.heads = nn.ModuleList([nn.Conv2d(3 * WIDTH, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        pooled = foot_pool(f, self.footprints)
        outs = [head(q) for head, q in zip(self.heads, pooled)]
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
