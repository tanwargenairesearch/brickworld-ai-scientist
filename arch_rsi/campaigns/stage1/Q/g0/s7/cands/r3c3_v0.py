import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
LH = 32
LO = 16


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
        self.Z = int(spec.Z)
        self.footprints = [(int(a), int(b)) for a, b in spec.footprints]
        self.layer_mlp = nn.Sequential(
            nn.Conv2d(12, LH, 1), nn.ReLU(),
            nn.Conv2d(LH, LO, 1), nn.ReLU())
        self.body = nn.Sequential(
            nn.Conv2d(2 * LO, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.heads = nn.ModuleList([nn.Conv2d(3 * WIDTH, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        bt = per[:, :, :2]
        zeros = bt.new_zeros(B, 1, 2, Y, X)
        below = torch.cat([zeros, bt[:, :-1]], dim=1)
        above = torch.cat([bt[:, 1:], zeros], dim=1)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, below, above, gl, zc], dim=2).reshape(B * Z, 12, Y, X)
        h = self.layer_mlp(x).reshape(B, Z, LO, Y, X)
        col = torch.cat([h.amax(1), h.mean(1)], dim=1)
        f = self.body(col)
        pooled = foot_pool(f, self.footprints)
        outs = [head(q) for head, q in zip(self.heads, pooled)]
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
