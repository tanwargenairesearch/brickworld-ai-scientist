import torch
import torch.nn as nn
import torch.nn.functional as F

W = 20
P = 12
HID = 16


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        g = 2.0 ** 0.5
        self.Z = int(spec.Z)
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        cin = 16
        self.body = nn.Sequential(
            init_layer(nn.Conv2d(cin, W, 3, padding=1), g), nn.ReLU(),
            init_layer(nn.Conv2d(W, W, 3, padding=1), g), nn.ReLU())
        self.proj = nn.Sequential(init_layer(nn.Conv2d(W, P, 1), g), nn.ReLU())
        heads = []
        for _ in self.footprints:
            heads.append(nn.Sequential(
                init_layer(nn.Conv2d(3 * P, HID, 1), g), nn.ReLU(),
                init_layer(nn.Conv2d(HID, 1, 1), 0.01)))
        self.type_heads = nn.ModuleList(heads)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * W, 64), g), nn.ReLU())
        self.done_head = init_layer(nn.Linear(64, 1), 0.01)
        self.value_head = init_layer(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        below = F.pad(s, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        above = F.pad(s, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:]
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zpos = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zpos = zpos.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([s, below, above, g3, zpos], dim=2).reshape(B * Z, 16, Y, X)
        f = self.body(x)
        p = self.proj(f)
        outs = []
        for fp, head in zip(self.footprints, self.type_heads):
            dy, dx = fp
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(q, (dy, dx), stride=1)
            av = F.avg_pool2d(q, (dy, dx), stride=1)
            mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, av, mn], dim=1)))
        score = torch.cat(outs, dim=1)
        T = score.shape[1]
        score = score.reshape(B, Z, T, Y, X)
        place = torch.logsumexp(score, dim=1).flatten(1)
        fz = f.reshape(B, Z, W, Y, X)
        pooled = torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
