import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
H = 32
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.layer_in = nn.Conv2d(6, H, 1)
        self.ctx = nn.Conv2d(WIDTH, H, 1)
        self.layer_emb = nn.Embedding(self.Z, H)
        self.mix = nn.Sequential(nn.ReLU(), nn.Conv2d(H, H, 1), nn.ReLU())
        self.footprints = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * H, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.footprints
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Yn, Xn = obs.shape
        Z = self.Z
        f = self.body(obs)
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        build_below = F.pad(build[:, :Z - 1], (0, 0, 0, 0, 1, 0), value=1.0)
        tgt_below = F.pad(tgt[:, :Z - 1], (0, 0, 0, 0, 1, 0), value=1.0)
        raw = torch.stack([build, tgt, miss, extra, build_below, tgt_below], dim=2)
        raw = raw.reshape(B * Z, 6, Yn, Xn)
        h = self.layer_in(raw).view(B, Z, H, Yn, Xn)
        h = h + self.ctx(f).unsqueeze(1) + self.layer_emb.weight.view(1, Z, H, 1, 1)
        h = self.mix(h.reshape(B * Z, H, Yn, Xn))
        outs = []
        for (dy, dx), head in zip(self.footprints, self.heads):
            if dy == 1 and dx == 1:
                mx = h
                mn = h
            else:
                p = F.pad(h, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(p, kernel_size=(dy, dx), stride=1)
                mn = -F.max_pool2d(-p, kernel_size=(dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn], dim=1)))
        per_layer = torch.cat(outs, dim=1).view(B, Z, len(self.footprints), Yn, Xn)
        place = torch.logsumexp(per_layer, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
