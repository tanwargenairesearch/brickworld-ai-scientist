import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 16
HID = 32


class MultiPoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in footprints]
        self.geoms = []
        for (dy, dx) in fps:
            shapes = [(dy, dx)] if dy == dx else [(dy, dx), (dx, dy)]
            g = []
            for (a, b) in shapes:
                for oy in sorted({0, a - 1}):
                    for ox in sorted({0, b - 1}):
                        g.append((a, b, oy, ox))
            self.geoms.append(g)
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(2 * proj * len(g), hid, 1) for g in self.geoms])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.geoms])

    def forward(self, f):
        p = self.proj(f)
        Y = p.shape[2]
        X = p.shape[3]
        res = []
        for g, mix, out in zip(self.geoms, self.mix, self.outs):
            feats = []
            for (a, b, oy, ox) in g:
                if a * b == 1:
                    feats.append(p)
                    feats.append(p)
                    continue
                q = F.pad(p, (b - 1, b - 1, a - 1, a - 1))
                mx = F.max_pool2d(q, (a, b), stride=1)[:, :, oy:oy + Y, ox:ox + X]
                av = F.avg_pool2d(q, (a, b), stride=1)[:, :, oy:oy + Y, ox:ox + X]
                feats.append(mx)
                feats.append(av)
            h = F.relu(mix(torch.cat(feats, dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.npl = int(spec.n_place)
        cin = spec.in_channels + self.nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = MultiPoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        m = mask[:, :self.npl].to(obs.dtype).reshape(B, self.nt, Y, X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
