import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 8, "n_minibatches": 8, "ppo_epochs": 4}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32


class FootHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(2 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(3 * hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        Y = p.shape[2]
        X = p.shape[3]
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            if dy * dx == 1:
                m = p
            else:
                m = F.max_pool2d(q, (dy, dx), stride=1)
            hs = []
            for i in range(dy):
                for j in range(dx):
                    cell = q[:, :, i:i + Y, j:j + X]
                    hs.append(F.relu(mix(torch.cat([cell, m], dim=1))))
            H = torch.stack(hs, dim=0)
            agg = torch.cat([H.amax(0), H.amin(0), H.mean(0)], dim=1)
            res.append(out(agg))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.X = int(spec.X)
        cin = spec.in_channels + 3 * self.Z + 1
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = FootHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        s = self.X // 2
        tgt = torch.cat([obs[:, Z:4 * Z], obs[:, 4 * Z + 1:4 * Z + 2]], dim=1)
        tgt = torch.cat([tgt[..., s:], tgt[..., :s]], dim=3)
        inp = torch.cat([obs, tgt], dim=1)
        h = self.stem(inp)
        x = F.relu(self.c1(h))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
