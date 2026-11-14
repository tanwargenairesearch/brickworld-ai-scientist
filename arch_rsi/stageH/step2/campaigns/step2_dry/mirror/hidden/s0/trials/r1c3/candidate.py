import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 32
P = 16
HID = 16

RECIPE = {"ppo_epochs": 3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.lay_local = nn.Conv2d(12, P, 1)
        self.lay_col = nn.Conv2d(WIDTH + COL, P, 1)
        self.lay_emb = nn.Parameter(torch.zeros(self.Z, P))
        self.lay2 = nn.Conv2d(P, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        vp = F.pad(vox, (0, 0, 0, 0, 1, 1))
        loc = torch.cat([vp[:, :, 0:Z], vp[:, :, 1:Z + 1], vp[:, :, 2:Z + 2]], dim=1)
        loc = loc.permute(0, 2, 1, 3, 4).reshape(B * Z, 12, Y, X)
        a = self.lay_local(loc).view(B, Z, P, Y, X)
        c = self.lay_col(torch.cat([f, s], dim=1)).unsqueeze(1)
        e = self.lay_emb.view(1, Z, P, 1, 1)
        h = F.relu(a + c + e).reshape(B * Z, P, Y, X)
        p = self.lay2(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            sc = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            sc = sc.reshape(B, Z, Y * X)
            res.append(torch.logsumexp(sc, dim=1))
        place = torch.cat(res, dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
