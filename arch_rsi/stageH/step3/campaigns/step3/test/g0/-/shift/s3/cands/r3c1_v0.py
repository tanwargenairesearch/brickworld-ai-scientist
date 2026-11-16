import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 32, "n_minibatches": 8, "ppo_epochs": 2, "lr": 0.001, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
TH = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.square = self.Y == self.X
        cin = int(spec.in_channels) + self.T
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, TH, 1) for _ in self.fps])
        K = 8 if self.square else 4
        self.final = nn.Conv2d(K * self.T * TH, self.T, 3, padding=1)
        self.act_bias = nn.Parameter(torch.zeros(int(spec.n_actions)))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].reshape(B, self.T, self.Y, self.X).to(obs.dtype)
        s = self.stem(torch.cat([obs, m], 1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(F.relu(out(h)))
        h = torch.cat(res, 1)
        views = [h, h.flip(2), h.flip(3), h.flip(2, 3)]
        if self.square:
            ht = h.transpose(2, 3)
            views = views + [ht, ht.flip(2), ht.flip(3), ht.flip(2, 3)]
        place = self.final(torch.cat(views, 1)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1) + self.act_bias
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
