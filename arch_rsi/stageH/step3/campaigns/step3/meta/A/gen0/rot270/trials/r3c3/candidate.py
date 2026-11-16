import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 64, "rollout_len": 8, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.95}

W = 48
P = 32
H = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.Y = int(spec.Y)
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + self.nt, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.hp = nn.Conv2d(W, P, 1)
        self.la = nn.ModuleList([nn.Conv2d(P, H, 1) for _ in self.fps])
        self.ca = nn.ModuleList([nn.Conv2d(W, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(2 * H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.nt, Y, X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        hp = self.hp(f)
        res = []
        for (dy, dx), la, ca, out in zip(self.fps, self.la, self.ca, self.outs):
            pad = (0, dx - 1, 0, dy - 1)
            if dy * dx == 1:
                lf = hp
            else:
                lf = F.max_pool2d(F.pad(hp, pad), (dy, dx), stride=1)
            a = la(lf)
            c = F.pad(ca(f), pad)
            hs = []
            for i in range(dy):
                for j in range(dx):
                    hs.append(F.relu(a + c[:, :, i:i + Y, j:j + X]))
            h = torch.stack(hs, dim=0)
            agg = torch.cat([h.amin(0), h.mean(0)], dim=1)
            res.append(out(agg))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
