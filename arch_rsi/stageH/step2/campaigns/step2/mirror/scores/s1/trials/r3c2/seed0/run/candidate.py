import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.003}

STEM = 64
COL = 48
WIDTH = 40
HEADC = 32
M = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.npl = int(spec.n_place)
        self.square = int(spec.Y) == int(spec.X)
        ng = 8 if self.square else 4
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = nn.Sequential(
            nn.Conv2d(WIDTH, HEADC, 3, padding=1), nn.ReLU(),
            nn.Conv2d(HEADC, M, 1))
        self.mixer = nn.Conv2d(M * ng + self.nt, HEADC, 1)
        self.out = nn.Conv2d(HEADC, self.nt, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        h = self.head(f)
        maps = [h, h.flip(2), h.flip(3), h.flip(2).flip(3)]
        if self.square:
            ht = h.transpose(2, 3)
            maps = maps + [ht, ht.flip(2), ht.flip(3), ht.flip(2).flip(3)]
        m = mask[:, :self.npl].float().view(B, self.nt, Y, X)
        z = F.relu(self.mixer(torch.cat(maps + [m], dim=1)))
        place = self.out(z).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
