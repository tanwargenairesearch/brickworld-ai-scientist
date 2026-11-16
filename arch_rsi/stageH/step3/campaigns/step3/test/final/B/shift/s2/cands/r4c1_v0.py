import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEADW = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        hw = self.Y * self.X
        self.hw = hw
        self.K = (self.n_place + hw - 1) // hw
        K = self.K
        self.stem = nn.Sequential(nn.Conv2d(int(spec.in_channels) + 2 + K, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1))
        self.mmix = nn.Linear(hw, hw)
        self.mstem = nn.Conv2d(K, WIDTH, 1)
        self.tmix = nn.Linear(hw, hw)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEADW, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADW, 1)
        self.amix = nn.Linear(hw, hw)
        self.hout = nn.Conv2d(HEADW, K, 1)
        self.hout_m = nn.Conv2d(HEADW, K, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K, hw = self.Y, self.X, self.K, self.hw
        mp = mask[:, :self.n_place].to(obs.dtype)
        if K * hw > self.n_place:
            mp = F.pad(mp, (0, K * hw - self.n_place))
        mflat = mp.reshape(B, K, hw)
        mg = mflat.reshape(B, K, Y, X)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        mm = self.mmix(mflat).reshape(B, K, Y, X)
        h = F.relu(self.stem(torch.cat([obs, ys, xs, mg], 1)) + self.mstem(mm))
        t = self.tmix(h.reshape(B, WIDTH, hw)).reshape(B, WIDTH, Y, X)
        h = F.relu(h + t)
        h = F.relu(h + self.r1(h))
        h = F.relu(h + self.r2(h))
        a = F.relu(self.h3(h) + self.h1(h))
        am = self.amix(a.reshape(B, HEADW, hw)).reshape(B, HEADW, Y, X)
        o = self.hout(a) + self.hout_m(am)
        place = o.reshape(B, K * hw)[:, :self.n_place]
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
