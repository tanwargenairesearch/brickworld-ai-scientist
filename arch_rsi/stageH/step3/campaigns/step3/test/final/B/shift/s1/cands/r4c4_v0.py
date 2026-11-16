import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 32
NH = 2
STEM = 64
WIDTH = 40
HEAD = 32
GH = 64
VGRAD = 0.01
RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        hw = self.Y * self.X
        self.hw = hw
        self.K = (self.n_place + hw - 1) // hw
        c_obs = int(spec.in_channels) + 2
        self.ostem = nn.Sequential(nn.Conv2d(c_obs, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, D, 1), nn.ReLU())
        self.opos = nn.Embedding(hw, D)
        self.apos = nn.Embedding(hw, D)
        self.mstem = nn.Linear(self.K, D)
        self.ln_o = nn.LayerNorm(D)
        self.ln_a = nn.LayerNorm(D)
        self.q = nn.Linear(D, D)
        self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D)
        self.proj = nn.Linear(D, D)
        self.direct = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.ReLU(), nn.Linear(2 * D, D))
        self.toc = nn.Conv2d(D + 2, WIDTH, 1)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        self.out = nn.Conv2d(HEAD, self.K, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2 * D, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K, hw = self.Y, self.X, self.K, self.hw
        dh = D // NH
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        of = self.ostem(torch.cat([obs, ys, xs], 1))
        idx = torch.arange(hw, device=obs.device)
        ot = of.flatten(2).transpose(1, 2) + self.opos(idx).unsqueeze(0)
        mp = mask[:, :self.n_place].to(obs.dtype)
        pad = K * hw - self.n_place
        if pad > 0:
            mp = F.pad(mp, (0, pad))
        mp = mp.reshape(B, K, hw).transpose(1, 2)
        at = self.mstem(mp) + self.apos(idx).unsqueeze(0)
        on = self.ln_o(ot)
        an = self.ln_a(at)
        q = self.q(an).reshape(B, hw, NH, dh).transpose(1, 2)
        k = self.k(on).reshape(B, hw, NH, dh).transpose(1, 2)
        v = self.v(on).reshape(B, hw, NH, dh).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        z = torch.matmul(att, v).transpose(1, 2).reshape(B, hw, D)
        h = at + self.proj(z) + self.direct(ot)
        h = h + self.mlp(self.ln2(h))
        hg = h.transpose(1, 2).reshape(B, D, Y, X)
        x = F.relu(self.toc(torch.cat([hg, ys, xs], 1)))
        x = F.relu(x + self.r1(x))
        x = F.relu(x + self.r2(x))
        hh = F.relu(self.h3(x) + self.h1(x))
        place = self.out(hh).reshape(B, -1)[:, :self.n_place]
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3)), of.mean((2, 3)), of.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        u = self.value_head(g).squeeze(1)
        value = u.detach() + VGRAD * (u - u.detach())
        return logits, value


def build_policy(spec):
    return Net(spec)
