import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
MCH = 8
RED = 16
GH = 64
OUT_GAIN = 0.01


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.n = ChanLN(c)
        self.g = nn.Linear(2 * c, c)

    def forward(self, f):
        y = self.conv(f) + self.g(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))[:, :, None, None]
        return F.relu(self.n(f + y))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        cells = self.Y * self.X
        self.obs_stem = nn.Sequential(nn.Conv2d(int(spec.in_channels) + 2, STEM, 1), ChanLN(STEM), nn.ReLU())
        self.mask_emb = nn.Linear(self.n_place, MCH * cells)
        self.mask_ln = ChanLN(MCH)
        self.stem2 = nn.Sequential(nn.Conv2d(STEM + MCH, WIDTH, 1), ChanLN(WIDTH), nn.ReLU())
        self.blocks = nn.ModuleList([Block(WIDTH, 1), Block(WIDTH, 1)])
        self.red = nn.Sequential(nn.Conv2d(WIDTH, RED, 1), ChanLN(RED), nn.ReLU())
        self.place_out = nn.Linear(RED * cells, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        for m in (self.place_out, self.done_head, self.value_head):
            nn.init.orthogonal_(m.weight, gain=OUT_GAIN)
            nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        o = self.obs_stem(torch.cat([obs, ys, xs], dim=1))
        m = mask[:, :self.n_place].to(obs.dtype)
        me = F.relu(self.mask_ln(self.mask_emb(m).reshape(B, MCH, Y, X)))
        f = self.stem2(torch.cat([o, me], dim=1))
        for blk in self.blocks:
            f = blk(f)
        place = self.place_out(self.red(f).flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
