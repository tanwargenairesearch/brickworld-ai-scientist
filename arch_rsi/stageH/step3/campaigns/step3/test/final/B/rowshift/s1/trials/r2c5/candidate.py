import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 40
MEMB = 8
DH = 16
RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


class CLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X = int(spec.Y), int(spec.X)
        self.n_place = int(spec.n_place)
        cells = self.Y * self.X
        self.obs_stem = nn.Conv2d(spec.in_channels + 2, STEM, 1)
        self.obs_ln = CLN(STEM)
        self.mask_emb = nn.Linear(self.n_place, MEMB * cells)
        self.mask_ln = CLN(MEMB)
        self.stem2 = nn.Conv2d(STEM + MEMB, WIDTH, 1)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(2)])
        self.res_ln = nn.ModuleList([CLN(WIDTH) for _ in range(2)])
        self.ctx = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in range(2)])
        self.hred = nn.Conv2d(WIDTH, DH, 1)
        self.hout = nn.Linear(DH * cells, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        for m in (self.hout, self.done_head, self.value_head):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).contiguous())

    def forward(self, obs, mask):
        B = obs.shape[0]
        coords = self.coords.expand(B, -1, -1, -1)
        o = F.relu(self.obs_ln(self.obs_stem(torch.cat([obs, coords], 1))))
        m = self.mask_emb(mask[:, :self.n_place].float()).reshape(B, MEMB, self.Y, self.X)
        m = F.relu(self.mask_ln(m))
        f = F.relu(self.stem2(torch.cat([o, m], 1)))
        for i in range(len(self.res)):
            u = self.res[i](self.res_ln[i](f))
            c = self.ctx[i](torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
            f = F.relu(f + u + c[:, :, None, None])
        h = F.relu(self.hred(f)).flatten(1)
        place = self.hout(h)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
