import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.95, "gae_lambda": 0.95, "ent_coef": 0.02}

STEM = 64
W = 40
PROJ = 32
GH = 64
NBLOCKS = 3
SQ = 8
FH = 128


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        cin = int(spec.in_channels) + self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.blocks = nn.ModuleList([Block(W) for _ in range(NBLOCKS)])
        self.norm = nn.GroupNorm(1, W)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.out = nn.Conv2d(PROJ, self.T, 3, padding=1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.sq = nn.Conv2d(W, SQ, 1)
        self.fl1 = nn.Linear(SQ * self.Y * self.X, FH)
        self.flm = nn.Linear(self.n_actions, FH)
        self.fl2 = nn.Linear(FH, self.n_actions)
        nn.init.normal_(self.fl2.weight, std=0.01)
        nn.init.zeros_(self.fl2.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        mf = mask.to(obs.dtype)
        m = mf[:, :self.n_place].reshape(B, T, Y, X)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = self.stem(torch.cat([obs, m, yy, xx], dim=1))
        for blk in self.blocks:
            h = blk(h)
        f = F.relu(self.norm(h))
        place = self.out(F.relu(self.proj(f))).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        spatial = torch.cat([place, self.done_head(g)], dim=1)
        fz = F.relu(self.sq(f)).flatten(1)
        fh = F.relu(self.fl1(fz) + self.flm(mf))
        logits = spatial + self.fl2(fh)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
