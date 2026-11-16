import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
GH = 64
OBS_T = ["id", "fy", "fx", "tr", "r180", "atr", "r90", "r270"]


def fwd_t(t, name):
    if name == "fy":
        return torch.flip(t, [2])
    if name == "fx":
        return torch.flip(t, [3])
    if name == "tr":
        return t.transpose(2, 3)
    if name == "r180":
        return torch.flip(t, [2, 3])
    if name == "atr":
        return torch.flip(t, [2, 3]).transpose(2, 3)
    if name == "r90":
        return torch.rot90(t, 1, [2, 3])
    if name == "r270":
        return torch.rot90(t, -1, [2, 3])
    return t


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        sq = self.Y == self.X
        self.obs_t = [s for s in OBS_T if sq or s in ("id", "fy", "fx", "r180")]
        cin = len(self.obs_t) * int(spec.in_channels) + self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out = nn.Conv2d(WIDTH, self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, self.Y, self.X)
        ys = torch.linspace(-1.0, 1.0, self.Y, device=obs.device, dtype=obs.dtype)
        xs = torch.linspace(-1.0, 1.0, self.X, device=obs.device, dtype=obs.dtype)
        ys = ys.view(1, 1, self.Y, 1).expand(B, 1, self.Y, self.X)
        xs = xs.view(1, 1, 1, self.X).expand(B, 1, self.Y, self.X)
        feats = [fwd_t(obs, s) for s in self.obs_t]
        feats = feats + [m, ys, xs]
        h = self.stem(torch.cat(feats, dim=1))
        x = F.relu(self.c1(h))
        f = F.relu(x + self.c2(x))
        place = self.out(f).reshape(B, -1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
