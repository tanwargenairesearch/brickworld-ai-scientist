import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEADW = 32
LAYOUTS = [(False, False, False, False)]


def to_grid(v, t, y, x, lay):
    minor, trans, fy, fx = lay
    b = v.shape[0]
    if minor:
        if trans:
            g = v.reshape(b, x, y, t).permute(0, 3, 2, 1)
        else:
            g = v.reshape(b, y, x, t).permute(0, 3, 1, 2)
    else:
        if trans:
            g = v.reshape(b, t, x, y).transpose(2, 3)
        else:
            g = v.reshape(b, t, y, x)
    dims = []
    if fy:
        dims.append(2)
    if fx:
        dims.append(3)
    if dims:
        g = torch.flip(g, dims)
    return g


def from_grid(g, lay):
    minor, trans, fy, fx = lay
    b = g.shape[0]
    dims = []
    if fy:
        dims.append(2)
    if fx:
        dims.append(3)
    if dims:
        g = torch.flip(g, dims)
    if minor:
        if trans:
            g = g.permute(0, 3, 2, 1)
        else:
            g = g.permute(0, 2, 3, 1)
    elif trans:
        g = g.transpose(2, 3)
    return g.reshape(b, -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.ny = int(spec.Y)
        self.nx = int(spec.X)
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        cin0 = int(spec.in_channels)
        cells = self.ny * self.nx
        self.grid_ok = self.n_place == self.nt * cells
        if self.grid_ok:
            self.lays = list(LAYOUTS)
            n_mask = len(self.lays) * self.nt
            n_out = len(self.lays)
        else:
            self.lays = []
            n_mask = 8
            self.mask_emb = nn.Linear(self.n_actions, 8 * cells)
            n_out = 1
            self.place_fc = nn.Linear(self.nt * cells, self.n_place)
        cin = cin0 + n_mask + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hk = nn.Conv2d(WIDTH, HEADW, 3, padding=1)
        self.hs = nn.Conv2d(WIDTH, HEADW, 1)
        self.outs = nn.ModuleList([nn.Conv2d(HEADW, self.nt, 1) for _ in range(n_out)])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        ny, nx, nt = self.ny, self.nx, self.nt
        m = mask.to(obs.dtype)
        feats = [obs]
        if self.grid_ok:
            v = m[:, :self.n_place]
            for lay in self.lays:
                feats.append(to_grid(v, nt, ny, nx, lay))
        else:
            feats.append(self.mask_emb(m).reshape(b, 8, ny, nx))
        yy = torch.linspace(-1.0, 1.0, ny, device=obs.device, dtype=obs.dtype).view(1, 1, ny, 1).expand(b, 1, ny, nx)
        xx = torch.linspace(-1.0, 1.0, nx, device=obs.device, dtype=obs.dtype).view(1, 1, 1, nx).expand(b, 1, ny, nx)
        feats.append(yy)
        feats.append(xx)
        s = self.stem(torch.cat(feats, 1))
        f = F.relu(s + self.r1(s))
        f = F.relu(f + self.r2(f))
        h = F.relu(self.hk(f) + self.hs(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        value = self.value_head(g).squeeze(1)
        if self.grid_ok:
            place = None
            for li, lay in enumerate(self.lays):
                p = from_grid(self.outs[li](h), lay)
                place = p if place is None else place + p
        else:
            place = self.place_fc(self.outs[0](h).reshape(b, -1))
        logits = torch.cat([place, self.done_head(g)], 1)
        if logits.shape[1] < self.n_actions:
            logits = F.pad(logits, (0, self.n_actions - logits.shape[1]))
        return logits, value


def build_policy(spec):
    return Net(spec)
