import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEADW = 32
HEAD_K = 3
WIDE = True
DONE_HEDGE = False
DENSE = 4


def make_layouts(wide):
    if not wide:
        return [(False, False, False, False), (False, True, False, False),
                (False, False, True, False), (False, False, False, True)]
    lays = []
    for minor in (False, True):
        for trans in (False, True):
            for fy in (False, True):
                for fx in (False, True):
                    lays.append((minor, trans, fy, fx))
    return lays


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
        offs = [0]
        if DONE_HEDGE and self.grid_ok and self.n_actions >= self.n_place + 1:
            offs = [0, 1]
        self.offs = offs
        self.lays = make_layouts(WIDE) if self.grid_ok else []
        if self.grid_ok:
            n_mask = len(offs) * len(self.lays) * self.nt
            n_out = len(offs) * len(self.lays)
        else:
            n_mask = 8
            self.mask_emb = nn.Linear(self.n_actions, 8 * cells)
            n_out = 1
            self.place_fc = nn.Linear(self.nt * cells, self.n_place)
        cin = cin0 + n_mask + 2
        if DENSE > 0:
            self.dense = nn.Linear(cin0 * cells, DENSE * cells)
            cin += DENSE
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hk = nn.Conv2d(WIDTH, HEADW, HEAD_K, padding=HEAD_K // 2)
        self.hs = nn.Conv2d(WIDTH, HEADW, 1)
        self.outs = nn.ModuleList([nn.Conv2d(HEADW, self.nt, 1) for _ in range(n_out)])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_heads = nn.ModuleList([nn.Linear(64, 1) for _ in offs])
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        ny, nx, nt = self.ny, self.nx, self.nt
        m = mask.to(obs.dtype)
        feats = [obs]
        if self.grid_ok:
            for off in self.offs:
                v = m[:, off:off + self.n_place]
                for lay in self.lays:
                    feats.append(to_grid(v, nt, ny, nx, lay))
        else:
            feats.append(self.mask_emb(m).reshape(b, 8, ny, nx))
        if DENSE > 0:
            feats.append(self.dense(obs.reshape(b, -1)).reshape(b, DENSE, ny, nx))
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
        logits = None
        if self.grid_ok:
            nl = len(self.lays)
            for oi, off in enumerate(self.offs):
                place = None
                for li, lay in enumerate(self.lays):
                    p = from_grid(self.outs[oi * nl + li](h), lay)
                    place = p if place is None else place + p
                done = self.done_heads[oi](g)
                if off == 0:
                    part = torch.cat([place, done], 1)
                else:
                    part = torch.cat([done, place], 1)
                if part.shape[1] < self.n_actions:
                    part = F.pad(part, (0, self.n_actions - part.shape[1]))
                logits = part if logits is None else logits + part
        else:
            place = self.place_fc(self.outs[0](h).reshape(b, -1))
            part = torch.cat([place, self.done_heads[0](g)], 1)
            if part.shape[1] < self.n_actions:
                part = F.pad(part, (0, self.n_actions - part.shape[1]))
            logits = part
        return logits, value


def build_policy(spec):
    return Net(spec)
