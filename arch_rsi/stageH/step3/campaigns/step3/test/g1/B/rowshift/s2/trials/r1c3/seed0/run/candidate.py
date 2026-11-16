import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HP = 32
HK = 5
MEMB = 8
OFFSETS = [0]
DENSE = False
OBS_SYM = False
LAYOUTS = [("TYX", ()), ("TXY", ()), ("TYX", (2,)), ("TYX", (3,))]


def to_grid(v, order, flips, T, Y, X):
    B = v.shape[0]
    if order == "TYX":
        g = v.reshape(B, T, Y, X)
    elif order == "TXY":
        g = v.reshape(B, T, X, Y).transpose(2, 3)
    elif order == "YXT":
        g = v.reshape(B, Y, X, T).permute(0, 3, 1, 2)
    else:
        g = v.reshape(B, X, Y, T).permute(0, 3, 2, 1)
    if len(flips) > 0:
        g = g.flip(list(flips))
    return g


def from_grid(m, order, flips):
    if len(flips) > 0:
        m = m.flip(list(flips))
    if order == "TYX":
        out = m
    elif order == "TXY":
        out = m.transpose(2, 3)
    elif order == "YXT":
        out = m.permute(0, 2, 3, 1)
    else:
        out = m.permute(0, 3, 2, 1)
    return out.reshape(m.shape[0], -1)


def coords(obs, Y, X):
    B = obs.shape[0]
    ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([ys, xs], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.grid_ok = (self.n_place == self.T * self.Y * self.X) and (self.n_actions == self.n_place + 1)
        self.obs_views = 1
        if OBS_SYM:
            self.obs_views = 4 if self.Y == self.X else 3
        cin = spec.in_channels * self.obs_views + 2
        if self.grid_ok:
            cin += len(OFFSETS) * len(LAYOUTS) * self.T
        else:
            self.mask_embed = nn.Linear(self.n_actions, MEMB * self.Y * self.X)
            cin += MEMB
        if DENSE:
            self.dense_in = nn.Linear(self.n_actions, MEMB * self.Y * self.X)
            cin += MEMB
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hconv = nn.Conv2d(WIDTH, HP, HK, padding=HK // 2)
        self.hskip = nn.Conv2d(WIDTH, HP, 1)
        if self.grid_ok:
            self.outs = nn.ModuleList([nn.Conv2d(HP, self.T, 1) for _ in range(len(OFFSETS) * len(LAYOUTS))])
            self.done_heads = nn.ModuleList([nn.Linear(64, 1) for _ in OFFSETS])
        else:
            self.out_proj = nn.Conv2d(HP, 4, 1)
            self.out_lin = nn.Linear(4 * self.Y * self.X, self.n_actions - 1)
            self.done_heads = nn.ModuleList([nn.Linear(64, 1)])
        if DENSE:
            self.dense_proj = nn.Conv2d(HP, 4, 1)
            self.dense_out = nn.Linear(4 * self.Y * self.X, self.n_actions)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask.float()
        if self.obs_views == 1:
            parts = [obs]
        elif self.obs_views == 4:
            parts = [obs, obs.flip(2), obs.flip(3), obs.transpose(2, 3)]
        else:
            parts = [obs, obs.flip(2), obs.flip(3)]
        parts.append(coords(obs, self.Y, self.X))
        if self.grid_ok:
            for off in OFFSETS:
                mp = m[:, off:off + self.n_place]
                for o, fl in LAYOUTS:
                    parts.append(to_grid(mp, o, fl, self.T, self.Y, self.X))
        else:
            parts.append(self.mask_embed(m).view(B, MEMB, self.Y, self.X))
        if DENSE:
            parts.append(self.dense_in(m).view(B, MEMB, self.Y, self.X))
        x = self.stem(torch.cat(parts, dim=1))
        x = F.relu(x + self.r1(x))
        f = F.relu(x + self.r2(x))
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        if self.grid_ok:
            logits = None
            k = 0
            for i, off in enumerate(OFFSETS):
                place = 0
                for o, fl in LAYOUTS:
                    place = place + from_grid(self.outs[k](h), o, fl)
                    k += 1
                d = self.done_heads[i](g)
                if off == 0:
                    part = torch.cat([place, d], dim=1)
                else:
                    part = torch.cat([d, place], dim=1)
                logits = part if logits is None else logits + part
        else:
            place = self.out_lin(self.out_proj(h).flatten(1))
            logits = torch.cat([place, self.done_heads[0](g)], dim=1)
        if DENSE:
            logits = logits + self.dense_out(self.dense_proj(h).flatten(1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
