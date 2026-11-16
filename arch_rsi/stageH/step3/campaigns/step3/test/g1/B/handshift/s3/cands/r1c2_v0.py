import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

# (type_minor, transposed, y_flip, x_flip)
MASK_LAYOUTS = [
    (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 1),
    (0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 0, 1), (0, 1, 1, 1),
    (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (1, 0, 1, 1),
    (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 0, 1), (1, 1, 1, 1),
]
# (transposed, y_flip, x_flip)
OBS_LAYOUTS = [(0, 0, 0)]
HEAD_K = 3
DONE_FIRST_HEDGE = False
STEM = 64
WIDTH = 40
PROJ = 32


def read_layout(m, K, Y, X, lay):
    minor, trans, fy, fx = lay
    B = m.shape[0]
    if trans:
        a, c = X, Y
    else:
        a, c = Y, X
    if minor:
        g = m.reshape(B, a, c, K).permute(0, 3, 1, 2)
    else:
        g = m.reshape(B, K, a, c)
    if trans:
        g = g.transpose(2, 3)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def write_layout(o, lay):
    minor, trans, fy, fx = lay
    if fy:
        o = o.flip(2)
    if fx:
        o = o.flip(3)
    if trans:
        o = o.transpose(2, 3)
    if minor:
        o = o.permute(0, 2, 3, 1)
    return o.reshape(o.shape[0], -1)


def obs_layout(obs, lay):
    trans, fy, fx = lay
    if trans:
        obs = obs.transpose(2, 3)
    if fy:
        obs = obs.flip(2)
    if fx:
        obs = obs.flip(3)
    return obs


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        yx = self.Y * self.X
        self.grid_ok = self.n_place > 0 and self.n_place % yx == 0
        self.K = self.n_place // yx if self.grid_ok else 4
        self.obs_lays = [l for l in OBS_LAYOUTS if self.Y == self.X or l[0] == 0]
        self.groups = [0, 1] if DONE_FIRST_HEDGE else [0]
        L = len(MASK_LAYOUTS)
        if self.grid_ok:
            mask_ch = L * self.K * len(self.groups)
        else:
            self.mask_embed = nn.Linear(self.n_actions, 4 * yx)
            mask_ch = 4
        cin = int(spec.in_channels) * len(self.obs_lays) + mask_ch + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hk = nn.Conv2d(WIDTH, PROJ, HEAD_K, padding=HEAD_K // 2)
        self.h1 = nn.Conv2d(WIDTH, PROJ, 1)
        if self.grid_ok:
            self.outs = nn.ModuleList([nn.Conv2d(PROJ, self.K, 1) for _ in range(L * len(self.groups))])
        else:
            self.out_map = nn.Conv2d(PROJ, 4, 1)
            self.out_lin = nn.Linear(4 * yx, self.n_actions - 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_heads = nn.ModuleList([nn.Linear(64, 1) for _ in self.groups])
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        m = mask.to(obs.dtype)
        if self.grid_ok:
            feats = []
            for off in self.groups:
                mp = m[:, off:off + self.n_place]
                for lay in MASK_LAYOUTS:
                    feats.append(read_layout(mp, self.K, Y, X, lay))
            mfeat = torch.cat(feats, 1)
        else:
            mfeat = self.mask_embed(m).reshape(B, 4, Y, X)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        obs_feats = [obs_layout(obs, l) for l in self.obs_lays]
        x = self.stem(torch.cat(obs_feats + [mfeat, ys, xs], 1))
        x = F.relu(x + self.r1(x))
        f = F.relu(x + self.r2(x))
        h = F.relu(self.hk(f) + self.h1(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        if self.grid_ok:
            logits = 0
            i = 0
            for gi, off in enumerate(self.groups):
                place = 0
                for lay in MASK_LAYOUTS:
                    place = place + write_layout(self.outs[i](h), lay)
                    i += 1
                d = self.done_heads[gi](g)
                if off == 0:
                    lg = torch.cat([place, d], 1)
                else:
                    lg = torch.cat([d, place], 1)
                logits = logits + lg
        else:
            place = self.out_lin(F.relu(self.out_map(h)).flatten(1))
            logits = torch.cat([place, self.done_heads[0](g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
