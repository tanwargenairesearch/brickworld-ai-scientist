import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

# (type_minor, transpose, flip_y, flip_x)
LAYOUTS = [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
DONE_FIRST = [0]
OBS_VIEWS = ["id"]
HEAD_K = 5
STEM = 64
WIDTH = 40
HEADC = 32
GLOB = 64
MASK_EMB = 16


def read_layout(m, lay, t, y, x):
    minor, tr, fy, fx = lay
    b = m.shape[0]
    if tr:
        a0, a1 = x, y
    else:
        a0, a1 = y, x
    if minor:
        g = m.reshape(b, a0, a1, t).permute(0, 3, 1, 2)
    else:
        g = m.reshape(b, t, a0, a1)
    if tr:
        g = g.transpose(2, 3)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def write_layout(g, lay):
    minor, tr, fy, fx = lay
    if fx:
        g = g.flip(3)
    if fy:
        g = g.flip(2)
    if tr:
        g = g.transpose(2, 3)
    if minor:
        g = g.permute(0, 2, 3, 1)
    return g.reshape(g.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.ny = int(spec.Y)
        self.nx = int(spec.X)
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.cin_obs = int(spec.in_channels)
        self.grid_ok = self.n_place == self.nt * self.ny * self.nx
        self.views = [v for v in OBS_VIEWS if v != "T" or self.nx == self.ny]
        self.combos = [(d, lay) for d in DONE_FIRST for lay in LAYOUTS]
        if self.grid_ok:
            n_mask = len(self.combos) * self.nt
        else:
            n_mask = MASK_EMB
            self.mask_emb = nn.Linear(self.n_actions, MASK_EMB * self.ny * self.nx)
        cin = len(self.views) * self.cin_obs + n_mask + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hk = nn.Conv2d(WIDTH, HEADC, HEAD_K, padding=HEAD_K // 2)
        self.hs = nn.Conv2d(WIDTH, HEADC, 1)
        if self.grid_ok:
            self.outs = nn.Conv2d(HEADC, len(self.combos) * self.nt, 1)
        else:
            self.fc_place = nn.Linear(HEADC * self.ny * self.nx, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, len(DONE_FIRST))
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        ny, nx, nt, npl = self.ny, self.nx, self.nt, self.n_place
        ins = []
        for v in self.views:
            if v == "id":
                ins.append(obs)
            elif v == "T":
                ins.append(obs.transpose(2, 3))
            else:
                ins.append(obs.reshape(b, ny, nx, self.cin_obs).permute(0, 3, 1, 2))
        m = mask.to(obs.dtype)
        if self.grid_ok:
            for d, lay in self.combos:
                if d:
                    mp = m[:, 1:1 + npl]
                else:
                    mp = m[:, :npl]
                ins.append(read_layout(mp, lay, nt, ny, nx))
        else:
            ins.append(self.mask_emb(m).view(b, MASK_EMB, ny, nx))
        cy = torch.linspace(-1.0, 1.0, ny, device=obs.device, dtype=obs.dtype).view(1, 1, ny, 1).expand(b, 1, ny, nx)
        cx = torch.linspace(-1.0, 1.0, nx, device=obs.device, dtype=obs.dtype).view(1, 1, 1, nx).expand(b, 1, ny, nx)
        ins.append(cy)
        ins.append(cx)
        z = torch.cat(ins, 1)
        f = self.stem(z)
        f = F.relu(f + self.r1(f))
        f = F.relu(f + self.r2(f))
        h = F.relu(self.hk(f) + self.hs(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        dn = self.done_head(g)
        if self.grid_ok:
            maps = self.outs(h)
            logits = None
            for j, d in enumerate(DONE_FIRST):
                pl = None
                for i, (dd, lay) in enumerate(self.combos):
                    if dd != d:
                        continue
                    part = write_layout(maps[:, i * nt:(i + 1) * nt], lay)
                    pl = part if pl is None else pl + part
                dj = dn[:, j:j + 1]
                if d:
                    full = torch.cat([dj, pl], 1)
                else:
                    full = torch.cat([pl, dj], 1)
                logits = full if logits is None else logits + full
        else:
            logits = torch.cat([self.fc_place(h.flatten(1)), dn[:, :1]], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
