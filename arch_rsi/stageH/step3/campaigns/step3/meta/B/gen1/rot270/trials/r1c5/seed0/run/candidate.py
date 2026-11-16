import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

WIDTH = 40
STEM = 64
HEADC = 32
HEAD_K = 3
DONE_FIRST = False
DENSE = True
MASK_SKIP = False
# layout = (type_minor, transposed, y_flip, x_flip)
LAYOUTS = [
    (False, False, False, False), (False, False, False, True), (False, False, True, False), (False, False, True, True),
    (False, True, False, False), (False, True, False, True), (False, True, True, False), (False, True, True, True),
    (True, False, False, False), (True, False, False, True), (True, False, True, False), (True, False, True, True),
    (True, True, False, False), (True, True, False, True), (True, True, True, False), (True, True, True, True)]


def to_grid(m, lay, T, Y, X):
    minor, trans, fy, fx = lay
    B = m.shape[0]
    if minor:
        if trans:
            g = m.reshape(B, X, Y, T).permute(0, 3, 2, 1)
        else:
            g = m.reshape(B, Y, X, T).permute(0, 3, 1, 2)
    else:
        if trans:
            g = m.reshape(B, T, X, Y).transpose(2, 3)
        else:
            g = m.reshape(B, T, Y, X)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def from_grid(g, lay):
    minor, trans, fy, fx = lay
    B = g.shape[0]
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    if minor:
        if trans:
            o = g.permute(0, 3, 2, 1)
        else:
            o = g.permute(0, 2, 3, 1)
    else:
        if trans:
            o = g.transpose(2, 3)
        else:
            o = g
    return o.reshape(B, -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_act = int(spec.n_actions)
        self.n_place = self.n_act - 1
        hw = self.Y * self.X
        self.grid = self.n_place > 0 and self.n_place % hw == 0
        self.T = self.n_place // hw if self.grid else 0
        mch = len(LAYOUTS) * self.T if self.grid else 0
        self.mch = mch
        cin = int(spec.in_channels) + 2 + mch
        self.use_emb = DENSE or not self.grid
        if self.use_emb:
            self.mask_emb = nn.Linear(self.n_place, 16)
            cin += 16
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        hin = WIDTH + (mch if MASK_SKIP else 0)
        self.hconv = nn.Conv2d(hin, HEADC, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(hin, HEADC, 1)
        if self.grid:
            self.outs = nn.ModuleList([nn.Conv2d(HEADC, self.T, 1) for _ in LAYOUTS])
        self.use_fc = DENSE or not self.grid
        if self.use_fc:
            self.fc_red = nn.Conv2d(HEADC, 8, 1)
            self.fc_out = nn.Linear(8 * hw, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        if DONE_FIRST:
            m = mask[:, 1:].to(obs.dtype)
        else:
            m = mask[:, :self.n_place].to(obs.dtype)
        parts = [obs, ys, xs]
        grids = []
        if self.grid:
            for lay in LAYOUTS:
                grids.append(to_grid(m, lay, self.T, Y, X))
            parts = parts + grids
        if self.use_emb:
            e = self.mask_emb(m)
            parts.append(e.view(B, -1, 1, 1).expand(B, e.shape[1], Y, X))
        h = self.stem(torch.cat(parts, 1))
        h = F.relu(h + self.r1(h))
        f = F.relu(h + self.r2(h))
        hin = f
        if MASK_SKIP and self.grid:
            hin = torch.cat([f] + grids, 1)
        hh = F.relu(self.hconv(hin) + self.hskip(hin))
        place = None
        if self.grid:
            for lay, out in zip(LAYOUTS, self.outs):
                p = from_grid(out(hh), lay)
                place = p if place is None else place + p
        if self.use_fc:
            p = self.fc_out(F.relu(self.fc_red(hh)).flatten(1))
            place = p if place is None else place + p
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        d = self.done_head(g)
        if DONE_FIRST:
            logits = torch.cat([d, place], 1)
        else:
            logits = torch.cat([place, d], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
