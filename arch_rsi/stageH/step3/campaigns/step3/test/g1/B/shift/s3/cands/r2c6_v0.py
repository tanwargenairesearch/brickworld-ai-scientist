import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = [("TYX", 0, 0), ("TYX", 0, 1), ("TYX", 1, 0), ("TYX", 1, 1), ("TXY", 0, 0), ("TXY", 0, 1), ("TXY", 1, 0), ("TXY", 1, 1), ("YXT", 0, 0), ("YXT", 0, 1), ("YXT", 1, 0), ("YXT", 1, 1), ("XYT", 0, 0), ("XYT", 0, 1), ("XYT", 1, 0), ("XYT", 1, 1), ("YTX", 0, 0), ("YTX", 0, 1), ("YTX", 1, 0), ("YTX", 1, 1), ("XTY", 0, 0), ("XTY", 0, 1), ("XTY", 1, 0), ("XTY", 1, 1)]
OBS_ORDERS = ["CYX", "CXY", "YXC", "XYC", "YCX", "XCY"]
AXIAL = True
DENSE = True
HEAD_K = 3
STEM = 64
WIDTH = 40
HEADC = 32
RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}


def decode(v, lay, T, Y, X):
    order, fy, fx = lay
    dims = {"T": T, "Y": Y, "X": X}
    g = v.reshape(v.shape[0], dims[order[0]], dims[order[1]], dims[order[2]])
    g = g.permute(0, 1 + order.index("T"), 1 + order.index("Y"), 1 + order.index("X"))
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def encode(m, lay):
    order, fy, fx = lay
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    m = m.permute(0, 1 + "TYX".index(order[0]), 1 + "TYX".index(order[1]), 1 + "TYX".index(order[2]))
    return m.reshape(m.shape[0], -1)


def obs_view(o, order, C, Y, X):
    if order == "CYX":
        return o
    dims = {"C": C, "Y": Y, "X": X}
    g = o.reshape(o.shape[0], dims[order[0]], dims[order[1]], dims[order[2]])
    return g.permute(0, 1 + order.index("C"), 1 + order.index("Y"), 1 + order.index("X"))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.T = int(spec.Y), int(spec.X), int(spec.n_types)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.ok = (self.n_place == self.T * self.Y * self.X) and (self.n_actions == self.n_place + 1)
        self.use_dense = DENSE or not self.ok
        cin = self.C * len(OBS_ORDERS) + 2
        if self.ok:
            cin += len(LAYOUTS) * self.T
        if self.use_dense:
            self.mask_embed = nn.Linear(self.n_actions, 4 * self.Y * self.X)
            cin += 4
        ys = torch.linspace(-1, 1, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1, 1, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).clone())
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        if AXIAL:
            self.ax = nn.ModuleList([nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 1), nn.Conv2d(WIDTH, WIDTH, 1), nn.Conv2d(WIDTH, WIDTH, 1)]) for _ in range(3)])
        self.hconv = nn.Conv2d(WIDTH, HEADC, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEADC, 1)
        if self.ok:
            self.outs = nn.Conv2d(HEADC, len(LAYOUTS) * self.T, 1)
        if self.use_dense:
            self.fb_red = nn.Conv2d(HEADC, 16, 1)
            self.fb_out = nn.Linear(16 * self.Y * self.X, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def mix(self, f, i):
        if not AXIAL:
            return f
        a, b, c = self.ax[i]
        return F.relu(f + a(f.mean(3, keepdim=True)) + b(f.mean(2, keepdim=True)) + c(f.amax((2, 3), keepdim=True)))

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask.float()
        feats = [obs_view(obs, o, self.C, self.Y, self.X) for o in OBS_ORDERS]
        if self.ok:
            v = m[:, :self.n_place]
            for lay in LAYOUTS:
                feats.append(decode(v, lay, self.T, self.Y, self.X))
        if self.use_dense:
            feats.append(self.mask_embed(m).reshape(B, 4, self.Y, self.X))
        feats.append(self.coords.expand(B, -1, -1, -1))
        x = torch.cat(feats, 1)
        x = F.relu(self.stem1(x))
        x = F.relu(self.stem2(x))
        x = self.mix(x, 0)
        f = F.relu(x + self.r1(x))
        f = self.mix(f, 1)
        f = F.relu(f + self.r2(f))
        f = self.mix(f, 2)
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        d = self.done_head(g)
        p = None
        if self.ok:
            maps = self.outs(h).split(self.T, dim=1)
            for k, lay in enumerate(LAYOUTS):
                e = encode(maps[k], lay)
                p = e if p is None else p + e
        if self.use_dense:
            q = self.fb_out(F.relu(self.fb_red(h)).flatten(1))
            p = q if p is None else p + q
        logits = torch.cat([p, d], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
