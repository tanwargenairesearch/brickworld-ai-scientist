import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 40
HEAD = 32
HEAD_K = 3
FREE = False
LAYOUTS = [(1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1)]
OFFSETS = [0, 1]
RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}


def to_grid(flat, lay, k, ny, nx):
    tm, tr, fy, fx = lay
    b = flat.shape[0]
    s1, s2 = (nx, ny) if tr else (ny, nx)
    if tm:
        g = flat.reshape(b, k, s1, s2)
    else:
        g = flat.reshape(b, s1, s2, k).permute(0, 3, 1, 2)
    if tr:
        g = g.transpose(2, 3)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def from_grid(g, lay):
    tm, tr, fy, fx = lay
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    if tr:
        g = g.transpose(2, 3)
    if not tm:
        g = g.permute(0, 2, 3, 1)
    return g.reshape(g.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X = int(spec.Y), int(spec.X)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        yx = self.Y * self.X
        self.grid_ok = self.n_place > 0 and self.n_place % yx == 0
        self.k = self.n_place // yx if self.grid_ok else 0
        if self.grid_ok:
            self.offsets = [o for o in OFFSETS if o + self.n_place < self.n_actions or (o == 0 and self.n_place < self.n_actions)]
            n_mask = len(LAYOUTS) * len(self.offsets) * self.k
        else:
            self.offsets = [0]
            self.mask_embed = nn.Linear(self.n_actions, 4 * yx)
            n_mask = 4
        cin = spec.in_channels + n_mask + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hk = nn.Conv2d(WIDTH, HEAD, HEAD_K, padding=HEAD_K // 2)
        self.hs = nn.Conv2d(WIDTH, HEAD, 1)
        if self.grid_ok:
            self.outs = nn.ModuleList([nn.Conv2d(HEAD, len(LAYOUTS) * self.k, 1) for _ in self.offsets])
        else:
            self.fb_proj = nn.Conv2d(HEAD, 4, 1)
            self.fb_out = nn.Linear(4 * yx, self.n_place)
        self.scale = (4.0 / len(LAYOUTS)) ** 0.5
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done = nn.ModuleList([nn.Linear(64, 1) for _ in self.offsets])
        self.value_head = nn.Linear(64, 1)
        if FREE:
            self.free_g = nn.Linear(64, self.n_actions)
            self.free_m = nn.Linear(self.n_actions, self.n_actions)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).contiguous())

    def forward(self, obs, mask):
        b = obs.shape[0]
        m = mask.float()
        feats = [obs]
        if self.grid_ok:
            for o in self.offsets:
                mp = m[:, o:o + self.n_place]
                for lay in LAYOUTS:
                    feats.append(to_grid(mp, lay, self.k, self.Y, self.X))
        else:
            feats.append(self.mask_embed(m).view(b, 4, self.Y, self.X))
        feats.append(self.coords.expand(b, -1, -1, -1))
        x = self.stem(torch.cat(feats, 1))
        x = F.relu(x + self.r1(x))
        f = F.relu(x + self.r2(x))
        h = F.relu(self.hk(f) + self.hs(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        rest = self.n_actions - self.n_place - 1
        logits = None
        if self.grid_ok:
            for o, out, dh in zip(self.offsets, self.outs, self.done):
                maps = out(h)
                place = None
                for i, lay in enumerate(LAYOUTS):
                    p = from_grid(maps[:, i * self.k:(i + 1) * self.k], lay)
                    place = p if place is None else place + p
                place = place * self.scale
                d = dh(g)
                pad = place.new_zeros(b, rest)
                if o == 0:
                    vec = torch.cat([place, pad, d], 1)
                else:
                    vec = torch.cat([d, place, pad], 1)
                logits = vec if logits is None else logits + vec
        else:
            place = self.fb_out(self.fb_proj(h).flatten(1))
            logits = torch.cat([place, place.new_zeros(b, rest), self.done[0](g)], 1)
        if FREE:
            logits = logits + self.free_g(g) + self.free_m(m)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
