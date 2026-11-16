import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

CONV = True
MLP = False
REMAP = True
ORDERS = ["major"]
DONE_FIRST = False
OBS_SYM = True
STEM = 64
WIDTH = 40
HEAD = 32
GLOB = 64
HID = 320
REMAP_IN = 8
REMAP_OUT = 16


def sym(g, t, fy, fx):
    if t:
        g = g.transpose(2, 3)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def read_grid(mp, order, k, ny, nx):
    b = mp.shape[0]
    if order == "minor":
        return mp.reshape(b, ny, nx, k).permute(0, 3, 1, 2)
    if order == "middle":
        return mp.reshape(b, ny, k, nx).permute(0, 2, 1, 3)
    return mp.reshape(b, k, ny, nx)


def write_flat(o, order):
    b = o.shape[0]
    if order == "minor":
        return o.permute(0, 2, 3, 1).reshape(b, -1)
    if order == "middle":
        return o.permute(0, 2, 1, 3).reshape(b, -1)
    return o.reshape(b, -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.ny = int(spec.Y)
        self.nx = int(spec.X)
        self.c_in = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_extra = int(spec.n_actions) - self.n_place
        cells = self.ny * self.nx
        sq = self.ny == self.nx
        self.grid_mask = (self.n_place % cells) == 0
        self.k = max(1, self.n_place // cells)
        self.orders = list(ORDERS) if self.grid_mask else ["major"]
        allsyms = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)]
        if sq:
            allsyms = allsyms + [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
        self.syms = allsyms if OBS_SYM else [(0, 0, 0)]
        self.use_conv = CONV or not MLP
        self.use_mlp = MLP
        if self.use_conv:
            no = len(self.orders)
            cin = self.c_in * len(self.syms) + 2
            if self.grid_mask:
                mch = self.k * no
                nout = self.k * no
            else:
                mch = 8
                nout = 8
                self.mask_embed = nn.Linear(self.n_place, 8 * cells)
                self.place_out = nn.Linear(8 * cells, self.n_place)
            self.mch = mch
            self.obs_stem = nn.Conv2d(cin, STEM, 1)
            self.mask_stem = nn.Conv2d(mch, STEM, 1)
            if REMAP:
                self.rm_in = nn.Conv2d(self.c_in, REMAP_IN, 1)
                self.rm_lin = nn.Linear(REMAP_IN * cells, REMAP_OUT * cells)
                self.rm_stem = nn.Conv2d(REMAP_OUT, STEM, 1)
            self.col = nn.Conv2d(STEM, WIDTH, 1)
            self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(2)])
            self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
            self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
            self.hout = nn.Conv2d(HEAD, nout, 1)
            self.glob = nn.Linear(2 * WIDTH, GLOB)
            self.done_head = nn.Linear(GLOB, self.n_extra)
            self.value_head = nn.Linear(GLOB, 1)
        if self.use_mlp:
            nin = self.c_in * cells + self.n_place
            self.m1 = nn.Linear(nin, HID)
            self.mln = nn.LayerNorm(HID)
            self.m2 = nn.Linear(HID, HID)
            self.m_place = nn.Linear(HID, self.n_place)
            self.m_done = nn.Linear(HID, self.n_extra)
            self.m_value = nn.Linear(HID, 1)
            nn.init.orthogonal_(self.m_place.weight, 0.01)
            nn.init.zeros_(self.m_place.bias)
            nn.init.orthogonal_(self.m_done.weight, 0.01)
            nn.init.zeros_(self.m_done.bias)

    def forward(self, obs, mask):
        b = obs.shape[0]
        ny = self.ny
        nx = self.nx
        mf = mask.to(obs.dtype)
        if DONE_FIRST:
            mp = mf[:, self.n_extra:self.n_extra + self.n_place]
        else:
            mp = mf[:, :self.n_place]
        place = None
        done = None
        value = None
        if self.use_conv:
            views = [sym(obs, t, fy, fx) for (t, fy, fx) in self.syms]
            yy = torch.linspace(-1.0, 1.0, ny, device=obs.device, dtype=obs.dtype).view(1, 1, ny, 1).expand(b, 1, ny, nx)
            xx = torch.linspace(-1.0, 1.0, nx, device=obs.device, dtype=obs.dtype).view(1, 1, 1, nx).expand(b, 1, ny, nx)
            xo = torch.cat(views + [yy, xx], dim=1)
            if self.grid_mask:
                xm = torch.cat([read_grid(mp, od, self.k, ny, nx) for od in self.orders], dim=1)
            else:
                xm = self.mask_embed(mp).reshape(b, self.mch, ny, nx)
            s = self.obs_stem(xo) + self.mask_stem(xm)
            if REMAP:
                r = F.relu(self.rm_in(obs)).reshape(b, -1)
                r = F.relu(self.rm_lin(r)).reshape(b, REMAP_OUT, ny, nx)
                s = s + self.rm_stem(r)
            s = F.relu(s)
            f = F.relu(self.col(s))
            for conv in self.res:
                f = F.relu(f + conv(f))
            h = F.relu(self.h3(f) + self.h1(f))
            o = self.hout(h)
            if self.grid_mask:
                for i, od in enumerate(self.orders):
                    c = write_flat(o[:, i * self.k:(i + 1) * self.k], od)
                    place = c if place is None else place + c
            else:
                place = self.place_out(o.reshape(b, -1))
            g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
            done = self.done_head(g)
            value = self.value_head(g).squeeze(1)
        if self.use_mlp:
            z = torch.cat([obs.reshape(b, -1), mp], dim=1)
            z = F.relu(self.mln(self.m1(z)))
            z = F.relu(self.m2(z))
            mpl = self.m_place(z)
            md = self.m_done(z)
            mv = self.m_value(z).squeeze(1)
            place = mpl if place is None else place + mpl
            done = md if done is None else done + md
            value = mv if value is None else value + mv
        if DONE_FIRST:
            logits = torch.cat([done, place], dim=1)
        else:
            logits = torch.cat([place, done], dim=1)
        return logits, value


def build_policy(spec):
    return Net(spec)
