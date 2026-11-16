import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]
DONE_POS = [1, 0]
HEAD_K = 5
HEAD_W = 24
MASK_TO_HEAD = True
STEM = 64
WIDTH = 40


def read_layout(v, lay, T, Y, X):
    tm, tr, fy, fx = lay
    B = v.shape[0]
    if tm:
        if tr:
            m = v.reshape(B, X, Y, T).permute(0, 3, 2, 1)
        else:
            m = v.reshape(B, Y, X, T).permute(0, 3, 1, 2)
    else:
        if tr:
            m = v.reshape(B, T, X, Y).transpose(2, 3)
        else:
            m = v.reshape(B, T, Y, X)
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    return m


def write_layout(m, lay):
    tm, tr, fy, fx = lay
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    if tm:
        if tr:
            m = m.permute(0, 3, 2, 1)
        else:
            m = m.permute(0, 2, 3, 1)
    else:
        if tr:
            m = m.transpose(2, 3)
    return m.reshape(m.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.n_act = int(spec.n_actions)
        self.ok = (self.n_place == self.T * self.Y * self.X) and (self.n_act == self.n_place + 1)
        self.combos = [(d, tuple(l)) for d in DONE_POS for l in LAYOUTS]
        if self.ok:
            self.m_in = len(self.combos) * self.T
        else:
            self.m_in = 8
            self.mask_embed = nn.Linear(self.n_act, self.m_in * self.Y * self.X)
        cin = int(spec.in_channels) + self.m_in + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hconv = nn.Conv2d(WIDTH, HEAD_W, HEAD_K, padding=HEAD_K // 2)
        skip_in = WIDTH + (self.m_in if MASK_TO_HEAD else 0)
        self.hskip = nn.Conv2d(skip_in, HEAD_W, 1)
        if self.ok:
            self.outs = nn.Conv2d(HEAD_W, len(self.combos) * self.T, 1)
            n_done = len(DONE_POS)
        else:
            self.out_conv = nn.Conv2d(HEAD_W, 4, 1)
            self.out_lin = nn.Linear(4 * self.Y * self.X, self.n_act - 1)
            n_done = 1
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, n_done)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        mf = mask.to(obs.dtype)
        if self.ok:
            maps = []
            for d, lay in self.combos:
                if d == 1:
                    v = mf[:, :self.n_place]
                else:
                    v = mf[:, 1:]
                maps.append(read_layout(v, lay, T, Y, X))
            mm = torch.cat(maps, dim=1)
        else:
            mm = self.mask_embed(mf).view(B, self.m_in, Y, X)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        s = self.stem(torch.cat([obs, mm, ys, xs], dim=1))
        h = F.relu(s + self.r1(s))
        f = F.relu(h + self.r2(h))
        if MASK_TO_HEAD:
            sk = self.hskip(torch.cat([f, mm], dim=1))
        else:
            sk = self.hskip(f)
        z = F.relu(self.hconv(f) + sk)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        dn = self.done_head(g)
        value = self.value_head(g).squeeze(1)
        if self.ok:
            o = self.outs(z)
            logits = None
            for i, (d, lay) in enumerate(self.combos):
                flat = write_layout(o[:, i * T:(i + 1) * T], lay)
                j = DONE_POS.index(d)
                if d == 1:
                    li = torch.cat([flat, dn[:, j:j + 1]], dim=1)
                else:
                    li = torch.cat([dn[:, j:j + 1], flat], dim=1)
                logits = li if logits is None else logits + li
        else:
            flat = self.out_lin(F.relu(self.out_conv(z)).flatten(1))
            logits = torch.cat([flat, dn[:, :1]], dim=1)
        return logits, value


def build_policy(spec):
    return Net(spec)
