import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.002, 'n_envs': 128, 'rollout_len': 16, 'n_minibatches': 8, 'ppo_epochs': 2, 'ent_coef': 0.002, 'gamma': 0.9, 'gae_lambda': 0.85}

LAYOUTS = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1), (2, 0, 0, 0), (2, 0, 0, 1), (2, 0, 1, 0), (2, 0, 1, 1), (2, 1, 0, 0), (2, 1, 0, 1), (2, 1, 1, 0), (2, 1, 1, 1)]
DONE_POS = [1]
OBS_ALT = False
DENSE = False
SEP_VALUE = False
GLOBAL_MIX = False
DILS = [1, 1]
HEAD_K = 3
HEAD_W = 32
STEM = 64
WIDTH = 40


def read_grid(v, lay, T, Y, X):
    tm, tr, fy, fx = lay
    B = v.shape[0]
    if tm == 0:
        if tr:
            m = v.reshape(B, T, X, Y).transpose(2, 3)
        else:
            m = v.reshape(B, T, Y, X)
    elif tm == 1:
        if tr:
            m = v.reshape(B, X, Y, T).permute(0, 3, 2, 1)
        else:
            m = v.reshape(B, Y, X, T).permute(0, 3, 1, 2)
    else:
        if tr:
            m = v.reshape(B, X, T, Y).permute(0, 2, 3, 1)
        else:
            m = v.reshape(B, Y, T, X).permute(0, 2, 1, 3)
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    return m


def write_grid(m, lay):
    tm, tr, fy, fx = lay
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    if tm == 0:
        if tr:
            m = m.transpose(2, 3)
    elif tm == 1:
        if tr:
            m = m.permute(0, 3, 2, 1)
        else:
            m = m.permute(0, 2, 3, 1)
    else:
        if tr:
            m = m.permute(0, 3, 1, 2)
        else:
            m = m.permute(0, 2, 1, 3)
    return m.reshape(m.shape[0], -1)


def read_compact(v, lay, blocks, Y, X):
    tm, tr, fy, fx = lay
    B = v.shape[0]
    maps = []
    off = 0
    for h, w in blocks:
        blk = v[:, off:off + h * w]
        off += h * w
        if tr:
            m = blk.reshape(B, w, h).transpose(1, 2)
        else:
            m = blk.reshape(B, h, w)
        m = F.pad(m, (0, X - w, 0, Y - h))
        if fy:
            m = m.flip(1)
        if fx:
            m = m.flip(2)
        maps.append(m)
    return torch.stack(maps, dim=1)


def write_compact(o, lay, blocks):
    tm, tr, fy, fx = lay
    outs = []
    for t, (h, w) in enumerate(blocks):
        m = o[:, t]
        if fy:
            m = m.flip(1)
        if fx:
            m = m.flip(2)
        m = m[:, :h, :w]
        if tr:
            m = m.transpose(1, 2)
        outs.append(m.reshape(m.shape[0], -1))
    return torch.cat(outs, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.Y = Y
        self.X = X
        self.n_place = int(spec.n_place)
        self.n_act = int(spec.n_actions)
        self.ne = self.n_act - self.n_place
        self.nd = max(self.ne, 1)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.blocks = [(Y - a + 1, X - b + 1) for a, b in fps]
        compact_n = sum(h * w for h, w in self.blocks)
        blocks_ok = len(self.blocks) > 0 and min(min(h, w) for h, w in self.blocks) > 0
        if self.n_place > 0 and self.n_place % (Y * X) == 0:
            self.mode = 0
            self.K = self.n_place // (Y * X)
            lays = [tuple(l) for l in LAYOUTS]
        elif blocks_ok and self.n_place == compact_n:
            self.mode = 1
            self.K = len(self.blocks)
            lays = []
            for l in LAYOUTS:
                c = (0, l[1], l[2], l[3])
                if c not in lays:
                    lays.append(c)
        else:
            self.mode = 2
            self.K = 0
            lays = []
        self.dpos = list(DONE_POS) if self.ne == 1 else [1]
        self.combos = [(d, l) for d in self.dpos for l in lays]
        if self.mode < 2:
            m_in = len(self.combos) * self.K
        else:
            m_in = 8
            self.mask_embed = nn.Linear(self.n_act, 8 * Y * X)
        cin = int(spec.in_channels) * (3 if OBS_ALT else 1) + m_in + 2
        if DENSE:
            self.dmask = nn.Linear(self.n_act, 4 * Y * X)
            cin += 4
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in DILS])
        if GLOBAL_MIX:
            self.gmix = nn.Linear(2 * WIDTH, WIDTH)
        self.hconv = nn.Conv2d(WIDTH, HEAD_W, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEAD_W, 1)
        if self.mode < 2:
            self.outs = nn.Conv2d(HEAD_W, len(self.combos) * self.K, 1)
        else:
            self.out_conv = nn.Conv2d(HEAD_W, 4, 1)
            self.out_lin = nn.Linear(4 * Y * X, max(self.n_place, 1))
        if DENSE:
            self.dout_conv = nn.Conv2d(HEAD_W, 4, 1)
            self.dout_lin = nn.Linear(4 * Y * X, self.n_act)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, len(self.dpos) * self.nd)
        if SEP_VALUE:
            self.vstem = nn.Sequential(nn.Conv2d(cin, 32, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
            self.vglob = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        ne, nd = self.ne, self.nd
        mf = mask.to(obs.dtype)
        parts = [obs]
        if OBS_ALT:
            C = obs.shape[1]
            parts.append(obs.reshape(B, Y, X, C).permute(0, 3, 1, 2))
            parts.append(obs.reshape(B, X, Y, C).permute(0, 3, 2, 1))
        if self.mode < 2:
            maps = []
            for d, lay in self.combos:
                if d == 1:
                    v = mf[:, :self.n_place]
                else:
                    v = mf[:, ne:ne + self.n_place]
                if self.mode == 0:
                    maps.append(read_grid(v, lay, K, Y, X))
                else:
                    maps.append(read_compact(v, lay, self.blocks, Y, X))
            parts.append(torch.cat(maps, dim=1))
        else:
            parts.append(self.mask_embed(mf).view(B, 8, Y, X))
        if DENSE:
            parts.append(self.dmask(mf).view(B, 4, Y, X))
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts.append(ys)
        parts.append(xs)
        xin = torch.cat(parts, dim=1)
        h = self.stem(xin)
        nmid = len(self.res) // 2 - 1
        for i, conv in enumerate(self.res):
            h = F.relu(h + conv(h))
            if GLOBAL_MIX and i == nmid:
                gm = self.gmix(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
                h = F.relu(h + gm[:, :, None, None])
        f = h
        z = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        dn = self.done_head(g)
        if self.mode < 2:
            o = self.outs(z)
            logits = None
            for i, (d, lay) in enumerate(self.combos):
                oi = o[:, i * K:(i + 1) * K]
                if self.mode == 0:
                    flat = write_grid(oi, lay)
                else:
                    flat = write_compact(oi, lay, self.blocks)
                j = self.dpos.index(d)
                ex = dn[:, j * nd:j * nd + ne]
                if d == 1:
                    li = torch.cat([flat, ex], dim=1)
                else:
                    li = torch.cat([ex, flat], dim=1)
                logits = li if logits is None else logits + li
        else:
            flat = self.out_lin(F.relu(self.out_conv(z)).flatten(1))[:, :self.n_place]
            logits = torch.cat([flat, dn[:, :ne]], dim=1)
        if DENSE:
            logits = logits + self.dout_lin(F.relu(self.dout_conv(z)).flatten(1))
        if SEP_VALUE:
            vf = self.vstem(xin)
            value = self.value_head(self.vglob(torch.cat([vf.mean((2, 3)), vf.amax((2, 3))], dim=1))).squeeze(1)
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
