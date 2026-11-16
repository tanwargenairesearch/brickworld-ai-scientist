import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.002, 'n_envs': 128, 'rollout_len': 16, 'n_minibatches': 8, 'ppo_epochs': 2, 'ent_coef': 0.01, 'gamma': 0.99, 'gae_lambda': 0.95}

LAYOUTS = [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
COMPRESS = True
POOL = False
VOX = False
SEP_VALUE = True
VGRAD = 0.1
HEAD_K = 3
HEAD_W = 32
STEM = 64
WIDTH = 40
PW = 16
PH = 24
VW = 8


def slog(u):
    return torch.sign(u) * torch.log1p(u.abs())


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
        Z = int(spec.Z)
        C = int(spec.in_channels)
        self.Y = Y
        self.X = X
        self.Z = Z
        self.n_place = int(spec.n_place)
        self.n_act = int(spec.n_actions)
        self.ne = self.n_act - self.n_place
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.fps = fps
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
        self.lays = lays
        if self.mode < 2:
            m_in = len(lays) * self.K
        else:
            m_in = 8
            self.mask_embed = nn.Linear(self.n_act, 8 * Y * X)
        if COMPRESS:
            self.comp = nn.Conv2d(C, 2 * C, 1, groups=C)
            obs_feat = 2 * C
        else:
            obs_feat = C
        self.vox_ok = VOX and C == 4 * Z + 3
        vox_feat = 0
        if self.vox_ok:
            self.vox1 = nn.Conv3d(4, VW, 3, padding=1)
            self.vox2 = nn.Conv3d(VW, VW, (3, 1, 1), padding=(1, 0, 0))
            vox_feat = VW * Z
        cin = obs_feat + vox_feat + m_in + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hconv = nn.Conv2d(WIDTH, HEAD_W, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEAD_W, 1)
        if self.mode < 2:
            self.outs = nn.Conv2d(HEAD_W, len(lays) * self.K, 1)
        else:
            self.out_conv = nn.Conv2d(HEAD_W, 4, 1)
            self.out_lin = nn.Linear(4 * Y * X, max(self.n_place, 1))
        fits = all(a <= Y and b <= X and a >= 1 and b >= 1 for a, b in fps)
        self.pool_ok = POOL and self.mode == 0 and self.K == len(fps) and fits
        if self.pool_ok:
            self.pproj = nn.Conv2d(WIDTH, PW, 1)
            self.pmix = nn.ModuleList([nn.Conv2d(6 * PW, PH, 1) for _ in fps])
            self.pout = nn.ModuleList([nn.Conv2d(PH, 1, 1) for _ in fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, max(self.ne, 1))
        if SEP_VALUE:
            self.vstem = nn.Sequential(nn.Conv2d(cin, 32, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
            self.vglob = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)

    def pooled(self, f):
        p = self.pproj(f)
        maps = []
        for (dy, dx), mix, out in zip(self.fps, self.pmix, self.pout):
            if dy * dx == 1:
                st = [p, p, p, p, p, p]
            else:
                qa = F.pad(p, (0, dx - 1, 0, dy - 1))
                qb = F.pad(p, (dx - 1, 0, dy - 1, 0))
                st = []
                for q in (qa, qb):
                    st.append(F.max_pool2d(q, (dy, dx), stride=1))
                    st.append(-F.max_pool2d(-q, (dy, dx), stride=1))
                    st.append(F.avg_pool2d(q, (dy, dx), stride=1))
            hh = F.relu(mix(torch.cat(st, dim=1)))
            maps.append(out(hh))
        return torch.cat(maps, dim=1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        ne = self.ne
        mf = mask.to(obs.dtype)
        if COMPRESS:
            parts = [slog(self.comp(obs))]
        else:
            parts = [obs]
        if self.vox_ok:
            v3 = obs[:, :4 * self.Z].reshape(B, 4, self.Z, Y, X)
            v3 = F.relu(self.vox1(v3))
            v3 = F.relu(self.vox2(v3))
            parts.append(v3.reshape(B, -1, Y, X))
        if self.mode < 2:
            v = mf[:, :self.n_place]
            maps = []
            for lay in self.lays:
                if self.mode == 0:
                    maps.append(read_grid(v, lay, K, Y, X))
                else:
                    maps.append(read_compact(v, lay, self.blocks, Y, X))
            parts.append(torch.cat(maps, dim=1))
        else:
            parts.append(self.mask_embed(mf).view(B, 8, Y, X))
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts.append(ys)
        parts.append(xs)
        xin = torch.cat(parts, dim=1)
        s = self.stem(xin)
        h = F.relu(s + self.r1(s))
        f = F.relu(h + self.r2(h))
        z = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        dn = self.done_head(g)[:, :ne]
        if self.mode < 2:
            o = self.outs(z)
            pm = self.pooled(f) if self.pool_ok else None
            flat = None
            for i, lay in enumerate(self.lays):
                oi = o[:, i * K:(i + 1) * K]
                if pm is not None:
                    oi = oi + pm
                if self.mode == 0:
                    fl = write_grid(oi, lay)
                else:
                    fl = write_compact(oi, lay, self.blocks)
                flat = fl if flat is None else flat + fl
            logits = torch.cat([flat, dn], dim=1)
        else:
            flat = self.out_lin(F.relu(self.out_conv(z)).flatten(1))[:, :self.n_place]
            logits = torch.cat([flat, dn], dim=1)
        if SEP_VALUE:
            vf = self.vstem(xin)
            value = self.value_head(self.vglob(torch.cat([vf.mean((2, 3)), vf.amax((2, 3))], dim=1))).squeeze(1)
        else:
            value = self.value_head(g).squeeze(1)
        if VGRAD < 1.0:
            value = value * VGRAD + (value * (1.0 - VGRAD)).detach()
        return logits, value


def build_policy(spec):
    return Net(spec)
