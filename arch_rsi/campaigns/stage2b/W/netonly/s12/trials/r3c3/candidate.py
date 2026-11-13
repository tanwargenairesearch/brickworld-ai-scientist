import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
N_BLOCKS = 3
K = 8
D = 32
NH = 4


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.nt = spec.n_types
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(c, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_BLOCKS)])
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.pos = nn.Parameter(torch.randn(1, spec.Y * spec.X, WIDTH) * 0.02)
        self.qkv = nn.Linear(WIDTH, 3 * D)
        self.proj = nn.Linear(D, WIDTH)
        self.cell = nn.Conv2d(WIDTH, self.nt * K, 1)
        self.mix = nn.Conv2d(self.nt * 3 * K, self.nt, 1, groups=self.nt)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def attend(self, f):
        b, w, ny, nx = f.shape
        n = ny * nx
        t = f.flatten(2).transpose(1, 2) + self.pos
        qkv = self.qkv(t).view(b, n, 3, NH, D // NH).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D // NH), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(b, n, D)
        o = self.proj(o).transpose(1, 2).reshape(b, w, ny, nx)
        return f + o

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for i, conv in enumerate(self.blocks):
            if i == 1:
                g0 = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
                f = f + F.relu(self.ctx(g0))[:, :, None, None]
            if i == 2:
                f = self.attend(f)
            f = f + F.relu(conv(f))
        b, _, ny, nx = f.shape
        h = self.cell(f).view(b, self.nt, K, ny, nx)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            ht = F.pad(h[:, t], (0, dx - 1, 0, dy - 1))
            sh = [ht[:, :, i:i + ny, j:j + nx] for i in range(dy) for j in range(dx)]
            st = torch.stack(sh, 0)
            outs += [st.amin(0), st.amax(0), st.mean(0)]
        place = self.mix(torch.cat(outs, dim=1)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
