import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
PROJ = 24
HID = 32
GLOB = 64
DET = 16
NQ = 4
AV = 24

RECIPE = {'ppo_epochs': 1, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 64, 'lr': 0.0015, 'gamma': 0.99, 'gae_lambda': 0.95, 'ent_coef': 0.004}


def init_layers(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2.0))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)


def coords(obs):
    bsz = obs.shape[0]
    ny = obs.shape[2]
    nx = obs.shape[3]
    yy = torch.linspace(-1.0, 1.0, ny, device=obs.device, dtype=obs.dtype).view(1, 1, ny, 1).expand(bsz, 1, ny, nx)
    xx = torch.linspace(-1.0, 1.0, nx, device=obs.device, dtype=obs.dtype).view(1, 1, 1, nx).expand(bsz, 1, ny, nx)
    return torch.cat([yy, xx], dim=1)


def mask_map(mask, nt, ny, nx, dtype):
    return mask[:, :nt * ny * nx].to(dtype).reshape(mask.shape[0], nt, ny, nx)


def win_feats(p, dy, dx):
    ny = p.shape[2]
    nx = p.shape[3]
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    parts = []
    for i in range(dy):
        for j in range(dx):
            parts.append(q[:, :, i:i + ny, j:j + nx])
    st = torch.stack(parts, 0)
    return torch.cat(parts + [st.amax(0), st.amin(0)], dim=1)


class WinHead(nn.Module):
    def __init__(self, fps, proj, hid):
        super().__init__()
        self.fps = fps
        self.mix1 = nn.ModuleList([nn.Conv2d((dy * dx + 2) * proj, hid, 1) for dy, dx in fps])
        self.mix2 = nn.ModuleList([nn.Conv2d(hid, hid, 1) for _ in fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in fps])

    def forward(self, p, gb):
        res = []
        for (dy, dx), m1, m2, o in zip(self.fps, self.mix1, self.mix2, self.outs):
            h = F.relu(m1(win_feats(p, dy, dx)) + gb)
            h = F.relu(m2(h))
            res.append(o(h))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(fps)
        self.nt = nt
        self.lna = math.log(float(spec.Y * spec.X))
        self.lnp = math.log(float(spec.n_place))
        cin = c + 2 + nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.GroupNorm(8, COL), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.proj = nn.Conv2d(WIDTH + SKIP, PROJ, 1)
        self.head = WinHead(fps, PROJ, HID)
        self.akey = nn.Conv2d(WIDTH + SKIP, NQ, 1)
        self.aval = nn.Conv2d(WIDTH + SKIP, AV, 1)
        self.det = nn.Conv2d(cin, DET, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2 * DET + NQ * AV, GLOB), nn.ReLU(), nn.Linear(GLOB, GLOB), nn.ReLU())
        self.gproj = nn.Linear(GLOB, HID)
        self.type_head = nn.Linear(GLOB, nt)
        self.done_head = nn.Linear(GLOB + 2, 1)
        self.value_head = nn.Linear(GLOB, 1)
        self.abias = nn.Parameter(torch.zeros(spec.n_actions))
        self.ltemp = nn.Parameter(torch.zeros(1))
        init_layers(self)
        for o in self.head.outs:
            init_out(o, 0.01)
        init_out(self.gproj, 0.1)
        init_out(self.type_head, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        bsz = obs.shape[0]
        ny = obs.shape[2]
        nx = obs.shape[3]
        x0 = torch.cat([obs, coords(obs), mask_map(mask, self.nt, ny, nx, obs.dtype)], dim=1)
        s = self.stem(x0)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        fs = torch.cat([f, self.skip(x0)], dim=1)
        p = self.proj(fs)
        att = torch.softmax(self.akey(fs).flatten(2), dim=2)
        val = self.aval(fs).flatten(2)
        pooled = torch.bmm(att, val.transpose(1, 2)).flatten(1)
        dm = F.relu(self.det(x0))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), dm.amax((2, 3)), dm.mean((2, 3)), pooled], dim=1))
        gb = self.gproj(g).view(bsz, -1, 1, 1)
        temp = torch.exp(self.ltemp.clamp(-3.0, 3.0))
        an = (self.head(p, gb) * temp).flatten(2)
        an = F.log_softmax(an, dim=2) + self.lna
        tl = self.type_head(g)
        place = (an + tl.unsqueeze(2)).flatten(1)
        pd = place.detach()
        dctx = torch.cat([g, pd.amax(1, keepdim=True), torch.logsumexp(pd, 1, keepdim=True) - self.lnp], dim=1)
        logits = torch.cat([place, self.done_head(dctx)], dim=1) + self.abias
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
