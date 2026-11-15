import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 32, "n_minibatches": 8, "lr": 0.001, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.003}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
DPROJ = 16
DHID = 256


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        nt = len(self.fps)
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.out = nn.Conv2d(hid * nt + nt, nt, 3, padding=1)

    def forward(self, f, mp):
        p = self.proj(f)
        hs = []
        for (dy, dx), mix in zip(self.fps, self.mix):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            hs.append(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
        hs.append(mp)
        return self.out(torch.cat(hs, dim=1)).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.nt = int(spec.n_types)
        self.np = int(spec.n_place)
        na = int(spec.n_actions)
        cin = int(spec.in_channels) + self.nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.dproj = nn.Conv2d(WIDTH, DPROJ, 1)
        self.dfc = nn.Linear(DPROJ * self.Y * self.X + na, DHID)
        self.dout = nn.Linear(DHID, na)
        nn.init.normal_(self.dout.weight, std=0.01)
        nn.init.zeros_(self.dout.bias)
        self.value_head = nn.Linear(64 + DHID, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask.to(obs.dtype)
        mp = m[:, :self.np].reshape(B, self.nt, self.Y, self.X)
        s = self.stem(torch.cat([obs, mp], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f, mp)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        conv_logits = torch.cat([place, self.done_head(g)], dim=1)
        d = F.relu(self.dproj(f)).flatten(1)
        hd = F.relu(self.dfc(torch.cat([d, m], dim=1)))
        logits = conv_logits + self.dout(hd)
        value = self.value_head(torch.cat([g, hd], dim=1)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
