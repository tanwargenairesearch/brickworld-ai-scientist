import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 32, "n_minibatches": 8, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95}

COLH = 96
E = 40
CTX = 24
H = 48
PROJ = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, COLH, 1), nn.ReLU(),
            nn.Conv2d(COLH, E, 1), nn.ReLU())
        self.ctx = nn.Conv2d(E, CTX, 3, padding=1)
        D = E + CTX
        self.win = nn.ModuleList([nn.Conv2d(D, H, (dy, dx)) for dy, dx in self.fps])
        self.proj = nn.Conv2d(D, PROJ, 1)
        self.pmix = nn.ModuleList([nn.Conv2d(3 * PROJ, H, 1, bias=False) for _ in self.fps])
        self.hid = nn.ModuleList([nn.Conv2d(H, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.gb = nn.Linear(64, H, bias=False)
        self.done_head = nn.Linear(64, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        e = self.col(obs)
        c = F.relu(self.ctx(e))
        f = torch.cat([e, c], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        gb = self.gb(g)[:, :, None, None]
        p = self.proj(f)
        res = []
        for (dy, dx), win, pmix, hid, out in zip(self.fps, self.win, self.pmix, self.hid, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            w = win(q)
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                pq = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(pq, (dy, dx), stride=1)
                mn = -F.max_pool2d(-pq, (dy, dx), stride=1)
                av = F.avg_pool2d(pq, (dy, dx), stride=1)
            h = F.relu(w + pmix(torch.cat([mx, mn, av], dim=1)) + gb)
            h = F.relu(hid(h))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
