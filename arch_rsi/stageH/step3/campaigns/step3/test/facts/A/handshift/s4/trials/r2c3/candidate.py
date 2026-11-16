import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 1e-3, 'n_envs': 128, 'rollout_len': 64, 'n_minibatches': 16, 'ppo_epochs': 1, 'ent_coef': 0.03, 'gamma': 0.995, 'gae_lambda': 0.97, 'clip_eps': 0.2}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
G = 64


def views(obs):
    v = [obs, obs.flip(2), obs.flip(3), obs.flip(2).flip(3)]
    if obs.shape[2] == obs.shape[3]:
        t = obs.transpose(2, 3)
        v = v + [t, t.flip(2), t.flip(3), t.flip(2).flip(3)]
    return torch.cat(v, dim=1)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f, gp):
        p = self.proj(f) + gp[:, :, None, None]
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        nv = 8 if spec.X == spec.Y else 4
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels * nv, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, G), nn.ReLU())
        self.gproj = nn.Linear(G, PROJ)
        self.done_head = nn.Linear(G, 1)
        self.act_bias = nn.Linear(G, spec.n_actions)
        nn.init.zeros_(self.act_bias.weight)
        nn.init.zeros_(self.act_bias.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        s = self.stem(views(obs))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(f, self.gproj(g))
        logits = torch.cat([place, self.done_head(g)], dim=1) + self.act_bias(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
