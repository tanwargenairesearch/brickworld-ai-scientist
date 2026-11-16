import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.001, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
WIDTH = 40
PROJ = 32
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.nt = int(spec.n_types)
        self.npl = int(spec.n_place)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = spec.in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(cin + self.nt, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH + cin, PROJ, 1)
        self.loc = nn.ModuleList([nn.Conv2d(PROJ + 1, HID, 3, padding=1) for _ in self.fps])
        self.pmix = nn.ModuleList([nn.Conv2d(2 * PROJ, HID, 1, bias=False) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.npl].to(obs.dtype).reshape(B, self.nt, self.Y, self.X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        p = self.proj(torch.cat([f, obs], dim=1))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                mx = p
                mn = p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(self.loc[i](torch.cat([p, m[:, i:i + 1]], dim=1)) + self.pmix[i](torch.cat([mx, mn], dim=1)))
            res.append(self.outs[i](h))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
