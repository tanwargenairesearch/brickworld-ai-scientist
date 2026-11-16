import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.005, "gae_lambda": 0.95}

W = 48
P = 24
HID = 48
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(int(spec.in_channels), 96, 1), nn.ReLU(),
            nn.Conv2d(96, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=2, dilation=2)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.film = nn.Linear(G, W)
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(dy * dx * P, HID, 1) for dy, dx in self.fps])
        self.mix2 = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.done_head = nn.Linear(G + 2, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        Y, X = obs.shape[2], obs.shape[3]
        s = self.stem(obs)
        x = F.relu(s + self.c1(s))
        x = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        f = F.relu(x + self.film(g)[:, :, None, None])
        p = self.proj(f)
        res = []
        for (dy, dx), mix, mix2, out in zip(self.fps, self.mix, self.mix2, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            parts = [q[:, :, i:i + Y, j:j + X] for i in range(dy) for j in range(dx)]
            h = F.relu(mix(torch.cat(parts, dim=1)))
            h = F.relu(h + mix2(h))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        stats = torch.cat([place.amax(1, keepdim=True),
                           torch.logsumexp(place, 1, keepdim=True)], dim=1)
        done = self.done_head(torch.cat([g, stats], dim=1))
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
