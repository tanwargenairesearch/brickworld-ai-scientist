import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 48
EMB = 24
HID = 48

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem1 = nn.Conv2d(spec.in_channels, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=2, dilation=2)
        self.emb = nn.Conv2d(2 * WIDTH, EMB, 1)
        self.mix = nn.ModuleList([nn.Conv2d(dy * dx * EMB, HID, 1) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        e = F.relu(self.emb(torch.cat([s, f], dim=1)))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            q = F.pad(e, (0, dx - 1, 0, dy - 1))
            cells = []
            for i in range(dy):
                for j in range(dx):
                    cells.append(q[:, :, i:i + Y, j:j + X])
            h = F.relu(mix(torch.cat(cells, dim=1)))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
