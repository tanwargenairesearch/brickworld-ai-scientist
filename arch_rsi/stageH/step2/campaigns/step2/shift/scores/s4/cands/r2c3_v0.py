import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 48
W = 32
P = 40
H = 48
GH = 64
FH = 128


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = int(spec.in_channels)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Conv2d(C + self.T, STEM, 1)
        self.c1 = nn.Conv2d(STEM, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.cell = nn.Conv2d(C + self.T + W, P, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.h1 = nn.ModuleList([nn.Conv2d(dy * dx * P, H, 1) for dy, dx in self.fps])
        self.gb = nn.ModuleList([nn.Linear(GH, H) for _ in self.fps])
        self.h2 = nn.ModuleList([nn.Conv2d(H, H, 1) for _ in self.fps])
        self.h3 = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.fc1 = nn.Linear(W * int(spec.Y) * int(spec.X), FH)
        self.fc2 = nn.Linear(FH, self.n_place)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        mm = mask[:, :self.n_place].reshape(B, self.T, Y, X).to(obs.dtype)
        inp = torch.cat([obs, mm], dim=1)
        s = F.relu(self.stem(inp))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = F.relu(self.cell(torch.cat([inp, f], dim=1)))
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        res = []
        for (dy, dx), l1, lb, l2, l3 in zip(self.fps, self.h1, self.gb, self.h2, self.h3):
            q = F.pad(g, (0, dx - 1, 0, dy - 1))
            parts = [q[:, :, oy:oy + Y, ox:ox + X] for oy in range(dy) for ox in range(dx)]
            h = torch.cat(parts, dim=1)
            h = F.relu(l1(h) + lb(gl)[:, :, None, None])
            h = F.relu(l2(h))
            res.append(l3(h).flatten(1))
        place = torch.cat(res, dim=1)
        place = place + self.fc2(F.relu(self.fc1(f.flatten(1))))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
