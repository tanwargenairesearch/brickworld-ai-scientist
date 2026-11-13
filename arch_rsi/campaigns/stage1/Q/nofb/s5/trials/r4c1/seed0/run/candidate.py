import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
RED = 32
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_types = spec.n_types
        self.n_place = spec.n_place
        c_in = spec.in_channels + spec.n_types + 2
        self.body = nn.Sequential(
            nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.reduce = nn.Sequential(nn.Conv2d(WIDTH, RED, 1), nn.ReLU())
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.fp_convs = nn.ModuleList([nn.Conv2d(RED, HID, (dy, dx)) for dy, dx in self.fps])
        self.fp_outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.n_place].reshape(B, self.n_types, Y, X).to(obs.dtype)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        f = self.body(torch.cat([obs, m, ys, xs], dim=1))
        r = self.reduce(f)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            fp = F.pad(r, (0, dx - 1, 0, dy - 1))
            h = F.relu(self.fp_convs[i](fp))
            outs.append(self.fp_outs[i](h))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
