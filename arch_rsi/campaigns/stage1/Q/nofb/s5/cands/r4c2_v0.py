import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_types = spec.n_types
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.fp_convs = nn.ModuleList([nn.Conv2d(WIDTH, HID, (dy, dx)) for dy, dx in self.fps])
        self.fp_outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.g_proj = nn.Linear(WIDTH, HID * spec.n_types)
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        f = self.body(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        gb = self.g_proj(g).view(B, self.n_types, HID)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            fp = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(self.fp_convs[i](fp) + gb[:, i].view(B, HID, 1, 1))
            outs.append(self.fp_outs[i](h))
        place = torch.cat(outs, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
