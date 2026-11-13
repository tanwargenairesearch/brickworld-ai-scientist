import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 36
N_RES = 3
EMB = 16
HID = 32
LAYER_CH = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.layer_conv = nn.Conv2d(8, LAYER_CH, 3, padding=1)
        self.mix = nn.Conv2d(self.Z * LAYER_CH, WIDTH, 1)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_RES)])
        self.embed = nn.Conv2d(WIDTH, EMB, 1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.type_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * EMB, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in range(spec.n_types)
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        cols = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype)
        zc = zc.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, cols, zc], dim=2).reshape(B * Z, 8, Y, X)
        x = F.relu(self.layer_conv(x)).reshape(B, Z * LAYER_CH, Y, X)
        h = F.relu(self.mix(x))
        for conv in self.res:
            h = h + F.relu(conv(h))
        e = self.embed(h)
        outs = []
        for t in range(len(self.fps)):
            dy, dx = self.fps[t]
            ep = F.pad(e, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(ep, kernel_size=(dy, dx), stride=1)
            av = F.avg_pool2d(ep, kernel_size=(dy, dx), stride=1)
            outs.append(self.type_heads[t](torch.cat([mx, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
