import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 60
K = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.nt = spec.n_types
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(c, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.cell = nn.Conv2d(WIDTH, self.nt * K, 1)
        self.mix = nn.Conv2d(self.nt * 3 * K, self.nt, 1, groups=self.nt)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        b, _, ny, nx = f.shape
        h = self.cell(f).view(b, self.nt, K, ny, nx)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            ht = F.pad(h[:, t], (0, dx - 1, 0, dy - 1))
            sh = [ht[:, :, i:i + ny, j:j + nx] for i in range(dy) for j in range(dx)]
            st = torch.stack(sh, 0)
            outs += [st.amin(0), st.amax(0), st.mean(0)]
        place = self.mix(torch.cat(outs, dim=1)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
