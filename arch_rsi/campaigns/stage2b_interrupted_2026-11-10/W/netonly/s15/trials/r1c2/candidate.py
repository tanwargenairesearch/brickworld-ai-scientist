import torch
import torch.nn as nn
import torch.nn.functional as Fn

WIDTH = 48
HID = 40


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.max_dy = max(dy for dy, dx in self.fps)
        self.max_dx = max(dx for dy, dx in self.fps)
        t = spec.n_types
        self.head1 = nn.Conv2d(t * 2 * WIDTH, t * HID, 1, groups=t)
        self.head2 = nn.Conv2d(t * HID, t, 1, groups=t)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        Y, X = f.shape[2], f.shape[3]
        fp = Fn.pad(f, (0, self.max_dx - 1, 0, self.max_dy - 1))
        feats = []
        for dy, dx in self.fps:
            shifts = [fp[:, :, i:i + Y, j:j + X] for i in range(dy) for j in range(dx)]
            s = torch.stack(shifts, dim=0)
            feats.append(s.amax(0))
            feats.append(s.mean(0))
        h = torch.relu(self.head1(torch.cat(feats, dim=1)))
        place = self.head2(h).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
