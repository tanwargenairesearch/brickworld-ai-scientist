import torch
import torch.nn as nn
import torch.nn.functional as F

COL = 64
WIDTH = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_types = spec.n_types
        self.Y = spec.Y
        self.X = spec.X
        self.n_place = spec.n_place
        c_in = spec.in_channels + spec.n_types
        self.col1 = nn.Conv2d(c_in, COL, 1)
        self.col2 = nn.Conv2d(COL, COL, 1)
        self.sp1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.sp2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.place_head = nn.Conv2d(WIDTH + COL, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, : self.n_place].reshape(B, self.n_types, self.Y, self.X).to(obs.dtype)
        x = torch.cat([obs, m], dim=1)
        c = F.relu(self.col1(x))
        c = F.relu(self.col2(c))
        f = F.relu(self.sp1(c))
        f = F.relu(self.sp2(f))
        place = self.place_head(torch.cat([f, c], dim=1)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
