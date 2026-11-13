import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_types, self.Y, self.X = spec.n_types, spec.Y, spec.X
        self.n_place = spec.n_place
        self.conv1 = nn.Conv2d(spec.in_channels + spec.n_types, WIDTH, 3, padding=1)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].reshape(B, self.n_types, self.Y, self.X).to(obs.dtype)
        x = torch.cat([obs, m], dim=1)
        f = F.relu(self.conv1(x))
        c = self.ctx(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = f + c[:, :, None, None]
        f = F.relu(self.conv2(f))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
