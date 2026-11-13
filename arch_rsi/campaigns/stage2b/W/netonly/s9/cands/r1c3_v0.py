import torch
import torch.nn as nn

WIDTH = 46


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        g = 2.0 ** 0.5
        self.stem = init_layer(nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), g)
        self.ln1 = ChanLN(WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), g)
        self.ln2 = ChanLN(WIDTH)
        self.ctx = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, WIDTH), g), nn.ReLU(),
                                 init_layer(nn.Linear(WIDTH, WIDTH), 1.0))
        self.c3 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), g)
        self.ln3 = ChanLN(WIDTH)
        self.place_head = init_layer(nn.Conv2d(WIDTH, spec.n_types, 1), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, 64), g), nn.ReLU())
        self.done_head = init_layer(nn.Linear(64, 1), 0.01)
        self.value_head = init_layer(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        x = torch.relu(self.ln1(self.stem(obs)))
        x = x + torch.relu(self.ln2(self.c2(x)))
        c = self.ctx(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        x = x + c[:, :, None, None]
        x = x + torch.relu(self.ln3(self.c3(x)))
        place = self.place_head(x).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
