"""Starting architecture: the main campaign's default policy (conv encoder, width 64, depth 2, ReLU,
no normalisation), rewritten as a candidate file. A 3x3 conv body, a 1x1 conv placement head
(one logit per brick type and anchor cell), and DONE and value heads on mean+max pooled features."""
import torch
import torch.nn as nn

WIDTH = 64
DEPTH = 2


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)


RECIPE = {'n_envs': 64, 'rollout_len': 16, 'n_minibatches': 8, 'ppo_epochs': 8, 'clip_eps': 0.3}
