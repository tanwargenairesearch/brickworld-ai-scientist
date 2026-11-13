"""Menu architecture: encoder=mlp, width=32, depth=1, activation=tanh, norm=none."""
import torch
import torch.nn as nn

ENCODER = "mlp"
WIDTH = 32
DEPTH = 1
ACTIVATION = "tanh"
NORM = "none"


def ACT():
    return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[ACTIVATION]()


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.kind = ENCODER
        if ENCODER == "conv":
            layers, c_in = [], spec.in_channels
            for _ in range(DEPTH):
                layers.append(nn.Conv2d(c_in, WIDTH, 3, padding=1))
                if NORM == "layernorm":
                    layers.append(nn.GroupNorm(1, WIDTH))
                layers.append(ACT())
                c_in = WIDTH
            self.body = nn.Sequential(*layers)
            self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
            self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), ACT())
            self.done_head = nn.Linear(WIDTH, 1)
            self.value_head = nn.Linear(WIDTH, 1)
        else:
            layers, d_in = [], spec.in_channels * spec.Y * spec.X
            for _ in range(DEPTH):
                layers.append(nn.Linear(d_in, WIDTH))
                if NORM == "layernorm":
                    layers.append(nn.LayerNorm(WIDTH))
                layers.append(ACT())
                d_in = WIDTH
            self.body = nn.Sequential(*layers)
            self.logit_head = nn.Linear(WIDTH, spec.n_actions)
            self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        if self.kind == "conv":
            f = self.body(obs)
            place = self.place_head(f).flatten(1)
            g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
            return torch.cat([place, self.done_head(g)], dim=1), self.value_head(g).squeeze(1)
        f = self.body(obs.flatten(1))
        return self.logit_head(f), self.value_head(f).squeeze(1)


def build_policy(spec):
    return Net(spec)


RECIPE = {'lr': 0.000111, 'n_envs': 64, 'rollout_len': 64, 'n_minibatches': 8, 'ppo_epochs': 5, 'ent_coef': 0.0062, 'gamma': 0.9546, 'gae_lambda': 0.8405, 'clip_eps': 0.2223}
