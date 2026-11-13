"""The search space: the only thing any proposer (random search or the AI scientist) may change.

Each knob is either a choice from a list or a log-uniform float range. The baseline is a
hand-written recipe fixed before any search ran.
"""
import json
import math
import random

SPACE = {
    # --- optimisation ---
    "lr": {"type": "logfloat", "low": 1e-4, "high": 3e-3,
           "desc": "Adam learning rate."},
    "lr_schedule": {"type": "choice", "values": ["constant", "linear_decay"],
                    "desc": "Learning-rate schedule over the training budget."},
    "n_envs": {"type": "choice", "values": [64, 128, 256, 512],
               "desc": "Parallel worlds per rollout."},
    "rollout_len": {"type": "choice", "values": [8, 16, 32],
                    "desc": "Steps per world per rollout (batch = n_envs x rollout_len)."},
    "ppo_epochs": {"type": "choice", "values": [1, 2, 4],
                   "desc": "Passes over each rollout batch."},
    "n_minibatches": {"type": "choice", "values": [2, 4, 8],
                      "desc": "Minibatches per epoch."},
    "gamma": {"type": "choice", "values": [0.9, 0.95, 0.99],
              "desc": "Discount factor."},
    "gae_lambda": {"type": "choice", "values": [0.8, 0.9, 0.95, 1.0],
                   "desc": "GAE lambda."},
    "clip_eps": {"type": "choice", "values": [0.1, 0.2, 0.3],
                 "desc": "PPO clipping range."},
    "ent_coef": {"type": "logfloat", "low": 1e-4, "high": 5e-2,
                 "desc": "Entropy bonus coefficient."},
    "vf_coef": {"type": "choice", "values": [0.25, 0.5, 1.0],
                "desc": "Value-loss coefficient."},
    "max_grad_norm": {"type": "choice", "values": [0.5, 1.0, 5.0],
                      "desc": "Gradient-norm clip."},
    "adv_norm": {"type": "choice", "values": [True, False],
                 "desc": "Normalise advantages per minibatch."},
    # --- architecture ---
    "encoder": {"type": "choice", "values": ["conv", "mlp"],
                "desc": "conv: fully convolutional, one logit per (brick type, anchor) cell; "
                        "mlp: flattened grid into an MLP."},
    "width": {"type": "choice", "values": [32, 64, 128],
              "desc": "Channels (conv) or hidden units (mlp)."},
    "depth": {"type": "choice", "values": [1, 2, 3, 4],
              "desc": "Number of conv or MLP layers."},
    "activation": {"type": "choice", "values": ["relu", "gelu", "tanh"],
                   "desc": "Activation function."},
    "norm": {"type": "choice", "values": ["none", "layernorm"],
             "desc": "Normalisation after each layer."},
    "obs_mode": {"type": "choice", "values": ["voxels", "heightmaps"],
                 "desc": "voxels: every cell of the current build and the target, layer by layer; "
                         "heightmaps: only the top-down height of each column (cheaper, smaller input)."},
    # --- reward shaping ---
    "reward_mode": {"type": "choice", "values": ["sparse", "cells", "iou_delta", "heightmap_delta"],
                    "desc": "sparse: reward only for a finished build that matches; "
                            "cells: +/- per placed cell inside/outside the target; "
                            "iou_delta: change in voxel overlap (IoU) with the target; "
                            "heightmap_delta: change in the share of columns whose height matches the target."},
    "success_bonus": {"type": "choice", "values": [0.0, 1.0, 5.0],
                      "desc": "Extra reward when the episode ends with the reward mode's notion of a match."},
    "step_penalty": {"type": "choice", "values": [0.0, 0.01, 0.05],
                     "desc": "Cost per placement."},
}

BASELINE = {
    "lr": 5e-4, "lr_schedule": "linear_decay", "n_envs": 256, "rollout_len": 16,
    "ppo_epochs": 2, "n_minibatches": 4, "gamma": 0.95, "gae_lambda": 0.9, "clip_eps": 0.2,
    "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 1.0, "adv_norm": True,
    "encoder": "conv", "width": 64, "depth": 2, "activation": "relu", "norm": "none",
    "obs_mode": "voxels", "reward_mode": "cells", "success_bonus": 1.0, "step_penalty": 0.0,
}

KNOBS = list(SPACE)


def sample(rng: random.Random) -> dict:
    cfg = {}
    for k, s in SPACE.items():
        if s["type"] == "choice":
            cfg[k] = rng.choice(s["values"])
        else:
            cfg[k] = math.exp(rng.uniform(math.log(s["low"]), math.log(s["high"])))
    return cfg


def _same(c, v) -> bool:
    if isinstance(c, bool) or isinstance(v, bool):
        return isinstance(c, bool) and isinstance(v, bool) and c == v
    if isinstance(c, (int, float)) and isinstance(v, (int, float)):
        return float(c) == float(v)
    return c == v


def validate(cfg: dict) -> dict:
    """Return a clean config or raise ValueError naming every problem."""
    errs, out = [], {}
    for k in cfg:
        if k not in SPACE:
            errs.append(f"unknown knob '{k}'")
    for k, s in SPACE.items():
        if k not in cfg:
            errs.append(f"missing knob '{k}'")
            continue
        v = cfg[k]
        if s["type"] == "choice":
            match = [c for c in s["values"] if _same(c, v)]
            if not match:
                errs.append(f"'{k}'={v!r} not in {s['values']}")
            else:
                out[k] = match[0]
        else:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                errs.append(f"'{k}'={v!r} is not a number")
            elif not (s["low"] <= float(v) <= s["high"]):
                errs.append(f"'{k}'={v!r} outside [{s['low']}, {s['high']}]")
            else:
                out[k] = float(v)
    if errs:
        raise ValueError("; ".join(errs))
    return out


def config_key(cfg: dict) -> str:
    """Stable identity of a config (floats rounded to 3 significant figures)."""
    norm = {k: (float(f"{v:.3g}") if SPACE[k]["type"] == "logfloat" else v) for k, v in cfg.items()}
    return json.dumps(norm, sort_keys=True)


def describe() -> str:
    lines = []
    for k, s in SPACE.items():
        if s["type"] == "choice":
            dom = "one of " + json.dumps(s["values"])
        else:
            dom = f"float, log-uniform in [{s['low']:g}, {s['high']:g}]"
        lines.append(f"- {k}: {dom}. {s['desc']}")
    return "\n".join(lines)


assert validate(BASELINE) == BASELINE
