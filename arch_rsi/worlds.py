"""World specs for the architecture-search campaign. Reuses the frozen env code unchanged."""
from brickworld.env import SCALES, WorldSpec

SPECS = {
    "large": SCALES["large"],                                    # 6x6x4, 4-12 bricks (meta-train problem)
    "xl": WorldSpec("xl", X=8, Y=8, Z=5, k_min=6, k_max=16),     # 8x8x5, 6-16 bricks (held-out problem)
}

# Training recipe held fixed for every architecture: the optimiser settings of sonnet-r0-t09,
# the main campaign's Sonnet run-0 winner, which kept the starting architecture.
RECIPE = {"lr": 0.001, "lr_schedule": "linear_decay", "n_envs": 64, "rollout_len": 16, "ppo_epochs": 4,
          "n_minibatches": 8, "gamma": 0.95, "gae_lambda": 0.9, "clip_eps": 0.2, "ent_coef": 0.01,
          "vf_coef": 0.5, "max_grad_norm": 1.0, "adv_norm": True, "obs_mode": "voxels",
          "reward_mode": "cells", "success_bonus": 1.0, "step_penalty": 0.01}

# Caps that apply to every candidate in every arm. The whole 144-option menu fits inside them
# (its largest network, conv width 128 depth 4, is about 10x the starting architecture's FLOPs).
# Training is compute-matched (train_arch.py): a network with r times the starting FLOPs per sample
# trains for 1/r of the steps, so the FLOP cap only bounds how few updates a network can get.
CAPS = {"max_params": 1_000_000, "flops_ratio": 12.0}
# "The policy must be learned", checked on the world's 256 check designs (targets/<world>_check.pt):
# - untrained: at most max_untrained_exact built exactly, and at most max_untrained_clean of target cells filled
#   (on average) before the first placement that puts a cell outside the target ("clean fill");
# - after a short training run of short_steps steps: at most max_short_exact built exactly. The starting
#   architecture needs about 256 updates to reach 22-55%, so 25% after 8 updates would mean a learner more than
#   30x faster; a hand-written score behind a zero-initialised gate reached 30-38%.
CAPS.update({"max_untrained_exact": 0.02, "max_untrained_clean": 0.10, "n_check": 256, "short_steps": 8192,
             "max_short_exact": 0.25})
