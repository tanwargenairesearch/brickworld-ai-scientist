"""Stage 2b helpers: the RECIPE block, its bounds, the fixed-FLOP budget, random recipes, and a launcher.

The RECIPE block is a top-level constant in a candidate file, read with ast.literal_eval (never executed here).
Budget (PROTOCOL_2B.md): steps = budget_steps * (E_REF + 1/3) / ((E + 1/3) * max(r, 1)), in whole updates of
n_envs * rollout_len steps, where r is the network's training-FLOP ratio to the start and E its PPO epochs.

usage (project root), launcher modes:
  .venv/bin/python -I arch_rsi/stage2/w2b.py train_cpu --spec run.json --out DIR   (frozen CPU trainer on W)
  .venv/bin/python -I arch_rsi/stage2/w2b.py eval --trial DIR --split dev --out OUT.json  (frozen evaluator on W)
"""
import ast
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

STAGE_G = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 4,
           "ent_coef": 0.01, "gamma": 0.95, "gae_lambda": 0.9, "clip_eps": 0.2}
BOUNDS = {"lr": (1e-4, 3e-3, float), "n_envs": (64, 1024, int), "rollout_len": (8, 64, int),
          "n_minibatches": (4, 64, int), "ppo_epochs": (1, 8, int), "ent_coef": (0.0, 0.05, float),
          "gamma": (0.9, 0.995, float), "gae_lambda": (0.8, 0.99, float), "clip_eps": (0.1, 0.3, float)}
E_REF = 4
MIN_UPDATES = 4
MIN_MB = 64
FROZEN_AS_BLOCK = {"lr": 0.001, "n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 4}  # worlds.RECIPE


def parse_recipe(src: str):
    """The RECIPE dict from a candidate's source, or None if there is none. Raises ValueError if malformed."""
    tree = ast.parse(src)
    found = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RECIPE" for t in node.targets):
            if found is not None:
                raise ValueError("RECIPE is assigned more than once")
            try:
                found = ast.literal_eval(node.value)
            except ValueError as e:
                raise ValueError("RECIPE must be a literal dict of numbers") from e
            if not isinstance(found, dict):
                raise ValueError("RECIPE must be a dict")
    return found


def strip_recipe(src: str) -> str:
    """The source's AST dump without the RECIPE assignment (to compare networks)."""
    tree = ast.parse(src)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and
                                              any(isinstance(t, ast.Name) and t.id == "RECIPE" for t in n.targets))]
    return ast.dump(tree)


def full_recipe(recipe) -> dict:
    return {**STAGE_G, **(recipe or {})}


def validate(recipe) -> list:
    """Errors in a RECIPE dict (bounds, types, minibatch rules). Budget is checked separately."""
    if recipe is None:
        return []
    errs = []
    for k, v in recipe.items():
        if k not in BOUNDS:
            errs.append(f"RECIPE key '{k}' is not allowed (allowed: {', '.join(BOUNDS)})")
            continue
        lo, hi, typ = BOUNDS[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (typ is int and not isinstance(v, int)):
            errs.append(f"RECIPE['{k}'] must be {'an integer' if typ is int else 'a number'}")
            continue
        if not (lo <= v <= hi) or (isinstance(v, float) and not math.isfinite(v)):
            errs.append(f"RECIPE['{k}'] = {v} is outside its bounds [{lo}, {hi}]")
    if errs:
        return errs
    rc = full_recipe(recipe)
    batch = rc["n_envs"] * rc["rollout_len"]
    if batch % rc["n_minibatches"]:
        errs.append(f"n_minibatches ({rc['n_minibatches']}) must divide n_envs * rollout_len ({batch})")
    elif batch // rc["n_minibatches"] < MIN_MB:
        errs.append(f"each minibatch must hold at least {MIN_MB} samples; n_envs * rollout_len / n_minibatches = "
                    f"{batch // rc['n_minibatches']}")
    return errs


def budget(budget_steps: int, flops_ratio: float, recipe) -> dict:
    """Steps and updates the fixed-FLOP budget allows for this network and recipe."""
    rc = full_recipe(recipe)
    E, batch = rc["ppo_epochs"], rc["n_envs"] * rc["rollout_len"]
    steps = budget_steps * (E_REF + 1 / 3) / ((E + 1 / 3) * max(flops_ratio, 1.0))
    n_updates = int(steps // batch)
    return {"allowed_steps": int(steps), "n_updates": n_updates, "batch": batch, "steps": n_updates * batch,
            "ok": n_updates >= MIN_UPDATES}


def random_recipe(rng) -> dict:
    """One draw of the random arm's recipe (PROTOCOL_2B.md); bounds re-checked by the caller."""
    return {"lr": round(math.exp(rng.uniform(math.log(1e-4), math.log(3e-3))), 6),
            "n_envs": rng.choice([64, 128, 256, 512, 1024]), "rollout_len": rng.choice([8, 16, 32, 64]),
            "n_minibatches": rng.choice([4, 8, 16, 32, 64]), "ppo_epochs": rng.randint(1, 8),
            "ent_coef": round(rng.uniform(0.0, 0.05), 4), "gamma": round(rng.uniform(0.9, 0.995), 4),
            "gae_lambda": round(rng.uniform(0.8, 0.99), 4), "clip_eps": round(rng.uniform(0.1, 0.3), 4)}


def with_recipe(src: str, recipe: dict) -> str:
    """A candidate file: the given network source plus a RECIPE block (keys in a fixed order)."""
    keys = [k for k in BOUNDS if k in recipe] + [k for k in recipe if k not in BOUNDS]  # unknown keys kept, so checks reject them
    body = ", ".join(f'{k!r}: {recipe[k]!r}' for k in keys)
    return src.rstrip() + "\n\n\nRECIPE = {" + body + "}\n"


def load_cpu_trainer():
    from arch_rsi.stage2 import w_world
    w_world.register()
    import arch_rsi.train_arch as ta
    orig = ta.heldout_hashes
    ta.heldout_hashes = lambda world: w_world.w_hashes() if world == "w" else orig(world)
    return ta


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    if mode == "train_cpu":
        load_cpu_trainer().main()
    elif mode == "eval":
        from arch_rsi.stage2 import w_world
        w_world.load_evaluator().main()
    else:
        raise SystemExit(f"unknown mode {mode}")
