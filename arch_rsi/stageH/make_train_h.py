"""Writes train_h.py from the frozen arch_rsi/stage2/train_2b.py by the exact replacements below (each must
match the stated number of times). Kept so the derivation can be re-run and diffed.
usage (project root): .venv/bin/python -I arch_rsi/stageH/make_train_h.py --check   (compares with train_h.py)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = '"""Stage H1 trainer (PROTOCOL_H1.md): arch_rsi/stage2/train_2b.py with a hidden rule\'s hooks, nothing else changed.\n\nusage (project root): .venv/bin/python -I arch_rsi/stageH/train_h.py --spec run.json --out DIR\nrun.json: {"candidate", "world": "ws", "rule": plain|shift|mirror|torus|late, "seed", "budget_steps", "device", "threads"}.\n\nDifferences from train_2b.py, each a one-line replacement (make_train_h.py, checked by test_identity_h.py):\n- W_s is registered (stage2/2c/ws_world.py, unchanged) and the training stream rejects h_heldout_hashes.pt\n  (W_s\'s dev designs and all four Stage H design sets);\n- the world and stream come from the rule (rules.Rule.make_world / make_stream);\n- the agent sees rule.obs(encode(...)), acts through rule.act, gets rule.mask(valid_mask()) and is paid rule.reward(...).\nWith rule "plain" every hook returns its input, so training is train_2b.py\'s, bit for bit.\n"""'
REPS = [('from train import TRAIN_SEED_BASE, reward_fn  # noqa: E402\n', 'from train import TRAIN_SEED_BASE, reward_fn  # noqa: E402\n\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nimport common  # noqa: E402\nfrom rules import Rule  # noqa: E402\n', 1), ('w_world.register()\n\n\ndef sync', 'common.register()\n\n\ndef sync', 1), ('    device = torch.device(run.get("device", "mps"))\n', '    device = torch.device(run.get("device", "mps"))\n    rule = Rule(run.get("rule", "plain"), spec, device)\n', 1), ('    hh = w_world.w_hashes() if spec.name == "w" else heldout_hashes(spec.name)\n', '    hh = torch.load(common.TARGETS / "h_heldout_hashes.pt")\n', 1), ('stream = StreamSource(spec, TRAIN_SEED_BASE + seed, hh)', 'stream = rule.make_stream(TRAIN_SEED_BASE + seed, hh)', 1), ('world = BrickWorld(spec, N, device, target_source=stream)', 'world = rule.make_world(N, target_source=stream)', 1), ('obs = encode(world, "voxels")', 'obs = rule.obs(encode(world, "voxels"))', 2), ('mask = world.valid_mask()', 'mask = rule.mask(world.valid_mask())', 1), ('info = world.step(a)', 'info = world.step(rule.act(a))', 1), ('r = reward_fn(rc, info, placed)', 'r = rule.reward(reward_fn(rc, info, placed), info["ended"])', 1), ('_, next_val = policy(obs, world.valid_mask())', '_, next_val = policy(obs, rule.mask(world.valid_mask()))', 1), ('"allowed_steps": bud["allowed_steps"]}', '"allowed_steps": bud["allowed_steps"], "rule": rule.name}', 1)]


def build() -> str:
    src = (ROOT / "arch_rsi" / "stage2" / "train_2b.py").read_text()
    body = src[src.index('"""', 3) + 3:]
    for old, new, n in REPS:
        if body.count(old) != n:
            raise SystemExit(f"{old!r}: expected {n} match(es), found {body.count(old)}")
        body = body.replace(old, new)
    return DOC + body


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "train_h.py"
    if "--check" in sys.argv:
        ok = out.read_text() == build()
        print("train_h.py matches its derivation" if ok else "train_h.py differs from its derivation")
        sys.exit(0 if ok else 1)
    if out.exists():
        raise SystemExit("train_h.py exists; refusing to overwrite")
    out.write_text(build())
