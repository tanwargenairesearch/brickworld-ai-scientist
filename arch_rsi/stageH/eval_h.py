"""Stage H1 evaluator (PROTOCOL_H1.md): arch_rsi/evaluate_arch.py's greedy run and NumPy scoring, through a hidden
rule's hooks, on the GPU, on a Stage H design set. It also records where the wrong cells are and the exact rate on
designs with hidden cells. With rule "plain" the greedy run is evaluate_arch.py's (checked by test_identity_h.py).

usage (project root): .venv/bin/python -I arch_rsi/stageH/eval_h.py --trial DIR --rule R --designs SET --out OUT.json
SET is hplain_dev or hplain_check (rules plain, shift, mirror, late) or htorus_dev or htorus_check (rule torus).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import torch  # noqa: E402

from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import load_candidate, sha  # noqa: E402
from arch_rsi.worlds import SPECS  # noqa: E402
from brickworld.model import encode  # noqa: E402
from evaluate import check  # noqa: E402
from rules import RULES, Rule, designs_hidden  # noqa: E402

SETS = ("hplain_dev", "hplain_check", "htorus_dev", "htorus_check")


def greedy_build(policy, rule, designs):
    """evaluate_arch.main's greedy loop through the rule's hooks -> (final occupancy, pressed DONE), both on the CPU."""
    spec, device = rule.spec, rule.device
    world = rule.make_world(len(designs), targets=designs, auto_reset=False)
    pressed_done = torch.zeros(len(designs), dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(spec.max_steps + 1):
            if bool(world.ended.all()):
                break
            logits, _ = policy(rule.obs(encode(world, "voxels")), rule.mask(world.valid_mask()))
            a = logits.argmax(dim=1)
            pressed_done |= (a == spec.done_action) & ~world.ended
            world.step(rule.act(a))
    return world.occ.cpu(), pressed_done.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--rule", required=True, choices=RULES)
    ap.add_argument("--designs", required=True, choices=SETS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if not args.designs.startswith(common.DESIGNS[args.rule]):
        raise SystemExit(f"rule {args.rule} is scored on {common.DESIGNS[args.rule]} designs, not {args.designs}")
    common.register()
    trial = Path(args.trial)
    src = (trial / "candidate.py").read_text()
    ck = torch.load(trial / "ckpt.pt", map_location="cpu")
    if ck["candidate_sha"] != sha(src):
        raise SystemExit("candidate.py does not match the checkpoint")
    spec = SPECS[ck["world"]]
    if spec.name != common.WORLD:
        raise SystemExit(f"Stage H trains on {common.WORLD}, not {spec.name}")
    aspec = arch_spec(spec)
    designs = torch.load(common.TARGETS / f"{args.designs}.pt")
    device = torch.device(args.device)
    torch.manual_seed(0)
    policy = CandidatePolicy(load_candidate(src)(aspec), aspec)
    policy.load_state_dict(ck["state_dict"])
    policy.to(device).eval()
    occ, pressed_done = greedy_build(policy, Rule(args.rule, spec, device), designs)
    sc = check(occ.numpy(), designs.numpy())
    hid = designs_hidden(designs).numpy()
    ex = sc["exact"]
    res = {"split": args.designs, "rule": args.rule, "n": int(len(designs)), "world": spec.name, "device": device.type,
           "exact": float(ex.mean()), "iou": float(sc["iou"].mean()), "pressed_done": float(pressed_done.float().mean()),
           "n_hidden": int(hid.sum()), "exact_hidden": float(ex[hid].mean()) if hid.any() else None,
           "wrong_cells_yx": (occ != designs).sum((0, 1)).tolist(),
           "per_design": {"exact": ex.astype(int).tolist()}}
    Path(args.out).write_text(json.dumps(res))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_design", "wrong_cells_yx")}))


if __name__ == "__main__":
    main()
