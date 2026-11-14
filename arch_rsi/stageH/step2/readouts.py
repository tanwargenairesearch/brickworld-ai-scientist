"""Step 2 readouts (PROTOCOL_STEP2.md, "Readouts"), computed by harness code from one trained run, on the GPU.

1 dev behaviour (aggregates only): placements per episode, share of episodes ending with DONE before any placement,
  mean IoU, cells placed inside and outside the target per episode
2 learning curve: training success and mean episode length at 25/50/75/100% of training (from train_log.json)
3 two greedy traces on the two diagnostic designs: the design's height map, the target-height channel as the agent
  observed it at the first step, each step's action, filled cells and reward, and the final height map
4 one untrained trace: the network at its seed initialisation, sampling as training does (sampling seed 0), design 1
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2/readouts.py --trial DIR --rule R --seed S --out OUT.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as C  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import load_candidate, sha  # noqa: E402
from arch_rsi.worlds import RECIPE, SPECS  # noqa: E402
from brickworld.env import heights_of  # noqa: E402
from brickworld.model import encode  # noqa: E402
from rules import Rule  # noqa: E402
from train import reward_fn  # noqa: E402

TYPES = ("1x1", "1x2", "2x1", "2x2")
LEGEND = (
    "Readouts come from each trial's seed-0 run and are computed by the harness. \"dev behaviour\" summarises the greedy "
    "runs on the dev designs (no design is shown). The learning curve gives training success and mean episode length at "
    "25%, 50%, 75% and 100% of training. Each trace runs the agent on one of two fixed diagnostic designs that are never "
    "scored: it shows the design's column heights, the target's column heights as the agent's observation showed them "
    "at the first step (channel 4Z+1 times Z), every step's action and the cells it filled (row y, column x, layer z) "
    "with its reward, and the final build's column heights. The greedy traces use the trained agent; the untrained trace "
    "uses the same network at its seed-0 initialisation, sampling actions as in training. Height maps list rows "
    "y = 0..7 from top to bottom, each a string of columns x = 0..7. Full readouts are shown for the starting "
    "architecture, the current best and the latest round; earlier trials show only the first two lines.")


def act_text(a: int, spec) -> str:
    if a == spec.done_action:
        return "DONE"
    t, pos = divmod(a, spec.Y * spec.X)
    y, x = divmod(pos, spec.X)
    return f"{TYPES[t]} at row {y}, col {x}"


def behaviour(policy, rule, designs) -> dict:
    spec, dev = rule.spec, rule.device
    w = rule.make_world(len(designs), targets=designs, auto_reset=False)
    n = len(designs)
    placed = torch.zeros(n, device=dev)
    first_done = torch.zeros(n, dtype=torch.bool, device=dev)
    cin = torch.zeros(n, device=dev)
    cout = torch.zeros(n, device=dev)
    with torch.no_grad():
        for _ in range(spec.max_steps + 1):
            if bool(w.ended.all()):
                break
            logits, _ = policy(rule.obs(encode(w, "voxels")), rule.mask(w.valid_mask()))
            a = logits.argmax(dim=1)
            active = ~w.ended
            is_done = a == spec.done_action
            first_done |= active & is_done & (placed == 0)
            info = w.step(rule.act(a))
            put = active & ~is_done & ~info["illegal"]
            placed += put.float()
            cin += torch.where(put, info["cells_in"].float(), torch.zeros_like(cin))
            cout += torch.where(put, info["cells_out"].float(), torch.zeros_like(cout))
    inter = (w.occ & w.target).flatten(1).sum(1).float()
    union = (w.occ | w.target).flatten(1).sum(1).float().clamp(min=1)
    return {"placements": round(float(placed.mean()), 3), "done_before_any": round(float(first_done.float().mean()), 4),
            "iou": round(float((inter / union).mean()), 4), "cells_in": round(float(cin.mean()), 3),
            "cells_out": round(float(cout.mean()), 3)}


def trace(policy, rule, design, greedy: bool) -> dict:
    spec, dev = rule.spec, rule.device
    w = rule.make_world(1, targets=design.unsqueeze(0), auto_reset=False)
    gen = torch.Generator().manual_seed(0)
    Z = spec.Z
    o = rule.obs(encode(w, "voxels"))
    seen = (o[0, 4 * Z + 1] * Z).round().long().cpu()
    steps = []
    with torch.no_grad():
        while not bool(w.ended.all()) and len(steps) <= spec.max_steps:
            o = rule.obs(encode(w, "voxels"))
            logits, _ = policy(o, rule.mask(w.valid_mask()))
            if greedy:
                a = logits.argmax(dim=1)
            else:
                a = torch.multinomial(F.softmax(logits, dim=-1).cpu(), 1, generator=gen).squeeze(1).to(dev)
            before = w.occ.clone()
            info = w.step(rule.act(a))
            r = reward_fn(RECIPE, info, a != spec.done_action)
            filled = (w.occ & ~before)[0].nonzero().tolist()
            steps.append({"action": act_text(int(a), spec), "filled": [[y, x, z] for z, y, x in filled],
                          "reward": round(float(r), 3)})
    return {"design": heights_of(design.unsqueeze(0))[0].tolist(), "seen": seen.tolist(),
            "final": heights_of(w.occ.cpu())[0].tolist(), "exact": bool((w.occ.cpu()[0] == design).all()), "steps": steps}


def curve_points(log: dict) -> list:
    c = log.get("curve") or []
    out = []
    for q in (0.25, 0.5, 0.75, 1.0):
        if c:
            p = c[max(0, math.ceil(q * len(c)) - 1)]
            out.append({"at": q, "train_success": round(p["train_exact"], 3), "ep_len": round(p["ep_len"], 1)})
    return out


def hm_rows(a, b=None, la="design", lb="as observed") -> list:
    rows = [f"    {la:<9}" + (f"  {lb}" if b is not None else "")]
    for y in range(len(a)):
        s = "".join(str(v) for v in a[y])
        rows.append(f"    row {y}: {s}" + (f"   {''.join(str(v) for v in b[y])}" if b is not None else ""))
    return rows


def render_trace(name: str, t: dict) -> list:
    lines = [f"- {name}: column heights of the design and of the target as observed at the first step"]
    lines += hm_rows(t["design"], t["seen"])
    for i, s in enumerate(t["steps"], 1):
        f = ", ".join(f"(row {y}, col {x}, layer {z})" for y, x, z in s["filled"]) or "nothing"
        lines.append(f"    step {i}: {s['action']} -> filled {f}; reward {s['reward']:+.3f}")
    lines.append(f"    final build column heights (exact build: {'yes' if t['exact'] else 'no'}):")
    lines += [f"    row {y}: {''.join(str(v) for v in r)}" for y, r in enumerate(t["final"])]
    return lines


def render(ro: dict, full: bool) -> str:
    b = ro["behaviour"]
    lines = [f"- dev behaviour: placements per episode {b['placements']:.2f}; episodes ending with DONE before any placement "
             f"{100 * b['done_before_any']:.1f}%; mean IoU {b['iou']:.3f}; cells placed inside / outside the target per "
             f"episode {b['cells_in']:.2f} / {b['cells_out']:.2f}",
             "- learning curve (training success, mean episode length): " +
             "; ".join(f"{int(100 * p['at'])}% {p['train_success']:.3f}, {p['ep_len']:.1f}" for p in ro["curve"])]
    if full:
        lines += render_trace("greedy trace, diagnostic design 1", ro["traces"][0])
        lines += render_trace("greedy trace, diagnostic design 2", ro["traces"][1])
        lines += render_trace("untrained trace (sampling), diagnostic design 1", ro["untrained"])
    return "\n".join(lines)


def compute(trial: Path, rule_name: str, seed: int, device="mps") -> dict:
    C.common.register()
    src = (trial / "candidate.py").read_text()
    ck = torch.load(trial / "ckpt.pt", map_location="cpu")
    if ck["candidate_sha"] != sha(src):
        raise SystemExit("candidate.py does not match the checkpoint")
    spec = SPECS[ck["world"]]
    aspec = arch_spec(spec)
    dev = torch.device(device)
    rule = Rule(rule_name, spec, dev)
    torch.manual_seed(0)
    pol = CandidatePolicy(load_candidate(src)(aspec), aspec)
    pol.load_state_dict(ck["state_dict"])
    pol.to(dev).eval()
    torch.manual_seed(seed)
    unt = CandidatePolicy(load_candidate(src)(aspec), aspec).to(dev).eval()
    dev_designs = torch.load(C.common.TARGETS / "hplain_dev.pt")
    diag = torch.load(C.DIAG)
    log = json.loads((trial / "train_log.json").read_text())
    ro = {"rule": rule_name, "seed": seed, "behaviour": behaviour(pol, rule, dev_designs), "curve": curve_points(log),
          "traces": [trace(pol, rule, diag[0], True), trace(pol, rule, diag[1], True)],
          "untrained": trace(unt, rule, diag[0], False)}
    ro["text_full"], ro["text_short"] = render(ro, True), render(ro, False)
    return ro


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    Path(args.out).write_text(json.dumps(compute(Path(args.trial), args.rule, args.seed)))
