"""Step 2S probes (PROTOCOL_STEP2S.md, "The sandbox"): check the AI scientist's probe requests, run them in the
search's rule world through Stage H1's rules.py on the GPU, and write the results in the format of Step 2's traces
minus the design's own height map. Nothing is trained. The world step, observation, reward and action names are Step
2's readouts.py objects, imported unchanged."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2s_common as C  # noqa: E402

import torch  # noqa: E402

import readouts as RO  # noqa: E402  (Step 2's)

TYPES = RO.TYPES
SPEC = C.common.register()
SCHEMA = {"type": "object", "properties": {"probes": {"type": "array", "items": {
    "type": "object", "properties": {
        "design": {"type": "integer"}, "purpose": {"type": "string"},
        "placements": {"type": "array", "items": {"type": "object", "properties": {
            "brick": {"type": "string", "enum": list(TYPES)}, "row": {"type": "integer"}, "col": {"type": "integer"}},
            "required": ["brick", "row", "col"], "additionalProperties": False}}},
    "required": ["design", "purpose", "placements"], "additionalProperties": False}}},
    "required": ["probes"], "additionalProperties": False}


def _int(v, lo, hi):
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def parse(obj) -> list:
    """The reply's probes, checked; raises ValueError (the call is then retried with the reason)."""
    if not isinstance(obj, dict) or not isinstance(obj.get("probes"), list):
        raise ValueError('expected an object with a "probes" list')
    if len(obj["probes"]) > C.MAX_PROBES:
        raise ValueError(f"at most {C.MAX_PROBES} probes, got {len(obj['probes'])}")
    out = []
    for i, p in enumerate(obj["probes"], 1):
        if not isinstance(p, dict) or not _int(p.get("design"), 1, C.N_PROBE_DESIGNS):
            raise ValueError(f'probe {i}: "design" must be an integer from 1 to {C.N_PROBE_DESIGNS}')
        pl = p.get("placements")
        if not isinstance(pl, list) or not 1 <= len(pl) <= C.MAX_PLACEMENTS:
            raise ValueError(f'probe {i}: "placements" must hold 1 to {C.MAX_PLACEMENTS} placements')
        steps = []
        for j, q in enumerate(pl, 1):
            if not isinstance(q, dict) or q.get("brick") not in TYPES:
                raise ValueError(f'probe {i}, placement {j}: "brick" must be one of {", ".join(TYPES)}')
            if not _int(q.get("row"), 0, SPEC.Y - 1) or not _int(q.get("col"), 0, SPEC.X - 1):
                raise ValueError(f'probe {i}, placement {j}: "row" and "col" must be integers from 0 to {SPEC.X - 1}')
            steps.append({"brick": q["brick"], "row": q["row"], "col": q["col"]})
        out.append({"design": p["design"], "purpose": str(p.get("purpose", ""))[:600], "placements": steps})
    return out


def action(brick: str, row: int, col: int) -> int:
    """The agent's action for a placement (the action-space order of brickworld.env)."""
    return TYPES.index(brick) * SPEC.Y * SPEC.X + row * SPEC.X + col


def load_designs():
    d = torch.load(C.PROBE_DESIGNS)
    assert d.shape[0] == C.N_PROBE_DESIGNS
    return d


def run(rule, design, placements) -> dict:
    """One probe from an empty board: the observed target-height map, then each placement as the world does it."""
    spec, Z = rule.spec, rule.spec.Z
    w = rule.make_world(1, targets=design.unsqueeze(0), auto_reset=False)
    seen = (rule.obs(RO.encode(w, "voxels"))[0, 4 * Z + 1] * Z).round().long().cpu().tolist()
    steps, stopped = [], None
    with torch.no_grad():
        for q in placements:
            a = torch.tensor([action(**q)], device=rule.device)
            before = w.occ.clone()
            info = w.step(rule.act(a))
            r = RO.reward_fn(RO.RECIPE, info, a != spec.done_action)
            filled = (w.occ & ~before)[0].nonzero().tolist()
            steps.append({**q, "filled": [[y, x, z] for z, y, x in filled], "cells_in": int(info["cells_in"][0]),
                          "cells_out": int(info["cells_out"][0]), "reward": round(float(r), 3),
                          "illegal": bool(info["illegal"][0])})
            if bool(w.ended[0]):
                stopped = "illegal placement: the probe ended here" if steps[-1]["illegal"] else "the episode ended here"
                break
    return {"seen": seen, "steps": steps, "final": RO.heights_of(w.occ.cpu())[0].tolist(), "stopped": stopped}


def _rows(hm) -> list:
    return [f"    row {y}: {''.join(str(v) for v in r)}" for y, r in enumerate(hm)]


def _act(q) -> str:
    return f"{q['brick']} at row {q['row']}, col {q['col']}"


def render_results(probes: list, results: list) -> str:
    if not probes:
        return "No probes were run."
    lines = []
    for i, (p, res) in enumerate(zip(probes, results), 1):
        lines.append(f"- probe {i}, probe design {p['design']}: column heights of the target as the agent observes them "
                     f"at the start")
        lines += _rows(res["seen"])
        for j, s in enumerate(res["steps"], 1):
            if s["illegal"]:
                lines.append(f"    placement {j}: {_act(s)} -> illegal, filled nothing; reward {s['reward']:+.3f}")
            else:
                f = ", ".join(f"(row {y}, col {x}, layer {z})" for y, x, z in s["filled"])
                lines.append(f"    placement {j}: {_act(s)} -> filled {f}; cells inside / outside the target "
                             f"{s['cells_in']} / {s['cells_out']}; reward {s['reward']:+.3f}")
        if res["stopped"]:
            lines.append(f"    {res['stopped']}")
        lines.append("    final build column heights:")
        lines += _rows(res["final"])
    return "\n".join(lines)


def render_requests(probes: list) -> str:
    if not probes:
        return "none"
    return "\n".join(f"- probe {i}, probe design {p['design']} (purpose: {p['purpose']}): "
                     + "; ".join(_act(q) for q in p["placements"]) for i, p in enumerate(probes, 1))
