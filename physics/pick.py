"""Pick the video's designs by fixed rules (no looking at renders). Writes picks.json."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def hidden(occ):
    o = np.array(occ, dtype=bool)
    above = np.flip(np.logical_or.accumulate(np.flip(o, 0), 0), 0)
    return bool((above & ~o).any())


ro = json.loads((HERE / "rollouts.json").read_text())
D, E = ro["designs"], ro["episodes"]
rec = [json.loads(l) for l in open(HERE / "records.jsonl")]

# Clip 1: first small design with hidden cells and >= 5 generator bricks that the starting
# recipe (baseline-s0) got wrong and Sonnet's run-0 winner (sonnet-r0-t09) got exactly right.
c1 = next(i for i, d in enumerate(D["small"])
          if hidden(d["occ"]) and len(d["bricks"]) >= 5
          and not E["small/baseline-s0"][i]["exact"] and E["small/sonnet-r0-t09"][i]["exact"])

# Clip 2: first small design whose exact build by sonnet-r0-t09 is not stable at every step
# (static analysis) and fell in the check's seed-0 replay. (The static condition was added after
# finding that some 'falls' of stable builds were sideways shifts from overlapping offsets.)
c2 = min(r["design"] for r in rec
         if r["scale"] == "small" and r["agent"] == "sonnet-r0-t09" and not r["mujoco"][0]
         and r["order_class"] != "stable")

# Clip 3: first large design with hidden cells and >= 8 generator bricks that the large
# starting recipe (xfer-large-00-s0) got wrong and the best transfer recipe (xfer-large-01-s0) got right.
c3 = next(i for i, d in enumerate(D["large"])
          if hidden(d["occ"]) and len(d["bricks"]) >= 8
          and not E["large/xfer-large-00-s0"][i]["exact"] and E["large/xfer-large-01-s0"][i]["exact"])

r2 = next(r for r in rec if r["scale"] == "small" and r["agent"] == "sonnet-r0-t09" and r["design"] == c2)
picks = {"clip1_design": c1, "clip2_design": c2, "clip2_agent": "sonnet-r0-t09", "clip3_design": c3,
         "clip2_static_class": r2["final_class"], "clip2_order_class": r2["order_class"],
         "clip2_mujoco_seeds": r2["mujoco"],
         "rules": "first design by index meeting each rule; see pick.py"}
(HERE / "picks.json").write_text(json.dumps(picks, indent=1))
print(json.dumps(picks, indent=1))
