"""Post-hoc sensitivity check (not in PLAN.md), added after run_check.py.

Diagnosis: every sampled 'stable' build that failed the planned MuJoCo criterion had a brick
shifted 1-6 mm sideways with no tilt. The planned offsets (up to 0.3 mm) exceed the 0.1 mm
side gap, so a brick can be released overlapping its neighbour and get pushed. This re-runs
every replay with the same seeds, checks the stored outcomes are reproduced exactly, and
adds a 'toppled' criterion: any brick tilted more than 5 degrees or more than 2 mm below its
lattice height.
  .venv/bin/python sensitivity.py
"""
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import brickphys as bp

HERE = Path(__file__).parent
SPECS = {"small": {"X": 5, "Y": 5, "Z": 3}, "large": {"X": 6, "Y": 6, "Z": 4}}


def seed(scale, agent, di, k):
    return int.from_bytes(hashlib.sha256(f"{scale}/{agent}/{di}/{k}".encode()).digest()[:4], "little")




def job_z(args):
    """Same replay, returning per-brick final z error (needs the data, so reimplemented)."""
    import mujoco
    r, bricks = args
    res = []
    for k in range(len(r["mujoco"])):
        spec = SPECS[r["scale"]]
        m = mujoco.MjModel.from_xml_string(bp.scene_xml(bricks, spec))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        rng = np.random.default_rng(seed(r["scale"], r["agent"], r["design"], k))
        offs = rng.uniform(-bp.JITTER, bp.JITTER, size=(len(bricks), 2))
        tg = [bp.lattice_pos(b, spec) for b in bricks]
        steps = lambda s: [mujoco.mj_step(m, d) for _ in range(int(round(s / bp.TIMESTEP)))]
        for i in range(len(bricks)):
            j = m.body(f"b{i}").jntadr[0]
            qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
            p = tg[i].copy(); p[:2] += offs[i]; p[2] += bp.LIFT
            d.qpos[qa:qa + 3] = p; d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]; d.qvel[va:va + 6] = 0
            mujoco.mj_forward(m, d)
            steps(bp.SETTLE)
        steps(bp.FINAL_SETTLE)
        pos_err, tilt, zdrop = [], [], []
        for i in range(len(bricks)):
            qa = m.jnt_qposadr[m.body(f"b{i}").jntadr[0]]
            pos_err.append(np.linalg.norm(d.qpos[qa:qa + 3] - tg[i]))
            tilt.append(bp._tilt(d.qpos[qa + 3:qa + 7]))
            zdrop.append(tg[i][2] - d.qpos[qa + 2])
        stands = all(e <= bp.POS_TOL and t <= bp.TILT_TOL for e, t in zip(pos_err, tilt))
        toppled = any(t > bp.TILT_TOL for t in tilt) or max(zdrop) > 0.002
        res.append({"stands": bool(stands), "toppled": bool(toppled),
                    "max_err_mm": float(max(pos_err) * 1000), "max_tilt_deg": float(np.degrees(max(tilt))),
                    "max_zdrop_mm": float(max(zdrop) * 1000)})
    return res


def main():
    ro = json.loads((HERE / "rollouts.json").read_text())
    rec = [json.loads(l) for l in open(HERE / "records.jsonl")]
    jobs = [(r, ro["episodes"][f"{r['scale']}/{r['agent']}"][r["design"]]["bricks"]) for r in rec]
    with Pool(10) as pool:
        outs = pool.map(job_z, jobs, chunksize=16)
    mism = sum(o["stands"] != s for r, os in zip(rec, outs) for o, s in zip(os, r["mujoco"]))
    summary = {"replays": sum(len(o) for o in outs), "mismatches_with_run_check": mism, "by_scale": {}}
    for scale in SPECS:
        sc = {}
        for c in ("stable", "marginal", "unstable"):
            rows = [o for r, os in zip(rec, outs) if r["scale"] == scale and r["order_class"] == c for o in os]
            sc[c] = {"n_replays": len(rows),
                     "stands_planned": float(np.mean([o["stands"] for o in rows])) if rows else None,
                     "not_toppled": float(np.mean([not o["toppled"] for o in rows])) if rows else None}
        groups = {}
        for g in sorted({r["group"] for r in rec if r["scale"] == scale}):
            rows = [o for r, os in zip(rec, outs) if r["scale"] == scale and r["group"] == g for o in os]
            groups[g] = {"n_replays": len(rows), "not_toppled": float(np.mean([not o["toppled"] for o in rows]))}
        summary["by_scale"][scale] = {"by_order_class": sc, "by_group": groups}
    with open(HERE / "sensitivity_records.jsonl", "w") as f:
        for r, os in zip(rec, outs):
            f.write(json.dumps({"scale": r["scale"], "agent": r["agent"], "design": r["design"], "replays": os}) + "\n")
    (HERE / "sensitivity.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
