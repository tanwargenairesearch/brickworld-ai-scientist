"""Tests of Step 2S's probes (PROTOCOL_STEP2S.md, "Integrity rules": probe tests), on the GPU, a few world steps each.
- the probe designs are the ones make_probe_designs.py makes, and none is a dev, check or diagnostic design;
- in the plain world, every placement's filled cells, cells inside / outside and reward match BrickWorld stepped
  directly on the CPU (the reference);
- in the mirrored-hands world, each brick fills exactly the cells the plain world fills for the mirrored anchor
  (column 8 - dx - x, the rule sentence); in the shifted-booklet world, the same cells as the plain world, and the
  observed target heights are the design's rolled 4 columns;
- an illegal placement ends the probe, as an episode ends; later placements don't run;
- malformed replies are refused; the rendered results carry no rule word.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2s/test_probes_2s.py   (exits 1 on any failure)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2s_common as C  # noqa: E402

import torch  # noqa: E402

import make_probe_designs as MK  # noqa: E402
import probes_2s as PB  # noqa: E402
import prompts as P  # noqa: E402  (Step 2's)
from brickworld.env import BRICK_TYPES, BrickWorld, design_hash, heights_of, zobrist_codes  # noqa: E402
from rules import Rule  # noqa: E402

FAILED = []
SP = PB.SPEC
GPU = torch.device("mps")
DESIGNS = PB.load_designs()
# a fixed placement list touching every brick type, both edges and a stack; the last one is illegal (1x2 at col 7)
PLACEMENTS = [{"brick": "1x1", "row": 0, "col": 0}, {"brick": "1x2", "row": 3, "col": 0}, {"brick": "2x1", "row": 5, "col": 6},
              {"brick": "2x2", "row": 2, "col": 3}, {"brick": "2x2", "row": 2, "col": 3}, {"brick": "1x2", "row": 7, "col": 6},
              {"brick": "1x1", "row": 4, "col": 7}, {"brick": "2x1", "row": 0, "col": 2}, {"brick": "1x2", "row": 1, "col": 7}]


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except AssertionError as e:
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, e)
    return fn


def reference(design, placements, mirror=False) -> list:
    """Plain BrickWorld on the CPU; with mirror=True each anchor column is replaced by 8 - dx - x."""
    w = BrickWorld(SP, 1, torch.device("cpu"), targets=design.unsqueeze(0), auto_reset=False)
    out = []
    for q in placements:
        dx = BRICK_TYPES[PB.TYPES.index(q["brick"])][1]
        col = SP.X - dx - q["col"] if mirror else q["col"]
        before = w.occ.clone()
        info = w.step(torch.tensor([PB.action(q["brick"], q["row"], col)]))
        r = PB.RO.reward_fn(PB.RO.RECIPE, info, torch.tensor([True]))
        out.append({"filled": sorted([y, x, z] for z, y, x in (w.occ & ~before)[0].nonzero().tolist()),
                    "cells_in": int(info["cells_in"][0]), "cells_out": int(info["cells_out"][0]),
                    "reward": round(float(r), 3), "illegal": bool(info["illegal"][0])})
        if bool(w.ended[0]):
            break
    return out


def got(rule_name, design, placements=PLACEMENTS):
    return PB.run(Rule(rule_name, SP, GPU), design, placements)


@test
def probe_designs_are_made_by_the_script_and_unseen():
    d, h = MK.make()
    assert torch.equal(d, DESIGNS)
    codes = zobrist_codes(SP)
    held = set(torch.load(C.common.TARGETS / "h_heldout_hashes.pt").tolist())
    assert not held & set(h.tolist())
    others = torch.cat([torch.load(C.common.TARGETS / "hplain_dev.pt"), torch.load(C.common.TARGETS / "hplain_check.pt"),
                        torch.load(C.C2.DIAG)])
    assert not set(design_hash(others, codes).tolist()) & set(h.tolist())
    for x in DESIGNS:
        assert not any(torch.equal(x, o) for o in others)


@test
def plain_world_matches_brickworld_on_the_cpu():
    for d in DESIGNS:
        res = got("plain", d)
        ref = reference(d, PLACEMENTS)
        mine = [{k: (sorted(s[k]) if k == "filled" else s[k]) for k in ("filled", "cells_in", "cells_out", "reward", "illegal")}
                for s in res["steps"]]
        assert mine == ref, (mine, ref)
        assert res["seen"] == heights_of(d.unsqueeze(0))[0].tolist()


@test
def mirror_world_fills_the_mirrored_cells():
    for d in DESIGNS:
        res, ref = got("mirror", d), reference(d, PLACEMENTS, mirror=True)
        assert [sorted(s["filled"]) for s in res["steps"]] == [s["filled"] for s in ref]
        assert [(s["cells_in"], s["cells_out"], s["illegal"]) for s in res["steps"]] == \
               [(s["cells_in"], s["cells_out"], s["illegal"]) for s in ref]
        assert res["seen"] == heights_of(d.unsqueeze(0))[0].tolist()   # the booklet isn't changed here


@test
def mirror_moves_a_brick_away_from_where_it_was_placed():
    res = got("mirror", DESIGNS[0], [{"brick": "1x2", "row": 3, "col": 0}])
    assert sorted(res["steps"][0]["filled"]) == [[3, 6, 0], [3, 7, 0]], res["steps"][0]["filled"]


@test
def shift_world_fills_the_same_cells_and_shows_rolled_heights():
    for d in DESIGNS:
        res, ref = got("shift", d), reference(d, PLACEMENTS)
        assert [sorted(s["filled"]) for s in res["steps"]] == [s["filled"] for s in ref]
        hm = heights_of(d.unsqueeze(0))[0]
        assert res["seen"] == torch.roll(hm, 4, dims=1).tolist()


@test
def illegal_placement_ends_the_probe():
    res = got("plain", DESIGNS[1], [{"brick": "1x1", "row": 0, "col": 0}] * 7 + [{"brick": "1x1", "row": 5, "col": 5}])
    assert len(res["steps"]) == 7 and res["steps"][-1]["illegal"] and res["steps"][-1]["filled"] == []
    assert res["stopped"].startswith("illegal placement") and res["final"][0][0] == SP.Z and res["final"][5][5] == 0
    res = got("plain", DESIGNS[1], PLACEMENTS)
    assert res["steps"][-1]["illegal"] and len(res["steps"]) == len(PLACEMENTS)


@test
def malformed_replies_are_refused():
    ok = {"design": 1, "purpose": "x", "placements": [{"brick": "1x1", "row": 0, "col": 0}]}
    assert PB.parse({"probes": []}) == [] and len(PB.parse({"probes": [ok] * 3})) == 3
    bad = [{"probes": [ok] * 4}, {"probes": [dict(ok, design=5)]}, {"probes": [dict(ok, design=0)]},
           {"probes": [dict(ok, design=True)]}, {"probes": [dict(ok, placements=[])]},
           {"probes": [dict(ok, placements=ok["placements"] * 17)]},
           {"probes": [dict(ok, placements=[{"brick": "3x1", "row": 0, "col": 0}])]},
           {"probes": [dict(ok, placements=[{"brick": "1x1", "row": 8, "col": 0}])]},
           {"probes": [dict(ok, placements=[{"brick": "1x1", "row": 0, "col": -1}])]},
           {"probes": [dict(ok, placements=[{"brick": "1x1", "row": 0.0, "col": 0}])]}, {"probe": []}, []]
    for b in bad:
        try:
            PB.parse(b)
            assert False, f"accepted {b}"
        except ValueError:
            pass


@test
def rendered_results_carry_no_rule_word():
    for w in ("shift", "mirror"):
        probes = [{"design": k + 1, "purpose": "see what happens", "placements": PLACEMENTS} for k in range(3)]
        text = PB.render_results(probes, [got(w, DESIGNS[k]) for k in range(3)]) + PB.render_requests(probes)
        assert not P.rule_words(text), P.rule_words(text)
        assert "probe design 3" in text and "illegal, filled nothing" in text


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
