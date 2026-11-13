"""Tests for analyze_2e.py's pure functions. No GPU, no model calls, no campaign needed.

usage (project root): .venv/bin/python -I arch_rsi/stage2/2e/test_analyze_2e.py
"""
import itertools
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_2e as AZ  # noqa: E402


def brute_p(d: list) -> float:
    obs = abs(sum(d))
    hits = sum(abs(sum(s * x for s, x in zip(signs, d))) >= obs for signs in itertools.product((1, -1), repeat=len(d)))
    return hits / 2 ** len(d)


def entry(pair, arm, builds_per_seed, label="r2c1", sha=None, seeds=(2001, 2002)):
    return {"kind": "search", "pair": pair, "arm": arm, "search": pair, "label": label, "name": "n",
            "sha": sha or f"{arm}{pair:02d}", "mean_selected": 0.5, "incumbent_2seed": label, "seeds": list(seeds),
            "dev": {str(s): k / 512 for s, k in zip(seeds, builds_per_seed)}, "status": {str(s): "ok" for s in seeds}}


def start(builds_per_seed, seeds=(2001, 2002)):
    return {"kind": "comparator", "arm": "start_g", "search": 0, "label": "start_g", "name": "start_g", "sha": "start",
            "seeds": list(seeds), "dev": {str(s): k / 512 for s, k in zip(seeds, builds_per_seed)},
            "status": {str(s): "ok" for s in seeds}}


def test_signflip():
    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(1, 12)
        d = [rng.choice([0, 0, 1, 2, 3, 5, 8, 40, 400]) * rng.choice([1, -1]) for _ in range(n)]
        assert abs(AZ.signflip_p(d) - brute_p(d)) < 1e-12, d
    assert AZ.signflip_p([1] * 5) == 2 / 32
    assert AZ.signflip_p([3, -3]) == 1.0 and AZ.signflip_p([0, 0, 0]) == 1.0
    assert AZ.signflip_p([7] * 12) == 2 / 4096 and AZ.signflip_p([-7] * 12) == 2 / 4096
    assert AZ.signflip_p([5, 0, 0, 0]) == 1.0                 # one non-zero pair can never be significant
    d = [rng.randint(-4000, 4000) for _ in range(48)]         # the real size: 48 pairs of counts out of 8,192
    p = AZ.signflip_p(d)
    assert 0 < p <= 1 and AZ.signflip_p([-x for x in d]) == p
    for bad in ([], [0.5, 1]):
        try:
            AZ.signflip_p(bad)
        except ValueError:
            continue
        raise AssertionError(f"signflip_p accepted {bad}")


def test_second_stage_rule():
    rng = random.Random(2)
    for _ in range(200):
        d = [rng.uniform(-0.6, 0.6) for _ in range(32)]
        a = AZ.second_stage_rule(d, 0.20)
        flipped = [x * rng.choice([1, -1]) for x in d]        # swapping labels in any pairs changes only signs
        b = AZ.second_stage_rule(flipped, 0.20)
        assert a["extend"] == b["extend"] and abs(a["rms"] - b["rms"]) < 1e-12
    assert not AZ.second_stage_rule([0.2, -0.2, 0.2], 0.20)["extend"]       # exactly at the threshold: no
    assert AZ.second_stage_rule([0.2, -0.2, 0.21], 0.20)["extend"]
    assert abs(AZ.rms([0.3, -0.4]) - (0.125) ** 0.5) < 1e-12


def test_reading_and_block():
    assert AZ.reading(0.12, 0.01, 0.2) == "pass" and AZ.reading(-0.12, 0.01, -0.02) == "worse"
    assert AZ.reading(0.02, 0.4, 0.099) == "bounded" and AZ.reading(0.02, 0.4, 0.10) == "open"
    assert AZ.reading(0.12, 0.05, 0.2) == "open"              # p must be below 0.05, not equal
    b = AZ.test_block([820] * 12, 8192)
    assert b["gate_passed"] and b["reading"] == "pass" and b["p"] == 2 / 4096 and abs(b["mean_d"] - 820 / 8192) < 1e-6
    b = AZ.test_block([-820] * 12, 8192)
    assert not b["gate_passed"] and b["reading"] == "worse"
    b = AZ.test_block([100, -100, 60, -60, 0, 0, 30, -30], 8192)
    assert not b["gate_passed"] and b["reading"] == "bounded" and b["n_zero"] == 2 and b["p"] == 1.0
    b = AZ.test_block([3000, -2500, 2800, -2600, 2400, -2000], 8192)
    assert b["reading"] == "open" and b["ci95_boot"][1] >= 0.10
    assert AZ.boot_interval([0.1, -0.2, 0.3, 0.0]) == AZ.boot_interval([0.1, -0.2, 0.3, 0.0])
    lo, hi = AZ.boot_interval([0.05] * 10)
    assert abs(lo - 0.05) < 1e-12 and abs(hi - 0.05) < 1e-12


def test_pair_rows():
    st = start([50, 60])
    ents = [entry(1, "seen", [300, 310]), entry(1, "hidden", [280, 290]),
            entry(2, "seen", [100, 100], sha="same"), entry(2, "hidden", [100, 100], sha="same"),
            entry(3, "seen", [400, 420], sha="bad"), entry(3, "hidden", [50, 60], label="start", sha="start"), st]
    full = {e["sha"]: {"verdict": "compliant"} for e in ents if e["kind"] == "search"} | {"bad": {"verdict": "violation"}}
    pr = AZ.pair_rows(ents, full)
    assert pr["denom"] == 1024 and pr["pending"] == 0
    r1, r2, r3 = pr["rows"]
    assert r1["d_builds"] == 40 and r2["d_builds"] == 0 and r2["same_pick"]
    assert r3["seen"]["builds"] == 820 and r3["seen"]["primary_builds"] == 110        # scored as the start
    assert r3["d_builds"] == 0 and r3["d_builds_as_computed"] == 710
    assert r3["hidden"]["verdict"] == "compliant (by construction)"
    assert AZ.pair_rows(ents, None)["pending"] == 5                                    # no review yet
    part = dict(full)
    del part["seen01"]
    assert AZ.pair_rows(ents, part)["pending"] == 1                                    # one pick unreviewed
    assert len(AZ.pair_rows(ents[:5] + [st], full)["rows"]) == 2                       # the last pair unfinished: left out...
    try:
        AZ.pair_rows([e for e in ents if e.get("pair") != 2], full)                    # ...but a gap in 1..N is an error
    except ValueError:
        pass
    else:
        raise AssertionError("a gap in the pairs was accepted")
    for bad in (0.1234, 1 / 3):
        try:
            AZ.builds(bad)
        except ValueError:
            continue
        raise AssertionError(f"builds accepted {bad}")
    assert AZ.builds(0.068359375) == 35 and AZ.pick_round("start") == 0 and AZ.pick_round("r12c3") == 12
    failed = entry(4, "seen", [300, 310])
    failed["status"]["2002"] = "timeout"
    assert AZ.seed_builds(failed) == [300, 0]


if __name__ == "__main__":
    for t in (test_signflip, test_second_stage_rule, test_reading_and_block, test_pair_rows):
        t()
        print("ok", t.__name__)
    print("all analyze_2e tests passed")
