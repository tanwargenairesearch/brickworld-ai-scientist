"""Tests of analyze_step2.py on fake campaigns (no runs, no calls).
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step2/test_analyze_step2.py   (exits 1 on any failure)
"""
import itertools
import json
import sys
import tempfile
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_step2 as A  # noqa: E402
import s2common as C  # noqa: E402

FAILED = []


def test(fn):
    try:
        fn()
        print("PASS", fn.__name__)
    except AssertionError as e:
        FAILED.append(fn.__name__)
        print("FAIL", fn.__name__, e)
    return fn


def exact_strat(ga, gb):
    NA, NB = sum(map(len, ga.values())), sum(map(len, gb.values()))
    obs = sum(map(sum, ga.values())) / NA - sum(map(sum, gb.values())) / NB
    tot = sum(sum(ga[w]) + sum(gb[w]) for w in ga)
    per = []
    for w in ga:
        v = ga[w] + gb[w]
        per.append([sum(v[i] for i in c) for c in itertools.combinations(range(len(v)), len(ga[w]))])
    hits = n = 0
    for combo in itertools.product(*per):
        sa = sum(combo)
        n += 1
        hits += abs(sa / NA - (tot - sa) / NB) >= abs(obs) - 1e-12
    return hits / n


def campaign(scores: dict, viol=(), n=12):
    """scores: {(world, arm): list of n search scores}. Writes summary/review/judge files; returns the directory."""
    d = Path(tempfile.mkdtemp())
    ents, rev = [], {}
    for (w, a), vals in scores.items():
        for i, v in enumerate(vals):
            label = "start" if v == 0 else f"r1c{i}"
            h = f"{w}{a}{i}"
            ents.append({"world": w, "arm": a, "search": i, "label": label, "name": "x", "sha": h, "hypothesis": "",
                         "mean_selected": v, "dev": {str(s): v for s in (5001, 5002, 5003, 5004)},
                         "status": {str(s): "ok" for s in (5001, 5002, 5003, 5004)}, "first_round_found": 1 if v >= C.FOUND else None,
                         "last_analysis": ""})
            if label != "start":
                rev[h] = {"verdict": "violation" if (w, a, i) in viol else "compliant"}
    starts = {w: {"confirm": {"5001": 0.0, "5002": 0.0, "5003": 0.0, "5004": 0.0}} for w in C.WORLDS}
    (d / "summary.json").write_text(json.dumps({"entries": ents, "starts": starts}))
    (d / "review.json").write_text(json.dumps({"verdicts": rev}))
    (d / "judge.json").write_text(json.dumps({f"{e['world']}/{e['arm']}/s{e['search']}": {"verdict": "no"} for e in ents}))
    return d


def arms(told, readouts, scores, hidden):
    out = {}
    for w in C.WORLDS:
        for a, k in zip(C.ARMS, (told, readouts, scores, hidden)):
            out[(w, a)] = [0.75] * k + [0.0] * (12 - k)
    return out


@test
def monte_carlo_p_matches_exact_enumeration():
    ga = {"shift": [0.7, 0.0, 0.6], "mirror": [0.75, 0.7, 0.0]}
    gb = {"shift": [0.0, 0.1, 0.0], "mirror": [0.0, 0.7, 0.0]}
    ex, mc = exact_strat(ga, gb), A.perm_p(ga, gb, n=200_000)
    assert abs(ex - mc) < 0.005, (ex, mc)


@test
def fisher_known_values():
    assert abs(A.fisher(14, 24, 6, 24) - 0.0392) < 5e-4 and abs(A.fisher(17, 24, 6, 24) - 0.0034) < 5e-4
    assert A.fisher(5, 10, 5, 10) == 1.0


@test
def clear_effect_passes_both_and_check():
    r = A.analyze(campaign(arms(11, 9, 2, 1)), n_perm=20_000)
    assert r["primary"]["P1"]["passes"] and r["primary"]["P2"]["passes"], r["summary_line"]
    assert r["check"] == {"found": 22, "n": 24, "needed": 18, "passes": True}
    assert r["ending"] == "both" and r["discovery_claim_allowed"] and r["ending_text"].endswith(A.CAVEAT)
    assert r["per_arm"]["readouts"]["found"] == 18 and r["per_arm"]["hidden"]["mean"] == round(0.75 * 2 / 24, 4)


@test
def no_effect_and_failed_check():
    r = A.analyze(campaign(arms(8, 3, 3, 3)), n_perm=20_000)
    assert not r["primary"]["P1"]["passes"] and not r["primary"]["P2"]["passes"] and r["ending"] == "none"
    assert r["check"]["found"] == 16 and not r["check"]["passes"] and not r["discovery_claim_allowed"]


@test
def only_p1():
    r = A.analyze(campaign(arms(11, 8, 6, 0)), n_perm=20_000)
    assert r["primary"]["P1"]["passes"] and not r["primary"]["P2"]["passes"] and r["ending"] == "p1", r["summary_line"]


@test
def violation_scores_as_start_and_holm():
    sc = arms(11, 9, 2, 1)
    r = A.analyze(campaign(sc, viol={("shift", "readouts", 0)}), n_perm=20_000)
    row = next(x for x in r["rows"] if (x["world"], x["arm"], x["search"]) == ("shift", "readouts", 0))
    assert row["violation"] and row["score"] == 0.0 and r["per_arm"]["readouts"]["found"] == 17
    ps = {k: v["p"] for k, v in r["primary"].items()}
    assert {k: v["p_holm"] for k, v in r["primary"].items()} == A.holm(ps)


if __name__ == "__main__":
    print("all passed" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
