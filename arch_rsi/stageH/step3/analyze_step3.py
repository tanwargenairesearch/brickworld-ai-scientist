"""Step 3 analysis (PROTOCOL_STEP3.md, "Analysis, fixed now" and "Endings, fixed now"). analyze(cdir) reads
summary.json, review.json, judge.json and program_review.json from a campaign directory and returns the results
(run_step3.py writes them to results.json).

Unit: one meta-test search on a held-out world, scored by the mean of its pick's 4 confirmation seeds (a failed seed
scores 0; a pick the rule review judges a violation scores as the start in that world). Contrasts pool the held-out
worlds: the difference in mean score between two arms, with Step 2's two-sided permutation test that shuffles arm
labels within each world (1,000,000 seeded shuffles), Step 2's bootstrap interval and Fisher test, and Stage 1's Holm.
Primary, Holm over 4: P1 g1 - g0, P2 g1 - facts, P3 g1 - nologs, P4 final - g1 (P4 uses only lineages whose final isn't
g0, with g1 from the same lineages; if none, P4 enters Holm with p = 1). Each passes if positive with Holm p < 0.05.
Check: at least 75% (9 of 12) of the check arm's searches score >= 38.3%. Endings A-E as fixed in the protocol.
usage (project root): .venv/bin/python -I -B arch_rsi/stageH/step3/analyze_step3.py --campaign DIR
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3common as C  # noqa: E402

import analyze_step2 as A2  # noqa: E402  (Step 2's tests, frozen)
from arch_rsi.stage1.analyze_stage1 import holm  # noqa: E402

N_PERM, ALPHA, CHECK_SHARE = A2.N_PERM, A2.ALPHA, C.CHECK_FOUND / C.CHECK_N
PRIMARY = {"P1": ("g1", "g0"), "P2": ("g1", "facts"), "P3": ("g1", "nologs"), "P4": ("final", "g1")}
ENDINGS = {
    "A": ("Rewriting its own research procedure, in worlds of other kinds, helped the AI scientist find fixes from scores "
          "alone in worlds of a kind it never saw. That beat a rewrite that could only add facts and one written without "
          "evidence, and the gain grew across its own rewrites."),
    "B": ("Rewriting its own research procedure, in worlds of other kinds, helped the AI scientist find fixes from scores "
          "alone in worlds of a kind it never saw. That beat a rewrite that could only add facts and one written without "
          "evidence. Further rewrites didn't measurably add to it."),
    "C": ("One rewrite of its own procedure helped the AI scientist find fixes from scores alone in worlds of a kind it "
          "never saw. The test didn't show that the procedure, rather than {which}, made the difference."),
    "D": ("Even with readouts, it couldn't reliably fix the held-out worlds, so this test can't say whether a better "
          "procedure would help."),
    "E": "Rewriting its own procedure didn't measurably help it find fixes from scores alone in worlds of a kind it never saw.",
}
C_WHICH = {"P2": "added facts", "P3": "the act of rewriting", "both": "added facts or the act of rewriting"}
CAVEAT = ("One model, whose weights never changed; two lineages, at toy scale; every rule re-indexes where things are, "
          "and one family was held out.")
CHANGE_WORDS = {"shift": r"\bshift\w*", "roll": r"\broll\w*", "flip": r"\bflip\w*", "mirror": r"\bmirror\w*",
                "reflect": r"\breflect\w*", "transpose": r"\btranspos\w*", "rotate": r"\brotat\w*", "turn": r"\bturn\w*",
                "permute": r"\bpermut\w*", "remap": r"\bremap\w*", "symmetr": r"\bsymmetr\w*",
                "one hypothesis": r"one (hypothesis|idea|change) per candidate"}


def rows(cdir: Path) -> list:
    summ = json.loads((cdir / "summary.json").read_text())
    rev = json.loads((cdir / "review.json").read_text())["verdicts"] if (cdir / "review.json").exists() else {}
    out = []
    for e in summ["entries"]:
        if e.get("phase") != "test":
            continue
        sc = e["score"]
        viol = e["label"] != "start" and rev.get(e["sha"], {}).get("verdict") == "violation"
        if viol:
            sc = mean(summ["starts"][e["world"]]["confirm"].values())
        out.append({"world": e["world"], "arm": e["arm"], "lineage": e["lineage"], "search": e["search"], "path": e["path"],
                    "label": e["label"], "name": e["name"], "score": sc, "found": sc >= C.FOUND, "violation": viol,
                    "first_round_found": e.get("first_round_found"), "n_trials": e.get("n_trials"),
                    "settings": e.get("settings"), "shared_with": e.get("shared_with"),
                    "reviewed": e["label"] == "start" or e["sha"] in rev})
    return out


def groups(rs: list, arm: str, lineages=None) -> dict:
    ws = sorted({r["world"] for r in rs})
    return {w: [r["score"] for r in rs if r["arm"] == arm and r["world"] == w
                and (lineages is None or r["lineage"] in lineages)] for w in ws}


def contrast(rs, a, b, n_perm, la=None, lb=None) -> dict:
    ga, gb = groups(rs, a, la), groups(rs, b, lb)
    if not all(ga.get(w) and gb.get(w) for w in set(ga) | set(gb)):
        return {"arms": [a, b], "diff": None, "p": 1.0, "ci95_boot": None, "per_world_found": {}, "note": "an arm has no searches"}
    d = mean(v for w in ga for v in ga[w]) - mean(v for w in gb for v in gb[w])
    fw = {}
    for w in ga:
        fa = sum(r["found"] for r in rs if r["arm"] == a and r["world"] == w and (la is None or r["lineage"] in la))
        fb = sum(r["found"] for r in rs if r["arm"] == b and r["world"] == w and (lb is None or r["lineage"] in lb))
        fw[w] = {"found": [fa, fb], "n": [len(ga[w]), len(gb[w])], "fisher_p": round(A2.fisher(fa, len(ga[w]), fb, len(gb[w])), 4)}
    return {"arms": [a, b], "diff": round(d, 4), "p": A2.perm_p(ga, gb, n_perm), "ci95_boot": A2.boot(ga, gb),
            "per_world_found": fw}


def lineage_facts(summ: dict) -> dict:
    out = {}
    for L, lin in summ["lineages"].items():
        gens = lin["generations"]
        kept_header = [g["generation"] for g in gens if g.get("kept") and g.get("header") != g.get("parent_header")]
        out[L] = {"final": lin["final"], "kept": [g["generation"] for g in gens if g.get("kept")],
                  "kept_header_change": kept_header,
                  "generations": [{k: g.get(k) for k in ("generation", "program", "mean", "search_means", "settings", "kept",
                                                        "parent", "parent_mean", "header_fallback")} for g in gens]}
    return out


def choose_ending(prim: dict, check: dict, header_kept: bool) -> tuple:
    p = {k: v["passes"] for k, v in prim.items()}
    if p["P1"] and p["P2"] and p["P3"] and p["P4"] and header_kept:
        return "A", ENDINGS["A"]
    if p["P1"] and p["P2"] and p["P3"]:
        return "B", ENDINGS["B"]
    if p["P1"]:
        which = C_WHICH["both"] if not p["P2"] and not p["P3"] else C_WHICH["P2"] if not p["P2"] else C_WHICH["P3"]
        return "C", ENDINGS["C"].format(which=which)
    if not check["passes"]:
        return "D", ENDINGS["D"]
    return "E", ENDINGS["E"]


def program_words(cdir: Path) -> dict:
    out = {}
    for p in sorted((cdir / "programs").glob("*/*.md")):
        t = p.read_text()
        out[f"{p.parent.name}/{p.stem}"] = {"words": len(t.split()),
                                            "named": {k: len(re.findall(w, t, re.I)) for k, w in CHANGE_WORDS.items()}}
    return out


def call_costs(cdir: Path) -> dict:
    out = {}
    for p in cdir.rglob("*.json"):
        rel = p.relative_to(cdir).parts
        if p.name.endswith(".failed.json"):
            continue
        if p.name.startswith("round_"):
            key = "search: " + ("meta" if rel[0] == "meta" else rel[1])
        elif p.name.startswith("call_"):
            key = "rewrite"
        elif rel[0] in ("review", "judge", "program_review") and len(rel) == 2:
            key = rel[0]
        else:
            continue
        rec = json.loads(p.read_text())
        if "reply" not in rec:
            continue
        u = rec.get("usage") or {}
        c = out.setdefault(key, {"calls": 0, "cost_usd": 0.0})
        c["calls"] += 1
        c["cost_usd"] = round(c["cost_usd"] + (u.get("notional_cost_usd") or 0.0), 4)
    return out


def analyze(cdir: Path, n_perm=N_PERM) -> dict:
    cdir = Path(cdir)
    summ = json.loads((cdir / "summary.json").read_text())
    rs = rows(cdir)
    lin = lineage_facts(summ)
    p4_lins = [L for L, v in lin.items() if v["final"] != "g0"]
    prim = {}
    for k, (a, b) in PRIMARY.items():
        if k == "P4":
            prim[k] = (contrast(rs, a, b, n_perm, p4_lins, p4_lins) if p4_lins else
                       {"arms": [a, b], "diff": None, "p": 1.0, "ci95_boot": None, "per_world_found": {},
                        "note": "not tested: no lineage kept a rewrite"})
            prim[k]["lineages"] = p4_lins
        else:
            prim[k] = contrast(rs, a, b, n_perm)
    adj = holm({k: v["p"] for k, v in prim.items()})
    for k, v in prim.items():
        v["p_holm"] = adj[k]
        v["passes"] = v["diff"] is not None and v["diff"] > 0 and adj[k] < ALPHA
    chk = [r for r in rs if r["arm"] == "check"]
    need = math.ceil(CHECK_SHARE * len(chk))
    check = {"found": sum(r["found"] for r in chk), "n": len(chk), "needed": need,
             "passes": len(chk) > 0 and sum(r["found"] for r in chk) >= need}
    header_kept = any(v["kept_header_change"] for v in lin.values())
    ending, text = choose_ending(prim, check, header_kept)
    sec = {"final - g0": contrast(rs, "final", "g0", n_perm)}
    for L in lin:
        for k, (a, b) in PRIMARY.items():
            if k == "P4" and L not in p4_lins:
                continue
            la = [L]
            sec[f"{k} lineage {L}"] = contrast(rs, a, b, n_perm, la, None if b == "g0" else la)
    adj2 = holm({k: v["p"] for k, v in sec.items()})
    for k, v in sec.items():
        v["p_holm"] = adj2[k]
    per_arm = {}
    jud = json.loads((cdir / "judge.json").read_text()) if (cdir / "judge.json").exists() else {}
    for a in C.TEST_ARMS:
        g = groups(rs, a)
        mine = [r for r in rs if r["arm"] == a]
        vs = [jud[r["path"]]["verdict"] for r in mine if r["path"] in jud and not r["shared_with"]]
        per_arm[a] = {"mean": round(mean(v for w in g for v in g[w]), 4) if any(g.values()) else None,
                      "by_world": {w: round(mean(v), 4) if v else None for w, v in g.items()},
                      "found": sum(r["found"] for r in mine), "n": len(mine),
                      "found_by_world": {w: sum(r["found"] for r in mine if r["world"] == w) for w in g},
                      "first_round_found": [r["first_round_found"] for r in mine],
                      "trials_per_search": [r["n_trials"] for r in mine],
                      "picks_start": sum(r["label"] == "start" for r in mine),
                      "violations": sum(r["violation"] for r in mine),
                      "judge": {x: vs.count(x) for x in ("yes", "partly", "no")}}
    prv = json.loads((cdir / "program_review.json").read_text()) if (cdir / "program_review.json").exists() else {}
    fd = lambda x: "n/a" if x is None else f"{x:+.3f}"  # noqa: E731
    line = ("; ".join(f"{k} {fd(v['diff'])} (Holm p {v['p_holm']:.4g})" for k, v in prim.items())
            + f"; check {check['found']}/{check['n']} ({'passes' if check['passes'] else 'fails'}); ending {ending}")
    return {"status": "final", "rows": rs, "primary": prim, "secondary": sec, "check": check, "per_arm": per_arm,
            "lineages": lin, "kept_rewrite_changed_header": header_kept, "ending": ending,
            "ending_text": text + " " + CAVEAT, "notes": summ.get("notes", {}),
            "all_reviewed": all(r["reviewed"] for r in rs if not r["shared_with"]),
            "program_review": {h: {"verdict": v["verdict"], "programs": v.get("programs")} for h, v in prv.items()},
            "programs": program_words(cdir), "calls": call_costs(cdir), "summary_line": line}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    res = analyze(Path(args.campaign))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1)[:6000])
