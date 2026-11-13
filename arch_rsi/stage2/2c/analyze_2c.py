"""Pre-registered analysis of Stage 2c (PROTOCOL_2C.md, "Analysis"). Written on 10 November 2026, before any Stage 2c
result, and before Stage 2b finished or was analysed.

usage (project root): .venv/bin/python -I arch_rsi/stage2/2c/analyze_2c.py --campaign arch_rsi/campaigns/stage2c
Reads W's dev results only, never W's final split. Writes results.json in the campaign directory. The exact
permutation tests are Stage 1's (every split enumerated), so with 16 searches per arm each takes several minutes.

Unit: one search on W, scored as the mean over its pick's 8 confirmation seeds (a failed seed scores 0).
Primary score: a pick that review_2c.py judges to break a rule scores as the start with Stage G's recipe on the same
seeds. Those are 2b's start_g comparator runs (same trainer, world, budget and seeds; 2c doesn't rerun them). The
as-computed score is reported beside it. Without review.json the primary family is "pending review".

Primary family, Holm over 4: P1 g1 - g0, P2 g1 - facts-only, P3 g1 - shuffled, P4 final - g1.
- P4 uses only lineages whose final program isn't g0 (if nothing was kept, P4 isn't tested for that lineage), with g1
  from the same lineages. If no lineage qualifies, P4 is "not tested" and enters Holm with p = 1, so the family
  stays at 4.
Tiers (their words are used only if met):
- "Level 2": P1, P2 and P3 all positive, each with Holm-adjusted p < 0.05;
- "recursive": Level 2, plus P4 positive with Holm-adjusted p < 0.05, plus at least one kept rewrite that changed
  the header.
Secondary (reported only; Holm within the family): final - g0; P1-P4 within each lineage; final - 2b's best arm
(the 2b search arm with the highest primary mean in 2b's results.json: a choice made after 2b, labelled as such).
Descriptive: the programs and their diffs from g0, header values and keep decisions per generation, training runs
spent per search, model calls and notional cost, time-outs, the g0 reuse check and header fallbacks.
"""
import argparse
import difflib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S2 = HERE.parent
A = S2.parent
ROOT = A.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import prog2c as P  # noqa: E402
from arch_rsi.sandbox import sha  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AS1 = _load("analyze_stage1", A / "stage1" / "analyze_stage1.py")
AN2 = _load("analyze_2b", S2 / "2b" / "analyze_2b.py")
CAMP_2B = A / "campaigns" / "stage2b"
mean = AS1.mean
PRIMARY = {"P1": ("g1", "g0"), "P2": ("g1", "facts"), "P3": ("g1", "shuffled"), "P4": ("final", "g1@p4")}
TIERS = {"level2": ("P1", "P2", "P3"), "recursive": ("P1", "P2", "P3", "P4")}


def conf_dir(cdir: Path, e: dict) -> Path:
    if e.get("reused_from") == "stage2b":
        return CAMP_2B / "confirm" / e["sha"]
    return cdir / "confirm" / "w" / e["sha"]


def seed_detail(d: Path, s, H: np.ndarray) -> dict:
    """AS1.seed_detail for a given confirmation directory."""
    f = d / f"seed{s}" / "dev.json"
    if not f.exists():
        return {"hidden": 0.0, "learned": False}
    p = np.array(json.loads(f.read_text())["per_design"]["exact"], bool)
    hid = float(p[H].mean())
    return {"hidden": hid, "learned": hid >= AS1.LEARNED}


def start_scores_2b() -> dict:
    """2b's start_g comparator, per confirmation seed (a failed seed scores 0), as analyze_2b uses it."""
    comp = {e["arm"]: e for e in json.loads((CAMP_2B / "confirm" / "summary.json").read_text()) if e["kind"] == "comparator"}
    sg = comp["start_g"]
    return {str(s): (sg["dev"].get(str(s)) if sg["status"].get(str(s)) == "ok" else 0.0) or 0.0 for s in sg["seeds"]}


def score_entries(cdir: Path, entries: list, review, H, start_by_seed: dict) -> list:
    rows = []
    for e in entries:
        seeds = [str(s) for s in e["seeds"]]
        sc = [(e["dev"].get(s) if e["status"].get(s) == "ok" else 0.0) or 0.0 for s in seeds]
        d = conf_dir(cdir, e)
        det = [seed_detail(d, s, H) if e["status"].get(s) == "ok" else {"hidden": 0.0, "learned": False} for s in seeds]
        if e["label"] == "start":
            verdict = "compliant (by construction)"
        elif review is None:
            verdict = "pending"
        elif e["sha"] not in review:
            raise SystemExit(f"review.json has no verdict for pick {e['sha']} ({e['arm']} {e.get('lineage')} s{e['search']})")
        else:
            verdict = review[e["sha"]]["verdict"]
        x = mean(sc)
        y = mean([start_by_seed[s] for s in seeds]) if verdict == "violation" else x
        src = (d / "candidate.py").read_text()
        rows.append({"arm": e["arm"], "lineage": e.get("lineage"), "search": e["search"], "pick": e["label"],
                     "name": e["name"], "sha": e["sha"], "verdict": verdict, "score": round(x, 4), "primary_score": round(y, 4),
                     "_x": x, "_y": y, "selection_mean": round(e["mean_selected"], 4), "per_seed": [round(v, 4) for v in sc],
                     "seeds_learned": sum(q["learned"] for q in det),
                     "timeouts": sum(e["status"].get(s) == "timeout" for s in seeds), "recipe": AN2.recipe_of(src),
                     "shared_with": e.get("shared_with"), "reused_from": e.get("reused_from")})
    return rows


def lineage_info(cdir: Path) -> dict:
    out = {}
    for f in sorted((cdir / "meta").glob("*/lineage.json")):
        lin = json.loads(f.read_text())
        L = lin["lineage"]
        pdir = cdir / "programs" / L
        header_kept = []
        for row in lin["generations"][1:]:
            if row.get("kept"):
                h_new = P.parse_program((pdir / f"{row['program']}.md").read_text())[0]
                h_par = P.parse_program((pdir / f"{row['parent']}.md").read_text())[0]
                header_kept.append({"program": row["program"], "header_changed": h_new != h_par})
        out[L] = {"final": lin["final"], "kept": header_kept,
                  "kept_header_change": any(k["header_changed"] for k in header_kept), "generations": lin["generations"]}
    return out


def family4(groups: dict, contrasts: dict) -> dict:
    """AS1.family with the family size fixed: a contrast that can't be tested enters Holm with p = 1."""
    out, ps = {}, {}
    for name, (a, b) in contrasts.items():
        ga, gb = groups.get(a) or [], groups.get(b) or []
        if ga and gb:
            p = AS1.perm_p(ga, gb)
            out[name] = {"contrast": f"{a} - {b}", "diff": round(mean(ga) - mean(gb), 4), "p": round(p, 4),
                         "ci95_boot": AS1.boot_diff(ga, gb), "n": [len(ga), len(gb)], "tested": True}
        else:
            p = 1.0
            out[name] = {"contrast": f"{a} - {b}", "diff": None, "p": None, "ci95_boot": None, "n": [len(ga), len(gb)],
                         "tested": False}
        ps[name] = p
        print(f"  {name} {a} - {b}: n {out[name]['n']}, p {out[name]['p']}", flush=True)
    for k, p in AS1.holm(ps).items():
        out[k]["p_holm"] = round(p, 4) if out[k]["tested"] else None
    return out


def tiers(prim: dict, lin: dict, pending: bool) -> dict:
    def ok(k):
        c = prim[k]
        return bool(c["tested"] and c["diff"] > 0 and c["p_holm"] < 0.05)
    if pending:
        return {"status": "pending review"}
    l2 = all(ok(k) for k in TIERS["level2"])
    header = any(v["kept_header_change"] for v in lin.values())
    rec = l2 and ok("P4") and header
    return {"status": "final", "level2": l2, "recursive": rec, "kept_rewrite_changed_header": header,
            "words_allowed": ("recursive" if rec else "Level 2" if l2 else "none: report as measured")}


def groups_for(rows: list, key: str, lin: dict) -> dict:
    g = {}
    for r in rows:
        g.setdefault(r["arm"], []).append(r[key])
        if r["lineage"]:
            g.setdefault(f"{r['arm']}@{r['lineage']}", []).append(r[key])
    p4 = [L for L, v in lin.items() if v["final"] != "g0"]
    g["g1@p4"] = [r[key] for r in rows if r["arm"] == "g1" and r["lineage"] in p4]
    g["final"] = [r[key] for r in rows if r["arm"] == "final" and r["lineage"] in p4]
    return g


def best_2b_arm():
    f = CAMP_2B / "results.json"
    if not f.exists():
        return None, []
    res = json.loads(f.read_text())
    arms = {a: d for a, d in res["per_arm"].items() if "searches" in d}
    if not arms:
        return None, []
    best = max(arms, key=lambda a: arms[a]["primary_mean"])
    return best, [s["primary_score"] for s in arms[best]["searches"]]


def per_arm(rows: list) -> dict:
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        ss = [r for r in rows if r["arm"] == arm]
        out[arm] = {"searches": [{k: v for k, v in r.items() if not k.startswith("_")} for r in ss],
                    "n": len(ss), "mean": round(mean([r["_x"] for r in ss]), 4),
                    "primary_mean": round(mean([r["_y"] for r in ss]), 4),
                    "sd_between_searches": round(statistics.stdev([r["_x"] for r in ss]), 4) if len(ss) > 1 else None,
                    "per_lineage_mean": {str(L): round(mean([r["_x"] for r in ss if r["lineage"] == L]), 4)
                                         for L in sorted({r["lineage"] for r in ss}, key=str)},
                    "share_seeds_learned": round(sum(r["seeds_learned"] for r in ss) / sum(len(r["per_seed"]) for r in ss), 4),
                    "selection_optimism": round(mean([r["selection_mean"] - r["score"] for r in ss]), 4),
                    "picked_start": sum(r["pick"] == "start" for r in ss),
                    "picks_with_recipe": sum(r["recipe"] is not None for r in ss),
                    "violations": sum(r["verdict"] == "violation" for r in ss),
                    "confirm_timeouts": sum(r["timeouts"] for r in ss)}
    return out


def programs(cdir: Path) -> dict:
    out = {}
    for pdir in sorted((cdir / "programs").glob("*")):
        g0 = (pdir / "g0.md").read_text()
        out[pdir.name] = {}
        for f in sorted(pdir.glob("*.md")):
            txt = f.read_text()
            h, body = P.parse_program(txt)
            rv = pdir / f"revision_{f.stem}.json"
            r = json.loads(rv.read_text()) if rv.exists() else {}
            diff = list(difflib.unified_diff(g0.splitlines(), txt.splitlines(), "g0", f.stem, lineterm="", n=1))
            out[pdir.name][f.stem] = {"sha": sha(txt), "words": len(txt.split()), "header": h, "settings_runs": P.runs_per_search(h),
                                      "diff_from_g0": diff[:600], "diff_lines": len(diff),
                                      "header_fallback": (r.get("result") or {}).get("header_fallback"),
                                      "written_from": r.get("parent"), "kind": r.get("kind", "start" if f.stem == "g0" else None)}
    return out


def spend(cdir: Path) -> dict:
    """Training runs per search (trial and retest seeds) and model calls with notional cost, by part of the campaign."""
    out = {"runs_per_search": {}, "calls": {}, "trial_timeouts": {}}

    def part(p: Path) -> str:
        rel = p.relative_to(cdir).parts
        return "meta-training (W_s)" if rel[0] == "meta" else f"W/{rel[1]}"
    for st_file in sorted(cdir.glob("**/state.json")):
        sdir = st_file.parent
        k = part(sdir)
        n = len(list(sdir.glob("trials/*/seed*/result.json")))
        out["runs_per_search"].setdefault(k, []).append(n)
        st = json.loads(st_file.read_text())
        out["trial_timeouts"][k] = out["trial_timeouts"].get(k, 0) + sum(t["result"]["status"] == "timeout" for t in st["trials"])
    out["runs_per_search"] = {k: {"searches": len(v), "mean": round(mean(v), 1), "min": min(v), "max": max(v)}
                              for k, v in out["runs_per_search"].items()}
    calls = [("search rounds and repairs", f, part(f.parent)) for f in cdir.glob("**/round_*.json")]
    calls += [("rewrites", f, "rewrites") for f in cdir.glob("programs/*/call_*.json")]
    calls += [("reviews", f, "reviews") for f in list(cdir.glob("review/*.json")) + list(cdir.glob("program_review/*.json"))]
    for kind, f, k in calls:
        if f.name.endswith(".failed.json"):
            continue
        u = json.loads(f.read_text()).get("usage", {}) or {}
        c = out["calls"].setdefault(k, {"calls": 0, "notional_usd": 0.0, "output_tokens": 0})
        c["calls"] += 1
        c["notional_usd"] = round(c["notional_usd"] + (u.get("notional_cost_usd") or 0.0), 2)
        c["output_tokens"] += u.get("output_tokens") or 0
    return out


def analyse(cdir: Path, entries: list, review, H, start_by_seed: dict) -> dict:
    lin = lineage_info(cdir)
    rows = score_entries(cdir, entries, review, H, start_by_seed)
    pending = review is None
    key = "_x" if pending else "_y"
    g = groups_for(rows, key, lin)
    gx = groups_for(rows, "_x", lin)
    print("primary family:", flush=True)
    prim = family4(g, PRIMARY)
    print("as-computed primary family:", flush=True)
    prim_x = family4(gx, PRIMARY) if not pending else prim
    sec_pairs = [("final", "g0")]
    for L in sorted(lin):
        sec_pairs += [(f"g1@{L}", "g0"), (f"g1@{L}", f"facts@{L}"), (f"g1@{L}", f"shuffled@{L}")]
        if lin[L]["final"] != "g0":
            sec_pairs.append((f"final@{L}", f"g1@{L}"))
    best, best_scores = best_2b_arm()
    if best:
        g[f"2b:{best}"] = best_scores
        sec_pairs.append(("final", f"2b:{best}"))
    print("secondary family:", flush=True)
    sec = AS1.family(g, sec_pairs)
    return {"per_arm": per_arm(rows),
            "primary_family": {"status": "pending review" if pending else "final", "contrasts": prim,
                               "p4_lineages": [L for L, v in lin.items() if v["final"] != "g0"]},
            "tiers": tiers(prim, lin, pending),
            "as_computed_primary_family": prim_x,
            "secondary_family": {"contrasts": sec, "best_2b_arm": best,
                                 "note": "2b's best arm was chosen after 2b's results (post hoc); reported only"},
            "lineages": lin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    cdir = Path(args.campaign).resolve()
    man = json.loads((cdir / "manifest.json").read_text())
    entries = json.loads((cdir / "confirm" / "summary.json").read_text())
    rv = cdir / "review.json"
    review = json.loads(rv.read_text())["verdicts"] if rv.exists() else None
    H = AN2.hidden_flags()
    res = {"campaign": str(cdir.relative_to(ROOT)) if cdir.is_relative_to(ROOT) else str(cdir), "preset": man["preset"],
           "split": "dev (W final locked)", "learned_threshold": AS1.LEARNED,
           "dev_designs_with_hidden_cells": int(H.sum()),
           "baselines": {w: json.loads((cdir / "baseline" / w / "summary.json").read_text()) for w in ("ws", "w")},
           "g0_reuse": json.loads((cdir / "W" / "g0_reuse.json").read_text()),
           "final_notes": json.loads((cdir / "W" / "final_notes.json").read_text())}
    res |= analyse(cdir, entries, review, H, start_scores_2b())
    res["programs"] = programs(cdir)
    res["spend"] = spend(cdir)
    pr = cdir / "program_review.json"
    res["program_review"] = json.loads(pr.read_text())["programs"] if pr.exists() else "not run"
    (cdir / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("primary_family", "tiers")}, indent=1))


if __name__ == "__main__":
    main()
