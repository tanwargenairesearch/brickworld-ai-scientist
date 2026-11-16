"""Finishes Step 3 with lineage B's final set to g2 (a departure, decided by the user on 16 Nov 2026).

Why: lineage B's g3 rewrite could not be written. All 9 attempts, over 3 runs, were over the protocol's 1,000-word
limit (1,019 to 1,250 words). The frozen runner (run_step3.py) needs every lineage to finish all 3 rewrites before
it writes summary.json, so it stopped. The user chose to treat B's g3 as not written and B's final as its last kept
program, g2, which is the protocol's own definition of final ("the last kept program").

What changes: only Step3.lineage(), copied here from the frozen run_step3.py (inspect.getsource) with the lines marked
"CHANGED". For lineage B the generation loop stops after g2 instead of g3, and the departure is written into
lineage.json, summary.json's notes and progress.log. Lineage A runs exactly as before.

Everything else is the frozen code, imported unchanged: the integrity check (hash_checks; this file is not in any frozen
set), the cached searches and calls (nothing finished is redone, and no saved call is repeated), final B's 12 new
searches, the rule review, the diagnosis judge, the program review, analyze_step3 and results.json.
usage (project root), under the usage-limit auto-restart:
  bash arch_rsi/stageH/step3/resume_on_limit.sh caffeinate -is nice -n 10 .venv/bin/python -I -B -u \
      arch_rsi/stageH/step3/finish_step3_b_final_g2.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_step3 as R  # noqa: E402  (frozen; imported, never changed)

PG, sha, C = R.PG, R.sha, R.C
STOP_AFTER = {"B": 2}
NOTE = ("g3 not written: all 9 attempts (3 runs) were over the 1,000-word limit (1,019-1,250 words), so B's final is "
        "its last kept program, g2 (departure decided by the user, 16 Nov 2026)")


class Finish(R.Step3):
    def lineage(self, L: str) -> dict:
        c = self.cfg
        pdir = self.dir / "programs" / L
        pdir.mkdir(parents=True, exist_ok=True)
        if (pdir / "g0.md").exists() and (pdir / "g0.md").read_text() != self.g0:
            raise SystemExit(f"{pdir / 'g0.md'} differs from g0")
        (pdir / "g0.md").write_text(self.g0)
        (pdir / "labels.json").write_text(json.dumps(self.labels[L], indent=1))
        parent, cur = {"name": "g0", "text": self.g0}, {"name": "g0", "text": self.g0}
        parent_row, hist, out = None, [], {"g0": self.g0}
        last = STOP_AFTER.get(L, c["rewrites"])  # CHANGED: B stops after g2 (its g3 could not be written)
        for g in range(last + 1):  # CHANGED: was range(c["rewrites"] + 1)
            row = self.generation(L, g, cur["name"], cur["text"])
            if g == 0:
                row["kept"], parent_row = None, row
            else:
                rv = json.loads((pdir / f"revision_{cur['name']}.json").read_text())
                row |= {"parent": parent["name"], "parent_mean": parent_row["mean"], "kept": row["mean"] > parent_row["mean"],
                        "changes": rv["result"]["changes"], "header_fallback": rv["result"].get("header_fallback", False),
                        "header": PG.parse_program(cur["text"])[0], "parent_header": PG.parse_program(parent["text"])[0]}
                if row["kept"]:
                    parent, parent_row = cur, row
            hist.append(row)
            self.log(f"lineage {L} generation {g} ({cur['name']}): mean confirmed {row['mean']:.3f}"
                     + ("" if g == 0 else f" vs {row['parent_mean']:.3f} for {row['parent']}: {'kept' if row['kept'] else 'not kept'}"))
            if g == last:  # CHANGED: was c["rewrites"]
                break
            if g == 0:
                with ThreadPoolExecutor(3) as ex:
                    fs = {k: ex.submit(self.rewrite, L, k, 0, parent, row, hist) for k in ("real", "facts", "nologs")}
                    texts = {k: f.result() for k, f in fs.items()}
                out |= {"g1": texts["real"], "facts": texts["facts"], "nologs": texts["nologs"]}
                for arm in ("g1", "facts", "nologs"):
                    self.submit_arm(arm, L, out[arm])
                cur = {"name": "g1", "text": texts["real"]}
            else:
                cur = {"name": f"g{g + 1}", "text": self.rewrite(L, "real", g, parent, row, hist)}
        out["final"], out["final_name"] = parent["text"], parent["name"]
        if sha(out["final"]) == sha(out["g0"]):
            self.notes[L] = "final is g0: nothing was kept"
        elif sha(out["final"]) == sha(out["g1"]):
            self.notes[L] = "final is g1: searches shared"
        else:
            self.submit_arm("final", L, out["final"])
        summary = {"lineage": L, "final": parent["name"], "labels": self.labels[L],
                   "programs": {k: sha(v) for k, v in out.items() if k != "final_name"},
                   "generations": [{k: v for k, v in r.items() if k != "searches"} for r in hist]}
        if L in STOP_AFTER:  # CHANGED: record the departure in lineage.json, summary.json and the log
            summary["departure"] = NOTE
            self.notes[L] = NOTE
            self.log(f"lineage {L}: {NOTE}")
        (self.dir / "meta" / L / "lineage.json").write_text(json.dumps(summary, indent=1))
        return out


def main():
    for f, what in ((C.STEP2_RESULTS, "Step 2"), (C.STEP3A_RESULTS, "Step 3a")):
        if not f.exists() or json.loads(f.read_text()).get("status") != "final":
            raise SystemExit(f"Step 3 needs {what}'s final results: {f}")
    Finish(C.STEP3 / R.PRESETS["step3"]["campaign"], "step3").run()


if __name__ == "__main__":
    main()
