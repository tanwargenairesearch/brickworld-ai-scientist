"""Step 3a legality check (PROTOCOL_STEP3A.md): stageH/check_h.py with the rules and their design sets taken from
rules3a.py and the short run made by train_3a.py, nothing else changed (make_copies_3a.py, checked by
test_identity_3a.py).

usage (project root): .venv/bin/python -I arch_rsi/stageH/step3a/check_3a.py --candidate FILE --rule R --out OUT.json
"""
import argparse
import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common  # noqa: E402

import torch  # noqa: E402

from arch_rsi.stage2 import w2b  # noqa: E402
from rules3a import DESIGNS, RULES, Rule  # noqa: E402

common.register()
_spec = importlib.util.spec_from_file_location("check_2b", common.ROOT / "arch_rsi" / "stage2" / "check_2b.py")
C2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C2)
DEVICE = torch.device("mps")
BUDGET = 131072
RULE = {"name": None}


def frozen_recipe_src(src: str) -> str:
    """The file with any RECIPE assignment removed and the frozen recipe appended as a RECIPE block."""
    tree = ast.parse(src)
    drop = set()
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RECIPE" for t in n.targets):
            drop.update(range(n.lineno - 1, n.end_lineno))
    lines = src.splitlines(keepends=True)
    return w2b.with_recipe("".join(s for i, s in enumerate(lines) if i not in drop), w2b.FROZEN_AS_BLOCK)


def check_metrics(pol) -> dict:
    """check_2b.check_metrics in the rule world, on the GPU."""
    spec = C2.SPECS[common.WORLD]
    rule = Rule(RULE["name"], spec, DEVICE)
    designs = torch.load(common.TARGETS / f"{DESIGNS[RULE['name']]}_check.pt")
    pol = pol.to(DEVICE).eval()
    w = rule.make_world(len(designs), targets=designs, auto_reset=False)
    n = len(designs)
    clean = torch.zeros(n, device=DEVICE)
    dirty = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        for _ in range(spec.max_steps + 1):
            if bool(w.ended.all()):
                break
            logits, _ = pol(rule.obs(C2.encode(w, "voxels")), rule.mask(w.valid_mask()))
            active = ~w.ended
            info = w.step(rule.act(logits.argmax(dim=1)))
            dirty |= (info["cells_out"] > 0) & active
            clean += torch.where(~dirty & active, info["cells_in"].float(), torch.zeros(n, device=DEVICE))
    exact = float(C2.np_check(w.occ.cpu().numpy(), designs.numpy())["exact"].mean())
    return {"exact": exact, "clean_fill": float((clean.cpu() / designs.flatten(1).sum(1).float()).mean())}


def learned_rule(src: str) -> tuple:
    """check_2b.learned_rule in the rule world: the short run is train_h.py on the GPU with the frozen recipe."""
    aspec = C2.arch_spec(C2.SPECS[common.WORLD])
    torch.manual_seed(0)
    u0 = check_metrics(C2.CandidatePolicy(C2.load_candidate(src)(aspec), aspec))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "c.py").write_text(frozen_recipe_src(src))
        (d / "run.json").write_text(json.dumps({"candidate": str(d / "c.py"), "world": common.WORLD, "rule": RULE["name"],
                                                "seed": 0, "budget_steps": C2.CAPS["short_steps"], "device": "mps",
                                                "threads": 1}))
        p = subprocess.run([sys.executable, "-I", str(HERE / "train_3a.py"), "--spec", str(d / "run.json"),
                            "--out", str(d / "t")], capture_output=True, text=True, timeout=900)
        logp = d / "t" / "train_log.json"
        log = json.loads(logp.read_text()) if logp.exists() else {"status": "crashed", "error": (p.stderr or "")[-400:]}
        if log["status"] != "ok":
            return u0, None, (f"a short training run ({C2.CAPS['short_steps']:,} steps) failed: {log['status']}: "
                              f"{str(log.get('error'))[:400]}")
        ck = torch.load(d / "t" / "ckpt.pt", map_location="cpu")
        torch.manual_seed(0)
        pol = C2.CandidatePolicy(C2.load_candidate(src)(aspec), aspec)
        pol.load_state_dict(ck["state_dict"])
        u1 = check_metrics(pol)
        u1["short_run_updates"] = log.get("n_updates")
    return u0, u1, None


C2.WORLD = common.WORLD
C2.check_metrics = check_metrics
C2.learned_rule = learned_rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--rule", required=True, choices=RULES)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    RULE["name"] = args.rule
    res = C2.check(Path(args.candidate).read_text(), "both", BUDGET)
    res["rule"] = args.rule
    Path(args.out).write_text(json.dumps(res))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
