"""Tests for the architecture-search harness. Run: .venv/bin/python arch_rsi/tests/test_arch.py (exits non-zero on failure)."""
import json
import subprocess
import sys
from pathlib import Path

import tempfile
import traceback
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arch_rsi.check import check  # noqa: E402
from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.menu import MENU, source  # noqa: E402
from arch_rsi.sandbox import load_candidate, static_check  # noqa: E402
from arch_rsi.worlds import SPECS  # noqa: E402
from brickworld.env import design_hash, zobrist_codes  # noqa: E402
from brickworld.model import Policy  # noqa: E402
from brickworld.space import BASELINE  # noqa: E402

HERE = ROOT / "arch_rsi"
PY = str(ROOT / ".venv" / "bin" / "python")
BASE = (HERE / "candidates" / "baseline.py").read_text()

ESCAPES = {
    "import os": "import os\ndef build_policy(spec):\n    return None\n",
    "import subprocess via from": "from subprocess import run\ndef build_policy(spec):\n    pass\n",
    "open": "def build_policy(spec):\n    open('x')\n",
    "dunder": "def build_policy(spec):\n    return ().__class__.__mro__\n",
    "torch.load": "import torch\ndef build_policy(spec):\n    return torch.load('x')\n",
    "aliased torch.save": "import torch as T\ndef build_policy(spec):\n    T.save(1, 'x')\n",
    "from torch import load": "from torch import load\ndef build_policy(spec):\n    pass\n",
    "from torch.nn import utils": "from torch.nn import utils\ndef build_policy(spec):\n    pass\n",
    "top-level call": "import torch\nx = torch.zeros(3)\ndef build_policy(spec):\n    pass\n",
    "str.format attribute leak": "def build_policy(spec):\n    return '{0.x}'.format(spec)\n",
    "monkeypatch": "import torch\ndef build_policy(spec):\n    torch.cat = None\n",
    "seeding": "import torch\ndef build_policy(spec):\n    torch.manual_seed(1)\n",
    "eval": "def build_policy(spec):\n    return eval('1')\n",
    "getattr": "def build_policy(spec):\n    return getattr(spec, 'X')\n",
    "global": "def build_policy(spec):\n    global q\n",
    "no build_policy": "import torch\n",
    "module reached through torch": "import torch\ndef build_policy(spec):\n    return torch.os.system\n",
    "module from-import through F": "from torch.nn.functional import warnings\ndef build_policy(spec):\n    pass\n",
}


def test_static_check_blocks():
    allowed = [n for n, s in ESCAPES.items() if not static_check(s)]
    assert not allowed, allowed


def test_restricted_builtins():
    from arch_rsi.sandbox import _safe_builtins
    b = _safe_builtins()
    for name in ("open", "eval", "exec", "compile", "getattr", "setattr", "globals", "input", "breakpoint"):
        assert name not in b, name


def test_ordinary_torch_calls_work_in_sandbox():
    src = ("import torch\nimport torch.nn as nn\n\nclass N(nn.Module):\n    def __init__(self, spec):\n"
           "        super().__init__()\n        self.w = nn.Parameter(torch.tensor([1.0, 2.0]))\n"
           "        self.lin = nn.Linear(spec.in_channels * spec.Y * spec.X, spec.n_actions + 1)\n"
           "    def forward(self, obs, mask):\n        z = torch.arange(3, dtype=obs.dtype)\n"
           "        o = self.lin(obs.flatten(1)) + self.w.sum() + z.sum()\n        return o[:, :-1], o[:, -1]\n\n"
           "def build_policy(spec):\n    return N(spec)\n")
    assert check(src, "large")["ok"]


def test_hand_written_policy_is_rejected():
    src = (HERE / "tests" / "fixtures" / "handcoded_prior.py").read_text()
    r = check(src, "large")
    assert not r["ok"] and "must be learned" in r["error"] and r["check_untrained"]["exact"] > 0.5, r
    # the same score with DONE left to learning builds nothing exactly, but fills target cells cleanly: caught
    nodone = src.replace("done = self.done_w[0] * nomiss + self.done_w[1] + RES_SCALE * self.done_res(g).squeeze(1)",
                         "done = self.done_w[1] * 0.0 - 50.0 + self.done_res(g).squeeze(1)")
    r = check(nodone, "large")
    assert not r["ok"] and r["check_untrained"]["exact"] <= 0.02 and r["check_untrained"]["clean_fill"] > 0.9, r
    # the score behind a learnable gate that starts at zero passes the untrained checks, but builds too much
    # after the short training run: caught
    gated = (src.replace("place = self.prior(obs) + RES_SCALE * self.res_head(f)",
                         "place = self.gate * self.prior(obs) + RES_SCALE * self.res_head(f)")
             .replace("        self.done_w = nn.Parameter(torch.tensor([8.0, -4.0]))",
                      "        self.done_w = nn.Parameter(torch.tensor([8.0, -4.0]))\n        self.gate = nn.Parameter(torch.zeros(()))"))
    r = check(gated, "large")
    assert r["check_untrained"]["clean_fill"] <= 0.10 and not r["ok"] and "training steps" in r["error"], r
    # Known limit (stated in the protocol): slower-opening gates or hand-computed features feeding a learned head
    # can pass the mechanical checks; the rule text and an independent review of every selected file cover them.


def test_baseline_candidate_equals_frozen_policy():
    spec = SPECS["large"]
    sp = arch_spec(spec)
    torch.manual_seed(3)
    ref = Policy(spec, dict(BASELINE))
    torch.manual_seed(3)
    cand = CandidatePolicy(load_candidate(BASE)(sp), sp)
    obs = torch.rand(9, sp.in_channels, sp.Y, sp.X)
    mask = torch.rand(9, sp.n_actions) < 0.6
    mask[:, -1] = True
    a, b = ref(obs, mask), cand(obs, mask)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_menu_is_144_valid_files_inside_caps():
    assert len(MENU) == 144
    assert all(not static_check(source(c)) for c in MENU)
    big = {"encoder": "conv", "width": 128, "depth": 4, "activation": "gelu", "norm": "layernorm"}
    assert check(source(big), "xl")["ok"]


def test_flop_cap_rejects_huge_network():
    src = BASE.replace("WIDTH = 64", "WIDTH = 300").replace("DEPTH = 2", "DEPTH = 3")
    r = check(src, "xl")
    assert not r["ok"] and ("cap" in r["error"])


def test_wrapper_masks_and_checks_shapes():
    sp = arch_spec(SPECS["xl"])
    pol = CandidatePolicy(load_candidate(BASE)(sp), sp)
    obs = torch.rand(4, sp.in_channels, sp.Y, sp.X)
    mask = torch.zeros(4, sp.n_actions, dtype=torch.bool)
    mask[:, -1] = True
    logits, _ = pol(obs, mask)
    assert (logits[:, :-1] == -1e9).all()
    bad = BASE.replace("return logits, self.value_head(g).squeeze(1)", "return logits[:, :-1], self.value_head(g).squeeze(1)")
    pol2 = CandidatePolicy(load_candidate(bad)(sp), sp)
    try:
        pol2(obs, torch.ones(4, sp.n_actions, dtype=torch.bool))
    except ValueError:
        return
    raise AssertionError("wrong logits shape was accepted")


def test_xl_splits_disjoint_and_hidden_from_stream():
    dev = torch.load(HERE / "targets" / "xl_dev.pt")
    fin = torch.load(HERE / "holdout" / "xl_final.pt")
    codes = zobrist_codes(SPECS["xl"])
    dh, fh = set(design_hash(dev, codes).tolist()), set(design_hash(fin, codes).tolist())
    assert len(dh) == 512 and len(fh) == 1024 and not dh & fh
    from arch_rsi.train_arch import heldout_hashes
    from brickworld.env import StreamSource
    from train import TRAIN_SEED_BASE
    stream = StreamSource(SPECS["xl"], TRAIN_SEED_BASE, heldout_hashes("xl"))
    got = set(design_hash(stream(5000), codes).tolist())
    assert not got & (dh | fh)


def test_train_and_evaluate_tiny(tmp_path):
    spec = tmp_path / "run.json"
    spec.write_text(json.dumps({"candidate": str(HERE / "candidates" / "baseline.py"), "world": "xl", "seed": 0,
                                "budget_steps": 2048, "threads": 1}))
    subprocess.run([PY, "-I", str(HERE / "train_arch.py"), "--spec", str(spec), "--out", str(tmp_path / "t")], check=True,
                   capture_output=True)
    log = json.loads((tmp_path / "t" / "train_log.json").read_text())
    assert log["status"] == "ok" and log["steps"] == 2048
    subprocess.run([PY, "-I", str(HERE / "evaluate_arch.py"), "--trial", str(tmp_path / "t"), "--split", "dev",
                    "--out", str(tmp_path / "dev.json")], check=True, capture_output=True)
    assert json.loads((tmp_path / "dev.json").read_text())["n"] == 512
    p = subprocess.run([PY, "-I", str(HERE / "evaluate_arch.py"), "--trial", str(tmp_path / "t"), "--split", "final",
                        "--out", str(tmp_path / "f.json"), "--campaign", str(tmp_path)], capture_output=True, text=True)
    assert p.returncode != 0 and "refused" in (p.stderr + p.stdout)
    # tampered candidate file is refused
    (tmp_path / "t" / "candidate.py").write_text(BASE + "\n# edit\n")
    p = subprocess.run([PY, "-I", str(HERE / "evaluate_arch.py"), "--trial", str(tmp_path / "t"), "--split", "dev",
                        "--out", str(tmp_path / "d2.json")], capture_output=True, text=True)
    assert p.returncode != 0


def test_compute_matched_steps(tmp_path):
    src = BASE.replace("DEPTH = 2", "DEPTH = 4")
    cand = tmp_path / "c.py"
    cand.write_text(src)
    spec = tmp_path / "run.json"
    spec.write_text(json.dumps({"candidate": str(cand), "world": "large", "seed": 0, "budget_steps": 8192}))
    subprocess.run([PY, "-I", str(HERE / "train_arch.py"), "--spec", str(spec), "--out", str(tmp_path / "t")], check=True,
                   capture_output=True)
    log = json.loads((tmp_path / "t" / "train_log.json").read_text())
    r = log["flops_ratio"]
    assert r > 1.5 and log["n_updates"] == int(8 * min(1.0, 1.0 / r)) and log["steps"] == 1024 * log["n_updates"]


def test_invalid_candidate_is_logged_not_trained(tmp_path):
    cand = tmp_path / "c.py"
    cand.write_text(ESCAPES["import os"])
    spec = tmp_path / "run.json"
    spec.write_text(json.dumps({"candidate": str(cand), "world": "large", "seed": 0, "budget_steps": 2048}))
    subprocess.run([PY, "-I", str(HERE / "train_arch.py"), "--spec", str(spec), "--out", str(tmp_path / "t")], check=True,
                   capture_output=True)
    assert json.loads((tmp_path / "t" / "train_log.json").read_text())["status"] == "invalid"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                    with tempfile.TemporaryDirectory() as d:
                        fn(Path(d))
                else:
                    fn()
                print("PASS", name)
            except Exception:  # noqa: BLE001
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
    print(f"{fails} failed")
    sys.exit(1 if fails else 0)
