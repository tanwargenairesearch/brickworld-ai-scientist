"""VISUALIZATION ONLY: the discovery story for LinkedIn's phone feed (4:5, 1080x1350, 24 fps). It follows the post
draft reachouts/prefrontal/linkedin-post-draft-2.md, scene for scene:
   1. the question, with dig.bench's finding (from its paper, arXiv 2608.12593 v2);
   2. the system: three loops;
   3. the problem, in the same 8x8x6 world as the rest: the strong network (reference A) building the design the
      hidden-rule scenes use, from Stage H1's plain-world run (seed 4000);
   4. the first test, as one card: +1.5 points from seeing scores in Stage 1 and Stage 2b (numbers checked against
      both REPORT.md files);
   5. two hidden-rule worlds, from Stage H1's saved runs: what the agent is shown, or where its bricks land, then the
      strong network and the fixed one building the same design;
   6. all four rules' calibration, passes and failures (Stage H1 results.json);
   7. the real test, Step 2's four arms, from its final results file (the scene stops unless that file selects
      ending A, whose wording it uses);
   8. along the way: the hand-coded rules caught, and suggestion 2 (the energy) with the budget caveat;
   9. what this isn't, and what's next.
Rules kept from make_2b_mobile.py (the render stops if one breaks): text at least 28 px, nothing outside the safe area,
no banned word on screen. Builds are recorded in the same batched greedy GPU run the evaluator makes (eval_h.py's
loop), so each must agree with its run's dev.json per-design result. The design shown in scenes 3 and 5 (fixed, and
stated on screen): the first H-plain dev design with hidden cells that the fix's seed-4000 run builds exactly; the
render stops if shift and mirror pick different designs. Nothing here scores or selects.
The video is encoded on the Mac's media engine (h264_videotoolbox).

usage (project root): nice -n 10 .venv/bin/python -I -B arch_rsi/viz/stageH/make_h_mobile.py [--stills-only] [--final]
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGEH = ROOT / "arch_rsi" / "stageH"
RUNS = STAGEH / "runs"
STEP2_RESULTS = STAGEH / "step2" / "campaigns" / "step2" / "results.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M2 = _load("make_2b_mobile", ROOT / "arch_rsi" / "viz" / "stage2b" / "make_2b_mobile.py")
R, PS, SYS = M2.R, M2.PS, M2.SYS
sys.path.insert(0, str(STAGEH))
import common  # noqa: E402  (Stage H: registers W_s)
import torch  # noqa: E402

from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import load_candidate, sha  # noqa: E402
from brickworld.model import encode  # noqa: E402
from rules import Rule, designs_hidden  # noqa: E402

W, H, FPS, M, TOP, BOT = M2.W, M2.H, M2.FPS, M2.M, M2.TOP, M2.BOT
INK, INK2, SURF, GRIDC = M2.INK, M2.INK2, M2.SURF, M2.GRIDC
ARM_COL, ORANGE_D, GREY_D = M2.ARM_COL, M2.ORANGE_D, M2.GREY_D
T, para, canvas, checked = M2.T, M2.para, M2.canvas, M2.checked
REF_COL, FIX_COL, SHOWN_COL = PS.START_COL, ARM_COL, (109, 91, 192)
NOTE_RED = (176, 48, 48)
DEV = torch.device("mps")
RULE_NAME = {"shift": "Shifted booklet", "mirror": "Mirrored hands", "torus": "Wraparound edges", "late": "Late feedback"}
CHANGE = {"shift": "the change: move the view back", "mirror": "the change: flip the input",
          "torus": "the change: circular padding", "late": "the change: longer-horizon discounting"}


def sec(s):
    return int(round(s * FPS))


# ---------------------------------------------------------------- scene 1: the question
def hook_frame():
    img, d = canvas()
    y = TOP + 10
    for ln in ("Can an AI scientist", "discover a rule", "nobody told it?"):
        T(d, (M, y), ln, 76, INK, True)
        y += 86
    y = para(d, (M, y + 34), "dig.bench, a benchmark of 70 games with hidden rules, asks this of agents that play.",
             34, INK, W - 2 * M)
    cw = (W - 2 * M) / 2
    yb = y + 40
    d.line([(M, yb), (W - M, yb)], fill=GRIDC, width=2)
    for k, (val, labs, col) in enumerate((("18 to 69", ("of 70 games won by one model,", "once told the rules"), ARM_COL),
                                          ("no better", ("agent harnesses against", "a plain loop, where compared"), INK))):
        T(d, (M + k * cw, yb + 30), val, 84, col, True)
        for j, lab in enumerate(labs):
            assert R.font(30).getlength(lab) <= cw - 16, lab
            T(d, (M + k * cw + 3, yb + 132 + j * 38), lab, 30, INK2)
    y = yb + 132 + 2 * 38 + 36
    d.line([(M, y), (W - M, y)], fill=GRIDC, width=2)
    y = para(d, (M, y + 34), "Discovery is the bottleneck. I asked the same question one level up: of an AI scientist "
                             "that designs learners.", 38, INK, W - 2 * M, True, 1.22)
    y = para(d, (M, y + 26), "Source: the dig.bench paper, arXiv 2608.12593.", 28, INK2, W - 2 * M)
    assert y <= BOT, y
    return checked(img)


# ---------------------------------------------------------------- scene 2: the system (make_2b_mobile's cards, new text)
CARDS = [
    {"kind": "agent", "head": "Loop 1 · the agent learns to build", "chip": ("RUNS IN EVERY EXPERIMENT", ORANGE_D),
     "does": "It places one brick at a time, gets a reward, and PPO updates its weights.",
     "learns": "Learns: the network's weights."},
    {"kind": "sci", "head": "Loop 2 · the AI scientist searches", "chip": ("STAGES 1–2b, STEP 2", ARM_COL),
     "does": "A hosted LLM reads its earlier experiments and writes 3 new candidates. Frozen code trains and scores "
             "them at equal compute. Control arms change what it gets to see.",
     "learns": "Learns: no weights. It adapts within one search."},
    {"kind": "frozen", "dotted": True, "head": "Loop 3 · the procedure improves itself", "chip": ("STEP 3 · NOT BUILT", GREY_D),
     "does": "It would rewrite its own research program across worlds whose rules differ, so only procedure could "
             "carry over. That is where \"recursive\" would begin.",
     "learns": "Would learn: the research program."}]


def system_frame(focus=None):
    """make_2b_mobile.system_frame's layout with this video's title and card text."""
    img, d = canvas()
    T(d, (M, TOP), "An AI scientist writes the network.", 50, INK, True)
    T(d, (M, TOP + 58), "Can it find what it isn't told?", 50, INK, True)
    T(d, (M, TOP + 128), "The system: three loops, each learning something different.", 30, INK2)
    links = [("each candidate is trained by Loop 1", ORANGE_D), ("confirmed picks would feed Loop 3", GREY_D)]
    pad, tw, bs, blh = 26, W - 2 * M - 52, 30, 1.27
    heights = [128 + len(SYS.wrap(c["does"], R.font(bs), tw)) * bs * blh + 8 + 36 + 22 for c in CARDS]
    top = TOP + 128 + 30 + 24
    gap = (BOT - 4 - top - sum(heights)) / 2
    assert gap >= 48, ("the three cards don't fit", gap)
    y0 = top
    for i, (c, ch) in enumerate(zip(CARDS, heights)):
        box = (M, y0, W - M, y0 + ch)
        fill, line = SYS.KIND[c["kind"]]
        on = focus in (None, i)
        d.rounded_rectangle(box, radius=18, fill=fill if on else R.mix(fill, SURF, 0.6),
                            outline=None if c.get("dotted") else line, width=6 if focus == i else 3)
        if c.get("dotted"):
            SYS.dashed_rect(d, box, line, width=6 if focus == i else 3, dash=12)
        T(d, (M + pad, y0 + 18), c["head"], 38, INK, True)
        cf = R.font(28, True)
        cw = cf.getlength(c["chip"][0])
        d.rounded_rectangle([M + pad, y0 + 70, M + pad + cw + 32, y0 + 116], radius=23, fill=c["chip"][1])
        T(d, (M + pad + 16, y0 + 93), c["chip"][0], 28, (255, 255, 255), True, anchor="lm")
        yy = para(d, (M + pad, y0 + 128), c["does"], bs, INK2, tw, lh=blh)
        T(d, (M + pad, yy + 8), c["learns"], 30, INK, True)
        if i < 2:
            lab, col = links[i]
            ym = y0 + ch + gap / 2
            (a, b) = (ym + 17, ym - 17) if i == 0 else (ym - 17, ym + 17)
            SYS.arrow(d, [(M + 40, a), (M + 40, b)], col=col, width=5, head=14)
            T(d, (M + 68, ym), lab, 28, col, True, anchor="lm")
        y0 += ch + gap
    return checked(img)


def system_frames():
    return [(system_frame(), sec(1.5))] + [(system_frame(i), sec(2.5)) for i in range(3)]


# ---------------------------------------------------------------- scene 3: the problem, in the plain 8x8x6 world
def plain_frame(Hd, nb, done, lift=0.0):
    """nb bricks placed; lift > 0 draws the newest brick that many layers up (one frame of its drop)."""
    img, d = canvas()
    sp, P = Hd["spec"], Hd["plain"]
    y = para(d, (M, TOP), "The problem: build the target design exactly, one brick at a time", 46, INK, W - 2 * M, True, 1.16)
    y = para(d, (M, y + 8), "A network sees the target and what is built so far, then places one brick or presses DONE. "
                            "Only an exact match counts.", 30, INK2, W - 2 * M)
    tgt = R.target_bricks(P["target"].numpy())
    top, th, zt = y + 18, 220, M2.top_layer(tgt)
    sc = iso_fit(sp, 440, th, zt)
    iso = R.Iso(sp, sc, 286, top + zt * sc)
    iso.ground(d)
    iso.bricks(d, tgt, lambda b: R.TARGET_COL)
    T(d, (540, top + 16), "Target design", 36, INK, True)
    para(d, (540, top + 68), f"{P['cells']} cells, some hidden under other bricks. One of {Hd['n']} dev designs, in an "
                             f"{sp.X}×{sp.Y}×{sp.Z} world.", 30, INK2, W - M - 540)
    y2 = top + th + 22
    d.line([(M, y2), (W - M, y2)], fill=GRIDC, width=2)
    T(d, (W / 2, y2 + 22), "The strong network, trained by PPO", 34, INK, True, anchor="ma")
    run = P["run"]
    zb = M2.top_layer(run["bricks"]) + 1                                      # one layer of room for the drop
    sc = iso_fit(sp, 700, 400, zb)
    iso = R.Iso(sp, sc, W / 2, y2 + 80 + zb * sc)
    iso.ground(d, footprint=P["target"].any(0).numpy())
    k = len(run["bricks"]) if done else min(nb, len(run["bricks"]))
    dropping = not done and lift > 0 and 0 < nb <= len(run["bricks"])
    iso.bricks(d, run["bricks"][:k], lambda b: R.WRONG if b["wrong"] else REF_COL, lift=(k - 1, lift) if dropping else None)
    yr = y2 + 80 + ((sp.X + sp.Y) * 0.5 + zb) * sc + 14
    if done:
        msg = "built exactly" if run["exact"] else f"not exact: {run['missing']} cells missing"
        w = R.font(30).getlength(msg)
        R.badge(d, W / 2 - w / 2 - 6, yr + 18, run["exact"], r=16)
        T(d, (W / 2 - w / 2 + 20, yr), msg, 30, INK)
        T(d, (W / 2, yr + 46), f"This run: {run['dev_exact'] * 100:.1f}% of all {Hd['n']} designs. Over 6 seeds: "
                               f"{P['mean'] * 100:.1f}%.", 28, INK2, anchor="ma")
    else:
        T(d, (W / 2, yr), f"bricks placed: {k}", 30, INK2, anchor="ma")
    yb = yr + 96
    d.line([(M, yb), (W - M, yb)], fill=GRIDC, width=2)
    yy = para(d, (M, yb + 14), "An illustration, not evidence. The design, used again in both hidden-rule worlds: the "
                               "first dev design with hidden cells that the changed network builds exactly.",
              28, INK2, W - 2 * M)
    assert yy <= BOT, ("problem scene too tall", yy)
    return checked(img)


def plain_frames(Hd):
    """The target and an empty board, the build brick by brick (each brick drops in), then the result held."""
    n = len(Hd["plain"]["run"]["bricks"])
    out = [(plain_frame(Hd, 0, False), sec(1.5))]
    for k in range(1, n + 1):
        out += [(plain_frame(Hd, k, False, lift), 1) for lift in (2.0, 1.1, 0.4)] + [(plain_frame(Hd, k, False), 3)]
    return out + [(plain_frame(Hd, n, True), sec(4.5))]


# ---------------------------------------------------------------- scene 4: the first test, as one card
FIRST_TEST = {   # each string must appear in its report's results table (checked in first_test_check)
    "stage1": {"report": "arch_rsi/stage1/REPORT.md", "rows": ("| +1.5 points | −9.1 to +12.0 |", "| +52.5 points |")},
    "stage2b": {"report": "arch_rsi/stage2/2b/REPORT.md", "rows": ("| +1.5 | −10.1 to +12.2 |", "| +43.6 |")}}


def first_test_check():
    for k, v in FIRST_TEST.items():
        txt = (ROOT / v["report"]).read_text()
        for row in v["rows"]:
            assert row in txt, (k, row)


def first_test_frame():
    first_test_check()
    img, d = canvas()
    y = para(d, (M, TOP), "First test: did seeing its scores help?", 48, INK, W - 2 * M, True, 1.16)
    y = para(d, (M, y + 12), "Pre-registered, twice. The AI scientist's networks beat the starting network by 52.5 and "
                             "43.6 points. Matched searches ran with every score hidden from it.", 34, INK2, W - 2 * M)
    y += 40
    d.line([(M, y), (W - M, y)], fill=GRIDC, width=2)
    T(d, (M, y + 44), "+1.5 points", 132, ARM_COL, True)
    y = para(d, (M, y + 210), "added by seeing its scores, in both tests", 40, INK, W - 2 * M, True)
    y = para(d, (M, y + 14), "95% intervals: −9.1 to +12.0, and −10.1 to +12.2.", 34, INK2, W - 2 * M)
    y = para(d, (M, y + 4), "Both include zero.", 34, INK2, W - 2 * M)
    y += 40
    d.line([(M, y), (W - M, y)], fill=GRIDC, width=2)
    y = para(d, (M, y + 36), "Prior knowledge did the work.", 44, INK, W - 2 * M, True, 1.22)
    y = para(d, (M, y + 14), "The strong network you just saw came from one of the searches where every score was "
                             "hidden from it.", 34, INK2, W - 2 * M)
    y = para(d, (M, y + 36), "Maybe nothing was left to discover.", 44, INK, W - 2 * M, True, 1.22)
    assert y <= BOT, ("first-test card too tall", y)
    return checked(img)


# ---------------------------------------------------------------- scene 5: two hidden-rule worlds
def recorded_run(run_dir: Path, rule_name: str, designs: torch.Tensor):
    """eval_h.greedy_build's loop (batched, GPU) recording the world action each design took at each step."""
    src = (run_dir / "run" / "candidate.py").read_text()
    ck = torch.load(run_dir / "run" / "ckpt.pt", map_location="cpu")
    assert ck["candidate_sha"] == sha(src), run_dir
    spec = common.register()
    aspec = arch_spec(spec)
    torch.manual_seed(0)
    pol = CandidatePolicy(load_candidate(src)(aspec), aspec)
    pol.load_state_dict(ck["state_dict"])
    pol.to(DEV).eval()
    rule = Rule(rule_name, spec, DEV)
    world = rule.make_world(len(designs), targets=designs, auto_reset=False)
    agent, wact, alive = [], [], []
    with torch.no_grad():
        for _ in range(spec.max_steps + 1):
            if bool(world.ended.all()):
                break
            logits, _ = pol(rule.obs(encode(world, "voxels")), rule.mask(world.valid_mask()))
            a = logits.argmax(dim=1)
            wa = rule.act(a)
            alive.append((~world.ended).cpu())
            agent.append(a.cpu())
            wact.append(wa.cpu())
            world.step(wa)
    return torch.stack(agent), torch.stack(wact), torch.stack(alive), world.occ.cpu()


def last_try(cell_dir: Path) -> Path:
    tries = sorted(cell_dir.glob("try*"))
    assert tries and (tries[-1] / "dev.json").exists(), cell_dir
    return tries[-1]


def build_of(run_dir, rule_name, designs, di, spec):
    agent, wact, alive, occ = recorded_run(run_dir, rule_name, designs)
    seq = [int(wact[t, di]) for t in range(len(wact)) if bool(alive[t, di])]
    bricks, end, rocc = R.replay(seq, designs[di], spec)
    assert torch.equal(rocc, occ[di]), f"{run_dir}: the replay disagrees with the batched run"
    dev = json.loads((run_dir / "dev.json").read_text())
    exact = bool((occ[di] == designs[di]).all())
    assert exact == bool(dev["per_design"]["exact"][di]), f"{run_dir}: the build disagrees with its dev.json"
    first = next((t for t in range(len(agent)) if bool(alive[t, di])), None)
    return {"bricks": bricks, "end": end, "exact": exact, "missing": int((designs[di] & ~occ[di]).sum()),
            "first_agent": int(agent[first, di]) if first is not None else None,
            "first_world": int(wact[first, di]) if first is not None else None, "dev_exact": dev["exact"]}


def h1_load():
    res = json.loads((RUNS / "results.json").read_text())
    assert res["status"] == "final", "Stage H1 results.json isn't final"
    spec = common.register()
    designs = torch.load(common.TARGETS / "hplain_dev.pt")
    hid = designs_hidden(designs)
    out = {"res": res, "spec": spec, "n": len(designs)}
    for rule in ("shift", "mirror"):
        fdir, adir = last_try(RUNS / "b131072" / f"{rule}_F_s4000"), last_try(RUNS / "b131072" / f"{rule}_A_s4000")
        f_ok = json.loads((fdir / "dev.json").read_text())["per_design"]["exact"]
        di = next(i for i in range(len(designs)) if bool(hid[i]) and f_ok[i])          # the stated rule
        out[rule] = {"di": di, "target": designs[di], "cells": int(designs[di].sum()),
                     "ref": build_of(adir, rule, designs, di, spec), "fix": build_of(fdir, rule, designs, di, spec),
                     "means": res["evaluation"]["rules"][rule]["means"]}
    di = out["shift"]["di"]
    assert out["mirror"]["di"] == di, "the problem scene's wording says both worlds use its design"
    pa = {res["evaluation"]["rules"][r]["means"]["pA"] for r in ("shift", "mirror")}
    assert len(pa) == 1, pa
    out["plain"] = {"di": di, "target": designs[di], "cells": int(designs[di].sum()), "mean": pa.pop(),
                    "run": build_of(last_try(RUNS / "b131072" / "plain_A_s4000"), "plain", designs, di, spec)}
    out["demo"] = mirror_demo(spec)
    return out


DEMO_ACTION = (1, 3, 0)   # a 1x2 brick, anchor row 3, column 0: placed at the left edge so its mirror is far away


def mirror_demo(spec):
    """One real step in the mirrored-hands world on an empty board: where DEMO_ACTION's cells are, and where the
    brick actually lands (read from the world's occupancy after the step)."""
    t, yy, xx = DEMO_ACTION
    a = t * spec.Y * spec.X + yy * spec.X + xx
    rule = Rule("mirror", spec, DEV)
    world = rule.make_world(1, targets=torch.zeros(1, spec.Z, spec.Y, spec.X, dtype=torch.bool), auto_reset=False)
    assert bool(rule.mask(world.valid_mask())[0, a])
    world.step(rule.act(torch.tensor([a], device=DEV)))
    occ = world.occ[0].cpu()
    landed = [(int(y), int(x)) for y, x in occ[0].nonzero().tolist()]
    assert not occ[1:].any() and len(landed) == 2
    return {"action": a, "chosen": footprint_cells(a, spec), "landed": landed}


def iso_fit(spec, cw, hh, zt):
    return min(cw / ((spec.X + spec.Y) * 0.8660254), hh / ((spec.X + spec.Y) * 0.5 + zt))


def builds_row(d, Hd, rule, y, nb, done, lift=0.0):
    """The strong network and the fixed one building the same design, side by side, from y; returns the bottom.
    nb bricks placed in each build; lift > 0 draws the newest brick that many layers up (one frame of its drop)."""
    sp, D = Hd["spec"], Hd[rule]
    cols = ((M + 242, "The strong network", D["ref"], REF_COL), (W - M - 242, "With one input change", D["fix"], FIX_COL))
    zb = max(M2.top_layer(D["ref"]["bricks"]), M2.top_layer(D["fix"]["bricks"])) + 1
    sc = iso_fit(sp, 452, 300, zb)
    yr = y + 52 + ((sp.X + sp.Y) * 0.5 + zb) * sc + 10
    for cx, head, run, col in cols:
        T(d, (cx, y), head, 32, INK, True, anchor="ma")
        iso = R.Iso(sp, sc, cx, y + 52 + zb * sc)
        iso.ground(d, footprint=D["target"].any(0).numpy())
        k = len(run["bricks"]) if done else min(nb, len(run["bricks"]))
        dropping = not done and lift > 0 and 0 < nb <= len(run["bricks"])
        iso.bricks(d, run["bricks"][:k], lambda b, c=col: R.WRONG if b["wrong"] else c,
                   lift=(k - 1, lift) if dropping else None)
        if not run["bricks"] and run["end"] == "pressed DONE":
            msg = "pressed DONE: built nothing"
        elif run["exact"]:
            msg = "built exactly"
        else:
            msg = f"not exact: {run['missing']} cells missing"
        show = done or not run["bricks"] or nb >= len(run["bricks"])
        if show:
            fw = R.font(28).getlength(msg) + 24
            x0 = min(max(M + 2, cx - fw / 2), W - M - 2 - fw)
            R.badge(d, x0 + 15, yr + 16, run["exact"], r=15)
            T(d, (x0 + 38, yr), msg, 28, INK)
    return yr + 46


def numbers_row(d, y, rule, means, n):
    T(d, (M, y), f"Share of {n} designs built exactly, over 6 seeds:", 30, INK, True)
    y += 48
    T(d, (M, y), f"{means['rA'] * 100:.1f}%", 84, REF_COL, True)
    x = M + R.font(84, True).getlength(f"{means['rA'] * 100:.1f}%") + 30
    SYS.arrow(d, [(x, y + 48), (x + 70, y + 48)], col=INK2, width=6, head=18)
    x += 70 + 34
    T(d, (x, y), f"{means['rF'] * 100:.1f}%", 84, FIX_COL, True)
    T(d, (M, y + 104), f"The strong network builds {means['pA'] * 100:.1f}% in the plain world.", 30, INK2)
    return y + 150


def shift_frame(Hd, nb, done, lift=0.0):
    img, d = canvas()
    sp, D = Hd["spec"], Hd["shift"]
    y = para(d, (M, TOP), "So I built worlds with something to discover", 46, INK, W - 2 * M, True, 1.16)
    y = para(d, (M, y + 8), "World 1, shifted booklet: the agent's view of the target is moved 4 columns, wrapping "
                            "round. The AI scientist isn't told.", 30, INK2, W - 2 * M)
    tgt = D["target"]
    shown = torch.roll(tgt, 4, dims=2)
    zt = max(M2.top_layer(R.target_bricks(tgt.numpy())), 1)
    sc = iso_fit(sp, 452, 250, zt)
    y0 = y + 22
    for cx, head, t, col in ((M + 242, "The real target", tgt, R.TARGET_COL),
                             (W - M - 242, "What the agent is shown", shown, SHOWN_COL)):
        T(d, (cx, y0), head, 32, INK, True, anchor="ma")
        iso = R.Iso(sp, sc, cx, y0 + 50 + zt * sc)
        iso.ground(d)
        iso.bricks(d, R.target_bricks(t.numpy()), lambda b, c=col: c)
    y = y0 + 50 + ((sp.X + sp.Y) * 0.5 + zt) * sc + 26
    d.line([(M, y), (W - M, y)], fill=GRIDC, width=2)
    y = builds_row(d, Hd, "shift", y + 20, nb, done, lift)
    if done:
        y = numbers_row(d, y + 6, "shift", D["means"], Hd["n"])
    assert y <= BOT, ("shift scene too tall", y)
    return checked(img)


def footprint_cells(a, spec):
    t, rem = divmod(a, spec.Y * spec.X)
    yy, xx = divmod(rem, spec.X)
    dy, dx = R.BRICK_TYPES[t]
    return [(yy + i, xx + j) for i in range(dy) for j in range(dx)]


def mirror_frame(Hd, phase, nb, done, lift=0.0):
    """phase 0: the chosen anchor only; phase k > 0: the brick dropping in at the mirror (k = lift step); then builds."""
    img, d = canvas()
    sp, D = Hd["spec"], Hd["mirror"]
    y = para(d, (M, TOP), "World 2, mirrored hands", 46, INK, W - 2 * M, True, 1.16)
    y = para(d, (M, y + 8), "Every brick lands at the left-right mirror of where the agent placed it. Again, the AI "
                            "scientist isn't told.", 30, INK2, W - 2 * M)
    chosen, landed = Hd["demo"]["chosen"], Hd["demo"]["landed"]
    sc = iso_fit(sp, W - 2 * M - 200, 290, 2)
    cx, y0 = W / 2, y + 26
    iso = R.Iso(sp, sc, cx, y0 + 2 * sc)
    iso.ground(d)
    soft = R.mix(ARM_COL, SURF, 0.72)
    for (yy, xx) in chosen:
        d.polygon([iso.p(xx, yy, 0), iso.p(xx + 1, yy, 0), iso.p(xx + 1, yy + 1, 0), iso.p(xx, yy + 1, 0)], fill=soft,
                  outline=ARM_COL, width=4)
    if phase > 0:
        dz = {1: 2.0, 2: 1.1, 3: 0.4}.get(phase)
        iso.bricks(d, [{"cells": [(0, yy, xx) for (yy, xx) in landed], "wrong": False}], lambda b: SHOWN_COL,
                   lift=(0, dz) if dz else None)
    pc = iso.p(np.mean([x + 0.5 for _, x in chosen]), np.mean([yy + 0.5 for yy, _ in chosen]), 0)
    pl = iso.p(np.mean([x + 0.5 for _, x in landed]), np.mean([yy + 0.5 for yy, _ in landed]), 0)
    left, right = (pc, "placed here", ARM_COL), (pl, "lands here", SHOWN_COL)
    if pc[0] > pl[0]:
        left, right = right, left
    for (px, py), lab, col in (left, right):
        if lab == "lands here" and phase == 0:
            continue
        lw = R.font(30, True).getlength(lab)
        tx = max(M, min(W - M - lw, px - lw / 2))
        T(d, (tx, py + 40), lab, 30, col, True)
    y = y0 + 2 * sc + (sp.X + sp.Y) * 0.5 * sc + 84
    d.line([(M, y), (W - M, y)], fill=GRIDC, width=2)
    if phase >= 4:
        y = builds_row(d, Hd, "mirror", y + 20, nb, done, lift)
        if done:
            y = numbers_row(d, y + 6, "mirror", D["means"], Hd["n"])
    assert y <= BOT, ("mirror scene too tall", y)
    return checked(img)


def rule_frames(Hd):
    out = []
    nf = len(Hd["shift"]["fix"]["bricks"])
    out.append((shift_frame(Hd, 0, False), sec(3)))
    for k in range(1, nf + 1):
        out += [(shift_frame(Hd, k, False, lift), 1) for lift in (2.0, 1.1, 0.4)] + [(shift_frame(Hd, k, False), 2)]
    out.append((shift_frame(Hd, nf, True), sec(4.5)))
    out.append((mirror_frame(Hd, 0, 0, False), sec(1.2)))
    out += [(mirror_frame(Hd, p, 0, False), 1) for p in (1, 2, 3)]
    out.append((mirror_frame(Hd, 3.5, 0, False), sec(1.8)))
    nm = len(Hd["mirror"]["fix"]["bricks"])
    out.append((mirror_frame(Hd, 4, 0, False), sec(1)))
    for k in range(1, nm + 1):
        out += [(mirror_frame(Hd, 4, k, False, lift), 1) for lift in (2.0, 1.1, 0.4)] + [(mirror_frame(Hd, 4, k, False), 2)]
    out.append((mirror_frame(Hd, 4, nm, True), sec(4.5)))
    return out


# ---------------------------------------------------------------- scene 6: all four rules
def calib_frame(Hd):
    img, d = canvas()
    ev = Hd["res"]["evaluation"]["rules"]
    y = para(d, (M, TOP), "Is there something to find? Four rules, checked first", 46, INK, W - 2 * M, True, 1.16)
    y = para(d, (M, y + 8), "For each hidden rule: the strong network, and the same network with one change aimed at "
                            "the rule. 512 designs, 6 seeds each.", 30, INK2, W - 2 * M)
    M2.legend(d, M, y + 18, [("line", REF_COL, "strong network"), ("line", FIX_COL, "with the change"),
                              ("dash", INK2, "plain world")], size=28, gap=26)
    y += 18 + 56
    bx0, bx1 = M + 4, W - M - 150
    for r in ("shift", "mirror", "torus", "late"):
        e = ev[r]
        mn = e["means"]
        d.line([(M, y), (W - M, y)], fill=GRIDC, width=2)
        T(d, (M, y + 16), RULE_NAME[r], 36, INK, True)
        chip, ccol = ("passes", ARM_COL) if e["passes"] else ("fails", GREY_D)
        cw = R.font(28, True).getlength(chip)
        x = M + R.font(36, True).getlength(RULE_NAME[r]) + 20
        d.rounded_rectangle([x, y + 18, x + cw + 28, y + 60], radius=21, fill=ccol)
        T(d, (x + 14, y + 39), chip, 28, (255, 255, 255), True, anchor="lm")
        note = CHANGE[r] + (f" · {e['budget']:,} steps" if e["budget"] != 131072 else "")
        T(d, (M, y + 72), note, 28, INK2)
        for j, (key, col) in enumerate((("rA", REF_COL), ("rF", FIX_COL))):
            yb = y + 110 + j * 38
            d.rounded_rectangle([bx0, yb, bx1, yb + 26], radius=8, fill=(236, 235, 229))
            if mn[key] > 0:
                d.rounded_rectangle([bx0, yb, bx0 + max(16, (bx1 - bx0) * mn[key]), yb + 26], radius=8, fill=col)
            T(d, (bx1 + 16, yb + 13), f"{mn[key] * 100:.1f}%", 30, INK, True, anchor="lm")
        xp = bx0 + (bx1 - bx0) * mn["pA"]
        for k in range(0, 80, 14):
            d.line([(xp, y + 104 + k), (xp, y + 104 + min(k + 8, 80))], fill=INK2, width=3)
        y += 200
    d.line([(M, y), (W - M, y)], fill=GRIDC, width=2)
    y = para(d, (M, y + 20), "Two rules pass and go on to the test. Two fail, and are reported as failures: a positive "
                             "control has to be earned.", 32, INK, W - 2 * M, True, 1.22)
    assert y <= BOT, ("calibration scene too tall", y)
    return checked(img)


# ---------------------------------------------------------------- scene 7: the real test (Step 2's results)
ARMS2 = [("told", "Told the rule", "one sentence states it"),
         ("readouts", "Readouts", "logs from its own training runs"),
         ("scores", "Scores only", "success per seed, as in the first test"),
         ("hidden", "Nothing", "every score hidden")]


def step2_load():
    """Step 2's numbers, from its final results file only. The scene's wording is ending A's, so it stops otherwise."""
    res = json.loads(STEP2_RESULTS.read_text())
    assert res["status"] == "final" and res["preset"] == "step2", "Step 2's results.json isn't the final full run"
    assert res["ending"] == "both" and res["check"]["passes"] and res["discovery_claim_allowed"], "not ending A"
    assert res["primary"]["P1"]["passes"] and res["primary"]["P2"]["passes"]
    assert len(res["rows"]) == 96
    arms = {}
    for a, _, _ in ARMS2:
        pa, rows = res["per_arm"][a], [r for r in res["rows"] if r["arm"] == a]
        assert pa["n"] == len(rows) == 24, a
        mean = sum(r["score"] for r in rows) / 24        # exact: per_arm's mean is rounded to 4 places (0.0065 for 0.647%)
        assert abs(mean - pa["mean"]) < 1e-4 and sum(r["found"] for r in rows) == pa["found"], a
        arms[a] = {"found": pa["found"], "n": pa["n"], "mean": mean}
    return {"arms": arms, "check": res["check"], "P1": res["primary"]["P1"], "P2": res["primary"]["P2"]}


def step2_frame(S, pA, shown=4, grow=1.0, foot=True):
    """The four arms; rows after the first `shown` are greyed out, and the last shown row's bar is `grow` of its length.
    pA is the strong network's plain-world mean (Stage H1): a working fix is a search scoring at least half of it."""
    img, d = canvas()
    y = para(d, (M, TOP), "The real test: does it find the rule?", 48, INK, W - 2 * M, True, 1.16)
    bar = f"{pA * 50:.1f}%"
    assert bar == "38.3%", bar                                      # PROTOCOL_STEP2.md's working-fix bar
    y = para(d, (M, y + 8), "96 searches from the strong network, in the two worlds that passed, 24 per arm. Bars: "
                            "designs built, averaged over the arm's searches. A working fix builds at least " + bar + ".",
             30, INK2, W - 2 * M)
    y += 22
    bx0, bx1 = M + 4, W - M - 150
    for i, (a, head, sub) in enumerate(ARMS2):
        A, on = S["arms"][a], i < shown
        g = grow if i == shown - 1 else 1.0
        ink, ink2 = (INK, INK2) if on else (R.mix(INK, SURF, 0.7), R.mix(INK2, SURF, 0.7))
        d.line([(M, y), (W - M, y)], fill=GRIDC, width=2)
        T(d, (M, y + 16), head, 36, ink, True)
        T(d, (M, y + 66), sub, 28, ink2)
        assert R.font(28).getlength(sub) <= W - 2 * M - 260, sub
        if on and g == 1.0:
            T(d, (W - M, y + 10), f"{A['found']} of {A['n']}", 52, ARM_COL if A["found"] else INK, True, anchor="ra")
            T(d, (W - M, y + 74), "working fixes", 28, INK2, anchor="ra")
        yb = y + 124
        d.rounded_rectangle([bx0, yb, bx1, yb + 26], radius=8, fill=(236, 235, 229))
        if on and A["mean"] > 0:
            d.rounded_rectangle([bx0, yb, bx0 + max(16, (bx1 - bx0) * A["mean"] * g), yb + 26], radius=8, fill=ARM_COL)
        if on and g == 1.0:
            T(d, (bx1 + 16, yb + 13), f"{A['mean'] * 100:.1f}%", 30, INK, True, anchor="lm")
        y += 174
    d.line([(M, y), (W - M, y)], fill=GRIDC, width=2)
    if foot:
        p1, p2, ck = S["P1"], S["P2"], S["check"]
        assert f"{p1['p_holm']:.6f}" == f"{p2['p_holm']:.6f}"
        y = para(d, (M, y + 20), f"Readouts beat scores only by {p2['diff'] * 100:.1f} points, and nothing by "
                                 f"{p1['diff'] * 100:.1f}. Both gaps passed the test fixed before the run "
                                 f"(Holm p {p1['p_holm']:.6f}).", 32, INK, W - 2 * M, True, 1.22)
        y = para(d, (M, y + 12), f"Told the rule, {ck['found']} of {ck['n']} wrote a working fix; {ck['needed']} were "
                                 "needed for the result to count. I designed the readouts knowing the rules: it read "
                                 "the evidence, it didn't choose it.", 28, INK2, W - 2 * M)
    assert y <= BOT, ("Step 2 scene too tall", y)
    return checked(img)


def step2_frames(S, pA):
    out = [(step2_frame(S, pA, 0, 1.0, False), sec(2))]
    for k in range(1, len(ARMS2) + 1):
        out += [(step2_frame(S, pA, k, g, False), 1) for g in (0.2, 0.45, 0.7, 0.9)]
        out.append((step2_frame(S, pA, k, 1.0, False), sec(1.1)))
    out.append((step2_frame(S, pA, len(ARMS2), 1.0, True), sec(8)))
    return out


# ---------------------------------------------------------------- scene 8: along the way
def along_frame():
    img, d = canvas()
    T(d, (M, TOP), "Along the way", 48, INK, True)
    y = TOP + 80
    d.line([(M, y), (W - M, y)], fill=GRIDC, width=2)
    y = para(d, (M, y + 22), "It once hand-coded the world's rules into a network.", 34, INK, W - 2 * M, True)
    y = para(d, (M, y + 8), "Untrained, that network built 97% of check designs. That became a rule: a self-improving "
                            "loop is only as honest as its evaluator.", 33, INK2, W - 2 * M)
    y += 30
    d.line([(M, y), (W - M, y)], fill=GRIDC, width=2)
    y = para(d, (M, y + 22), "Suggestion 2, made outside the search", 34, INK, W - 2 * M, True)
    y = para(d, (M, y + 8), "Prompted by Prefrontal's CRAM paper: a small pairwise energy, counted from 4,000 solved "
                            "designs. It built", 33, INK2, W - 2 * M)
    T(d, (M, y + 10), "2,046 of 2,048", 84, ARM_COL, True)
    y = para(d, (M, y + 112), "designs it never trained or tuned on. Once the rules are known, this task is an easy "
                              "puzzle.", 33, INK2, W - 2 * M)
    y = para(d, (M, y + 22), "Not a ranking against RL: the networks trained on a small budget. With 4× the budget and "
                             "a better recipe, the starting network builds 95.7% in the 8×8×5 world.", 33, INK,
             W - 2 * M)
    assert y <= BOT, ("along-the-way scene too tall", y)
    return checked(img)


# ---------------------------------------------------------------- scene 9: limits and next
def limits_frame():
    img, d = canvas()
    T(d, (M, TOP), "What this isn't, and what's next", 48, INK, True)
    y = TOP + 92
    for head, items in (("Not a self-improving system yet", ("One laptop, toy worlds.",
                                                             "Two hidden rules, both of one kind.",
                                                             "One model, whose weights never changed.")),
                        ("Next", ("A sandbox where it can experiment on the world before designing: dig.bench's "
                                  "creative mode, one level up.",
                                  "Then rewriting its own procedure across worlds whose rules differ, so only "
                                  "procedure can carry over."))):
        d.line([(M, y), (W - M, y)], fill=GRIDC, width=2)
        y = para(d, (M, y + 26), head, 42, INK, W - 2 * M, True)
        for it in items:
            d.ellipse([M + 4, y + 36, M + 22, y + 54], fill=ARM_COL)
            y = para(d, (M + 44, y + 22), it, 37, INK2, W - 2 * M - 44)
        y += 44
    assert y <= BOT, ("limits scene too tall", y)
    return checked(img)


# ---------------------------------------------------------------- assembly
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills-only", action="store_true")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.final:
        raise SystemExit("not yet: the post now waits for Step 3 (reachouts/prefrontal/linkedin-post-draft-3.md); Loop 3's "
                         "scene and the rewritten system and last cards come from its results")
    out = HERE / "mobile"
    out.mkdir(exist_ok=True)
    Hd = h1_load()
    S2 = step2_load()
    stills = {"h01_hook": hook_frame(), "h02_system": system_frame(),
              "h03_problem": plain_frame(Hd, len(Hd["plain"]["run"]["bricks"]), True), "h04_first_test": first_test_frame(),
              "h05_shift": shift_frame(Hd, len(Hd["shift"]["fix"]["bricks"]), True),
              "h06_mirror": mirror_frame(Hd, 4, len(Hd["mirror"]["fix"]["bricks"]), True),
              "h07_calibration": calib_frame(Hd), "h08_step2": step2_frame(S2, Hd["plain"]["mean"]),
              "h09_along": along_frame(), "h10_limits": limits_frame()}
    for old in out.glob("h*.png"):          # this script's own stills, renamed when scenes changed
        old.unlink()
    for k, img in stills.items():
        img.save(out / f"{k}.png")
    if a.stills_only:
        M2.banned_check()
        print(json.dumps({"stills": sorted(stills), "design": Hd["plain"]["di"],
                          "plain_build": {k: Hd["plain"]["run"][k] for k in ("end", "exact", "dev_exact")},
                          "plain_bricks": len(Hd["plain"]["run"]["bricks"])}))
        return
    path = out / "discovery_mobile_DRAFT.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv,"
                           "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
                           "-c:v", "h264_videotoolbox", "-profile:v", "high", "-b:v", "8M", "-pix_fmt", "yuv420p",
                           "-g", str(FPS), "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                           "-color_range", "tv", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    secs = {}

    def write(part, frames):
        n_f = 0
        for img, times in frames:
            buf = np.asarray(checked(img).convert("RGB"))
            assert buf.shape == (H, W, 3), (part, buf.shape)
            for _ in range(times):
                ff.stdin.write(buf.tobytes())
            n_f += times
        secs[part] = round(n_f / FPS, 2)

    write("1_question", [(stills["h01_hook"], sec(7))])
    write("2_system", system_frames())
    write("3_problem", plain_frames(Hd))
    write("4_first_test", [(stills["h04_first_test"], sec(8))])
    write("5_hidden_rules", rule_frames(Hd))
    write("6_calibration", [(stills["h07_calibration"], sec(8))])
    write("7_step2", step2_frames(S2, Hd["plain"]["mean"]))
    write("8_along_the_way", [(stills["h09_along"], sec(10))])
    write("9_limits_next", [(stills["h10_limits"], sec(8))])
    ff.stdin.close()
    assert ff.wait() == 0
    M2.banned_check()
    pr = M2.probe(path)
    v = pr["streams"][0]
    assert len(pr["streams"]) == 1 and (v["codec_name"], v["pix_fmt"], v["width"], v["height"]) == ("h264", "yuv420p", W, H), v
    assert v["r_frame_rate"] == v["avg_frame_rate"] == f"{FPS}/1", v
    kbps = int(pr["format"]["bit_rate"]) / 1000
    assert 2 * 192 <= kbps <= 30000, kbps
    total = round(sum(secs.values()), 2)
    assert abs(float(pr["format"]["duration"]) - total) < 0.1, (pr["format"]["duration"], secs)
    data = {"status": "DRAFT: Step 2's scene is from its final results; Loop 3's scene isn't written yet (the post waits for Step 3); not for posting", "size": [W, H], "fps": FPS,
            "parts_seconds": secs, "video_seconds": total,
            "h1_design": Hd["plain"]["di"],
            "plain_build": {"run": "arch_rsi/stageH/runs/b131072/plain_A_s4000", "bricks": len(Hd["plain"]["run"]["bricks"]),
                            "end": Hd["plain"]["run"]["end"], "exact": Hd["plain"]["run"]["exact"]},
            "h1_builds": {r: {k: {"bricks": len(Hd[r][k]["bricks"]), "end": Hd[r][k]["end"], "exact": Hd[r][k]["exact"]}
                              for k in ("ref", "fix")} for r in ("shift", "mirror")},
            "mirror_demo": Hd["demo"], "step2": S2["arms"],
            "file": {"codec": f"{v['codec_name']} {v.get('profile')}", "encoder": "h264_videotoolbox", "kbps": round(kbps),
                     "bytes": int(pr["format"]["size"])},
            "sources": {"dig.bench": "arXiv 2608.12593 v2, Section 3, Figure 5B", "first_test": {k: v["report"] for k, v in FIRST_TEST.items()},
                        "reference_A": "arch_rsi/campaigns/stage1/results.json per_arm.nofb, search 9, pick r4c3",
                        "stage_h1": "arch_rsi/stageH/runs/results.json", "step2": "arch_rsi/stageH/step2/campaigns/step2/results.json", "along_the_way": "energy_check/README.md; arch_rsi/gpu/stageG/REPORT.md G2"},
            "checks": {"smallest_text_px": M2.MIN_PX, "safe_area": {"sides": M, "top": TOP, "bottom": H - BOT},
                       "banned_words_absent": list(M2.BANNED), "strings_checked": len(M2.SEEN)}}
    (out / "discovery_mobile_data.json").write_text(json.dumps(data, indent=1))
    print(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
