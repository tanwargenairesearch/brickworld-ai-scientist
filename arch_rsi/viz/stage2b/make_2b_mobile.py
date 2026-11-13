"""VISUALIZATION ONLY: the Stage 2b article video laid out for a phone (4:5, 1080x1350), for LinkedIn's mobile feed.

Same story, same data and same checks as the landscape video (make_2b_video.py), in the same order:
  1. the system: three loops, as three stacked cards, each pointed out in turn;
  2. the problem: a target design, and the start and one searched network building it;
  3. how it searches: one search, with the build, its evaluation and the progress chart stacked;
  4. whether the experiments guided the search: the two compared arms' searches, then every arm's picks with the
     answer, then the result again with the limits.
Nothing here scores or selects: every number comes from the loaders the landscape scenes use.

Rules this file enforces on itself (the render stops if one breaks):
  - no text smaller than MIN_PX (28 px in the frame, about 10 points on a phone showing the video 390 points wide);
  - nothing drawn outside the safe area (side margins M, nothing above TOP or below BOT: the feed player draws its
    controls over the bottom edge);
  - no on-screen string contains a word in BANNED (the user's wording rules, 10 November 2026);
  - the answer's numbers are recomputed and must appear in search_chart_2b.outcome_lines' own sentence.

usage (project root):
  nice -n 19 .venv/bin/python -I arch_rsi/viz/stage2b/make_2b_mobile.py [--stills-only]
"""
import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAMP = ROOT / "arch_rsi" / "campaigns" / "stage2b"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PS = _load("problem_scene", HERE / "problem_scene.py")
SB, R = PS.SB, PS.R
SC = _load("search_chart_2b", HERE / "search_chart_2b.py")
MV = _load("make_2b_video", HERE / "make_2b_video.py")
SYS = _load("make_system", HERE.parent / "stages" / "system" / "make_system.py")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

W, H, FPS, DPI, M = 1080, 1350, 24, 200, 56
TOP, BOT = 56, 1250          # nothing is drawn above TOP or below BOT
MIN_PX = 28                  # smallest text allowed in the frame
BANNED = ("gate", "lego", "claude", "opus", "anthropic", "openai", "gpt", "gemini")
INK, INK2, GRIDC, SURF = SB.INK, SB.INK2, SB.GRIDC, SB.SURF
ARM_COL, OTHER, EMPTY = SB.ARM_COL, SB.OTHER, SB.EMPTY
ORANGE_D, GREY_D = (176, 98, 20), (104, 102, 97)   # darker than the brick orange and the muted grey: text contrast
TINT = SYS.KIND["sci"][0]
SEEN = set()                 # every string drawn, for the BANNED check


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def T(d, xy, s, size, col=INK, bold=False, anchor="la", stroke=0):
    assert size >= MIN_PX, (s, size)
    SEEN.add(s)
    d.text(xy, s, font=R.font(size, bold), fill=col, anchor=anchor, stroke_width=stroke, stroke_fill=SURF)


def para(d, xy, s, size, col, width, bold=False, lh=1.3):
    assert size >= MIN_PX, (s, size)
    SEEN.add(s)
    return SYS.para(d, xy, s, size, col, width, bold, lh)


def fit(s, size, width, bold=False):
    """s, cut with an ellipsis if it is wider than width."""
    f = R.font(size, bold)
    if f.getlength(s) <= width:
        return s
    while s and f.getlength(s + "…") > width:
        s = s[:-1]
    return s.rstrip() + "…"


def minus(s):
    """Hyphen-minus before a number becomes a true minus sign."""
    return re.sub(r"(?<![A-Za-z0-9])-(?=\d)", "−", s)


def canvas():
    img = Image.new("RGB", (W, H), SURF)
    return img, ImageDraw.Draw(img)


def checked(img):
    """The frame, after checking that nothing was drawn outside the safe area."""
    a = np.asarray(img)
    surf = np.array(SURF, dtype=a.dtype)
    for name, band in (("top", a[:TOP - 6]), ("bottom", a[BOT:]), ("left", a[:, :M - 6]), ("right", a[:, W - M + 6:])):
        assert (band == surf).all(), f"something is drawn in the {name} margin"
    return img


def legend(d, x, y, items, size=28, gap=30):
    """A row of marker and label pairs."""
    f = R.font(size)
    cy = y + size * 0.56
    for kind, col, lab in items:
        if kind == "dot":
            d.ellipse([x, cy - 12, x + 24, cy + 12], fill=col)
            mw = 24
        elif kind == "small":
            d.ellipse([x + 3, cy - 9, x + 21, cy + 9], fill=col)
            mw = 24
        elif kind == "line":
            d.line([(x, cy), (x + 40, cy)], fill=col, width=6)
            mw = 40
        elif kind == "dash":
            for k in (0, 15, 30):
                d.line([(x + k, cy), (x + k + 9, cy)], fill=col, width=3)
            mw = 40
        else:  # diamond
            d.polygon([(x + 11, cy - 12), (x + 22, cy), (x + 11, cy + 12), (x, cy)], fill=col)
            mw = 22
        T(d, (x + mw + 10, y), lab, size, INK)
        x += mw + 10 + f.getlength(lab) + gap
    assert x - gap <= W - M, ("legend row too wide", [i[2] for i in items])


def top_layer(bricks):
    return 1 + max((z for b in bricks for (z, _, _) in b["cells"]), default=0)


# ---------------------------------------------------------------- the cover: a thumbnail for the video, not a frame of it
def cover_frame(P, O):
    """The article's 16:9 cover (make_cover.py) re-laid out at 4:5, to upload as the video's thumbnail: the headline,
    the problem scene's two builds (same rule and checks) and the three numbers."""
    img, d = canvas()
    y = TOP
    for ln in ("An AI scientist", "writes the network.", "Did its experiments", "guide the search?"):
        T(d, (M, y), ln, 72, INK, True)
        y += 80
    sp = P["spec"]
    y2, bh = y + 44, 350
    zb = max(top_layer(P["start"]["bricks"]), top_layer(P["pick"]["bricks"]))
    sc = min(456 / ((sp.X + sp.Y) * 0.866), bh / ((sp.X + sp.Y) * 0.5 + zb))
    yr = y2 + 54 + ((sp.X + sp.Y) * 0.5 + zb) * sc + 12
    for cx, head, run, col in ((M + 242, "The starting network", P["start"], PS.START_COL),
                               (W - M - 242, "The AI scientist's network", P["pick"], PS.PICK_COL)):
        T(d, (cx, y2), head, 34, INK, True, anchor="ma")
        iso = R.Iso(sp, sc, cx, y2 + 54 + zb * sc)
        iso.ground(d, footprint=P["target"].any(0).numpy())
        iso.bricks(d, run["bricks"], lambda b, c=col: R.WRONG if b["wrong"] else c)
        msg = "built exactly" if run["exact"] else f"not exact: {run['missing']} cells missing"
        w = R.font(30).getlength(msg)
        R.badge(d, cx - w / 2 - 6, yr + 18, run["exact"], r=16)
        T(d, (cx - w / 2 + 20, yr), msg, 30, INK)
    yb = yr + 76
    d.line([(M, yb), (W - M, yb)], fill=GRIDC, width=2)
    cw = (W - 2 * M) / 3
    for k, (val, labs, col) in enumerate(((f"{P['start_fresh'] * 100:.1f}%", ("the starting", "network"), ORANGE_D),
                                          (O["seen"], ("AI scientist,", "seeing its scores"), ARM_COL),
                                          (O["hidden"], ("the same search,", "scores hidden"), INK))):
        T(d, (M + k * cw, yb + 26), val, 88, col, True)
        for j, lab in enumerate(labs):
            assert R.font(30).getlength(lab) <= cw - 20, lab
            T(d, (M + k * cw + 3, yb + 130 + j * 38), lab, 30, INK)
    yy = para(d, (M, yb + 130 + 2 * 38 + 30), f"Share of {P['n_designs']} designs built exactly, on fresh seeds. The builds are one "
                               "illustration: the first dev design with hidden cells that this network builds and the "
                               "start doesn't.", 30, INK2, W - 2 * M)
    assert yy <= BOT, yy
    return checked(img)


# ---------------------------------------------------------------- part 1: the system
def system_frame(focus=None):
    """The three loops as stacked cards. focus = the loop being pointed out (None: all three drawn the same)."""
    img, d = canvas()
    T(d, (M, TOP), "An AI scientist writes the network.", 50, INK, True)
    T(d, (M, TOP + 58), "Did its experiments guide the search?", 50, INK, True)
    T(d, (M, TOP + 128), "The system: three loops, each learning something different.", 30, INK2)
    cards = [
        {"kind": "agent", "head": "Loop 1 · the agent learns to build", "chip": ("RUNS IN EVERY EXPERIMENT", ORANGE_D),
         "does": "It places one brick at a time, gets a reward, and PPO updates its weights.",
         "learns": "Learns: the network's weights."},
        {"kind": "sci", "head": "Loop 2 · the AI scientist searches", "chip": ("STAGES 1–2b · DONE", ARM_COL),
         "does": "A hosted LLM reads its earlier experiments and writes 3 new candidates. Frozen code trains and scores "
                 "them. A control arm hides the scores.",
         "learns": "Learns: no weights. It adapts within one search."},
        {"kind": "frozen", "dotted": True, "head": "Loop 3 · the procedure improves itself",
         "chip": ("STAGE 2c · NOT RUN", GREY_D),
         "does": "It would rewrite its own research program: where \"recursive\" would begin. It runs only if seeing "
                 "its scores helps Loop 2's search. Stage 2b didn't show that.",
         "learns": "Would learn: the research program."}]
    links = [("each candidate is trained by Loop 1", ORANGE_D), ("confirmed picks would feed Loop 3", GREY_D)]
    pad, tw, bs, blh = 26, W - 2 * M - 52, 30, 1.27
    heights = [128 + len(SYS.wrap(c["does"], R.font(bs), tw)) * bs * blh + 8 + 36 + 22 for c in cards]
    top = TOP + 128 + 30 + 24
    gap = (BOT - 4 - top - sum(heights)) / 2
    assert gap >= 48, ("the three cards don't fit", gap)
    y0 = top
    for i, (c, ch) in enumerate(zip(cards, heights)):
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
            (a, b) = (ym + 17, ym - 17) if i == 0 else (ym - 17, ym + 17)     # up into Loop 1; down toward Loop 3
            SYS.arrow(d, [(M + 40, a), (M + 40, b)], col=col, width=5, head=14)
            T(d, (M + 68, ym), lab, 28, col, True, anchor="lm")
        y0 += ch + gap
    return checked(img)


def system_frames():
    """All three loops drawn the same, then each pointed out in turn for the same time."""
    return [(system_frame(), int(1.5 * FPS))] + [(system_frame(i), int(3.5 * FPS)) for i in range(3)]


# ---------------------------------------------------------------- part 2: the problem
def problem_frame(P, nb, done, lift=0.0):
    """nb bricks placed in each build; lift > 0 draws the newest brick that many layers up (one frame of its drop)."""
    img, d = canvas()
    sp = P["spec"]
    y = para(d, (M, TOP), "The problem: build the target design exactly, one brick at a time", 46, INK, W - 2 * M, True, 1.16)
    y = para(d, (M, y + 8), "A network sees the target and what is built so far, then places one brick or presses DONE. "
                            "Only an exact match counts.", 30, INK2, W - 2 * M)
    tgt = R.target_bricks(P["target"].numpy())
    top, th, zt = y + 14, 250, top_layer(tgt)
    sc = min(th / ((sp.X + sp.Y) * 0.5 + zt), 440 / ((sp.X + sp.Y) * 0.866))
    iso = R.Iso(sp, sc, 286, top + zt * sc)
    iso.ground(d)
    iso.bricks(d, tgt, lambda b: R.TARGET_COL)
    T(d, (540, top + 16), "Target design", 36, INK, True)
    para(d, (540, top + 68), f"{P['cells']} cells, some hidden under other bricks. One of {P['n_designs']} dev designs, in a "
                             f"{sp.X}×{sp.Y}×{sp.Z} world.", 30, INK2, W - M - 540)
    y2, bh = top + th + 22, 344
    zb = max(top_layer(P["start"]["bricks"]), top_layer(P["pick"]["bricks"])) + 1      # one layer of room for the drop
    sc = min(456 / ((sp.X + sp.Y) * 0.866), bh / ((sp.X + sp.Y) * 0.5 + zb))
    for cx, head, sub, run, col, fresh in (
            (M + 242, "The starting network", "trained by PPO", P["start"], PS.START_COL, P["start_fresh"]),
            (W - M - 242, "The AI scientist's network", "its own recipe, equal compute", P["pick"], PS.PICK_COL,
             P["pick_fresh"])):
        T(d, (cx, y2), head, 32, INK, True, anchor="ma")
        T(d, (cx, y2 + 42), sub, 28, INK2, anchor="ma")
        iso = R.Iso(sp, sc, cx, y2 + 84 + zb * sc)
        iso.ground(d, footprint=P["target"].any(0).numpy())
        bricks = run["bricks"][:nb]
        dropping = lift > 0 and 0 < nb <= len(run["bricks"])
        iso.bricks(d, bricks, lambda b: R.WRONG if b["wrong"] else col, lift=(len(bricks) - 1, lift) if dropping else None)
        yr = y2 + 84 + bh + 12
        if done:
            msg = "built exactly" if run["exact"] else (f"not exact: {run['missing']} missing"
                                                        + (f", {run['extra']} extra" if run["extra"] else ""))
            w = R.font(30).getlength(msg)
            R.badge(d, cx - w / 2 - 6, yr + 18, run["exact"], r=16)
            T(d, (cx - w / 2 + 20, yr), msg, 30, INK)
            T(d, (cx, yr + 44), f"this run: {run['exact_all'] * 100:.1f}% of all {P['n_designs']} designs", 28, INK2, anchor="ma")
            T(d, (cx, yr + 80), f"on fresh seeds: {fresh * 100:.1f}%", 28, INK2, anchor="ma")
        else:
            T(d, (cx, yr), f"bricks placed: {len(bricks)}", 30, INK2, anchor="ma")
    yb = y2 + 84 + bh + 12 + 124
    d.line([(M, yb), (W - M, yb)], fill=GRIDC, width=2)
    yy = para(d, (M, yb + 12), "An illustration, not evidence. The rule: the first dev design with hidden cells that this "
                               "network builds and the start doesn't. The test comes next.", 28, INK2, W - 2 * M)
    assert yy <= BOT, yy
    return checked(img)


def problem_frames(P):
    """The target alone, both builds brick by brick (each brick drops in), then the result held."""
    n = max(len(P["start"]["bricks"]), len(P["pick"]["bricks"]))
    out = [(problem_frame(P, 0, False), 2 * FPS)]
    for k in range(1, n + 1):
        out += [(problem_frame(P, k, False, lift), 1) for lift in (2.0, 1.1, 0.4)] + [(problem_frame(P, k, False), 5)]
    return out + [(problem_frame(P, n, True), 7 * FPS)]


# ---------------------------------------------------------------- part 3: one search
def search_chart(d, D, cur, show_cur):
    y0 = 912
    legend(d, M, y0, [("dot", ARM_COL, "kept: a new best"), ("small", OTHER, "discarded"), ("line", ARM_COL, "running best")])
    px0, py0, px1, py1 = M + 92, y0 + 62, W - M - 16, 1196
    n = D["n"]
    X = lambda i: px0 + (px1 - px0) * i / (n + 0.5)  # noqa: E731
    Y = lambda v: py1 - (py1 - py0) * v  # noqa: E731
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        d.line([(px0, Y(v)), (px1, Y(v))], fill=GRIDC, width=2)
    for v in (0, 0.5, 1.0):
        T(d, (px0 - 14, Y(v)), f"{v * 100:.0f}%", 28, INK2, anchor="rm")
    rounds = sorted({r["round"] for r in D["rows"]})
    for rd in rounds:
        idx = [r["i"] for r in D["rows"] if r["round"] == rd]
        T(d, ((X(idx[0]) + X(idx[-1])) / 2, py1 + 10), f"round {rd}", 28, INK2, anchor="ma")
        if rd != rounds[-1]:
            xs = X(idx[-1] + 0.5)
            d.line([(xs, py0), (xs, py1)], fill=GRIDC, width=2)
    shown = [r for r in D["rows"] if r["i"] < cur or (r["i"] == cur and show_cur)]
    pts, best = [(X(0), Y(D["bar"]))], D["bar"]
    for r in shown:
        if r["kept"]:
            pts += [(X(r["i"]), Y(best)), (X(r["i"]), Y(r["mean"]))]
            best = r["mean"]
    pts.append((X(shown[-1]["i"]) if shown else X(0), Y(best)))
    d.line(pts, fill=ARM_COL, width=5)
    for r in shown:
        if not r["ok"]:
            xx = X(r["i"])
            d.line([(xx - 9, Y(0) - 9), (xx + 9, Y(0) + 9)], fill=INK2, width=4)
            d.line([(xx - 9, Y(0) + 9), (xx + 9, Y(0) - 9)], fill=INK2, width=4)
        elif not r["kept"]:
            xx, yy = X(r["i"]), Y(r["mean"])
            d.ellipse([xx - 9, yy - 9, xx + 9, yy + 9], fill=OTHER)
    d.ellipse([X(0) - 11, Y(D["bar"]) - 11, X(0) + 11, Y(D["bar"]) + 11], fill=INK2, outline=SURF, width=2)
    T(d, (X(0) - 12, Y(D["bar"]) - 16), "start", 28, INK2, anchor="lb", stroke=3)
    for r in shown:
        if r["kept"]:
            xx, yy = X(r["i"]), Y(r["mean"])
            d.ellipse([xx - 13, yy - 13, xx + 13, yy + 13], fill=ARM_COL, outline=SURF, width=2)
            up = 25 if show_cur and r["i"] == cur else 17          # clear of the ring round the current experiment
            T(d, (xx, yy - up), f"{100 * r['mean']:.0f}%", 28, ARM_COL, True, anchor="mb", stroke=4)
    if show_cur:
        r = D["rows"][cur - 1]
        yy = Y(r["mean"]) if r["ok"] else Y(0)
        d.ellipse([X(cur) - 21, yy - 21, X(cur) + 21, yy + 21], outline=INK, width=4)


def search_frame(D, cur, nb, fill, show_cur, lift=0.0):
    img, d = canvas()
    r, sp = D["rows"][cur - 1], D["spec"]
    T(d, (M, TOP), f"One search: experiment {cur} of {D['n']}", 46, INK, True)
    T(d, (M, TOP + 60), fit(f"Round {r['round']} · {r['name']}", 30, W - 2 * M), 30, INK2)
    tgt = R.target_bricks(D["target"].numpy())
    yb, hb = 160, 328
    zb = max(top_layer(x.get("bricks", [])) for x in D["rows"]) + 1               # one layer of room for the drop
    sc = min(hb / ((sp.X + sp.Y) * 0.5 + zb), 600 / ((sp.X + sp.Y) * 0.866))
    iso = R.Iso(sp, sc, M + sp.Y * sc * 0.866 + 40, yb + zb * sc)
    iso.ground(d, footprint=D["target"].any(0).numpy())
    T(d, (878, yb), "Target design", 28, INK2, anchor="ma")
    zt = top_layer(tgt)
    st = min(250 / ((sp.X + sp.Y) * 0.5 + zt), 280 / ((sp.X + sp.Y) * 0.866))
    iso_t = R.Iso(sp, st, 878, yb + 44 + zt * st)
    iso_t.ground(d)
    iso_t.bricks(d, tgt, lambda b: R.TARGET_COL)
    ys = yb + hb + 12
    if r["ok"]:
        bricks = r["bricks"][:nb]
        iso.bricks(d, bricks, lambda b: R.WRONG if b["wrong"] else ARM_COL,
                   lift=(len(bricks) - 1, lift) if lift > 0 and bricks else None)
        if nb >= len(r["bricks"]) and fill > 0:
            msg = "built exactly" if r["exact"] else (f"not exact: {r['missing']} cells missing"
                                                      + (f", {r['extra']} extra" if r["extra"] else ""))
            R.badge(d, M + 16, ys + 18, r["exact"], r=16)
            T(d, (M + 42, ys), msg, 30, INK)
        else:
            T(d, (M, ys), f"placing bricks: {len(bricks)}", 30, INK2)
    else:
        T(d, (M, ys), "This experiment failed and scores as failed.", 30, INK2)
    gy0 = ys + 46
    T(d, (M, gy0), f"Evaluation: {D['n_designs']} dev designs. Filled = built exactly.", 30, INK, True)
    cols, cell, pitch = 32, 13, 14
    assert len(D["seeds"]) == 2 and D["n_designs"] % cols == 0
    for s in range(2):
        gx, gy = (M, W - M - cols * pitch)[s], gy0 + 84
        g = r["grid"][s] if r["ok"] else [0] * D["n_designs"]
        shown = min(fill, len(g))
        lab = f"Seed {D['seeds'][s]} run: {100 * sum(g[:shown]) / len(g):.1f}%" if r["ok"] else f"Seed {D['seeds'][s]} run: failed"
        T(d, (gx, gy - 40), lab, 28, INK2)
        for k in range(len(g)):
            cy, cx = divmod(k, cols)
            xx, yy = gx + cx * pitch, gy + cy * pitch
            d.rectangle([xx, yy, xx + cell - 1, yy + cell - 1], fill=ARM_COL if (k < shown and g[k]) else EMPTY)
            if s == 0 and k == D["di"] and k < shown:
                d.rectangle([xx - 3, yy - 3, xx + cell + 2, yy + cell + 2], outline=INK, width=3)
    if r["ok"] and fill >= D["n_designs"]:
        T(d, (M, gy0 + 84 + (D["n_designs"] // cols) * pitch + 10),
          f"Score: {100 * r['mean']:.1f}%, the mean of the two seeds. Ringed: the design above.", 28, INK)
    search_chart(d, D, cur, show_cur)
    return checked(img)


def search_frames(D):
    """The landscape scene's timing (search_build.video_frames) at twice the frame rate: each brick drops in over two
    frames, and the evaluation fills in 16 steps."""
    for r in D["rows"]:
        k, nb = r["i"], len(r.get("bricks", []))
        yield search_frame(D, k, 0, 0, False), 8
        for j in range(1, nb + 1):
            yield search_frame(D, k, j, 0, False, 1.2), 1
            yield search_frame(D, k, j, 0, False), 1
        for j in range(1, 17):
            yield search_frame(D, k, nb, D["n_designs"] * j // 16, False), 1
        yield search_frame(D, k, nb, D["n_designs"], True), 48 if r["kept"] else 20


# ---------------------------------------------------------------- part 4: the compared searches, the answer, the limits
def panel(draw, w, h, **adjust):
    """One axes, drawn by the landscape chart's own function, as a w x h image. DPI is set so its text is phone-sized."""
    fig, ax = plt.subplots(figsize=(w / DPI, h / DPI), dpi=DPI)
    fig.patch.set_facecolor(SC.SURFACE)
    draw(ax)
    for loc in ("left", "center", "right"):
        ax.set_title("", loc=loc)                                  # titles are drawn on the frame, at the margin
    fig.subplots_adjust(**adjust)
    for t in fig.findobj(mtext.Text):
        if t.get_text().strip() and t.get_visible():
            assert t.get_fontsize() * DPI / 72 >= MIN_PX, (t.get_text(), t.get_fontsize())
            SEEN.add(t.get_text())
    buf = SC.to_rgb(fig)
    plt.close(fig)
    assert buf.shape == (h, w, 3), buf.shape
    return Image.fromarray(buf)


def outcome(C):
    """The primary contrast, recomputed from unrounded primary scores as search_chart_2b.outcome_lines does. Each piece
    shown on the phone must appear in that function's own sentence, so the two videos can't disagree."""
    res = C["res"]
    assert res["primary_family"]["status"] == "final"
    assert not res["primary_family"]["gate_passed"], "the wording in part 4 is for this result only"
    m = {a: float(np.mean([s["primary"] for s in C["data"][a]])) for a in ("g0", "nofb")}
    for a in m:
        assert abs(m[a] - res["per_arm"][a]["primary_mean"]) < 6e-5, a
    p1, p2 = res["primary_family"]["contrasts"]["g0 - nofb"], res["primary_family"]["contrasts"]["g0 - refA"]
    d1 = m["g0"] - m["nofb"]
    assert abs(d1 - p1["diff"]) < 6e-5
    d2 = m["g0"] - float(np.mean(C["comps"]["refA"]))
    assert abs(d2 - p2["diff"]) < 6e-5
    lo, hi = p1["ci95_boot"]
    assert lo < 0 < hi, "the wording in part 4 says the interval includes zero"
    o = {"seen": f"{100 * m['g0']:.1f}%", "hidden": f"{100 * m['nofb']:.1f}%", "diff": f"{SC.pts(d1)} points",
         "ci": f"{SC.pts(lo)} to {SC.pts(hi)}", "hi": SC.pts(hi), "p": f"Holm-adjusted p {SC.pval(p1['p_holm'])}",
         "retest": f"{SC.pts(d2)} points (Holm-adjusted p {SC.pval(p2['p_holm'])})"}
    l1, l2 = SC.outcome_lines(C)
    assert all(o[k] in l1 for k in ("diff", "ci", "p")) and o["retest"] in l2, (o, l1, l2)
    return o


def results_frame(C, nd, upto, confirm, caption):
    """The two compared arms' searches, candidate by candidate."""
    img, d = canvas()
    T(d, (M, TOP), "Did seeing its scores help the search?", 48, INK, True)
    y = para(d, (M, TOP + 66), f"{C['n_search']} searches per arm. Each dot is one candidate, scored as the share of {nd} "
                               "dev designs built exactly.", 28, INK2, W - 2 * M)
    blue, other, ink2 = rgb(SC.BLUE), rgb(SC.OTHER), rgb(SC.INK2)
    start = 100 * float(np.mean(C["comps"]["start_g"]))
    st = {r["status"] for a in ("g0", "nofb") for s in C["data"][a] for r in s["rows"]}
    assert st == {"ok"}, f"the legend has no entry for {st - {'ok'}}"
    legend(d, M, y + 6, [("dot", blue, "new best in its search"), ("small", other, "not a new best")]
           + ([("diamond", ink2, "its pick")] if confirm else []))
    legend(d, M, y + 44, [("line", blue, f"mean best so far, {C['n_search']} searches"), ("dash", ink2, f"start: {start:.1f}%")])
    yp, ph = y + 92, 372

    def draw(ax, arm):
        SC.search_panel(ax, C, arm, upto, confirm, True)
        ax.set_xlim(-0.1, C["ncand"] + 4.6)                         # room for the mean's label

    for k, arm in enumerate(("g0", "nofb")):
        yk = int(yp + k * (ph + 46))
        T(d, (M, yk), SC.TITLES[arm], 34, INK, True)
        img.paste(panel(lambda ax, arm=arm: draw(ax, arm), W - 2 * M, ph, left=0.105, right=0.995, top=0.975, bottom=0.25),
                  (M, yk + 46))
    yy = para(d, (M, yp + 2 * (ph + 46) + 4), caption, 32, INK, W - 2 * M, True, 1.25)
    assert yy <= BOT, yy
    return checked(img)


def answer_frame(C, O):
    """Every arm's picks, with the answer in large type above them."""
    img, d = canvas()
    T(d, (M, TOP), "Did seeing its scores help the search?", 48, INK, True)
    y, tw = TOP + 72, (W - 2 * M - 24) // 2
    for x, lab, val in ((M, "Sees its scores", O["seen"]), (W - M - tw, "Scores hidden", O["hidden"])):
        d.rounded_rectangle([x, y, x + tw, y + 216], radius=18, fill=TINT)
        T(d, (x + 24, y + 18), lab, 34, INK, True)
        T(d, (x + 24, y + 58), val, 96, ARM_COL, True)
        T(d, (x + 24, y + 166), f"{C['n_search']} picks, {C['n_conf']} fresh seeds each", 28, INK2)
    y += 232
    T(d, (M, y), minus(f"Difference: {O['diff']}"), 38, INK, True)
    T(d, (M, y + 50), minus(f"95% interval {O['ci']}  ·  {O['p']}"), 30, INK2)
    y = para(d, (M, y + 96), "The interval includes zero. This test doesn't show that seeing its scores helped the search.",
             32, INK, W - 2 * M, True, 1.25)
    T(d, (M, y + 14), f"Every arm: each search's pick on {C['n_conf']} fresh seeds", 32, INK, True)

    def draw(ax):
        SC.answer_panel(ax, C, True)
        ax.set_yticklabels([f"{v}%" for v in (0, 20, 40, 60, 80, 100)], fontsize=11)
        ax.tick_params(labelleft=True)

    y = int(y)
    ph = int(BOT - (y + 58) - 6 - 2 * 28 * 1.3 - 6)                # the chart takes the room that is left
    img.paste(panel(draw, W - 2 * M, ph, left=0.105, right=0.995, top=0.975, bottom=0.215), (M, y + 58))
    yy = para(d, (M, y + 58 + ph + 6), f"Right of the line: two fixed networks, {C['n_comp']} fresh seeds each. Against the "
                                       f"retest's best: {minus(O['retest'])}.", 28, INK2, W - 2 * M)
    assert yy <= BOT, yy
    return checked(img)


def limits_frame(C, O):
    """The result again, then what it does and doesn't show."""
    img, d = canvas()
    T(d, (M, TOP), "What this does and doesn't show", 48, INK, True)
    y, bh = TOP + 80, 316
    d.rounded_rectangle([M, y, W - M, y + bh], radius=18, fill=TINT)
    f = R.font(104, True)
    x = M + 28
    for val, lab in ((O["seen"], "scores seen"), (O["hidden"], "scores hidden")):
        T(d, (x, y + 16), val, 104, ARM_COL, True)
        T(d, (x + 4, y + 130), lab, 32, INK2)
        x += f.getlength(val) + 26
        if lab == "scores seen":
            T(d, (x, y + 58), "vs", 36, INK2)
            x += R.font(36).getlength("vs") + 26
    assert x <= W - M, x
    T(d, (M + 28, y + 190), minus(f"{O['diff']}  ·  95% interval {O['ci']}"), 38, INK, True)
    T(d, (M + 28, y + 244), minus(f"The interval still allows a gain of up to {O['hi']} points."), 30, INK2)
    lines = ["One toy world, one AI scientist (an LLM), dev designs only. The held-out designs are still locked.",
             "Its research program and weights never changed. This tests a search, not a self-improving system.",
             "The next stage, where it rewrites its own research program, hasn't run: this test didn't show that "
             "seeing its scores helps."]
    size, lh, tw = 38, 1.3, W - 2 * M - 44
    n_lines = [len(SYS.wrap(t, R.font(size), tw)) for t in lines]
    y += bh + 48
    gap = min(64, (BOT - y - sum(n_lines) * size * lh) / (len(lines) - 1))
    assert gap >= 24, ("the limits don't fit", gap)
    for t in lines:
        d.ellipse([M + 4, y + 17, M + 20, y + 33], fill=ARM_COL)
        y = para(d, (M + 44, y), t, size, INK, tw, lh=lh) + gap
    assert y - gap <= BOT, y
    return checked(img)


def results_frames(C, O, nd):
    ncand = C["ncand"]
    yield results_frame(C, nd, 0, False, "Every search starts from the start's score on seeds 0-1: the bar to beat."), 2 * FPS
    for k in range(1, ncand + 1):
        yield results_frame(C, nd, k, False, f"Round {(k - 1) // C['K'] + 1}: candidate {k} of {ncand} in every search."), \
            int(0.75 * FPS)
    yield results_frame(C, nd, ncand, True, f"Each search's pick is retrained on {C['n_conf']} fresh seeds."), 3 * FPS
    yield answer_frame(C, O), 12 * FPS
    yield limits_frame(C, O), 11 * FPS


# ---------------------------------------------------------------- assembly
def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def banned_check():
    bad = sorted(s for s in SEEN if any(b in s.lower() for b in BANNED))
    assert SEEN and not bad, f"banned wording on screen: {bad}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills-only", action="store_true")
    a = ap.parse_args()
    out = HERE / "mobile"
    out.mkdir(exist_ok=True)
    ch = MV.chosen_search()                                   # the search fixed by README.md's rule
    P = PS.load(ch["search"])
    spec = R.SPECS["w"]
    designs = torch.load(ROOT / "arch_rsi" / "stage2" / "targets" / "w_dev.pt")
    di = int(R.has_hidden_cells(designs).nonzero()[0, 0])     # the landscape scene's fixed design
    D = SB.load_search(CAMP / "W" / "g0" / f"s{ch['search']}", json.loads((CAMP / "baseline" / "summary.json").read_text()),
                       "w", designs, di, spec)
    C = SC.load(CAMP)
    O = outcome(C)
    nd = len(designs)
    assert nd == D["n_designs"] == P["n_designs"]
    cover_frame(P, O).save(out / "mobile_0_cover.png")           # a thumbnail to upload with the video
    system_frame().save(out / "mobile_1_system.png")
    n = max(len(P["start"]["bricks"]), len(P["pick"]["bricks"]))
    problem_frame(P, n, True).save(out / "mobile_2_problem.png")
    last = D["rows"][-1]
    search_frame(D, D["n"], len(last.get("bricks", [])), nd, True).save(out / "mobile_3_search.png")
    answer_frame(C, O).save(out / "mobile_4_answer.png")
    limits_frame(C, O).save(out / "mobile_5_limits.png")
    if a.stills_only:
        banned_check()
        return
    path = out / "stage2b_mobile.mp4"
    # H.264 High, 4:2:0, constant 24 frames a second, BT.709 tagged, a keyframe every second, index at the front.
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv,"
                           "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv", "-c:v", "libx264",
                           "-profile:v", "high", "-threads", "2", "-pix_fmt", "yuv420p", "-crf", "14", "-preset", "slow",
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
        secs[part] = round(n_f / FPS, 1)

    write("system", system_frames())
    write("problem", problem_frames(P))
    write("one_search", search_frames(D))
    write("all_searches_answer_limits", results_frames(C, O, nd))
    ff.stdin.close()
    assert ff.wait() == 0
    banned_check()
    pr = probe(path)
    v = pr["streams"][0]
    assert len(pr["streams"]) == 1 and (v["codec_name"], v["pix_fmt"], v["width"], v["height"]) == ("h264", "yuv420p", W, H), v
    assert v["r_frame_rate"] == v["avg_frame_rate"] == f"{FPS}/1", v
    assert {v.get(k) for k in ("color_space", "color_primaries", "color_transfer")} == {"bt709"}, v
    kbps = int(pr["format"]["bit_rate"]) / 1000
    assert 2 * 192 <= kbps <= 30000, kbps       # LinkedIn's help page lists 192 Kbps to 30 Mbps; stay well inside it
    total = round(sum(secs.values()), 1)
    assert abs(float(pr["format"]["duration"]) - total) < 0.1 and 95.5 <= total < 96.5, (pr["format"]["duration"], secs)
    data = {"chosen_search": ch, "size": [W, H], "fps": FPS, "parts_seconds": secs, "video_seconds": total,
            "design_index_problem": P["di"], "design_index_search": di, "cover": "mobile_0_cover.png",
            "file": {"codec": f"{v['codec_name']} {v['profile']}", "pix_fmt": v["pix_fmt"], "audio": "none",
                     "kbps": round(kbps), "bytes": int(pr["format"]["size"])},
            "checks": {"smallest_text_px": MIN_PX, "safe_area": {"sides": M, "top": TOP, "bottom": H - BOT},
                       "banned_words_absent": list(BANNED), "strings_checked": len(SEEN)}}
    (out / "stage2b_mobile_data.json").write_text(json.dumps(data, indent=1))
    print(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
