"""VISUALIZATION ONLY: the whole system at a glance, as three loops, each learning something different.
Generic wording (no model or person names). Facts are the protocol's own: 131,072 PPO steps per run on world W,
4 rounds of 3 candidates per search, 2 training seeds, 512 dev designs, scores hidden in the control arm,
Stage 2c not run because Stage 2b's gate failed.

usage (project root): nice -n 19 .venv/bin/python -I arch_rsi/viz/stages/system/make_system.py
render() takes the loops' status chips and an optional highlighted loop. The defaults were updated on 10 November 2026,
after Stage 2b's gate failed: Loop 2 done, Loop 3 not run (the Stage 2b video highlights Loop 2).
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 1920, 1080
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
SURF, INK, INK2, MUTED = (252, 252, 251), (11, 11, 11), (82, 81, 78), (137, 135, 129)
KIND = {  # fill, outline
    "sci": ((230, 240, 251), (42, 120, 214)),      # the AI scientist (an LLM)
    "frozen": ((241, 240, 236), (138, 138, 133)),  # frozen code it can't edit
    "agent": ((253, 240, 225), (217, 130, 43)),    # the learning agent and its world
}
CUT = (230, 100, 50)


def font(size, bold=False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def wrap(s, f, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if f.getlength(t) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur] if cur else lines


def para(d, xy, s, size, col=INK2, width=500, bold=False, lh=1.3):
    f = font(size, bold)
    x, y = xy
    for ln in wrap(s, f, width):
        d.text((x, y), ln, font=f, fill=col)
        y += size * lh
    return y


def dashed_rect(d, box, col, width=2, dash=10, r=0):
    x0, y0, x1, y1 = box
    for (a, b, c, e) in [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]:
        n = int(max(abs(c - a), abs(e - b)) / dash)
        for k in range(0, n, 2):
            t0, t1 = k / n, min(1, (k + 1) / n)
            d.line([(a + (c - a) * t0, b + (e - b) * t0), (a + (c - a) * t1, b + (e - b) * t1)], fill=col, width=width)


def node(d, box, kind, title, body, dotted=False, lock=False):
    fill, line = KIND[kind]
    d.rounded_rectangle(box, radius=12, fill=fill, outline=None if dotted else line, width=2)
    if dotted:
        dashed_rect(d, box, line)
    x0, y0, x1, y1 = box
    d.text((x0 + 14, y0 + 12), title, font=font(18, True), fill=INK)
    para(d, (x0 + 14, y0 + 42), body, 16, INK2, width=x1 - x0 - 28)
    if lock:  # small padlock: frozen
        cx, cy = x1 - 22, y0 + 20
        d.rounded_rectangle([cx - 8, cy - 2, cx + 8, cy + 11], radius=2, fill=(138, 138, 133))
        d.arc([cx - 6, cy - 11, cx + 6, cy + 3], 180, 360, fill=(138, 138, 133), width=3)


def arrow(d, pts, col=INK2, width=3, head=12):
    d.line(pts, fill=col, width=width, joint="curve")
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    import math
    ang = math.atan2(yb - ya, xb - xa)
    p1 = (xb - head * math.cos(ang - 0.45), yb - head * math.sin(ang - 0.45))
    p2 = (xb - head * math.cos(ang + 0.45), yb - head * math.sin(ang + 0.45))
    d.polygon([(xb, yb), p1, p2], fill=col)


def loop_column(d, x0, spec):
    """A loop as four nodes in a clockwise square: top-left -> top-right -> bottom-right -> bottom-left -> back."""
    cw, nw, nh, gap_y = 560, 250, 170, 110
    yA, yB = 250, 250 + nh + gap_y
    tl, tr = (x0, yA, x0 + nw, yA + nh), (x0 + cw - nw, yA, x0 + cw, yA + nh)
    bl, br = (x0, yB, x0 + nw, yB + nh), (x0 + cw - nw, yB, x0 + cw, yB + nh)
    d.text((x0, 128), spec["head"], font=font(22, True), fill=INK)
    d.text((x0, 160), spec["sub"], font=font(16), fill=INK2)
    chip_f = font(14, True)
    tw = chip_f.getlength(spec["status"])
    d.rounded_rectangle([x0, 192, x0 + tw + 20, 220], radius=14, fill=spec["chip"])
    d.text((x0 + 10, 197), spec["status"], font=chip_f, fill=(255, 255, 255))
    for box, n in zip((tl, tr, br, bl), spec["nodes"]):
        node(d, box, *n, dotted=spec.get("dotted", False))
    mid = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)  # noqa: E731
    arrow(d, [(tl[2] + 6, mid(tl)[1]), (tr[0] - 8, mid(tr)[1])])
    arrow(d, [(mid(tr)[0], tr[3] + 6), (mid(br)[0], br[1] - 8)])
    arrow(d, [(br[0] - 6, mid(br)[1]), (bl[2] + 8, mid(bl)[1])])
    arrow(d, [(mid(bl)[0], bl[1] - 6), (mid(tl)[0], tl[3] + 8)])
    if spec.get("back_label"):
        para(d, (mid(bl)[0] + 34, (tl[3] + bl[1]) / 2 - 18), spec["back_label"], 14, INK2, width=160)
    if spec.get("cut"):
        x, y = mid(bl)[0], (tl[3] + bl[1]) / 2 + 8
        for dy in (-7, 7):
            d.line([(x - 18, y + dy + 6), (x + 18, y + dy - 6)], fill=CUT, width=4)
        para(d, (x - 120, y - 6), "control arm:\nscores hidden", 14, CUT, width=100, bold=True)
    center_y = yB + nh + 56
    para(d, (x0, center_y), spec["learns"], 16, INK, width=cw, bold=True)
    para(d, (x0, center_y + 30), spec["note"], 15, INK2, width=cw)
    return tl, tr, br, bl


def render(loop2_status="STAGES 1–2b · DONE", loop3_status="STAGE 2c · NOT RUN: GATE FAILED", loop3_run=False,
           highlight=None):
    img = Image.new("RGB", (W, H), SURF)
    d = ImageDraw.Draw(img)
    d.text((40, 26), "What I'm building: three loops, each learning something different", font=font(30, True), fill=INK)
    d.text((40, 70), "A brick-assembly world on one laptop GPU. Each loop runs inside the next one out. "
                     "Only Loop 1 changes any network's weights.", font=font(17), fill=INK2)
    cols = [
        {"head": "Loop 1 · the agent learns to build", "sub": "Reinforcement learning (PPO), inside every experiment",
         "status": "RUNS IN EVERY EXPERIMENT", "chip": (217, 130, 43),
         "nodes": [("agent", "Brick world", "A target design. The agent sees the grid it has built so far."),
                   ("agent", "Policy network", "Its architecture was written by the AI scientist in Loop 2."),
                   ("agent", "Action and reward", "Place one brick or press DONE. Reward for matching the target."),
                   ("agent", "PPO update", "The network's weights change, from its own trial and error.")],
         "learns": "Learns: the network's weights.",
         "note": "Every candidate gets the same compute: 131,072 steps for the starting network, fewer for "
                 "costlier networks or recipes. On the laptop's GPU."},
        {"head": "Loop 2 · the AI scientist searches", "sub": "Architecture and recipe search, 4 rounds of 3",
         "status": loop2_status, "chip": (42, 120, 214),
         "nodes": [("sci", "Read", "Its research program, and every earlier experiment's code, hypothesis and score."),
                   ("sci", "Write 3 candidates", "Each a network and a training recipe, with a hypothesis that could fail."),
                   ("frozen", "Check, then train", "Sandbox and caps; trained by Loop 1 on 2 seeds."),
                   ("frozen", "Evaluate", "The share of 512 dev designs built exactly. Frozen code.")],
         "back_label": "scores fed back after each round", "cut": True,
         "learns": "Learns: nothing's weights. It adapts within one search.",
         "note": "After 4 rounds the best candidate is retested and confirmed on fresh seeds. The control arm runs the "
                 "same search with every score hidden: if the scores guide the search, hiding them should hurt."},
        {"head": "Loop 3 · the procedure improves itself", "sub": "The AI scientist rewrites its own research program",
         "status": loop3_status, "chip": (42, 120, 214) if loop3_run else (137, 135, 129), "dotted": not loop3_run,
         "nodes": [("frozen", "Confirmed results", "From many finished searches in Loop 2."),
                   ("sci", "Rewrite the program", "The AI scientist edits the advice and settings it searches with."),
                   ("frozen", "Test it", "New program against the old one, on fresh searches."),
                   ("frozen", "Keep it only if better", "Then the next generation starts from it.")],
         "learns": "Learns: the research program. Where \"recursive\" would begin.",
         "note": ("The AI scientist's own weights never change. Tested on a larger world it never saw while rewriting."
                  if loop3_run else
                  "The AI scientist's own weights never change. Not run: Stage 2b didn't show that the scores guide "
                  "the search.")},
    ]
    xs = [40, 680, 1320]
    boxes = [loop_column(d, x, c) for x, c in zip(xs, cols)]
    # links between loops (in the gutters)
    tl1, tr1, br1, bl1 = boxes[0]
    tl2, tr2, br2, bl2 = boxes[1]
    tl3 = boxes[2][0]
    ylink = br2[3] + 22
    arrow(d, [((br2[0] + br2[2]) / 2, br2[3] + 6), ((br2[0] + br2[2]) / 2, ylink), (640, ylink),
              (640, (tr1[1] + tr1[3]) / 2), (tr1[2] + 8, (tr1[1] + tr1[3]) / 2)], col=KIND["agent"][1])
    para(d, (700, ylink + 8), "each candidate is trained by Loop 1", 14, KIND["agent"][1], width=400, bold=True)
    ymid = (tr2[3] + br2[1]) / 2
    arrow(d, [(tr2[2] - 30, ymid), (1280, ymid), (1280, (tl3[1] + tl3[3]) / 2), (tl3[0] - 8, (tl3[1] + tl3[3]) / 2)],
          col=MUTED)
    # the label sits in the gutter; it moves aside when a highlight outline would cross it
    para(d, ({1: 1262, 2: 1228}.get(highlight, 1244), ymid + 10), "confirmed picks", 14, MUTED, width=72, bold=True)
    # frozen footer
    d.line([(40, 1010), (W - 40, 1010)], fill=(225, 224, 217), width=1)
    d.text((40, 1026), "Frozen, never edited by the AI scientist: the evaluator, the training code, the compute budget and "
                       "the time limit. Held-out designs stay locked until the end.", font=font(16), fill=INK2)
    lx = 1560
    for k, (kind, lab) in enumerate([("sci", "the AI scientist (an LLM, no tools)"), ("frozen", "frozen code"),
                                     ("agent", "the learning agent")]):
        fill, line = KIND[kind]
        y = 22 + k * 26
        d.rounded_rectangle([lx, y, lx + 22, y + 18], radius=4, fill=fill, outline=line, width=2)
        d.text((lx + 32, y), lab, font=font(15), fill=INK)
    if highlight is not None:  # fade the other loops, outline the highlighted one
        x = xs[highlight]
        for box in ((0, 120, x - 16, 1000), (x + 578, 120, W, 1000)):
            if box[2] > box[0]:
                veil = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), SURF)
                img.paste(Image.blend(img.crop(box), veil, 0.6), box[:2])
        ImageDraw.Draw(img).rounded_rectangle((x - 14, 118, x + 574, 1002), radius=16, outline=(42, 120, 214), width=4)
    return img


def main():
    out = HERE / "system_loops.png"
    render().save(out)
    print(out)


if __name__ == "__main__":
    main()
