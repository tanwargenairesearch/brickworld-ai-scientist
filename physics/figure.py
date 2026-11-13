"""Figure for the article: stills from the replay plus the static-stability shares.
  .venv/bin/python figure.py   (after run_check.py and render.py)
Writes media/05-physics.png (1200 px wide, drawn for a phone).
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
MEDIA = HERE / "media"
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
SYMBOLS = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
W = 1200
INK, INK2, MUTED, HAIR = (31, 35, 40), (87, 96, 106), (110, 119, 129), (225, 224, 217)
GOOD, WARN, CRIT = (12, 163, 12), (250, 178, 25), (208, 59, 59)
BG = (255, 255, 255)


def f(size, bold=False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def text_w(dr, s, font):
    return dr.textlength(s, font=font)


def wrap(dr, s, font, width):
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if text_w(dr, t, font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def main():
    res = json.loads((HERE / "results.json").read_text())
    meta = json.loads((MEDIA / "render_meta.json").read_text())
    img = Image.new("RGB", (W, 3000), BG)
    dr = ImageDraw.Draw(img)
    x0, y = 48, 44
    for line in wrap(dr, "Exact on the grid. Would it stand with loose bricks?", f(46, True), W - 2 * x0):
        dr.text((x0, y), line, font=f(46, True), fill=INK)
        y += 58
    y += 6
    for line in wrap(dr, "Exploratory check, planned before the physics ran. Top: one exact build replayed in MuJoCo. "
                         "Bottom: force-balance analysis of every exact build, and of the target designs, on 512 new "
                         "designs per world.", f(27), W - 2 * x0):
        dr.text((x0, y), line, font=f(27), fill=INK2)
        y += 37
    y += 22

    # stills
    welded = Image.open(MEDIA / "clip2_welded_raw.png").convert("RGB")
    loose = Image.open(MEDIA / "clip2_loose_raw.png").convert("RGB")
    boxes = [Image.eval(im, lambda v: 255 - v).convert("L").point(lambda v: 255 if v > 12 else 0).getbbox()
             for im in (welded, loose)]
    bx = (min(b[0] for b in boxes) - 20, min(b[1] for b in boxes) - 20,
          max(b[2] for b in boxes) + 20, max(b[3] for b in boxes) + 20)
    welded, loose = welded.crop(bx), loose.crop(bx)
    pw = (W - 2 * x0 - 24) // 2
    ph = int(welded.height * pw / welded.width)
    for k, (im, title, status, ok) in enumerate((
            (welded, "Snapped in place", "Stands", True),
            (loose, "Loose bricks", "Falls", meta["clips"]["clip2"]["loose_stands"]))):
        px = x0 + k * (pw + 24)
        dr.text((px, y), title, font=f(30, True), fill=INK)
        img.paste(im.resize((pw, ph), Image.LANCZOS), (px, y + 44))
        col = GOOD if (status == "Stands") else CRIT
        if k == 1 and ok:
            status, col = "Stands", GOOD
        dr.text((px, y + 44 + ph + 4), "✓" if status == "Stands" else "✗", font=ImageFont.truetype(SYMBOLS, 30), fill=col)
        dr.text((px + 34, y + 44 + ph + 6), status, font=f(27, True), fill=col)
    y += 44 + ph + 52
    for line in wrap(dr, "The same exact build by Sonnet's winning recipe. Loose bricks were released with the check's "
                         "±0.3 mm offsets.", f(24), W - 2 * x0):
        dr.text((x0, y), line, font=f(24), fill=MUTED)
        y += 33
    y += 30
    dr.line([(x0, y), (W - x0, y)], fill=HAIR, width=2)
    y += 34

    # bars
    dr.text((x0, y), "Which would stand as loose bricks?", font=f(32, True), fill=INK)
    y += 46
    dr.text((x0, y), "Force-balance analysis of each finished build", font=f(25), fill=INK2)
    y += 44
    lx = x0
    for col, lab in ((GOOD, "Stable"), (WARN, "Knife-edge"), (CRIT, "Unstable")):
        dr.rounded_rectangle([lx, y + 6, lx + 24, y + 30], radius=4, fill=col)
        dr.text((lx + 34, y + 2), lab, font=f(25), fill=INK)
        lx += 34 + text_w(dr, lab, f(25)) + 40
    y += 56
    rows = [
        ("Small world, 5×5×3", None),
        ("The target designs themselves", ("small", "designs")),
        ("Starting recipe's exact builds", ("small", "baseline")),
        ("Random-search winners' exact builds", ("small", "random")),
        ("Claude's winners' exact builds", ("small", "claude")),
        ("Larger world, 6×6×4", None),
        ("The target designs themselves", ("large", "designs")),
        ("Starting recipe's exact builds", ("large", "baseline")),
        ("Best recipe's exact builds", ("large", "best_transfer")),
    ]
    bar_x, bar_w = x0, W - 2 * x0
    for label, key in rows:
        if key is None:
            y += 10
            dr.text((x0, y), label, font=f(27, True), fill=INK2)
            y += 44
            continue
        if key[1] == "designs":
            g = res["by_scale"][key[0]]["designs_generator_bricks"]
            cc, n = g["class_counts"], g["n"]
            ntxt = f"{n:,} designs"
        else:
            g = res["by_scale"][key[0]]["groups"][key[1]]
            cc, n = g["class_counts"], g["n_builds"]
            ntxt = f"{n:,} builds"
        dr.text((x0, y), label, font=f(26), fill=INK)
        dr.text((W - x0 - text_w(dr, ntxt, f(23)), y + 3), ntxt, font=f(23), fill=MUTED)
        y += 38
        cx = bar_x
        segs = [(cc["stable"], GOOD, (255, 255, 255)), (cc["marginal"], WARN, (11, 11, 11)), (cc["unstable"], CRIT, (255, 255, 255))]
        for i, (c, colr, tcol) in enumerate(segs):
            if c == 0:
                continue
            w_ = bar_w * c / n
            dr.rounded_rectangle([cx, y, cx + w_ - 2, y + 40], radius=4, fill=colr)
            pct = f"{100 * c / n:.0f}%"
            if w_ > text_w(dr, pct, f(24, True)) + 16:
                dr.text((cx + 10, y + 7), pct, font=f(24, True), fill=tcol)
            cx += w_
        y += 48
        if key[1] == "baseline":
            dr.text((x0, y), "All on designs without overhangs: it built none with an overhang exactly",
                    font=f(22), fill=MUTED)
            y += 30
        y += 14
    y += 10
    notes = [
        "Knife-edge: a brick's centre of mass sits exactly over the edge of what holds it up, so any nudge toward "
        "the overhang tips it. Unstable: it falls without a nudge. Target designs are scored with the bricks "
        "the generator used to make them.",
        "Real bricks also grip each other by their studs, which this check leaves out.",
    ]
    for nte in notes:
        for line in wrap(dr, nte, f(24), W - 2 * x0):
            dr.text((x0, y), line, font=f(24), fill=MUTED)
            y += 33
        y += 8
    y += 30
    out = img.crop((0, 0, W, y))
    out.save(MEDIA / "05-physics.png")
    print("saved", out.size)


if __name__ == "__main__":
    main()
