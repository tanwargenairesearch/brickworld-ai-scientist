"""VISUALIZATION ONLY: a 15-second teaser for the LinkedIn post (4:5, 1080x1350, 30 fps, silent, seamless loop).

The user's brief (16 Nov 2026): the media is a hook to bring people to the blog, with no results and no numbers.
So the frame holds only three questions and the three nested loops: train (inner), design (middle) and
rewrite itself (outer), with a small brick stack being built at the centre.

Motion, all periodic in 15 s so the clip loops without a jump:
- each loop's dot circles at its own speed: inner 3 s, middle 7.5 s, outer 15 s per turn;
- when a dot completes a turn, a ripple travels outward to the next ring (a result feeding the loop above);
- one brick lands at the centre on each turn of the inner loop;
- the emphasis moves outward with the text: lines 1 and 2 light the inner rings, then the outer one.
Rules kept from the earlier videos: text at least 28 px, nothing outside the safe area, no banned word on screen.
usage (project root): nice -n 10 .venv/bin/python -I -B arch_rsi/viz/teaser/make_teaser.py
"""
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H, FPS, DUR, S = 1080, 1350, 30, 15.0, 2
N = int(FPS * DUR)
MARGIN, SAFE_TOP, SAFE_BOT = 64, 96, 1350 - 96
FONTS = Path.home() / "Library" / "Fonts"


def font(name, px):
    assert px >= 28, "text below 28 px"
    return ImageFont.truetype(str(FONTS / name), int(px * S))


F_HEAD, F_SUB = font("Inter-SemiBold.otf", 64), font("Inter-Regular.otf", 32)
F_MONO, F_BRAND = font("JetBrainsMono-Regular.ttf", 28), font("JetBrainsMono-Regular.ttf", 28)

BG, INK, INK2, MUTED = (8, 9, 10), (247, 248, 248), (208, 214, 224), (138, 143, 152)
LINE, PANEL, ACC, OK, RUN, BRICK = (72, 74, 82), (17, 18, 20), (128, 136, 244), (76, 183, 130), (242, 201, 76), (76, 141, 240)
CX, CY = 540, 760
R = (138, 246, 354)
PERIOD = (3.0, 7.5, 15.0)
DOT = (OK, RUN, ACC)
LABELS = ("train", "design", "rewrite itself")
LABEL_ANGLE = (12, -14, 152)  # degrees clockwise from 12 o'clock; negative puts the pill on the left
LINES = ((0.0, 5.0, "It designs the learners.", None),
         (5.0, 10.0, "Then it rewrites how it does research.", None),
         (10.0, 15.0, "Can an AI scientist learn to discover?", "The full experiment is on my blog."))
CTA = "Read it · link in the comments"
BANNED = ("llm", "gate", "claude", "gpt")
for s in [l[2] for l in LINES] + [l[3] for l in LINES if l[3]] + list(LABELS) + [CTA, "BRICKWORLD AI SCIENTIST"]:
    assert not any(b in s.lower() for b in BANNED), s

# emphasis per ring in each text phase; transitions take 0.6 s at each phase start, so 15 s wraps to 0 smoothly
EMPH = ((1.0, 1.0, 0.32), (0.55, 0.55, 1.0), (1.0, 1.0, 1.0))


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def emphasis(t):
    k = 0 if t < 5 else (1 if t < 10 else 2)
    start, prev = (0.0, 2) if k == 0 else ((5.0, 0) if k == 1 else (10.0, 1))
    a = ease((t - start) / 0.6)
    return [EMPH[prev][i] + (EMPH[k][i] - EMPH[prev][i]) * a for i in range(3)]


def background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx - CX, yy - CY)
    img = np.empty((H, W, 3), np.float32)
    img[:] = BG
    glow = 0.20 * np.exp(-(d / 470.0) ** 2)
    img += glow[..., None] * (np.array(ACC, np.float32) - img)
    grid = ((xx % 54) < 1) | ((yy % 54) < 1)
    fade = np.clip(1 - d / 720.0, 0, 1) * 0.28
    img = np.where(grid[..., None], img + fade[..., None] * (np.array(LINE, np.float32) - img), img)
    return Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGB").convert("RGBA")


def rgba(c, a):
    return (c[0], c[1], c[2], int(255 * max(0.0, min(1.0, a))))


def mix(c, f):
    t = 255 if f > 0 else 0
    return tuple(int(v + (t - v) * abs(f)) for v in c)


def pos(r, ang):  # ang in radians, clockwise from 12 o'clock
    return CX + r * math.sin(ang), CY - r * math.cos(ang)


def draw_rings(d, em):
    for i, r in enumerate(R):
        box = [(CX - r) * S, (CY - r) * S, (CX + r) * S, (CY + r) * S]
        if i < 2:
            d.ellipse(box, outline=rgba(mix(LINE, 0.25), 0.35 + 0.65 * em[i]), width=2 * S)
        else:
            for k in range(96):
                a0 = k * 3.75
                d.arc(box, a0, a0 + 1.6, fill=rgba(ACC, 0.25 + 0.6 * em[i]), width=2 * S)


def draw_ripples(d, t, em):
    for i, r in enumerate(R):
        age = t % PERIOD[i]
        if age > 0.9:
            continue
        r2 = R[i + 1] if i < 2 else R[2] + 70
        rr = r + (r2 - r) * ease(age / 0.9)
        a = (1 - age / 0.9) * 0.55 * em[i]
        d.ellipse([(CX - rr) * S, (CY - rr) * S, (CX + rr) * S, (CY + rr) * S], outline=rgba(DOT[i], a), width=3 * S)


def draw_dots(d, t, em):
    for i, r in enumerate(R):
        ang = 2 * math.pi * ((t / PERIOD[i]) % 1.0)
        for j in range(16, 0, -1):
            x, y = pos(r, ang - j * 9.0 / r)
            rad = 7.5 * (1 - j / 18)
            d.ellipse([(x - rad) * S, (y - rad) * S, (x + rad) * S, (y + rad) * S], fill=rgba(DOT[i], (1 - j / 16) * 0.42 * em[i]))
        x, y = pos(r, ang)
        d.ellipse([(x - 21) * S, (y - 21) * S, (x + 21) * S, (y + 21) * S], fill=rgba(DOT[i], 0.18 * em[i]))
        d.ellipse([(x - 9.5) * S, (y - 9.5) * S, (x + 9.5) * S, (y + 9.5) * S], fill=rgba(DOT[i], 0.35 + 0.65 * em[i]))


HW, HH, CH = 34, 17, 34
CUBES = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1))


def iso(x, y, z):
    return CX + (x - y) * HW, CY + 6 + (x + y) * HH - z * CH - 22


def draw_bricks(d, t, em):
    for gx in range(-1, 3):
        for gy in range(-1, 3):
            pts = [iso(gx, gy, 0), iso(gx + 1, gy, 0), iso(gx + 1, gy + 1, 0), iso(gx, gy + 1, 0)]
            d.polygon([(px * S, py * S) for px, py in pts], outline=rgba(LINE, 0.55), width=1 * S)
    fade = 1 - ease((t - 14.5) / 0.5)
    for k, (x, y, z) in sorted(enumerate(CUBES), key=lambda kc: (kc[1][0] + kc[1][1], kc[1][2])):
        t0 = 0.25 + 3.0 * k
        if t < t0:
            continue
        p = ease((t - t0) / 0.4)
        lift = (1 - p) * 1.6
        a = min(1.0, p * 1.8) * fade * (0.55 + 0.45 * em[0])
        zz = z + lift
        edge = rgba(mix(BRICK, -0.45), a)
        faces = (([iso(x + 1, y, zz), iso(x + 1, y + 1, zz), iso(x + 1, y + 1, zz + 1), iso(x + 1, y, zz + 1)], mix(BRICK, -0.25)),
                 ([iso(x, y + 1, zz), iso(x + 1, y + 1, zz), iso(x + 1, y + 1, zz + 1), iso(x, y + 1, zz + 1)], BRICK),
                 ([iso(x, y, zz + 1), iso(x + 1, y, zz + 1), iso(x + 1, y + 1, zz + 1), iso(x, y + 1, zz + 1)], mix(BRICK, 0.25)))
        for pts, col in faces:
            d.polygon([(px * S, py * S) for px, py in pts], fill=rgba(col, a), outline=edge, width=1 * S)


def pill(d, x, y, text, alpha, anchor_left):
    w = d.textlength(text, font=F_MONO) / S + 28
    x0 = x - w if not anchor_left else x
    box = [x0 * S, (y - 22) * S, (x0 + w) * S, (y + 22) * S]
    assert MARGIN - 8 <= x0 and x0 + w <= W - MARGIN + 8, ("label outside the safe area", text)
    d.rounded_rectangle(box, radius=10 * S, fill=rgba(PANEL, 0.92 * alpha), outline=rgba(LINE, alpha), width=1 * S)
    d.text(((x0 + 14) * S, y * S), text, font=F_MONO, fill=rgba(INK2, alpha), anchor="lm")
    return [(x0, y - 22), (x0 + w, y - 22), (x0, y + 22), (x0 + w, y + 22)]


def draw_labels(d, em):
    for i, r in enumerate(R):
        ang = math.radians(LABEL_ANGLE[i])
        x, y = pos(r, ang)
        left = LABEL_ANGLE[i] > 0
        corners = pill(d, x + (14 if left else -14), y, LABELS[i], 0.3 + 0.7 * em[i], left)
        rs = [math.hypot(px - CX, py - CY) for px, py in corners]
        # keep each label clear of the other loops' dots (halo radius 21)
        assert i == 2 or max(rs) <= R[i + 1] - 22, ("label in the next loop's path", LABELS[i])
        assert i == 0 or min(rs) >= R[i - 1] + 22, ("label in the inner loop's path", LABELS[i])


def wrap(d, text, f, width):
    """One line if it fits; otherwise the two-line break with the most even line lengths (no orphaned word)."""
    if d.textlength(text, font=f) / S <= width:
        return [text]
    words, best = text.split(), None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        wa, wb = d.textlength(a, font=f) / S, d.textlength(b, font=f) / S
        if max(wa, wb) <= width and (best is None or max(wa, wb) < best[0]):
            best = (max(wa, wb), [a, b])
    assert best, ("headline needs more than two lines", text)
    return best[1]


def draw_text(d, t):
    for (a, b, head, sub) in LINES:
        if not a <= t < b + 1e-9:
            continue
        alpha = ease((t - a) / 0.5) * (1 - ease((t - (b - 0.45)) / 0.45))
        rise = (1 - ease((t - a) / 0.6)) * 14
        y = 170 + rise
        for line in wrap(d, head, F_HEAD, W - 2 * MARGIN):
            d.text((W / 2 * S, y * S), line, font=F_HEAD, fill=rgba(INK, alpha), anchor="mm")
            y += 78
        if sub:
            d.text((W / 2 * S, (y + 8) * S), sub, font=F_SUB, fill=rgba(INK2, alpha), anchor="mm")
        assert y + 40 < CY - R[2] - 10, "text runs into the rings"
    a3 = ease((t - 10.3) / 0.5) * (1 - ease((t - 14.55) / 0.45)) if t >= 10.3 else 0.0
    if a3 > 0:
        tw = d.textlength(CTA, font=F_MONO) / S + 44
        box = [(W / 2 - tw / 2) * S, 1148 * S, (W / 2 + tw / 2) * S, 1204 * S]
        d.rounded_rectangle(box, radius=28 * S, fill=rgba(ACC, 0.16 * a3), outline=rgba(ACC, 0.9 * a3), width=2 * S)
        d.text((W / 2 * S, 1176 * S), CTA, font=F_MONO, fill=rgba(INK, a3), anchor="mm")
    d.text((W / 2 * S, 1230 * S), "BRICKWORLD AI SCIENTIST", font=F_BRAND, fill=rgba(MUTED, 0.9), anchor="mm")
    assert 1230 + 16 <= SAFE_BOT


def frame(bg, t):
    ov = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    em = emphasis(t)
    draw_rings(d, em)
    draw_ripples(d, t, em)
    draw_labels(d, em)
    draw_bricks(d, t, em)
    draw_dots(d, t, em)
    draw_text(d, t)
    return Image.alpha_composite(bg, ov.resize((W, H), Image.LANCZOS)).convert("RGB")


def main():
    bg = background()
    out = HERE / "teaser_1080x1350.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-g", str(FPS * 2),
                           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                           "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    stills = {int(2.5 * FPS): "still_1.png", int(7.5 * FPS): "still_2.png", int(12.5 * FPS): "cover.png"}
    for n in range(N):
        im = frame(bg, n / FPS)
        ff.stdin.write(im.tobytes())
        if n in stills:
            im.save(HERE / stills[n])
    ff.stdin.close()
    assert ff.wait() == 0
    print("wrote", out, out.stat().st_size, "bytes;", N, "frames")


if __name__ == "__main__":
    main()
