"""Two-panel still (snapped vs loose) for the Claude Doc: media/05-stills.png."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
HERE = Path(__file__).parent; MEDIA = HERE / "media"
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"; SYM = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
f = lambda s, b=False: ImageFont.truetype(FONT, s, index=1 if b else 0)
W, X0, GAP = 1200, 40, 24
INK, GOOD, CRIT = (31, 35, 40), (12, 163, 12), (208, 59, 59)
a = Image.open(MEDIA / "clip2_welded_raw.png").convert("RGB"); b = Image.open(MEDIA / "clip2_loose_raw.png").convert("RGB")
bb = [Image.eval(im, lambda v: 255 - v).convert("L").point(lambda v: 255 if v > 12 else 0).getbbox() for im in (a, b)]
bx = (min(x[0] for x in bb) - 20, min(x[1] for x in bb) - 20, max(x[2] for x in bb) + 20, max(x[3] for x in bb) + 20)
a, b = a.crop(bx), b.crop(bx)
pw = (W - 2 * X0 - GAP) // 2; ph = int(a.height * pw / a.width)
img = Image.new("RGB", (W, ph + 150), (255, 255, 255)); dr = ImageDraw.Draw(img)
for k, (im, t, s, c, m) in enumerate(((a, "Snapped in place", "Stands", GOOD, "✓"), (b, "Loose bricks", "Falls", CRIT, "✗"))):
    x = X0 + k * (pw + GAP)
    dr.text((x, 24), t, font=f(32, True), fill=INK)
    img.paste(im.resize((pw, ph), Image.LANCZOS), (x, 76))
    dr.text((x, 76 + ph + 8), m, font=ImageFont.truetype(SYM, 30), fill=c)
    dr.text((x + 34, 76 + ph + 10), s, font=f(28, True), fill=c)
img = img.crop((0, 0, W, 76 + ph + 56)); img.save(MEDIA / "05-stills.png"); print(img.size)
