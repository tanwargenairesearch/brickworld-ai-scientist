"""VISUALIZATION ONLY: the article's cover image (1920x1080). One design, the start's build and one searched network's
build (the problem scene's illustration, same rule and checks), above the three numbers the article turns on.
Numbers come from campaigns/stage2b/results.json. Generic wording; nothing here scores or selects.

usage (project root): nice -n 19 .venv/bin/python -I arch_rsi/viz/stage2b/make_cover.py [--out ../linkedin/images/v15]
"""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PS = _load("problem_scene", HERE / "problem_scene.py")
MV = _load("make_2b_video", HERE / "make_2b_video.py")
SYS = _load("make_system", HERE.parent / "stages" / "system" / "make_system.py")
SB, R = PS.SB, PS.R
from PIL import Image, ImageDraw  # noqa: E402

W, H, M = 1920, 1080, 64
INK, INK2, MUTED, SURF = SB.INK, SB.INK2, SB.MUTED, SB.SURF


def render():
    res = json.loads((PS.CAMP / "results.json").read_text())
    assert res["primary_family"]["status"] == "final" and not res["primary_family"]["gate_passed"]
    arm = res["per_arm"]
    P = PS.load(MV.chosen_search()["search"])
    sp = P["spec"]
    img = Image.new("RGB", (W, H), SURF)
    d = ImageDraw.Draw(img)
    y = SYS.para(d, (M, 50), "An AI scientist writes the network.", 58, INK, W - 2 * M, bold=True, lh=1.16)
    y = SYS.para(d, (M, y), "Did its experiments guide the search?", 58, INK, W - 2 * M, bold=True, lh=1.16)
    span = (sp.X + sp.Y) * 0.5 + sp.Z
    top, bh = y + 64, 400
    sc = min(540 / ((sp.X + sp.Y) * 0.866), bh / span)
    cols = ((330, "Target design", None, R.TARGET_COL, f"{P['cells']} cells, some hidden"),
            (960, "The starting network", P["start"], PS.START_COL, None),
            (1590, "A network the AI scientist wrote", P["pick"], PS.PICK_COL, None))
    for cx, head, run, col, note in cols:
        SB.text(d, (cx, top - 44), head, 30, INK, bold=True, anchor="ma")
        iso = R.Iso(sp, sc, cx, top + sp.Z * sc)
        if run is None:
            iso.ground(d)
            iso.bricks(d, R.target_bricks(P["target"].numpy()), lambda b: col)
            SB.text(d, (cx, top + bh + 6), note, 26, INK2, anchor="ma")
        else:
            iso.ground(d, footprint=P["target"].any(0).numpy())
            iso.bricks(d, run["bricks"], lambda b, c=col: R.WRONG if b["wrong"] else c)
            msg = "built exactly" if run["exact"] else f"not exact: {run['missing']} cells missing"
            w = R.font(26).getlength(msg)
            R.badge(d, cx - w / 2 - 20, top + bh + 22, run["exact"], r=14)
            SB.text(d, (cx - w / 2 + 4, top + bh + 6), msg, 26, INK)
    yb = top + bh + 76
    d.line([(M, yb), (W - M, yb)], fill=(225, 224, 217), width=2)
    stats = ((arm["start_g"]["mean"], "the starting network", PS.START_COL),
             (arm["g0"]["primary_mean"], "AI scientist, seeing its scores", PS.PICK_COL),
             (arm["nofb"]["primary_mean"], "the same search, scores hidden", INK))
    for k, (v, lab, col) in enumerate(stats):
        x = M + k * (W - 2 * M) / 3
        SB.text(d, (x, yb + 26), f"{v * 100:.1f}%", 76, col, bold=True)
        SB.text(d, (x + 4, yb + 118), lab, 28, INK)
    SB.text(d, (M, H - 62), f"Share of {P['n_designs']} designs built exactly, on fresh seeds, in a {sp.X}×{sp.Y}×{sp.Z} "
                            "brick-assembly world. The builds above are one illustration.", 22, MUTED)
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "linkedin" / "images" / "v15"))
    a = ap.parse_args()
    out = Path(a.out) / "00-cover.png"
    render().save(out)
    print(out)
