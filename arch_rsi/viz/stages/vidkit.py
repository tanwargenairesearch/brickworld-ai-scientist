"""Shared style and video writer for the per-stage article videos. Visualization only.

Every stage video: 1920x1080, 12 fps, H.264 (yuv420p, faststart), no audio, captions on screen, the same palette and
type as Stage 1's package (arch_rsi/viz/stage1/). Render with low priority while Stage 2b runs:
    nice -n 19 .venv/bin/python -I arch_rsi/viz/stages/<stage>/make_<stage>.py
The writer limits ffmpeg to 2 threads.

usage inside a stage script:
    from arch_rsi.viz.stages import vidkit as vk      (with the project root on sys.path)
    fig = vk.new_fig(); vk.header(fig, "Title", "Subtitle"); ... ; vk.caption(fig, "One line of caption")
    with vk.Writer(out / "stage.mp4") as w:
        w.add(fig, seconds=3.0)              # closes the figure
        w.add(vk.card("Title", ["line 1", "line 2"]), seconds=5)
"""
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

W_PX, H_PX, FPS = 1920, 1080, 12
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"      # Claude sees its scores / Claude's choice
GREEN = "#1baf7a"     # scores hidden from Claude
GREY = "#6e7781"      # the start / baselines
OTHER = "#a3a29b"     # not a new best, background marks
ORANGE = "#d9822b"    # random search / a second comparison
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial"], "font.size": 14, "text.color": INK,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "svg.fonttype": "none"})


def new_fig():
    fig = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    return fig


def header(fig, title, subtitle=None, stage=None):
    """Stage label (small caps line), title and subtitle at the top-left."""
    y = 0.955
    if stage:
        fig.text(0.05, y, stage.upper(), fontsize=13, color=MUTED, ha="left", fontweight="bold")
        y -= 0.045
    fig.text(0.05, y, title, fontsize=28, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.05, y - 0.06, subtitle, fontsize=15, color=INK2, ha="left", va="top")


def caption(fig, text, y=0.045, bold=True):
    fig.text(0.05, y, text, fontsize=16, color=INK, ha="left", fontweight="bold" if bold else None)


def style_axes(ax, ylim=(0, 100), yticks=(0, 20, 40, 60, 80, 100), pct=True):
    ax.set_facecolor(SURFACE)
    for y in yticks:
        ax.axhline(y, color=GRID, lw=1, zorder=0)
    ax.set_ylim(ylim[0] - 4, ylim[1] + 4)
    ax.set_yticks(list(yticks))
    ax.set_yticklabels([f"{v}%" if pct else str(v) for v in yticks], fontsize=13)
    ax.tick_params(length=0, pad=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def card(title, lines, stage=None, foot=None):
    """A text-only frame: a title and up to ~6 short lines."""
    fig = new_fig()
    if stage:
        fig.text(0.08, 0.84, stage.upper(), fontsize=15, color=MUTED, ha="left", fontweight="bold")
    fig.text(0.08, 0.76, title, fontsize=36, fontweight="bold", color=INK, ha="left")
    for k, t in enumerate(lines):
        fig.text(0.08, 0.62 - 0.095 * k, t, fontsize=22, color=INK2, ha="left")
    if foot:
        fig.text(0.08, 0.08, foot, fontsize=14, color=MUTED, ha="left")
    return fig


def to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


class Writer:
    """ffmpeg pipe writer. add(fig, seconds) renders the figure once, repeats it, and closes it."""

    def __init__(self, path, fps=FPS):
        self.path, self.fps, self.frames = Path(path), fps, 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_PX}x{H_PX}",
             "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-crf", "18",
             "-movflags", "+faststart", str(self.path)], stdin=subprocess.PIPE)
        return self

    def add(self, fig, seconds):
        buf = to_rgb(fig)
        plt.close(fig)
        assert buf.shape == (H_PX, W_PX, 3), buf.shape
        for _ in range(int(round(seconds * self.fps))):
            self.ff.stdin.write(buf.tobytes())
            self.frames += 1

    def __exit__(self, *exc):
        self.ff.stdin.close()
        assert self.ff.wait() == 0, "ffmpeg failed"
        return False

    @property
    def seconds(self):
        return self.frames / self.fps
