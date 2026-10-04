"""Generate the report's architecture and historical-window diagrams."""

from pathlib import Path
import os
import shutil
import subprocess


REPORT_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPORT_DIR / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(REPORT_DIR / "build" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPORT_DIR / "build" / "cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as DrawingPath


BLUE = "#1D4ED8"
INK = "#1F2937"
GREY = "#6B7280"
LIGHT = "#EFF6FF"
FONT_SIZE = 10 * 72 / 72.27  # Match LaTeX's 10 pt small text.


def report_fonts():
    """Use the same Latin Modern fonts as the report's TeX distribution."""
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich is None:
        mac_tex = Path("/Library/TeX/texbin/kpsewhich")
        if mac_tex.is_file():
            kpsewhich = str(mac_tex)
        else:
            raise RuntimeError("Add your TeX distribution's bin directory to PATH.")
    fonts = {}
    for weight in ("regular", "bold"):
        filename = f"lmroman10-{weight}.otf"
        path = subprocess.check_output([kpsewhich, filename], text=True).strip()
        if not path or not Path(path).is_file():
            raise RuntimeError(f"TeX font not found: {filename}")
        fonts[weight] = FontProperties(fname=path, size=FONT_SIZE)
    return fonts


def canvas(width, height):
    """Size the drawing in millimeters to preserve its report footprint."""
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, width), ylim=(0, height))
    ax.set_axis_off()
    return fig, ax


def box(ax, fonts, x, y, width, height, title, body):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0,rounding_size=0.7",
        linewidth=0.6, edgecolor=BLUE, facecolor=LIGHT, zorder=2,
    ))
    lines = [(line, "bold") for line in title.split("\n")]
    lines += [(line, "regular") for line in body.split("\n")]
    for index, (line, weight) in enumerate(lines):
        line_y = y + height / 2 + ((len(lines) - 1) / 2 - index) * 4.22
        ax.text(x + width / 2, line_y, line, ha="center", va="center",
                color=INK, fontproperties=fonts[weight], zorder=3)


def arrow(ax, start, end, *, color=BLUE, dashed=False):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=0.7,
        color=color,
        shrinkA=0, shrinkB=0, zorder=1,
    )
    ax.add_patch(patch)
    if dashed:
        # Split the standard arrow so the dash pattern affects only its shaft.
        ax.figure.canvas.draw()
        path = patch.get_path()
        head_start = path.codes.tolist().index(DrawingPath.MOVETO, 1)
        patch.remove()
        for vertices, codes, fill, style in (
            (path.vertices[:head_start], path.codes[:head_start], "none", (0, (3.7, 4))),
            (path.vertices[head_start:], path.codes[head_start:], color, "-"),
        ):
            ax.add_patch(PathPatch(
                DrawingPath(vertices, codes), facecolor=fill, edgecolor=color,
                linewidth=0.7, linestyle=style, capstyle=patch.get_capstyle(),
                joinstyle=patch.get_joinstyle(), zorder=1,
            ))


def save(fig, name, title):
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, facecolor="white", metadata={
        "Title": title, "Software": "report/scripts/generate_assets.py",
    })
    plt.close(fig)
    print(path.relative_to(REPORT_DIR.parent))


def framework_architecture(fonts):
    fig, ax = canvas(130, 93)
    box(ax, fonts, 0.2, 78.3, 61, 14.5, "Sources and reviewed evidence",
        "Prices, membership, sectors,\nidentities and events")
    box(ax, fonts, 68.8, 78.3, 61, 14.5, "Acquisition and preparation",
        "Deterministic datasets\nand validated manifests")
    box(ax, fonts, 0.2, 57.8, 61, 14.5, "Configuration and research",
        "JSON parameters, grids\nand verified checkpoints")
    box(ax, fonts, 68.8, 57.8, 61, 14.5, "Interchangeable strategies",
        "Frozen signals, selection,\nsizing and decision audits")
    box(ax, fonts, 0.2, 38.5, 127, 13,
        "Shared portfolio accounting and lifecycle",
        "Turnover filtering, share orders, costs, financing and corporate actions")
    box(ax, fonts, 0.2, 19.3, 127, 13,
        "Monthly backtest and competition-period simulation",
        "NAV, holdings and trades; performance, risk and reconciled Brinson attribution")
    box(ax, fonts, 0.2, 0.2, 127, 13, "Efficient-frontier diagnostics",
        "Saved live formation targets and historical returns; no change to trades")

    arrow(ax, (61.2, 85.55), (68.8, 85.55))
    arrow(ax, (99.3, 78.3), (99.3, 72.3))
    arrow(ax, (61.2, 65.05), (68.8, 65.05))
    arrow(ax, (99.3, 57.8), (99.3, 51.5))
    arrow(ax, (63.7, 38.5), (63.7, 32.3))
    arrow(ax, (63.7, 19.3), (63.7, 13.2), dashed=True)
    save(fig, "framework_architecture", "Framework architecture")


def historical_evaluation_timeline(fonts):
    fig, ax = canvas(133, 18.5)
    box(ax, fonts, 0.2, 2.1, 26.5, 14.3, "2014", "Signal\nwarm-up")
    box(ax, fonts, 32.2, 4.6, 27.5, 9.3, "2015-2017", "Initial research")
    box(ax, fonts, 65.8, 4.6, 28.5, 9.3, "2018-2023", "Six annual folds")
    box(ax, fonts, 100.2, 0.25, 32.6, 18, "Feb. 2024-\nJan. 2026",
        "Stability and\nselection")
    for left, right in ((26.7, 32.2), (59.7, 65.8), (94.3, 100.2)):
        arrow(ax, (left, 9.25), (right, 9.25), color=GREY)
    save(fig, "historical_evaluation_timeline", "Historical evaluation timeline")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fonts = report_fonts()
    framework_architecture(fonts)
    historical_evaluation_timeline(fonts)


if __name__ == "__main__":
    main()
