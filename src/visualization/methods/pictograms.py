"""Clean, monochrome pictograms of mouse behaviours.

The pictograms are drawn from scratch (uniform stroke, black) with the same
mouse primitive used by the Methods figures, so that the behavioural figures
share one visual language.  Outputs are written as SVG and PNG bundles.

Usage:
    python scripts/build_behavior_pictograms.py --out-dir DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import Circle, Ellipse, PathPatch, Polygon  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

INK = "#000000"
LW = 1.15                      # single, uniform line width for every stroke
FONT = "Arial"

BODY_VERTS = [
    (1.24, -0.02),
    (1.04, 0.15), (0.88, 0.27), (0.70, 0.30),
    (0.52, 0.34), (0.28, 0.36), (0.03, 0.35),
    (-0.30, 0.34), (-0.53, 0.24), (-0.63, 0.04),
    (-0.71, -0.15), (-0.55, -0.28), (-0.28, -0.30),
    (0.05, -0.33), (0.55, -0.30), (0.88, -0.16),
    (1.06, -0.09), (1.18, -0.06), (1.24, -0.02),
]
BODY_CODES = [MplPath.MOVETO] + [MplPath.CURVE4] * 18

TAIL_VERTS = [
    (-0.61, 0.04),
    (-0.88, 0.20), (-1.06, 0.00), (-1.26, 0.18),
    (-1.42, 0.32), (-1.52, 0.44), (-1.58, 0.54),
]
TAIL_CODES = [MplPath.MOVETO] + [MplPath.CURVE4] * 6


def mouse(ax, x=0.0, y=0.0, s=1.0, rot=0.0, face=1, pose="walk", z=5):
    """Draw one mouse in a given pose (walk, still, rear, stretched, ball)."""
    body_scale_y = 1.22
    if pose == "stretched":
        tr = (Affine2D().scale(1.35 * face, 0.78).rotate_deg(rot)
              .translate(x, y) + ax.transData)
        s = s * 1.0
    else:
        tr = (Affine2D().scale(s * face, s * body_scale_y).rotate_deg(rot)
              .translate(x, y) + ax.transData)

    if pose == "rear":
        tr = (Affine2D().scale(s * face, s).rotate_deg(rot - 78)
              .translate(x, y) + ax.transData)
    elif pose == "ball":
        ax.add_patch(Circle((x, y), 0.32 * s, transform=ax.transData,
                            facecolor="none", edgecolor=INK, lw=LW, zorder=z))
        ax.add_patch(MplPathPatchTail(ax, x, y, s, rot, z))
        return

    # ear sits on the back line, tangent to the head
    ear_tr = (Affine2D().scale(s * face, s).rotate_deg(rot).translate(x, y)
              + ax.transData)
    ear_center = (0.60, 0.46) if pose != "stretched" else (0.62, 0.40)
    ear_radius = 0.155 if pose != "stretched" else 0.135
    ax.add_patch(Circle(ear_center, ear_radius, transform=ear_tr,
                        facecolor="none", edgecolor=INK, lw=LW, zorder=z))
    ax.add_patch(Circle((ear_center[0] - 0.02, ear_center[1] - 0.015),
                        ear_radius * 0.55, transform=ear_tr, facecolor="none",
                        edgecolor=INK, lw=LW * 0.7, zorder=z))
    ax.add_patch(PathPatch(MplPath(TAIL_VERTS, TAIL_CODES), transform=tr,
                           fill=False, edgecolor=INK, lw=LW, capstyle="round",
                           zorder=z - 1))
    for px in (0.34, -0.20):
        ax.add_patch(Ellipse((px, -0.31), 0.17, 0.11, transform=tr,
                             facecolor="none", edgecolor=INK, lw=LW * 0.85,
                             zorder=z + 1))
    ax.add_patch(PathPatch(MplPath(BODY_VERTS, BODY_CODES), transform=tr,
                           facecolor="none", edgecolor=INK, lw=LW, zorder=z))
    ax.add_patch(Circle((0.92, 0.09), 0.05, transform=tr, facecolor=INK,
                        edgecolor="none", zorder=z + 2))
    for tip in ((1.56, 0.12), (1.60, -0.02), (1.52, -0.15)):
        ax.add_patch(PathPatch(
            MplPath([(1.12, -0.03), tip], [MplPath.MOVETO, MplPath.LINETO]),
            transform=tr, fill=False, edgecolor=INK, lw=LW * 0.7, zorder=z))
    if pose == "walk":
        return


def MplPathPatchTail(ax, x, y, s, rot, z):
    from matplotlib.patches import PathPatch

    tr = Affine2D().scale(s, s).rotate_deg(rot).translate(x, y) + ax.transData
    return PathPatch(MplPath([(-0.30, 0.10), (-0.55, 0.28), (-0.75, 0.10),
                              (-0.95, 0.30)],
                             [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4,
                              MplPath.CURVE4]),
                     transform=tr, fill=False, edgecolor=INK, lw=LW,
                     capstyle="round", zorder=z)


def arrow(ax, p0, p1, rad=0.0, z=8):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=LW,
                                shrinkA=0, shrinkB=0,
                                connectionstyle="arc3,rad=%.2f" % rad),
                zorder=z)


def dashed_circle(ax, x, y, r, z=2):
    ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor=INK, lw=LW * 0.8,
                        ls=(0, (3, 3)), zorder=z))


def new_axes(w, h, xlim, ylim):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------
# pictograms (each returns a function drawing into an axes)
# --------------------------------------------------------------------------
def pic_move_alone(ax):
    mouse(ax, 0.55, 0.45, s=0.52)
    arrow(ax, (0.05, 0.18), (0.05, 0.80))


def pic_stop_alone(ax):
    mouse(ax, 0.55, 0.45, s=0.52, pose="still")
    ax.plot([0.10, 0.34], [0.82, 0.82], color=INK, lw=LW, zorder=8)


def pic_rear_alone(ax):
    mouse(ax, 0.55, 0.28, s=0.52, pose="rear")


def pic_jump_wall(ax):
    mouse(ax, 0.42, 0.32, s=0.50, pose="rear")
    ax.plot([0.86, 0.86], [0.10, 0.92], color=INK, lw=LW, zorder=3)


def pic_huddled(ax):
    mouse(ax, 0.42, 0.50, s=0.52, pose="ball")
    mouse(ax, 0.30, 0.28, s=0.42)


def pic_contact(ax):
    mouse(ax, 0.52, 0.62, s=0.46, rot=-6)
    mouse(ax, 0.52, 0.26, s=0.46, rot=6, face=-1)


def pic_contact_nose(ax):
    mouse(ax, 0.38, 0.45, s=0.46, rot=4)
    mouse(ax, 0.92, 0.45, s=0.46, rot=4, face=-1)


def pic_rear_in_contact(ax):
    mouse(ax, 0.36, 0.28, s=0.46, pose="rear")
    mouse(ax, 0.80, 0.30, s=0.46)


def pic_group(ax, n=4):
    xs = [0.30, 0.62, 0.94, 1.10][:n]
    for i, x in enumerate(xs):
        mouse(ax, x, 0.36 + 0.10 * (i % 2), s=0.42,
              rot=(-14 if i % 2 else 12))


def pic_nest(ax, n=4):
    layout = [(0.62, 0.44), (0.36, 0.60), (0.74, 0.66), (0.97, 0.44)][:n]
    for i, (x, y) in enumerate(layout):
        mouse(ax, x, y, s=0.40, rot=(-30 + 60 * (i % 2)), pose="still")


def pic_train(ax, n=3):
    for i in range(n):
        mouse(ax, 0.30 + 0.46 * i, 0.42 + 0.02 * i, s=0.42, rot=-12,
              face=1 if i == 0 else -1)


def pic_approach(ax):
    dashed_circle(ax, 0.40, 0.45, 0.34)
    mouse(ax, 0.34, 0.34, s=0.40, rot=-18)
    mouse(ax, 0.84, 0.62, s=0.40, rot=10, face=-1)
    arrow(ax, (0.52, 0.44), (0.70, 0.56))


def pic_move_away(ax):
    dashed_circle(ax, 0.62, 0.45, 0.34)
    mouse(ax, 0.50, 0.44, s=0.40, rot=-16)
    mouse(ax, 0.98, 0.72, s=0.40, rot=14, face=-1)
    arrow(ax, (0.36, 0.36), (0.16, 0.20))


def pic_break_contact(ax):
    mouse(ax, 0.32, 0.46, s=0.40, rot=-10, face=-1)
    mouse(ax, 0.86, 0.46, s=0.40, rot=10)
    arrow(ax, (0.30, 0.78), (0.12, 0.92))
    arrow(ax, (0.90, 0.78), (1.08, 0.92))


def pic_follow(ax):
    mouse(ax, 0.34, 0.34, s=0.40, rot=-14)
    mouse(ax, 0.74, 0.62, s=0.40, rot=-14)
    ax.annotate("", xy=(0.62, 0.60), xytext=(0.44, 0.42),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=LW,
                                shrinkA=1, shrinkB=1,
                                connectionstyle="arc3,rad=0.25"), zorder=8)


def pic_make_group(ax):
    mouse(ax, 0.36, 0.36, s=0.40, rot=10)
    mouse(ax, 0.78, 0.62, s=0.40, rot=-16, face=-1)
    arrow(ax, (0.62, 0.34), (0.86, 0.50))


def pic_break_group(ax):
    mouse(ax, 0.36, 0.52, s=0.40, rot=-14, face=-1)
    mouse(ax, 0.80, 0.40, s=0.40, rot=12)
    arrow(ax, (0.44, 0.30), (0.22, 0.16))


PICTOGRAMS = {
    "move_alone": pic_move_alone,
    "stop_alone": pic_stop_alone,
    "rear_alone": pic_rear_alone,
    "jump_wall": pic_jump_wall,
    "huddled": pic_huddled,
    "contact_side_by_side": pic_contact,
    "contact_nose_nose": pic_contact_nose,
    "rear_in_contact": pic_rear_in_contact,
    "group2": lambda ax: pic_group(ax, 2),
    "group3": lambda ax: pic_group(ax, 3),
    "group4": lambda ax: pic_group(ax, 4),
    "nest3": lambda ax: pic_nest(ax, 3),
    "nest4": lambda ax: pic_nest(ax, 4),
    "train2": lambda ax: pic_train(ax, 2),
    "train3": lambda ax: pic_train(ax, 3),
    "train4": lambda ax: pic_train(ax, 4),
    "approach": pic_approach,
    "move_away": pic_move_away,
    "break_contact": pic_break_contact,
    "follow": pic_follow,
    "make_group": pic_make_group,
    "break_group": pic_break_group,
}

LAYOUT = [
    ("Individual events", ["move_alone", "stop_alone", "rear_alone",
                           "huddled", "jump_wall"]),
    ("Dyadic state events", ["contact_side_by_side", "contact_nose_nose",
                             "rear_in_contact"]),
    ("Dyadic dynamic events", ["approach", "follow", "move_away",
                               "break_contact"]),
    ("Social configuration events", ["group2", "group3", "group4", "nest3",
                                     "nest4", "train2", "train3", "train4"]),
    ("Group making/breaking events", ["make_group", "break_group"]),
]


def render_pictogram(name, fn, out_dir, size=400):
    fig, ax = new_axes(1.0, 1.0, (0, 1.2), (0, 1.0))
    fn(ax)
    for suffix in ("png", "svg"):
        fig.savefig(out_dir / f"{name}.{suffix}", dpi=size,
                    transparent=False, facecolor="white")
    plt.close(fig)


def render_repertoire(out_dir, width=7.2):
    rows = len(LAYOUT)
    heights = [1.0, 1.05, 1.05, 1.15, 1.05]
    fig_w = width
    fig_h = sum(heights) * 1.05 + 0.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    y = fig_h - 0.35
    for (title, names), h in zip(LAYOUT, heights):
        ax = fig.add_axes([0.0, (y - h) / fig_h, 1.0, h / fig_h])
        ax.set_xlim(0, len(names) * 1.2)
        ax.set_ylim(0, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.02, 1.28, title, fontsize=9, fontweight="bold",
                family=FONT, color=INK, va="top")
        for i, n in enumerate(names):
            inner = ax.inset_axes([i * 1.2 / (len(names) * 1.2), 0.30 / 1.35,
                                   1.0 / (len(names) * 1.2), 0.90 / 1.35])
            inner.set_xlim(0, 1.2)
            inner.set_ylim(0, 1.0)
            inner.set_aspect("equal")
            inner.axis("off")
            PICTOGRAMS[n](inner)
            ax.text((i + 0.5) * 1.2, 0.16, n.replace("_", " "), fontsize=6.5,
                    family=FONT, color=INK, ha="center", va="top")
        y -= h * 1.05

    # frames around the three blocks of the original figure
    fig.text(0.5, 0.985, "Behavioral repertoire classified by the "
             "Live Mouse Tracker", fontsize=11, family=FONT, color=INK,
             ha="center", va="top")
    fig.text(0.005, 0.006,
             "Drawings follow the behavioral categories classified by the "
             "Live Mouse Tracker (de Chaumont et al., 2019).",
             fontsize=8, family=FONT, color="#5A646D", ha="left", va="bottom")
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out_dir / f"figure_methods_02_repertoire.{suffix}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "methods"


def build(out_dir: Path | None = None) -> Path:
    """Render pictograms + repertoire (standalone or via build_all)."""
    out = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    (out / "pictograms").mkdir(parents=True, exist_ok=True)
    for name, fn in PICTOGRAMS.items():
        render_pictogram(name, fn, out / "pictograms")
    render_repertoire(out)
    print("pictograms written to", out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    build(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
