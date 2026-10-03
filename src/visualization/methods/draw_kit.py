# -*- coding: utf-8 -*-
"""Strumenti di disegno condivisi delle figure Methods."""
from __future__ import annotations

from pathlib import Path

import matplotlib

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Circle, Ellipse, FancyBboxPatch, PathPatch, Polygon, Rectangle,)
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D


INK = "#111111"
INK_SOFT = "#5A646D"
GRID = "#E5E7EB"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
MAGENTA = "#CC79A7"
YELLOW = "#F2C94C"

BODY_FILL = "#B8BEC6"
BODY_BELLY = "#E9ECEF"
EAR_FILL = "#E2A9A9"
EAR_DARK = "#C98A8A"
PLATE_FILL = "#4D565E"
GLASS_FILL = "#E8F2F8"
GLASS_EDGE = "#56B4E9"
BEDDING_FILL = "#F7E9B5"
BEDDING_EDGE = "#D9C67A"
NEST_FILL = MAGENTA
BOTTLE_FILL = "#D6ECF8"
FOOD_FILL = "#F7E9B5"

ID_COLORS = [BLUE, ORANGE, GREEN, MAGENTA]

SMALL = 8.0
LETTER = 10.0
TITLE = 11.0


def set_paper_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "axes.axisbelow": True,
        }
    )


def save_bundle(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def add_panel(fig, rect, x0, x1, y0=0.0):
    """Panel axes whose data limits match the box, so equal aspect holds."""
    box_w = rect[2] * fig.get_figwidth()
    box_h = rect[3] * fig.get_figheight()
    ax = fig.add_axes(rect)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y0 + (box_h / box_w) * (x1 - x0))
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def note(ax, text, xy, xytext, fontsize=SMALL, ha="left", va="center",
         color=INK, leader=True, z=20, weight="normal", lw=0.7, ls="-"):
    props = None
    if leader:
        props = dict(arrowstyle="-", color="#8A939B", lw=lw, shrinkA=0.0,
                     shrinkB=2.0, linestyle=ls)
    ax.annotate(
        text, xy=xy, xytext=xytext, ha=ha, va=va, fontsize=fontsize,
        color=color, zorder=z, fontweight=weight, arrowprops=props,
        annotation_clip=False, linespacing=1.35,
    )


def panel_letter(ax, letter, x, y):
    ax.text(x, y, letter, fontsize=LETTER, fontweight="bold", ha="left",
            va="top", color=INK, zorder=30)


def capsule(ax, p0, p1, w, transform, fc, ec=INK, lw=0.8, z=5, alpha=1.0):
    """Rounded capsule running from p0 to p1 in local coordinates."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = max(float(np.hypot(dx, dy)), 1e-6)
    ang = float(np.degrees(np.arctan2(dy, dx)))
    tr = Affine2D().rotate_deg(ang).translate(x0, y0) + transform
    ax.add_patch(FancyBboxPatch(
        (-w / 2.0, -w / 2.0), length + w, w,
        boxstyle="round,pad=0,rounding_size=%f" % (w / 2.0), transform=tr,
        facecolor=fc, edgecolor=ec, lw=lw, zorder=z, alpha=alpha))


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


def draw_mouse(ax, x, y, s=1.0, rot=0.0, face=1, body=BODY_FILL, ink=INK,
               lw=0.8, z=5, alpha=1.0, tail=True, eye="open", whiskers=True,
               shade=True):
    tr = (Affine2D().scale(s * face, s).rotate_deg(rot).translate(x, y)
          + ax.transData)

    ax.add_patch(Circle((0.58, 0.30), 0.185, transform=tr, facecolor=EAR_FILL,
                        edgecolor=ink, lw=lw, zorder=z, alpha=alpha))
    ax.add_patch(Circle((0.56, 0.29), 0.100, transform=tr, facecolor="#F0C9C9",
                        edgecolor="none", zorder=z, alpha=alpha))

    if tail:
        ax.add_patch(PathPatch(MplPath(TAIL_VERTS, TAIL_CODES), transform=tr,
                               fill=False, edgecolor=EAR_DARK, lw=1.5 * s,
                               capstyle="round", zorder=z - 1, alpha=alpha))

    for px in (0.34, -0.20):
        ax.add_patch(Ellipse((px, -0.29), 0.17, 0.11, transform=tr,
                             facecolor=BODY_BELLY, edgecolor=ink, lw=lw * 0.8,
                             zorder=z + 1, alpha=alpha))

    body_patch = PathPatch(MplPath(BODY_VERTS, BODY_CODES), transform=tr,
                           facecolor=body, edgecolor=ink, lw=lw, zorder=z,
                           alpha=alpha)
    ax.add_patch(body_patch)

    if shade:
        belly = Ellipse((0.02, -0.17), 0.95, 0.22, transform=tr,
                        facecolor=BODY_BELLY, edgecolor="none", zorder=z + 1,
                        alpha=alpha)
        ax.add_patch(belly)
        belly.set_clip_path(body_patch)

    if eye == "open":
        ax.add_patch(Circle((0.90, 0.08), 0.045, transform=tr, facecolor=ink,
                            edgecolor="none", zorder=z + 2, alpha=alpha))
    elif eye == "closed":
        ax.add_patch(PathPatch(
            MplPath([(0.85, 0.07), (0.92, 0.04), (0.99, 0.07)],
                    [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]),
            transform=tr, fill=False, edgecolor=ink, lw=0.9, zorder=z + 2,
            alpha=alpha))

    ax.add_patch(Circle((1.235, -0.02), 0.05, transform=tr, facecolor=EAR_DARK,
                        edgecolor="none", zorder=z + 2, alpha=alpha))

    if whiskers:
        for tip in ((1.56, 0.12), (1.60, -0.02), (1.52, -0.15)):
            ax.add_patch(PathPatch(
                MplPath([(1.12, -0.03), tip],
                        [MplPath.MOVETO, MplPath.LINETO]),
                transform=tr, fill=False, edgecolor="#8A939B", lw=0.6,
                zorder=z, alpha=alpha))


def draw_hand(ax, x, y, s=1.0, rot=0.0, z=12, alpha=1.0):
    """A gloved hand seen from the side, fingers curling to the left."""
    tr = Affine2D().scale(s, s).rotate_deg(rot).translate(x, y) + ax.transData
    glove = "#BBD6E8"
    glove_d = "#96BCD6"

    ax.add_patch(FancyBboxPatch(
        (0.92, -0.52), 0.95, 1.34, boxstyle="round,pad=0,rounding_size=0.18",
        transform=tr, facecolor=glove_d, edgecolor=INK, lw=0.8, zorder=z,
        alpha=alpha))
    ax.add_patch(FancyBboxPatch(
        (-0.10, -0.50), 1.15, 1.40,
        boxstyle="round,pad=0,rounding_size=0.30", transform=tr,
        facecolor=glove, edgecolor=INK, lw=0.8, zorder=z + 1, alpha=alpha))

    fingers = [
        ((-0.05, 0.58), (-0.92, 0.50), 0.26),
        ((-0.05, 0.30), (-1.02, 0.24), 0.27),
        ((-0.05, 0.03), (-0.94, -0.03), 0.26),
        ((-0.02, -0.24), (-0.78, -0.32), 0.24),
    ]
    for p0, p1, w in fingers:
        capsule(ax, p0, p1, w, tr, glove, z=z + 2, alpha=alpha)
    capsule(ax, (0.18, -0.44), (-0.30, -0.80), 0.30, tr, glove_d, z=z + 2,
            alpha=alpha)


def draw_nest_box(ax, x, y, w, h, z=4):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=NEST_FILL, edgecolor="#A85C86", lw=0.8, alpha=0.35,
        zorder=z))
    ax.add_patch(FancyBboxPatch(
        (x + 0.16, y), w - 0.32, h * 0.52,
        boxstyle="round,pad=0,rounding_size=0.07",
        facecolor="#F3DCEC", edgecolor="#A85C86", lw=0.7, alpha=0.95,
        zorder=z + 1))


def draw_water_food(ax, x, y, s=1.0, z=4):
    tr = Affine2D().scale(s, s).translate(x, y) + ax.transData
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.62), 0.60, 1.22, boxstyle="round,pad=0,rounding_size=0.10",
        transform=tr, facecolor=BOTTLE_FILL, edgecolor=GLASS_EDGE, lw=0.8,
        zorder=z))
    ax.add_patch(Polygon([(0.10, 0.62), (0.50, 0.62), (0.40, 0.34),
                          (0.20, 0.34)], closed=True, transform=tr,
                         facecolor=BOTTLE_FILL, edgecolor=GLASS_EDGE, lw=0.8,
                         zorder=z))
    ax.add_patch(Rectangle((0.25, 0.06), 0.10, 0.30, transform=tr,
                           facecolor="#8A939B", edgecolor="none", zorder=z))
    ax.add_patch(Polygon([(0.76, 0.08), (1.58, 0.08), (1.44, 0.70),
                          (0.90, 0.70)], closed=True, transform=tr,
                         facecolor=FOOD_FILL, edgecolor=BEDDING_EDGE, lw=0.8,
                         zorder=z))
    for i in range(3):
        ax.add_patch(Rectangle((0.84, 0.20 + 0.15 * i), 0.64, 0.045,
                               transform=tr, facecolor="#D9C67A",
                               edgecolor="none", zorder=z + 1))


def draw_camera(ax, x, y, w=1.95, h=0.74, z=8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.10",
        facecolor="#F2F4F6", edgecolor=INK, lw=0.9, zorder=z))
    cx = x + w / 2.0
    ax.add_patch(Rectangle((cx - 0.30, y - 0.17), 0.60, 0.19,
                           facecolor=PLATE_FILL, edgecolor=INK, lw=0.7,
                           zorder=z))
    ax.add_patch(Circle((cx, y - 0.16), 0.135, facecolor="#2B3238",
                        edgecolor=INK, lw=0.6, zorder=z + 1))
    ax.add_patch(Circle((cx, y - 0.16), 0.055, facecolor=GLASS_EDGE,
                        edgecolor="none", zorder=z + 2))
    ax.add_patch(Rectangle((cx - 0.16, y + h), 0.32, 0.80, facecolor=GRID,
                           edgecolor=INK, lw=0.7, zorder=z - 1))


def draw_tube(ax, x, y, s=1.0, rot=0.0, z=8, fill_alpha=0.55):
    tr = Affine2D().scale(s, s).rotate_deg(rot).translate(x, y) + ax.transData
    fc, ec = "#FFFFFF", "#8A939B"
    ax.add_patch(Polygon([(3.30, 0.0), (3.30, 1.05), (3.94, 0.80),
                          (3.94, 0.25)], closed=True, transform=tr,
                         facecolor=fc, edgecolor=ec, lw=0.9, alpha=0.75,
                         zorder=z))
    ax.add_patch(FancyBboxPatch(
        (3.94, 0.22), 0.34, 0.61, boxstyle="round,pad=0,rounding_size=0.07",
        transform=tr, facecolor="#E9ECEF", edgecolor=ec, lw=0.9, zorder=z + 1))
    for i in range(4):
        ax.add_patch(Circle((4.11, 0.33 + 0.13 * i), 0.042, transform=tr,
                            facecolor="#8A939B", edgecolor="none",
                            zorder=z + 2))
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 3.30, 1.05, boxstyle="round,pad=0,rounding_size=0.12",
        transform=tr, facecolor="#FFFFFF", edgecolor=ec, lw=0.9,
        alpha=fill_alpha, zorder=z + 3))


def draw_syringe(ax, x, y, s=1.0, rot=-40.0, z=14):
    """Needle tip sits at local (-1.55, 0) before the transform."""
    tr = Affine2D().scale(s, s).rotate_deg(rot).translate(x, y) + ax.transData
    ax.add_patch(Rectangle((-1.55, -0.03), 1.60, 0.06, transform=tr,
                           facecolor="#B8BEC6", edgecolor=INK, lw=0.5,
                           zorder=z))
    ax.add_patch(FancyBboxPatch(
        (0.0, -0.20), 1.90, 0.40, boxstyle="round,pad=0,rounding_size=0.05",
        transform=tr, facecolor="#FFFFFF", edgecolor=INK, lw=0.8, zorder=z))
    ax.add_patch(Rectangle((0.12, -0.14), 0.98, 0.28, transform=tr,
                           facecolor="#D6ECF8", edgecolor="none", zorder=z + 1))
    ax.add_patch(Rectangle((1.90, -0.07), 0.52, 0.14, transform=tr,
                           facecolor="#E9ECEF", edgecolor=INK, lw=0.7,
                           zorder=z))
    ax.add_patch(Rectangle((2.42, -0.25), 0.15, 0.50, transform=tr,
                           facecolor="#E9ECEF", edgecolor=INK, lw=0.7,
                           zorder=z))
    ax.add_patch(Rectangle((-0.15, -0.12), 0.17, 0.24, transform=tr,
                           facecolor="#DCE3E9", edgecolor=INK, lw=0.6,
                           zorder=z))


def draw_transponder(ax, x, y, s=1.0, rot=0.0, z=16):
    tr = Affine2D().scale(s, s).rotate_deg(rot).translate(x, y) + ax.transData
    ax.add_patch(FancyBboxPatch(
        (-0.30, -0.08), 0.60, 0.16,
        boxstyle="round,pad=0,rounding_size=0.08", transform=tr,
        facecolor=BLUE, edgecolor=INK, lw=0.7, zorder=z))


def draw_timer(ax, x, y, s=1.0, z=16, label="30 min"):
    tr = Affine2D().scale(s, s).translate(x, y) + ax.transData
    ax.add_patch(Circle((0, 0), 0.40, transform=tr, facecolor="white",
                        edgecolor=INK, lw=0.8, zorder=z))
    ax.add_patch(PathPatch(
        MplPath([(0, 0), (0, 0.26)], [MplPath.MOVETO, MplPath.LINETO]),
        transform=tr, fill=False, edgecolor=INK, lw=0.9, zorder=z + 1))
    ax.add_patch(PathPatch(
        MplPath([(0, 0), (0.20, -0.09)], [MplPath.MOVETO, MplPath.LINETO]),
        transform=tr, fill=False, edgecolor=INK, lw=0.9, zorder=z + 1))
    ax.text(0, -0.78, label, transform=tr, ha="center", va="center",
            fontsize=SMALL, color=INK, zorder=z + 2)


def draw_tray(ax, x, y, w, h, mice, z=1, bedding_h=0.60):
    """Open cage / home-cage tray seen from the side."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.14",
        facecolor="#F7FAFC", edgecolor=GLASS_EDGE, lw=0.9, zorder=z))
    ax.add_patch(Rectangle((x + 0.15, y + 0.15), w - 0.30, bedding_h,
                           facecolor=BEDDING_FILL, edgecolor=BEDDING_EDGE,
                           lw=0.7, zorder=z + 1))
    for (mx, my, mrot, mface) in mice:
        draw_mouse(ax, mx, my, s=0.44, rot=mrot, face=mface, z=z + 3)


def panel_rect(fig_w, fig_h, x_frac, w_frac, data_ratio, y_frac):
    """Rect with a box aspect that matches the panel data limits exactly."""
    height_in = w_frac * fig_w * data_ratio
    return [x_frac, y_frac, w_frac, height_in / fig_h]


# --------------------------------------------------------------------------
# Figure 1 - Live Mouse Tracker setup
# --------------------------------------------------------------------------

