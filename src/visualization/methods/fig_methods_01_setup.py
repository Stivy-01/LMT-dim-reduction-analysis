# -*- coding: utf-8 -*-
"""Figure 1: setup di registrazione LMT."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization.methods.draw_kit import *  # noqa: F401,F403
from src.visualization.methods.draw_kit import save_bundle, set_paper_style

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "methods"


def build_figure_1(out_dir: Path) -> None:
    fig_w, fig_h = 7.2, 4.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ratio = 1.06

    axA = add_panel(fig, panel_rect(fig_w, fig_h, 0.012, 0.474, ratio, 0.055),
                    0, 10)
    axB = add_panel(fig, panel_rect(fig_w, fig_h, 0.508, 0.474, ratio, 0.055),
                    0, 10)
    panel_letter(axA, "A", 0.15, 10.45)
    panel_letter(axB, "B", 0.15, 10.45)

    # ---- panel A: side view of the recording setup
    axA.add_patch(Rectangle((0.35, 0.78), 9.30, 0.40, facecolor="#F2F4F6",
                            edgecolor="#B8BEC6", lw=0.8, zorder=1))
    axA.add_patch(Rectangle((0.55, 0.20), 0.30, 0.58, facecolor=GRID,
                            edgecolor="none", zorder=1))
    axA.add_patch(Rectangle((9.30, 0.20), 0.30, 0.58, facecolor=GRID,
                            edgecolor="none", zorder=1))

    axA.add_patch(FancyBboxPatch(
        (1.05, 1.18), 4.30, 0.34, boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=PLATE_FILL, edgecolor="#2B3238", lw=0.8, zorder=2))
    for i in range(6):
        axA.add_patch(PathPatch(
            MplPath([(1.45 + 0.40 * i, 1.26), (1.64 + 0.40 * i, 1.43),
                     (1.45 + 0.40 * i, 1.26)],
                    [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]),
            fill=False, edgecolor="#93A2AE", lw=0.7, zorder=3))
    axA.text(3.20, 1.35, "RFID antennas", ha="center", va="center",
             fontsize=7, color="#DDE4E9", zorder=5)

    cage_x, cage_y, cage_w, cage_h = 0.95, 1.52, 4.40, 3.73
    axA.add_patch(FancyBboxPatch(
        (cage_x, cage_y), cage_w, cage_h,
        boxstyle="round,pad=0,rounding_size=0.10", facecolor=GLASS_FILL,
        edgecolor=GLASS_EDGE, lw=0.9, alpha=0.55, zorder=2))

    bed_verts, bed_codes = [], []
    xs = np.linspace(cage_x, cage_x + cage_w, 70)
    bed_verts.append((xs[0], cage_y + 0.02))
    bed_codes.append(MplPath.MOVETO)
    for i in range(len(xs) - 1):
        bed_verts.append((xs[i + 1],
                          cage_y + 0.26 + 0.075 * np.sin(xs[i] * 2.6)))
        bed_codes.append(MplPath.LINETO)
    bed_verts += [(cage_x + cage_w, cage_y + 0.02), (cage_x, cage_y + 0.02)]
    bed_codes += [MplPath.LINETO, MplPath.CLOSEPOLY]
    axA.add_patch(PathPatch(MplPath(bed_verts, bed_codes),
                            facecolor=BEDDING_FILL, edgecolor=BEDDING_EDGE,
                            lw=0.7, zorder=3))

    draw_nest_box(axA, 1.10, 1.78, 1.25, 0.80, z=4)
    draw_water_food(axA, 3.62, 3.42, s=0.92, z=4)

    draw_mouse(axA, 2.55, 2.25, s=0.50, rot=-6, z=6)
    draw_mouse(axA, 3.15, 2.18, s=0.50, rot=4, face=-1, z=7)
    draw_mouse(axA, 4.05, 2.25, s=0.48, rot=-12, face=-1, z=6)
    draw_mouse(axA, 4.50, 2.62, s=0.48, rot=10, z=7)

    axA.add_patch(Rectangle((0.45, 9.30), 8.30, 0.16, facecolor=GRID,
                            edgecolor=INK, lw=0.7, zorder=6))
    draw_camera(axA, 2.10, 7.60, w=1.95, h=0.74, z=8)

    axA.add_patch(FancyBboxPatch(
        (7.50, 1.45), 2.25, 1.60, boxstyle="round,pad=0,rounding_size=0.08",
        facecolor="#F2F4F6", edgecolor=INK, lw=0.9, zorder=6))
    axA.add_patch(Rectangle((7.63, 1.58), 1.99, 1.34, facecolor="#2B3238",
                            edgecolor=INK, lw=0.6, zorder=7))
    axA.add_patch(Rectangle((7.74, 1.69), 1.77, 1.12, facecolor="white",
                            edgecolor="none", zorder=8))
    axA.add_patch(Rectangle((7.86, 1.81), 0.80, 0.88, facecolor="#E8F2F8",
                            edgecolor=GLASS_EDGE, lw=0.6, zorder=9))
    for i, (mx, my) in enumerate([(0.12, 0.22), (0.38, 0.24), (0.22, 0.60),
                                  (0.58, 0.52)]):
        axA.add_patch(Circle((7.86 + mx, 1.81 + my), 0.045,
                             facecolor=ID_COLORS[i], edgecolor="none",
                             zorder=10))
    axA.add_patch(Rectangle((8.76, 1.81), 0.66, 0.88, facecolor="white",
                            edgecolor=GRID, lw=0.6, zorder=9))
    tt = np.linspace(0, 1, 40)
    axA.plot(8.76 + 0.09 + 0.48 * tt,
             1.81 + 0.44 + 0.30 * np.sin(tt * 2.6), color=BLUE, lw=0.9,
             zorder=10)
    axA.plot(8.76 + 0.11 + 0.46 * tt,
             1.81 + 0.44 - 0.28 * np.sin(tt * 2.2 + 1.1), color=ORANGE, lw=0.9,
             zorder=10)
    axA.add_patch(Rectangle((8.45, 1.20), 0.35, 0.25, facecolor=GRID,
                            edgecolor=INK, lw=0.6, zorder=6))

    # cage dimensions
    axA.plot([0.95, 0.95], [0.35, 0.60], color="#8A939B", lw=0.7, zorder=3)
    axA.plot([5.35, 5.35], [0.35, 0.60], color="#8A939B", lw=0.7, zorder=3)
    axA.plot([0.95, 2.70], [0.35, 0.35], color="#8A939B", lw=0.7, zorder=3)
    axA.plot([3.60, 5.35], [0.35, 0.35], color="#8A939B", lw=0.7, zorder=3)
    axA.text(3.15, 0.35, "50 cm", ha="center", va="center", fontsize=7,
             color=INK_SOFT, zorder=4)
    axA.plot([5.62, 5.62], [1.52, 2.86], color="#8A939B", lw=0.7, zorder=3)
    axA.plot([5.62, 5.62], [4.44, 5.25], color="#8A939B", lw=0.7, zorder=3)
    axA.plot([5.62, 5.72], [1.52, 1.52], color="#8A939B", lw=0.7, zorder=3)
    axA.plot([5.62, 5.72], [5.25, 5.25], color="#8A939B", lw=0.7, zorder=3)
    axA.text(5.86, 3.65, "40 cm", ha="center", va="center", fontsize=7,
             color=INK_SOFT, rotation=90, zorder=4)

    note(axA, "Depth camera", (3.80, 8.04), (4.20, 8.85))
    note(axA, "LMT acquisition", (8.62, 3.05), (8.45, 3.40), ha="center",
         leader=False)
    note(axA, "Food / water", (4.30, 5.05), (3.05, 5.62))
    note(axA, "Nest box", (1.60, 2.62), (1.05, 3.15))
    note(axA, "4 cage mates", (2.60, 2.40), (2.00, 4.55))

    # ---- panel B: top view with identities and trajectories
    bx, by, bw = 1.70, 2.10, 5.30
    axB.add_patch(FancyBboxPatch(
        (bx, by), bw, bw, boxstyle="round,pad=0,rounding_size=0.12",
        facecolor="#F7FAFC", edgecolor=GLASS_EDGE, lw=0.9, zorder=1))

    def traj(pts, color, ls="-"):
        axB.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.1,
                 ls=ls, alpha=0.85, solid_capstyle="round", zorder=2,
                 dash_capstyle="round")

    traj([(3.10, 2.60), (3.25, 3.30), (3.05, 3.90), (3.62, 4.28)], BLUE)
    traj([(2.30, 3.10), (2.95, 3.20), (3.20, 4.00), (4.05, 5.05)], ORANGE)
    traj([(6.30, 4.00), (6.85, 4.60), (6.35, 5.60), (6.05, 6.40)], GREEN,
         ls=(0, (3, 2)))

    draw_nest_box(axB, 1.90, 2.30, 1.35, 1.05, z=4)
    axB.add_patch(Rectangle((5.90, 6.25), 0.95, 0.95, facecolor=FOOD_FILL,
                            edgecolor=BEDDING_EDGE, lw=0.7, zorder=3))
    for i in range(4):
        axB.add_patch(Rectangle((6.02, 6.36 + 0.19 * i), 0.71, 0.055,
                                facecolor="#D9C67A", edgecolor="none",
                                zorder=4))

    mouse_pos = [(4.35, 5.25), (4.80, 5.62), (3.75, 4.35), (6.30, 4.00)]
    draw_mouse(axB, 4.35, 5.25, s=0.44, rot=12, z=8)
    draw_mouse(axB, 4.80, 5.62, s=0.44, rot=-28, z=8)
    draw_mouse(axB, 3.75, 4.35, s=0.44, rot=34, z=8)
    draw_mouse(axB, 6.30, 4.00, s=0.44, rot=-14, z=8)

    badge_off = [(0.02, 0.50), (0.36, 0.44), (-0.16, 0.48), (0.20, 0.50)]
    for i, (px, py) in enumerate(mouse_pos):
        ox, oy = badge_off[i]
        axB.add_patch(Circle((px + ox, py + oy), 0.23, facecolor=ID_COLORS[i],
                             edgecolor="white", lw=0.8, zorder=14))
        axB.text(px + ox, py + oy, str(i + 1), ha="center", va="center",
                 fontsize=6.5, color="white", fontweight="bold", zorder=15)

    note(axB, "Huddle", (4.55, 5.75), (3.20, 8.20))
    note(axB, "Social\ncontact", (3.68, 4.50), (0.30, 5.40))
    note(axB, "Isolated\nmovement", (6.55, 4.05), (7.95, 4.35))
    note(axB, "Nest box", (2.35, 2.35), (1.05, 1.15))
    note(axB, "Food and\nwater", (6.90, 6.40), (7.25, 6.70))

    for i in range(4):
        axB.add_patch(Circle((4.10 + 0.55 * i, 1.05), 0.20,
                             facecolor=ID_COLORS[i], edgecolor="white", lw=0.8,
                             zorder=14))
        axB.text(4.10 + 0.55 * i, 1.05, str(i + 1), ha="center", va="center",
                 fontsize=6.0, color="white", fontweight="bold", zorder=15)
    axB.text(4.92, 0.40, "RFID-identified cage mates", ha="center",
             va="center", fontsize=SMALL, color=INK_SOFT, zorder=14)

    fig.text(0.5, 0.988,
             "Live Mouse Tracker recording setup for group-housed mice",
             ha="center", va="top", fontsize=TITLE, color=INK)
    fig.text(0.012, 0.012,
             "Numbers 1-4 label the four RFID-implanted cage mates; all "
             "features come from the same continuous recording.",
             ha="left", va="bottom", fontsize=SMALL, color=INK)
    save_bundle(fig, out_dir, "figure_methods_01_lmt_setup")


# --------------------------------------------------------------------------
# Figure 2 - mouse manipulation and experimental design
# --------------------------------------------------------------------------


def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build figure_methods_01_lmt_setup (standalone or via build_all)."""
    set_paper_style()
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    build_figure_1(out_dir)
    print(f"wrote figure_methods_01_lmt_setup to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build methods figure.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(None, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

