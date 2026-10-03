# -*- coding: utf-8 -*-
"""Figure 3: manipolazione e disegno sperimentale."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization.methods.draw_kit import *  # noqa: F401,F403
from src.visualization.methods.draw_kit import save_bundle, set_paper_style

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "methods"


def build_figure_2(out_dir: Path) -> None:
    fig_w, fig_h = 7.2, 4.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ratio = 1.079

    axT = fig.add_axes([0.028, 0.600, 0.944, 0.290])
    axT.set_xlim(0, 100)
    axT.set_ylim(0, 100)
    axT.set_axis_off()

    axB = add_panel(fig, panel_rect(fig_w, fig_h, 0.018, 0.315, ratio, 0.050),
                    0, 10)
    axC = add_panel(fig, panel_rect(fig_w, fig_h, 0.348, 0.315, ratio, 0.050),
                    0, 10)
    axD = add_panel(fig, panel_rect(fig_w, fig_h, 0.678, 0.315, ratio, 0.050),
                    0, 10)
    for ax, letter in ((axB, "B"), (axC, "C"), (axD, "D")):
        panel_letter(ax, letter, 0.15, 10.65)

    # ---- panel A: timeline
    y, h = 52.0, 20.0
    segs = [
        (0.8, 17.0, "#F2F4F6", "#B8BEC6", "RFID implantation +\n"
         "recovery (1-2 weeks)", INK),
        (19.0, 13.5, "#F2F4F6", "#B8BEC6", "Habituation\n24 h", INK),
        (32.5, 25.5, GLASS_FILL, GLASS_EDGE, "Baseline recording\n2 nights",
         INK),
        (58.0, 11.0, ORANGE, ORANGE, "Restraint\n30 min", "white"),
        (69.0, 30.0, GLASS_FILL, GLASS_EDGE, "Post-stress recording\n"
         "3 nights", INK),
    ]
    for x0, w, fc, ec, label, tc in segs:
        axT.add_patch(FancyBboxPatch(
            (x0, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, lw=0.9, zorder=3))
        axT.text(x0 + w / 2.0, y + h / 2.0, label, ha="center", va="center",
                 fontsize=SMALL, color=tc, zorder=5, linespacing=1.30)

    for x, lbl, ha in [(19.0, "Day 1, 19:00\nstart of baseline", "left"),
                       (58.0, "Day 3, 18:30\nacute stress", "center"),
                       (99.0, "Day 6, 19:00\nend of recording", "right")]:
        axT.plot([x, x], [y - 4.0, y], color=INK, lw=0.8, zorder=4)
        axT.text(x, y - 7.5, lbl, ha=ha, va="top", fontsize=7.5,
                 color=INK_SOFT, linespacing=1.35)
    axT.text(0.8, y - 7.5, "Day -14 to -1", ha="left", va="top", fontsize=7.5,
             color=INK_SOFT)
    axT.text(50.0, y + h + 14.0,
             "Continuous 24 h recording; 12 h dark-phase windows used for "
             "analysis", ha="center", va="center", fontsize=SMALL,
             color=INK_SOFT)
    axT.text(0.5, 99.0, "A", ha="left", va="top", fontsize=LETTER,
             fontweight="bold", color=INK)

    # ---- panel B: RFID transponder implantation
    axB.add_patch(FancyBboxPatch(
        (0.40, 8.20), 3.30, 2.40, boxstyle="round,pad=0,rounding_size=0.14",
        facecolor="#F2F4F6", edgecolor="#B8BEC6", lw=0.9, zorder=3))
    axB.add_patch(Rectangle((0.60, 8.40), 0.90, 0.45, facecolor=GRID,
                            edgecolor=INK, lw=0.6, zorder=4))
    axB.text(2.30, 9.70, "Isoflurane\nanaesthesia", fontsize=SMALL,
             color=INK_SOFT, ha="center", va="center", zorder=6,
             linespacing=1.35)
    axB.add_patch(PathPatch(
        MplPath([(2.20, 8.20), (2.38, 7.35), (2.60, 6.35), (2.85, 5.45)],
                [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.CURVE4]),
        fill=False, edgecolor=INK_SOFT, lw=1.1, zorder=5))
    axB.add_patch(Polygon([(4.00, 5.45), (2.85, 5.97), (2.85, 4.93)],
                          closed=True, facecolor="#E9ECEF", edgecolor=INK,
                          lw=0.8, alpha=0.95, zorder=12))
    axB.add_patch(FancyBboxPatch(
        (2.75, 4.80), 3.95, 0.36, boxstyle="round,pad=0,rounding_size=0.10",
        facecolor="#F2F4F6", edgecolor=GRID, lw=0.7, zorder=4))
    draw_mouse(axB, 4.90, 5.45, s=0.72, face=-1, z=8, eye="closed")
    draw_transponder(axB, 5.00, 5.55, s=0.90, rot=8, z=13)
    axB.add_patch(Circle((5.00, 5.55), 0.62, facecolor="none", edgecolor=BLUE,
                         lw=0.8, ls=(0, (2.5, 2)), zorder=14))
    draw_syringe(axB, 5.78, 6.36, s=0.62, rot=45, z=16)

    note(axB, "Subcutaneous RFID transponder\ninserted dorsolaterally",
         (4.60, 5.35), (0.35, 3.45))
    axB.text(5.00, 1.60,
             "Implanted 1-2 weeks before\nbehavioral recording",
             ha="center", va="center", fontsize=SMALL, color=INK_SOFT,
             zorder=10, linespacing=1.35)

    # ---- panel C: handling and transfer
    draw_tray(axC, 0.70, 0.90, 7.60, 3.80,
              [(2.30, 2.55, 8, 1), (3.05, 2.40, -10, -1), (6.20, 2.55, -6, -1)])
    draw_mouse(axC, 4.90, 7.20, s=0.62, rot=6, z=8)
    draw_hand(axC, 5.90, 7.40, s=0.62, rot=-6, z=12)
    axC.add_patch(PathPatch(
        MplPath([(5.05, 6.15), (5.05, 5.30), (4.85, 4.60), (4.70, 4.00)],
                [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.CURVE4]),
        fill=False, edgecolor=BLUE, lw=1.0, zorder=6, linestyle=(0, (4, 2))))
    axC.add_patch(Polygon([(4.66, 4.34), (4.98, 3.92), (4.42, 3.90)],
                          closed=True, facecolor=BLUE, edgecolor="none",
                          zorder=6))
    note(axC, "Manual handling", (4.60, 7.55), (0.35, 9.10))
    note(axC, "Transfer to the\nLMT cage", (4.82, 4.90), (5.25, 4.90))
    note(axC, "LMT cage", (7.60, 0.95), (8.20, 0.35), ha="right")

    # ---- panel D: acute restraint stress
    draw_tube(axD, 0.55, 6.60, s=0.95, z=6, fill_alpha=0.70)
    draw_mouse(axD, 1.60, 7.08, s=0.56, z=8)
    draw_mouse(axD, 2.70, 7.06, s=0.56, face=-1, z=8)
    draw_tube(axD, 0.55, 6.60, s=0.95, z=12, fill_alpha=0.10)
    axD.add_patch(FancyBboxPatch(
        (5.05, 6.35), 1.95, 1.05, boxstyle="round,pad=0,rounding_size=0.12",
        facecolor="white", edgecolor=ORANGE, lw=0.9, zorder=4))
    axD.text(6.02, 6.88, "2 of 4 mice", ha="center", va="center",
             fontsize=SMALL, color=ORANGE, fontweight="bold", zorder=6)
    draw_timer(axD, 8.20, 7.05, s=0.80, z=8, label="30 min")
    note(axD, "Ventilated 50 ml\nconical tube", (1.60, 7.65), (0.90, 9.20))

    draw_tray(axD, 0.70, 0.90, 7.60, 3.80,
              [(3.10, 2.55, 8, 1), (4.10, 2.40, -10, -1)])
    draw_water_food(axD, 6.55, 1.95, s=0.60, z=4)
    axD.text(4.50, 5.40, "2 cage mates: handled and\nkept in the home cage",
             ha="center", va="center", fontsize=SMALL, color=INK_SOFT,
             fontweight="bold", zorder=6, linespacing=1.35)
    axD.text(4.50, 0.55, "At 19:00 all four mice were\nreturned to the LMT cage",
             ha="center", va="center", fontsize=SMALL, color=INK_SOFT,
             zorder=6, linespacing=1.35)

    fig.text(0.5, 0.988, "Mouse manipulation and experimental design",
             ha="center", va="top", fontsize=TITLE, color=INK)
    fig.text(0.012, 0.012,
             "Behavior was summarized in 12 h dark-phase windows (19:00-07:00), "
             "which were used for all analyses.",
             ha="left", va="bottom", fontsize=SMALL, color=INK)
    save_bundle(fig, out_dir, "figure_methods_03_manipulation")




def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build figure_methods_03_manipulation (standalone or via build_all)."""
    set_paper_style()
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    build_figure_2(out_dir)
    print(f"wrote figure_methods_03_manipulation to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build methods figure.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(None, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

