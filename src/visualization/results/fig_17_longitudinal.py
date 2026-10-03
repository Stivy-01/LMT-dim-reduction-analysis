"""Thesis Figure 16: night-by-night trajectory of post-manipulation changes.

Layout follows the thesis style (Arial/Helvetica, grid #E5E7EB, no top/right
spines, 300 dpi PNG + SVG + PDF), with one row per metric family and one
column per treatment group.
"""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.paper_style import save_bundle, set_paper_style
try:
    from scripts.build_window_comparison_report import (
        FAMILIES,
        load_inputs,
        per_mouse_pct,
    )
except ImportError:  # direct script execution
    from build_window_comparison_report import (  # noqa: E402
        FAMILIES,
        load_inputs,
        per_mouse_pct,
    )

NIGHTS = [-2, -1, 1, 2, 3]
GROUP_COLOURS = {"control": "#0072B2", "stressed": "#D55E00"}
FEATURE_LINE = "#B8BCC4"


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "paper"
RUN_DIR = DEFAULT_INPUT_DIR


def parse_args() -> argparse.Namespace:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory. Defaults to src/visualization/output/paper.")
    return parser.parse_args()


def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build Figure 17: night-by-night longitudinal trajectory (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else RUN_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mouse, families = load_inputs(run_dir)
    metrics = ["count", "mean_duration", "std_duration"]
    fig, panels = plt.subplots(len(metrics), 2, figsize=(7.4, 8.0),
                               sharex=True)
    for row, metric in enumerate(metrics):
        features = families[metric]
        for col, treatment in enumerate(("control", "stressed")):
            ax = panels[row, col]
            curves = {}
            counts = {}
            for night in NIGHTS:
                values = per_mouse_pct(mouse, features, treatment, night=night)
                curves[night] = values.mean(axis=0)
                counts[night] = int(values.notna().all(axis=1).sum())
            table = pd.DataFrame(curves)[NIGHTS]
            for _, series in table.iterrows():
                ax.plot(NIGHTS, series.to_numpy(), color=FEATURE_LINE,
                        linewidth=0.45, alpha=0.55, zorder=1)
            median = table.median(axis=0).to_numpy()
            q1 = table.quantile(0.25, axis=0).to_numpy()
            q3 = table.quantile(0.75, axis=0).to_numpy()
            colour = GROUP_COLOURS[treatment]
            ax.fill_between(NIGHTS, q1, q3, color=colour, alpha=0.22,
                            linewidth=0, zorder=2)
            ax.plot(NIGHTS, median, color=colour, linewidth=1.9, zorder=3,
                    marker="o", markersize=3.2)
            ax.axhline(0, color="#111111", linewidth=0.8, zorder=1)
            ax.axvline(-0.5, color="#9aa0a6", linewidth=0.9,
                       linestyle=(0, (4, 3)), zorder=1)
            ax.text(0.012, 0.96, "Control" if treatment == "control"
                    else "Stressed", transform=ax.transAxes, ha="left",
                    va="top", fontsize=10)
            if col == 0:
                ax.set_ylabel("change from night −1 (%)", fontsize=9)
            if row == len(metrics) - 1:
                ax.set_xticks(NIGHTS)
                ax.set_xticklabels(["%+d\nn=%d" % (n, counts[n])
                                    for n in NIGHTS], fontsize=8)
        # shared, robust limits per metric across both groups
        values = []
        for col, treatment in enumerate(("control", "stressed")):
            for night in NIGHTS:
                values.append(per_mouse_pct(mouse, families[metric], treatment,
                                            night=night).mean(axis=0))
        stacked = pd.concat(values, axis=1).to_numpy()
        limit = float(np.nanpercentile(np.abs(stacked), 97.5)) * 1.15
        for col in range(2):
            panels[row, col].set_ylim(-limit, limit)
    row_titles = {
        "count": "Event counts",
        "mean_duration": "Mean event duration",
        "std_duration": "Duration variability",
    }
    for row, metric in enumerate(metrics):
        panels[row, 0].set_title(row_titles[metric], loc="left", fontsize=10,
                                 pad=6)
    panels[-1, 0].set_xlabel("night relative to the manipulation",
                             fontsize=9, labelpad=8)
    panels[-1, 1].set_xlabel("night relative to the manipulation",
                             fontsize=9, labelpad=8)
    fig.tight_layout(rect=(0, 0.0, 1, 1.0), h_pad=2.2)
    save_bundle(fig, out_dir, "thesis_figure_16_longitudinal_trajectory")
    print("saved", out_dir / "thesis_figure_16_longitudinal_trajectory.png")
    return out_dir


def main() -> int:
    args = parse_args()
    build(args.run_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
