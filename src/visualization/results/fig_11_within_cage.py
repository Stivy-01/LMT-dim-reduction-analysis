"""Thesis figure: stressed and control animals within the same cage.

Two metrics (control response from baseline, stressed-control separation
change) shown for all cages (top row) and excluding the influential WT cage
wt_10132 (bottom row), in the same style as
``paper_figure_03_group_trajectories`` (Arial/Helvetica, grid #E5E7EB, no
top/right spines, stress window shaded, thin per-cage lines plus source means
with SEM ribbons). Output: 300 dpi PNG + SVG + PDF.
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

from src.visualization.paper_style import (
    SOURCE_COLORS,
    SOURCE_LABELS,
    SOURCE_ORDER,
    save_bundle,
    set_paper_style,
)

RUN_DIR = Path(__file__).resolve().parents[3] / "src" / "visualization" / "data"
STEM = "thesis_figure_stress_control_within_cage"
NIGHT_TICKS = [-3, -2, -1, 1, 2, 3]
PANELS = [
    ("control_response", "Control response from baseline"),
    ("stressed_control_separation_delta", "Stressed-control separation change"),
]
ROWS = [
    ("All cages (n = 13)", None),
    ("Excluding wt_10132 (n = 12)", "wt_10132"),
]


def _annotate_outlier(ax, frame: pd.DataFrame, metric: str, x_col: str) -> None:
    """Label the dominant cage when it dwarfs every other cage, as Figure 10."""
    ranked = frame.dropna(subset=[metric]).copy()
    if ranked.empty:
        return
    ranked["abs_metric"] = ranked[metric].abs()
    cage_rank = (
        ranked.sort_values("abs_metric", ascending=False)
        .drop_duplicates("cage_id")
        .sort_values("abs_metric", ascending=False)
    )
    if len(cage_rank) < 2:
        return
    if cage_rank.iloc[0]["abs_metric"] <= 3 * cage_rank.iloc[1]["abs_metric"]:
        return
    outlier_cage = cage_rank.iloc[0]["cage_id"]
    outlier = (
        ranked.loc[ranked["cage_id"].eq(outlier_cage)]
        .sort_values("abs_metric", ascending=False)
        .iloc[0]
    )
    ax.annotate(
        outlier_cage,
        (outlier[x_col], outlier[metric]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=7,
        color=SOURCE_COLORS.get(outlier["source_sheet"], "#111111"),
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
        },
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "paper"


def parse_args() -> argparse.Namespace:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory. Defaults to src/visualization/output/paper.")
    return parser.parse_args()


def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build Figure 11: stressed/control within the same cage (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else RUN_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    group = pd.read_csv(run_dir / "group_lineage_night_metrics.csv")
    x_col = "stress_aligned_night"

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), sharex=True)
    for row, (scope_label, excluded) in enumerate(ROWS):
        frame = group.loc[~group["cage_id"].eq(excluded)] if excluded else group
        for col, (metric, title) in enumerate(PANELS):
            ax = axes[row, col]
            ax.axvspan(0.5, 1.5, color="#F2C94C", alpha=0.18, linewidth=0)
            ax.axhline(0, color="#111111", linewidth=0.7)
            for source in SOURCE_ORDER:
                sub = frame.loc[frame["source_sheet"].eq(source)].copy()
                color = SOURCE_COLORS[source]
                for _, cage in sub.groupby("cage_id", sort=False):
                    ordered = cage.sort_values(x_col)
                    ax.plot(
                        ordered[x_col],
                        ordered[metric],
                        color=color,
                        alpha=0.18,
                        linewidth=0.8,
                    )
                mean = sub.groupby(x_col)[metric].mean()
                sem = sub.groupby(x_col)[metric].sem()
                x = mean.index.to_numpy(dtype=float)
                y = mean.to_numpy(dtype=float)
                err = sem.reindex(mean.index).fillna(0).to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=2.0,
                    marker="o",
                    markersize=4,
                    label=SOURCE_LABELS[source],
                )
                ax.fill_between(
                    x,
                    y - err,
                    y + err,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
            _annotate_outlier(ax, frame, metric, x_col)
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel("LDA-space units")
            ax.set_xticks(NIGHT_TICKS)
            if row == len(ROWS) - 1:
                ax.set_xlabel("Night relative to stress")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("Stressed and control animals within the same cage", y=1.0,
                 fontsize=11, fontweight="bold")
    fig.subplots_adjust(wspace=0.28, hspace=0.30, top=0.85, bottom=0.09,
                        left=0.10)
    for row, (scope_label, _) in enumerate(ROWS):
        position = axes[row, 0].get_position()
        fig.text(0.012, position.y1 + 0.055, scope_label, ha="left",
                 va="bottom", fontsize=9, fontweight="bold")
    save_bundle(fig, out_dir, STEM)
    print("saved", out_dir / f"{STEM}.png")
    return out_dir


def main() -> int:
    args = parse_args()
    build(args.run_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
