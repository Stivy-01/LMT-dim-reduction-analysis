# -*- coding: utf-8 -*-
"""Figure 3: group trajectories."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.paper_style import (
    PHASE_LABELS,
    PHASE_MARKERS,
    SIGNIFICANCE_ALPHA,
    SOURCE_COLORS,
    SOURCE_LABELS,
    SOURCE_ORDER,
    bool_series,
    bootstrap_mean_ci,
    clean_label,
    p_text,
    save_bundle,
    set_paper_style,
    sign_flip_p,
    star,
)
from src.visualization.figure_data import (
    genotype_pvalues,
    model_observed_points,
    summarize_group_response,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "paper"


def _build(run_dir: Path, out_dir: Path) -> None:
    group = pd.read_csv(run_dir / "group_lineage_night_metrics.csv")
    panels = [
        ("baseline_centroid_distance", "Whole-group displacement"),
        ("dispersion_delta", "Dispersion change"),
        ("pairwise_distance_delta", "Pairwise distance change"),
        ("synchronization_similarity", "Trajectory synchronization"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), sharex=True)
    axes = axes.ravel()
    x_col = "stress_aligned_night"
    for ax, (metric, title) in zip(axes, panels):
        ax.axvspan(0.5, 1.5, color="#F2C94C", alpha=0.18, linewidth=0)
        ax.axhline(0, color="#111111", linewidth=0.7)
        for src in SOURCE_ORDER:
            sub = group.loc[group["source_sheet"].eq(src)].copy()
            color = SOURCE_COLORS[src]
            for _, cage in sub.groupby("cage_id", sort=False):
                ordered = cage.sort_values(x_col)
                ax.plot(ordered[x_col], ordered[metric], color=color, alpha=0.18, linewidth=0.8)
            mean = sub.groupby(x_col)[metric].mean()
            sem = sub.groupby(x_col)[metric].sem()
            x = mean.index.to_numpy(dtype=float)
            y = mean.to_numpy(dtype=float)
            e = sem.reindex(mean.index).fillna(0).to_numpy(dtype=float)
            ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=4, label=SOURCE_LABELS[src])
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.12, linewidth=0)
        ranked = group.dropna(subset=[metric]).copy()
        ranked["abs_metric"] = ranked[metric].abs()
        cage_rank = (
            ranked.sort_values("abs_metric", ascending=False)
            .drop_duplicates("cage_id")
            .sort_values("abs_metric", ascending=False)
        )
        if len(cage_rank) >= 2 and cage_rank.iloc[0]["abs_metric"] > 3 * cage_rank.iloc[1]["abs_metric"]:
            outlier_cage = cage_rank.iloc[0]["cage_id"]
            outlier = ranked.loc[ranked["cage_id"].eq(outlier_cage)].sort_values("abs_metric", ascending=False).iloc[0]
            color = SOURCE_COLORS.get(outlier["source_sheet"], "#111111")
            ax.annotate(
                outlier["cage_id"],
                (outlier[x_col], outlier[metric]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color=color,
                clip_on=False,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )
        ax.set_title(title)
        ax.set_xlabel("Night relative to stress")
        ax.set_ylabel("LDA-space units")
        ax.set_xticks([-3, -2, -1, 1, 2, 3])
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Stress-aligned group trajectories", y=0.98, fontsize=11, fontweight="bold")
    fig.subplots_adjust(hspace=0.42, wspace=0.28, top=0.90, bottom=0.09)
    save_bundle(fig, out_dir, "paper_figure_03_group_trajectories")



def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build figure 3 (standalone or via build_all)."""
    set_paper_style()
    input_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _build(input_dir, out_dir)
    print(f"wrote figure 3 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build figure 3.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

