# -*- coding: utf-8 -*-
"""Supplementary S3: identity trajectories."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.visualization.paper_style import save_bundle, set_paper_style
from src.visualization.paper_style import SOURCE_COLORS, SOURCE_LABELS, SOURCE_ORDER
from src.visualization.supplementary.common import STRESS_BAND
from src.visualization.supplementary.common import NIGHT_TICKS, OUTLIER

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "supplementary"


def figure_s3_trajectories(run_dir: Path, out_dir: Path) -> None:
    group = pd.read_csv(run_dir / "group_lineage_night_metrics.csv")
    panels = [
        ("All cages (n = 13)", None),
        ("Excluding wt_10132 (n = 12)", OUTLIER),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    metric = "mean_identity_composite_delta"
    for ax, (title, excluded) in zip(axes, panels):
        frame = group.loc[~group["cage_id"].eq(excluded)] if excluded else group
        ax.axvspan(0.5, 1.5, **STRESS_BAND)
        ax.axhline(0, color="#111111", linewidth=0.8)
        for source in SOURCE_ORDER:
            sub = frame.loc[frame["source_sheet"].eq(source)].copy()
            color = SOURCE_COLORS[source]
            for _, cage in sub.groupby("cage_id", sort=False):
                ordered = cage.sort_values("stress_aligned_night")
                ax.plot(ordered["stress_aligned_night"], ordered[metric],
                        color=color, alpha=0.20, linewidth=0.8)
            mean = sub.groupby("stress_aligned_night")[metric].mean()
            sem = sub.groupby("stress_aligned_night")[metric].sem()
            x = mean.index.to_numpy(dtype=float)
            y = mean.to_numpy(dtype=float)
            err = sem.reindex(mean.index).fillna(0).to_numpy(dtype=float)
            ax.plot(x, y, color=color, linewidth=2.0, marker="o",
                    markersize=4, label=SOURCE_LABELS[source])
            ax.fill_between(x, y - err, y + err, color=color, alpha=0.12,
                            linewidth=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Night relative to the stress event")
        ax.set_xticks(NIGHT_TICKS)
    axes[0].set_ylabel("Identity-composite change\n(from own baseline,"
                       " LDA units)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Stress-aligned identity-composite trajectories", y=1.03,
                 fontsize=11, fontweight="bold")
    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.20)
    save_bundle(fig, out_dir, "Figure_S03_identity_trajectories")




def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build S-03 (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_s3_trajectories(run_dir, out_dir)
    print(f"wrote S-03 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S-03.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

