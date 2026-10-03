# -*- coding: utf-8 -*-
"""Supplementary S7: group response heatmap."""
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
from src.visualization.paper_style import SOURCE_LABELS
from src.visualization.supplementary.common import GROUP_METRICS
from src.visualization.supplementary.common import NIGHT_TICKS, OUTLIER

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "supplementary"


def figure_s7_group_heatmap(run_dir: Path, out_dir: Path) -> None:
    response = pd.read_csv(run_dir / "group_lineage_response_summary.csv")
    sources = ["16p", "CD del", "wt"]
    panels = [
        ("All cages (n = 13)", response),
        ("Excluding wt_10132 (n = 12)",
         response.loc[~response["cage_id"].eq(OUTLIER)]),
    ]
    labels = [label for _, label in GROUP_METRICS]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.4))
    image = None
    for ax, (title, frame) in zip(axes, panels):
        means = np.full((len(GROUP_METRICS), len(sources)), np.nan)
        counts = np.zeros_like(means)
        for j, source in enumerate(sources):
            sub = frame.loc[frame["source_sheet"].eq(source)]
            for i, (metric, _) in enumerate(GROUP_METRICS):
                values = pd.to_numeric(sub[metric], errors="coerce").dropna()
                if len(values):
                    means[i, j] = float(values.mean())
                    counts[i, j] = len(values)
        scaled = means / np.nanmax(np.abs(means), axis=1, keepdims=True)
        image = ax.imshow(scaled, cmap="RdBu_r", vmin=-1, vmax=1,
                          aspect="auto")
        for i in range(len(GROUP_METRICS)):
            for j in range(len(sources)):
                if not np.isfinite(means[i, j]):
                    continue
                ax.text(j, i, f"{means[i, j]:+.2f}\nn={int(counts[i, j])}",
                        ha="center", va="center", fontsize=7.5)
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([SOURCE_LABELS[s] for s in sources], fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.grid(False)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[1].set_yticks([])
    fig.suptitle("Group-level response signature by line", y=1.0,
                 fontsize=11, fontweight="bold")
    bar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label("Row-scaled response direction", fontsize=8)
    bar.ax.tick_params(labelsize=8)
    fig.text(0.01, -0.03, "Colours are scaled within each metric row;"
             " annotations are raw means. The WT column is influenced by the"
             " single cage wt_10132 (see Figure 11 and Figures 15-16).",
             fontsize=7.5)
    save_bundle(fig, out_dir, "Figure_S07_group_response_heatmap")


FAMILY_COLORS = {"count": "#0072B2", "mean_duration": "#D55E00",
                 "std_duration": "#009E73"}




def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build S-07 (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_s7_group_heatmap(run_dir, out_dir)
    print(f"wrote S-07 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S-07.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

