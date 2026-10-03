# -*- coding: utf-8 -*-
"""Figures 7-8: group identity space."""
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
    for include_outlier, stem, title in [
        (True, "paper_figure_07_group_identity_space", "Group centroids in identity-related space"),
        (False, "paper_figure_08_group_identity_space_without_wt10132", "Group centroids excluding wt_10132"),
    ]:
        plot_data = group.copy()
        if not include_outlier:
            plot_data = plot_data.loc[~plot_data["cage_id"].eq("wt_10132")].copy()
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        ax.axhline(0, color="#D1D5DB", linewidth=0.8)
        ax.axvline(0, color="#D1D5DB", linewidth=0.8)
        for src in SOURCE_ORDER:
            sub = plot_data.loc[plot_data["source_sheet"].eq(src)].copy()
            color = SOURCE_COLORS[src]
            for cage_id, cage in sub.groupby("cage_id", sort=False):
                ordered = cage.sort_values("stress_aligned_night")
                ax.plot(ordered["mean_id_1"], ordered["mean_id_2"], color=color, alpha=0.25, linewidth=1.0)
                for phase in ["baseline", "post_stress"]:
                    phase_data = ordered.loc[ordered["phase"].eq(phase)]
                    if phase_data.empty:
                        continue
                    ax.scatter(
                        phase_data["mean_id_1"],
                        phase_data["mean_id_2"],
                        color=color,
                        marker=PHASE_MARKERS[phase],
                        s=28,
                        alpha=0.82,
                        edgecolor="white",
                        linewidth=0.4,
                    )
                if cage_id in {"wt_10132", "wt_10565", "wt_10572", "16p_9333"} and len(ordered):
                    last = ordered.iloc[-1]
                    ax.annotate(cage_id, (last["mean_id_1"], last["mean_id_2"]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)
        ax.set_xlabel("Group centroid ID1")
        ax.set_ylabel("Group centroid ID2")
        ax.set_title(title)
        handles = [
            plt.Line2D([0], [0], marker="o", color=SOURCE_COLORS[src], linestyle="", label=SOURCE_LABELS[src])
            for src in SOURCE_ORDER
        ]
        handles.extend(
            [
                plt.Line2D([0], [0], marker="o", color="#555555", linestyle="", label="Baseline"),
                plt.Line2D([0], [0], marker="s", color="#555555", linestyle="", label="Post-stress"),
            ]
        )
        ax.legend(handles=handles, frameon=False, loc="best", ncol=1)
        save_bundle(fig, out_dir, stem)



def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build figures 7-8 (standalone or via build_all)."""
    set_paper_style()
    input_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _build(input_dir, out_dir)
    print(f"wrote figures 7-8 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build figures 7-8.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

