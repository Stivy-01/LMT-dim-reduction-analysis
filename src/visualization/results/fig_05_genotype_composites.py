# -*- coding: utf-8 -*-
"""Figure 5: genotype composite deltas."""
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
    summary = pd.read_csv(run_dir / "genotype_composite_delta_summary.csv")
    deltas = pd.read_csv(run_dir / "genotype_composite_paired_deltas.csv")
    pvals = genotype_pvalues(deltas, ["source_sheet", "phase", "composite"])
    pvals.to_csv(run_dir / "paper_genotype_composite_signflip_pvalues.csv", index=False)
    summary = summary.merge(pvals, on=["source_sheet", "phase", "composite"], how="left")
    composites = [
        "Social proximity counts",
        "Social bout duration",
        "Isolation counts",
        "Isolation duration",
        "Exploration / rearing",
        "Stop / fragmentation",
    ]
    panel_groups = [("16p", "baseline"), ("16p", "post_stress"), ("CD del", "baseline"), ("CD del", "post_stress")]
    fig, axes = plt.subplots(1, 4, figsize=(8.2, 4.8), sharey=True, sharex=True)
    y_pos = np.arange(len(composites))[::-1]
    for ax, (src, phase) in zip(axes, panel_groups):
        ax.axvline(0, color="#111111", linewidth=0.7)
        sub = summary.loc[summary["source_sheet"].eq(src) & summary["phase"].eq(phase)]
        for _, row in sub.iterrows():
            if row["composite"] not in composites:
                continue
            y = y_pos[composites.index(row["composite"])]
            color = SOURCE_COLORS[src]
            ax.errorbar(
                row["mean"],
                y,
                xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
                fmt=PHASE_MARKERS[phase],
                color=color,
                ecolor=color,
                capsize=3,
                markersize=5,
                elinewidth=1.2,
            )
            if star(row["p_value"]):
                ax.text(
                    0.97,
                    y,
                    "*",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
        ax.set_title(f"{SOURCE_LABELS[src]}\n{PHASE_LABELS[phase]}")
        ax.set_xlabel("TG - WT")
        ax.set_yticks(y_pos)
        if ax is axes[0]:
            ax.set_yticklabels(composites)
        else:
            ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Genotype differences in behavioral composites", y=0.98, fontsize=11, fontweight="bold")
    fig.text(0.01, -0.02, "* p < .005, exact sign-flip test on paired TG-WT deltas.", fontsize=8)
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.18, top=0.82, wspace=0.52)
    save_bundle(fig, out_dir, "paper_figure_05_genotype_composite_deltas")



def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build figure 5 (standalone or via build_all)."""
    set_paper_style()
    input_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _build(input_dir, out_dir)
    print(f"wrote figure 5 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build figure 5.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

