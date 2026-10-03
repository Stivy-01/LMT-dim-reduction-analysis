# -*- coding: utf-8 -*-
"""Figure 4: genotype axis deltas."""
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
    summary = pd.read_csv(run_dir / "genotype_axis_delta_summary.csv")
    deltas = pd.read_csv(run_dir / "genotype_axis_paired_deltas.csv")
    pvals = genotype_pvalues(deltas, ["source_sheet", "phase", "response"])
    pvals.to_csv(run_dir / "paper_genotype_axis_signflip_pvalues.csv", index=False)
    summary = summary.merge(pvals, on=["source_sheet", "phase", "response"], how="left")
    responses = ["identity_composite", "id_1", "id_2"]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.5), sharey=False)
    rng = np.random.default_rng(20240612)
    x_groups = [("16p", "baseline"), ("16p", "post_stress"), ("CD del", "baseline"), ("CD del", "post_stress")]
    x_labels = [
        "16p\nBase",
        "16p\nPost",
        "CD-del\nBase",
        "CD-del\nPost",
    ]
    for ax, response in zip(axes, responses):
        ax.axhline(0, color="#111111", linewidth=0.7)
        for i, (src, phase) in enumerate(x_groups):
            raw = deltas.loc[
                deltas["source_sheet"].eq(src) & deltas["phase"].eq(phase) & deltas["response"].eq(response),
                "tg_minus_wt",
            ]
            color = SOURCE_COLORS[src]
            if len(raw):
                jitter = rng.normal(0, 0.035, size=len(raw))
                ax.scatter(np.full(len(raw), i) + jitter, raw, color=color, alpha=0.28, s=12, linewidth=0)
            row = summary.loc[
                summary["source_sheet"].eq(src) & summary["phase"].eq(phase) & summary["response"].eq(response)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.errorbar(
                i,
                row["mean"],
                yerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                markersize=5,
                elinewidth=1.2,
            )
            if star(row["p_value"]):
                ax.text(
                    i,
                    0.94,
                    "*",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
        ax.set_title(clean_label(response))
        ax.set_xticks(range(len(x_groups)))
        ax.set_xticklabels(x_labels, rotation=0)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylabel("TG - WT mean delta")
    fig.suptitle("Genotype differences in identity/projection axes", y=0.98, fontsize=11, fontweight="bold")
    fig.text(0.01, -0.02, "* p < .005, exact sign-flip test on paired TG-WT deltas.", fontsize=8)
    fig.subplots_adjust(bottom=0.22, wspace=0.34, top=0.84)
    save_bundle(fig, out_dir, "paper_figure_04_genotype_axis_deltas")



def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build figure 4 (standalone or via build_all)."""
    set_paper_style()
    input_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _build(input_dir, out_dir)
    print(f"wrote figure 4 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build figure 4.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

