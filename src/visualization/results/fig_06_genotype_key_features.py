# -*- coding: utf-8 -*-
"""Figure 6: genotype key features."""
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
    summary = pd.read_csv(run_dir / "genotype_feature_delta_summary.csv")
    deltas = pd.read_csv(run_dir / "genotype_feature_paired_deltas.csv")
    pvals = genotype_pvalues(deltas, ["source_sheet", "phase", "feature"])
    pvals.to_csv(run_dir / "paper_genotype_feature_signflip_pvalues.csv", index=False)
    summary = summary.merge(pvals, on=["source_sheet", "phase", "feature"], how="left")
    sig = summary.loc[summary["p_value"].lt(SIGNIFICANCE_ALPHA)].copy()
    if sig.empty:
        return
    sig["abs_mean"] = sig["mean"].abs()
    sig = sig.sort_values(["p_value", "abs_mean"], ascending=[True, False]).head(16)
    sig["label"] = sig.apply(
        lambda row: f"{SOURCE_LABELS[row['source_sheet']]} {PHASE_LABELS[row['phase']]}: {clean_label(row['feature'])}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    y_pos = np.arange(len(sig))[::-1]
    for y, (_, row) in zip(y_pos, sig.iterrows()):
        color = SOURCE_COLORS[row["source_sheet"]]
        ax.errorbar(
            row["mean"],
            y,
            xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
            fmt=PHASE_MARKERS[row["phase"]],
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
            elinewidth=1.2,
        )
        ax.text(
            0.985,
            y,
            "*",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.axvline(0, color="#111111", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sig["label"])
    ax.set_xlabel("TG - WT mean delta (log-transformed feature scale)")
    ax.set_title("Key genotype-linked behavioral features")
    ax.text(
        0.01,
        -0.12,
        "* p < .005, exact sign-flip test on paired TG-WT deltas; top significant features shown.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    save_bundle(fig, out_dir, "paper_figure_06_genotype_key_features")



def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build figure 6 (standalone or via build_all)."""
    set_paper_style()
    input_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _build(input_dir, out_dir)
    print(f"wrote figure 6 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build figure 6.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

