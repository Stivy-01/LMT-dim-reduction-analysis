# -*- coding: utf-8 -*-
"""Supplementary S6: genotype heatmap."""
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
from src.visualization.supplementary.common import NIGHT_TICKS, OUTLIER

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "supplementary"


def figure_s6_genotype(run_dir: Path, out_dir: Path) -> None:
    summary = pd.read_csv(run_dir / "genotype_composite_delta_summary.csv")
    pvalues = pd.read_csv(run_dir / "paper_genotype_composite_signflip_pvalues.csv")
    columns = [("16p", "baseline", "16p\nBaseline"),
               ("16p", "post_stress", "16p\nPost-stress"),
               ("CD del", "baseline", "CD-del\nBaseline"),
               ("CD del", "post_stress", "CD-del\nPost-stress")]
    composites = list(dict.fromkeys(summary["composite"]))
    means = np.full((len(composites), len(columns)), np.nan)
    stars = np.empty_like(means, dtype=object)
    for j, (source, phase, _) in enumerate(columns):
        for i, composite in enumerate(composites):
            row = summary.loc[summary["source_sheet"].eq(source)
                              & summary["phase"].eq(phase)
                              & summary["composite"].eq(composite)]
            p_row = pvalues.loc[pvalues["source_sheet"].eq(source)
                                & pvalues["phase"].eq(phase)
                                & pvalues["composite"].eq(composite)]
            if len(row):
                means[i, j] = float(row["mean"].iloc[0])
            p = float(p_row["p_value"].iloc[0]) if len(p_row) else np.nan
            stars[i, j] = "*" if np.isfinite(p) and p < 0.005 else ""
    limit = float(np.nanmax(np.abs(means)))
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    image = ax.imshow(means, cmap="RdBu_r", vmin=-limit, vmax=limit,
                      aspect="auto")
    for i in range(len(composites)):
        for j in range(len(columns)):
            ax.text(j, i, f"{means[i, j]:+.2f}{stars[i, j]}", ha="center",
                    va="center", fontsize=8)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c[2] for c in columns], fontsize=8.5)
    ax.set_yticks(range(len(composites)))
    ax.set_yticklabels(composites, fontsize=8.5)
    ax.set_title("Genotype-linked behavioral signature\n(TG - WT paired"
                 " log1p deltas)", fontsize=10)
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("TG - WT mean delta", fontsize=8)
    bar.ax.tick_params(labelsize=8)
    fig.text(0.01, -0.02, "* exact sign-flip p < .005. Deltas are paired"
             " within the same cage, night and treatment.", fontsize=7.5)
    save_bundle(fig, out_dir, "Figure_S06_genotype_behavior_heatmap")




def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build S-06 (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_s6_genotype(run_dir, out_dir)
    print(f"wrote S-06 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S-06.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

