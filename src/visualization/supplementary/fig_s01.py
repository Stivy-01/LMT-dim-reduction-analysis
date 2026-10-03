# -*- coding: utf-8 -*-
"""Supplementary S1: recording coverage and QC."""
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
from src.visualization.supplementary.common import STRESS_BAND, _cage_labels, _cage_order
from src.visualization.supplementary.common import NIGHT_TICKS, OUTLIER

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "supplementary"


def figure_s1_coverage(run_dir: Path, out_dir: Path) -> None:
    qc = pd.read_csv(run_dir / "quality_control.csv")
    cages = _cage_order(qc)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axvspan(0.5, 1.5, **STRESS_BAND)
    legend_done: set[str] = set()
    for y, cage in enumerate(cages):
        sub = qc.loc[qc["cage_id"].eq(cage)]
        for night, group in sub.groupby("stress_aligned_night"):
            total = len(group)
            kept = int(group["projection_eligible"].sum())
            if kept == total:
                color, marker, key = "#0072B2", "s", "primary"
            elif kept == 0:
                color, marker, key = "#D55E00", "x", "excluded"
            else:
                color, marker, key = "#E9C46A", "s", "partly retained"
            label = None if key in legend_done else {
                "primary": "Retained rows",
                "excluded": "Excluded / flagged rows",
                "partly retained": "Partly retained",
            }[key]
            legend_done.add(key)
            ax.scatter(night, y, marker=marker, s=28, color=color,
                       linewidth=1.0, zorder=3, label=label)
    ax.set_yticks(range(len(cages)))
    ax.set_yticklabels(_cage_labels(qc, cages), fontsize=8)
    ax.set_ylim(len(cages) - 0.5, -0.5)
    ax.set_xticks(NIGHT_TICKS)
    ax.set_xlabel("Night relative to the stress event")
    ax.set_ylabel("Cage (source)")
    ax.set_title("Recording coverage and quality control by cage",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.text(0.0, -0.32, "* influential WT cage (see Figure 11 and Figures"
            " 15-16).", transform=ax.transAxes, fontsize=7.5)
    save_bundle(fig, out_dir, "Figure_S01_recording_coverage_qc")




def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build S-01 (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_s1_coverage(run_dir, out_dir)
    print(f"wrote S-01 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S-01.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

