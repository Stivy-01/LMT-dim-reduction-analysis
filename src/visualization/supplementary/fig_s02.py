# -*- coding: utf-8 -*-
"""Supplementary S2: retained observations."""
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
from src.visualization.supplementary.common import _cage_labels, _cage_order
from src.visualization.supplementary.common import NIGHT_TICKS, OUTLIER

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "supplementary"


def figure_s2_availability(run_dir: Path, out_dir: Path) -> None:
    qc = pd.read_csv(run_dir / "quality_control.csv")
    cages = _cage_order(qc)
    nights = sorted(qc["stress_aligned_night"].dropna().unique())
    counts = (
        qc.pivot_table(index="cage_id", columns="stress_aligned_night",
                       values="projection_eligible", aggfunc="sum")
        .reindex(index=cages, columns=nights)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(counts.to_numpy(), cmap="Blues", vmin=0, vmax=4,
                      aspect="auto")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            value = counts.iloc[i, j]
            ax.text(j, i, f"{int(value)}", ha="center", va="center",
                    fontsize=7, color="white" if value >= 3 else "#111111")
    ax.set_xticks(range(len(nights)))
    ax.set_xticklabels([f"{int(n):+d}" for n in nights], fontsize=8)
    ax.set_yticks(range(len(cages)))
    ax.set_yticklabels(_cage_labels(qc, cages), fontsize=8)
    ax.set_xlabel("Night relative to the stress event")
    ax.set_ylabel("Cage (source)")
    ax.set_title("Retained mouse-night observations per cage and night",
                 fontsize=10)
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("Mice retained (of 4 per cage)", fontsize=8)
    bar.ax.tick_params(labelsize=8)
    save_bundle(fig, out_dir, "Figure_S02_observation_availability")




def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build S-02 (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_s2_availability(run_dir, out_dir)
    print(f"wrote S-02 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S-02.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

