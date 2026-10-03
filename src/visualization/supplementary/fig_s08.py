# -*- coding: utf-8 -*-
"""Supplementary S8: acute vs extended window."""
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
from src.visualization.supplementary.common import FAMILY_COLORS
from src.visualization.supplementary.common import NIGHT_TICKS, OUTLIER
try:
    from scripts.build_window_comparison_report import FAMILIES, load_inputs, per_mouse_pct
except ImportError:
    from build_window_comparison_report import FAMILIES, load_inputs, per_mouse_pct

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "supplementary"


def figure_s8_agreement(out_dir: Path, mouse: pd.DataFrame,
                        families: dict[str, list[str]]) -> tuple[float, float]:
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    agreement: list[float] = []
    all_x: list[float] = []
    all_y: list[float] = []
    for metric, features in families.items():
        for treatment in ("control", "stressed"):
            acute = per_mouse_pct(mouse, features, treatment,
                                  night=1).mean(axis=0).to_numpy()
            window = per_mouse_pct(mouse, features, treatment,
                                   window=True).mean(axis=0).to_numpy()
            all_x.extend(acute)
            all_y.extend(window)
            ax.scatter(acute, window, s=13, color=FAMILY_COLORS[metric],
                       alpha=0.8, linewidth=0, zorder=3,
                       label=FAMILIES[metric][1] if treatment == "control"
                       else None)
    stacked = np.abs(np.concatenate([all_x, all_y]))
    limit = float(np.nanpercentile(stacked, 98)) * 1.1
    outside = int((stacked > limit).sum())
    ax.plot([-limit, limit], [-limit, limit], color="#9AA0A6", linewidth=0.9,
            linestyle=(0, (4, 3)), zorder=2)
    ax.axhline(0, color="#111111", linewidth=0.7)
    ax.axvline(0, color="#111111", linewidth=0.7)
    r = float(stats.pearsonr(all_x, all_y)[0])
    slope = float(np.polyfit(all_x, all_y, 1)[0])
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Acute effect, night +1 (%)")
    ax.set_ylabel("Extended-window effect (%)")
    ax.set_title("Acute versus extended-window feature effects", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.03, 0.03, f"r = {r:.2f}, slope = {slope:.2f}"
            + (f"\n{outside} points outside the axis range" if outside else ""),
            transform=ax.transAxes, fontsize=8, va="bottom")
    fig.text(0.01, -0.05, "Each point is one behavioral feature; colours"
             " separate the three metric families and both treatment groups"
             " are pooled.", fontsize=7.5)
    save_bundle(fig, out_dir, "Figure_S08_acute_vs_extended_scatter")
    return r, slope




def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build S-08 (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mouse, families = load_inputs(run_dir)
    figure_s8_agreement(out_dir, mouse, families)
    print(f"wrote S-08 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S-08.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

