# -*- coding: utf-8 -*-
"""Supplementary S9: peak night distribution."""
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
try:
    from scripts.build_window_comparison_report import FAMILIES, load_inputs, per_mouse_pct
except ImportError:
    from build_window_comparison_report import FAMILIES, load_inputs, per_mouse_pct

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "supplementary"


def figure_s9_peak_night(out_dir: Path, mouse: pd.DataFrame,
                         families: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for metric, features in families.items():
        for treatment in ("control", "stressed"):
            columns = {night: per_mouse_pct(mouse, features, treatment,
                                            night=night).mean(axis=0)
                       for night in (1, 2, 3)}
            table = pd.DataFrame(columns)
            share = table.abs().idxmax(axis=1).value_counts(normalize=True) * 100
            for night in (1, 2, 3):
                rows.append({"family": FAMILIES[metric][1],
                             "treatment": treatment,
                             "night": f"+{night}",
                             "share": float(share.get(night, 0.0))})
    data = pd.DataFrame(rows)
    order = ["counts", "mean duration", "duration variability"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    width = 0.13
    night_colors = {"+1": "#0072B2", "+2": "#56B4E9", "+3": "#9AA0A6"}
    for k, (treatment, hatch) in enumerate((("control", None),
                                            ("stressed", "//"))):
        for j, night in enumerate(("+1", "+2", "+3")):
            heights = []
            for family in order:
                sel = data.loc[data["family"].eq(family)
                               & data["treatment"].eq(treatment)
                               & data["night"].eq(night)]
                heights.append(float(sel["share"].iloc[0]) if len(sel) else 0.0)
            xs = np.arange(len(order)) + (k * 3 + j - 2.5) * width
            ax.bar(xs, heights, width=width, color=night_colors[night],
                   hatch=hatch, edgecolor="white", linewidth=0.4,
                   label=f"{treatment.capitalize()}, night {night}")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, fontsize=8.5)
    ax.set_ylabel("Features with largest\nabsolute effect (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Night of maximum feature-level change", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    fig.text(0.01, -0.30, "Solid bars: control mice; hatched bars: stressed"
             " mice.", fontsize=7.5)
    save_bundle(fig, out_dir, "Figure_S09_peak_night_distribution")
    return data




def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build S-09 (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mouse, families = load_inputs(run_dir)
    figure_s9_peak_night(out_dir, mouse, families)
    print(f"wrote S-09 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S-09.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

