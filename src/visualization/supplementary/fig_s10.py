# -*- coding: utf-8 -*-
"""Supplementary S10: model terms acute vs window."""
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
from src.visualization.supplementary.common import PRIMARY_RESPONSES, _pretty
from src.visualization.supplementary.common import NIGHT_TICKS, OUTLIER

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "supplementary"


def figure_s10_model_terms(run_dir: Path, out_dir: Path) -> None:
    table = pd.read_csv(run_dir / "longitudinal_window" /
                        "model_terms_acute_vs_window.csv")
    table = table.loc[table["response"].isin(PRIMARY_RESPONSES)]
    designs = [("acute (-1/+1)", "#0072B2", "o", "Acute (-1/+1)"),
               ("window (-2..-1 / +1..+3)", "#D55E00", "s",
                "Extended window")]
    rows = (table[["response", "term"]].drop_duplicates().reset_index(drop=True))
    rows["label"] = [f"{_pretty(r['response'])}: {r['term']}"
                     for _, r in rows.iterrows()]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    positions = np.arange(len(rows))[::-1]
    ax.axvline(0, color="#111111", linewidth=0.8)
    for offset, (design, color, marker, label) in zip(
            (0.16, -0.16), designs):
        for y, (_, row) in zip(positions, rows.iterrows()):
            match = table.loc[table["design"].eq(design)
                              & table["response"].eq(row["response"])
                              & table["term"].eq(row["term"])]
            if not len(match):
                continue
            estimate = float(match["estimate"].iloc[0])
            low = float(match["ci_low"].iloc[0])
            high = float(match["ci_high"].iloc[0])
            ax.errorbar(estimate, y + offset,
                        xerr=[[estimate - low], [high - estimate]], fmt=marker,
                        color=color, ecolor=color, elinewidth=1.2, capsize=3,
                        markersize=5, label=label if y == positions[0] else None,
                        zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(rows["label"], fontsize=8)
    ax.set_xlabel("Model estimate (95% CI)")
    ax.set_title("Individual identity-axis model terms under the two"
                 " contrasts", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.01, -0.06, "The acute contrast uses nights -1 and +1; the"
             " extended window compares the mean of -2/-1 with the mean of"
             " +1/+2/+3.", fontsize=7.5)
    save_bundle(fig, out_dir, "Figure_S10_model_terms_acute_vs_window")




def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build S-10 (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_s10_model_terms(run_dir, out_dir)
    print(f"wrote S-10 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S-10.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

