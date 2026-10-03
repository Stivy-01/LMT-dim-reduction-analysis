# -*- coding: utf-8 -*-
"""Supplementary S4: feature interactions."""
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
from src.visualization.supplementary.common import _pretty
from src.visualization.supplementary.common import NIGHT_TICKS, OUTLIER

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "supplementary"


def figure_s4_feature_interactions(run_dir: Path, out_dir: Path) -> None:
    effects = pd.read_csv(run_dir / "feature_effects.csv")
    scopes = [
        ("primary", "Primary (all cages)"),
        ("exclude_wt_10132", "Excluding wt_10132"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.4))
    for ax, (scope, label) in zip(axes, scopes):
        sub = effects.loc[effects["analysis_scope"].eq(scope)].dropna(
            subset=["interaction_estimate"]).copy()
        sub["abs_estimate"] = sub["interaction_estimate"].abs()
        top = sub.sort_values(["interaction_p_value", "abs_estimate"],
                              ascending=[True, False]).head(12)
        y = np.arange(len(top))[::-1]
        ax.errorbar(
            top["interaction_estimate"], y,
            xerr=[top["interaction_estimate"] - top["interaction_ci_low"],
                  top["interaction_ci_high"] - top["interaction_estimate"]],
            fmt="o", color="#0072B2", ecolor="#0072B2", elinewidth=1.2,
            capsize=3, markersize=4.5, zorder=3,
        )
        ax.axvline(0, color="#111111", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([_pretty(f) for f in top["feature"]], fontsize=7.5)
        minimum_q = sub["interaction_q_value"].min()
        n_sig = int((sub["interaction_q_value"] < 0.05).sum())
        ax.set_title(f"{label}\n{n_sig} of {len(sub)} features q < 0.05;"
                     f" min q = {minimum_q:.3f}", fontsize=9)
        ax.set_xlabel("Phase x treatment estimate (95% CI)")
    axes[0].set_ylabel("Feature (strongest effects)")
    fig.suptitle("Feature-level phase-by-treatment interactions", y=1.02,
                 fontsize=11, fontweight="bold")
    fig.subplots_adjust(wspace=0.62, top=0.80, bottom=0.14)
    save_bundle(fig, out_dir, "Figure_S04_feature_interaction_forest")




def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build S-04 (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_s4_feature_interactions(run_dir, out_dir)
    print(f"wrote S-04 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S-04.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

