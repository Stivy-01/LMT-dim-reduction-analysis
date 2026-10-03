# -*- coding: utf-8 -*-
"""Figure 1: individual model effects."""
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
    model = pd.read_csv(run_dir / "model_effects.csv")
    terms = {
        "phase_post_stress": "Post-stress phase",
        "phase_post_stress:treatment_stressed": "Post-stress x stressed",
    }
    responses = ["identity_composite", "id_1", "id_2"]
    scopes = [("primary", "Primary dataset"), ("exclude_wt_10132", "Sensitivity: excluding wt_10132")]
    observed = model_observed_points(run_dir, responses, scopes)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True, sharey=True)
    y_labels = [f"{clean_label(resp)}\n{terms[term]}" for resp in responses for term in terms]
    y_pos = np.arange(len(y_labels))[::-1]
    rng = np.random.default_rng(20240612)

    for ax, (scope, title) in zip(axes, scopes):
        subset = model.loc[
            model["analysis_scope"].eq(scope)
            & model["response"].isin(responses)
            & model["term"].isin(terms)
        ].copy()
        row_lookup = {(row.response, row.term): row for row in subset.itertuples(index=False)}
        ax.axvline(0, color="#111111", linewidth=0.8)
        for i, (resp, term) in enumerate((resp, term) for resp in responses for term in terms):
            row = row_lookup.get((resp, term))
            if row is None:
                continue
            y = y_pos[i]
            color = "#0072B2" if term == "phase_post_stress" else "#D55E00"
            points = observed.loc[
                observed["analysis_scope"].eq(scope)
                & observed["response"].eq(resp)
                & observed["term"].eq(term)
            ]
            if not points.empty:
                jitter = rng.normal(0, 0.055, size=len(points))
                ax.scatter(
                    points["value"],
                    np.full(len(points), y) + jitter,
                    color=color,
                    alpha=0.22,
                    s=13,
                    linewidth=0,
                    zorder=1,
                )
            ax.errorbar(
                row.estimate,
                y,
                xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.3,
                capsize=3,
                markersize=5,
                zorder=3,
            )
            if star(row.p_value):
                # significance markers sit in a dedicated column at the right
                # edge of the panel, aligned with their row, so they are easy
                # to read and never collide with the estimate or its CI
                ax.annotate("*", xy=(0.985, y), xycoords=("axes fraction",
                                                          "data"),
                            ha="right", va="center", fontsize=13,
                            fontweight="bold", color="#111111", zorder=5)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
    axes[0].set_ylabel("Response and model term")
    fig.supxlabel("Model estimate (95% CI)", y=0.04, fontsize=10)
    fig.suptitle("Individual identity-axis model effects", y=1.03, fontsize=11, fontweight="bold")
    fig.text(0.01, -0.03, "Faint points show mouse-level observed deltas aligned to each model term.", fontsize=8)
    fig.subplots_adjust(bottom=0.22, wspace=0.22)
    save_bundle(fig, out_dir, "paper_figure_01_individual_model_effects")



def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build figure 1 (standalone or via build_all)."""
    set_paper_style()
    input_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _build(input_dir, out_dir)
    print(f"wrote figure 1 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build figure 1.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

