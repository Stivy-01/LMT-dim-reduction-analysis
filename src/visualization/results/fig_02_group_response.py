# -*- coding: utf-8 -*-
"""Figure 2: group response summary."""
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
    response = pd.read_csv(run_dir / "group_lineage_response_summary.csv")
    summary = summarize_group_response(response)
    summary.to_csv(run_dir / "paper_group_response_signflip_summary.csv", index=False)
    metrics = summary.loc[summary["source_sheet"].eq("ALL"), "metric"].tolist()
    metric_labels = dict(zip(summary["metric"], summary["metric_label"]))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    offsets = {"ALL": 0.27, "16p": 0.09, "CD del": -0.09, "wt": -0.27}
    markers = {"ALL": "D", "16p": "o", "CD del": "s", "wt": "^"}
    y_base = np.arange(len(metrics))[::-1]
    rng = np.random.default_rng(20240612)
    for source in ["ALL", "16p", "CD del", "wt"]:
        sub = summary.loc[summary["source_sheet"].eq(source)]
        for _, row in sub.iterrows():
            y = y_base[metrics.index(row["metric"])] + offsets[source]
            color = SOURCE_COLORS[source]
            raw = response if source == "ALL" else response.loc[response["source_sheet"].eq(source)]
            raw_values = pd.to_numeric(raw[row["metric"]], errors="coerce").dropna()
            if not raw_values.empty:
                jitter = rng.normal(0, 0.025, size=len(raw_values))
                ax.scatter(
                    raw_values,
                    np.full(len(raw_values), y) + jitter,
                    marker=markers[source],
                    color=color,
                    alpha=0.22 if source == "ALL" else 0.28,
                    s=14,
                    linewidth=0,
                    zorder=1,
                )
            ax.errorbar(
                row["mean"],
                y,
                xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
                fmt=markers[source],
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                markersize=5 if source != "ALL" else 5.5,
                alpha=0.95 if source == "ALL" else 0.82,
                zorder=3,
            )
            if row["sig"]:
                ax.text(row["ci_high"] + 0.35, y, "*", ha="left", va="center", fontsize=17, fontweight="bold")
    ax.axvline(0, color="#111111", linewidth=0.8)
    ax.set_yticks(y_base)
    ax.set_yticklabels([metric_labels[m] for m in metrics])
    ax.set_xlabel("Cage-level post-stress response (mean, 95% CI)")
    ax.set_ylabel("Group metric")
    ax.set_title("Group-level behavioral response")
    handles = [
        plt.Line2D([0], [0], marker=markers[src], color=SOURCE_COLORS[src], linestyle="", label=SOURCE_LABELS[src])
        for src in ["ALL", "16p", "CD del", "wt"]
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    ax.text(
        0.01,
        -0.16,
        "* p < .005, exact sign-flip test for signed metrics only; distance metrics are descriptive.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    save_bundle(fig, out_dir, "paper_figure_02_group_response_summary")



def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build figure 2 (standalone or via build_all)."""
    set_paper_style()
    input_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _build(input_dir, out_dir)
    print(f"wrote figure 2 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build figure 2.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

