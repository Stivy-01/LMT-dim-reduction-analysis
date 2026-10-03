# -*- coding: utf-8 -*-
"""Supplementary S5: axis loadings."""
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


def figure_s5_loadings(run_dir: Path, out_dir: Path) -> None:
    pca = pd.read_csv(run_dir / "pca_loadings.csv")
    identity = pd.read_csv(run_dir / "identity_loadings.csv")
    axis_order = ["pca_1", "pca_2", "id_1", "id_2"]
    table = pd.concat([
        pca.loc[pca["component"].isin(["pca_1", "pca_2"]),
                ["component", "feature", "loading"]],
        identity.loc[identity["component"].isin(["id_1", "id_2"]),
                     ["component", "feature", "loading"]],
    ])
    matrix = (table.pivot_table(index="feature", columns="component",
                                values="loading")
              .reindex(columns=axis_order).dropna(how="all"))
    scaled = matrix.div(matrix.abs().max(axis=0), axis=1)
    keep = scaled.abs().max(axis=1).sort_values(ascending=False).head(24).index
    scaled = scaled.loc[keep]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    image = ax.imshow(scaled.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1,
                      aspect="auto")
    ax.set_xticks(range(len(axis_order)))
    ax.set_xticklabels(["PCA1", "PCA2", "ID1", "ID2"], fontsize=9)
    ax.set_yticks(range(len(scaled)))
    ax.set_yticklabels([_pretty(f) for f in scaled.index], fontsize=7.5)
    ax.set_title("Feature loadings on the projection axes", fontsize=10)
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("Relative loading (scaled per axis)", fontsize=8)
    bar.ax.tick_params(labelsize=8)
    fig.text(0.01, -0.02, "Loadings are scaled by each axis's own maximum"
             " absolute loading.", fontsize=7.5)
    save_bundle(fig, out_dir, "Figure_S05_axis_feature_loadings")




def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build S-05 (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_s5_loadings(run_dir, out_dir)
    print(f"wrote S-05 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S-05.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

