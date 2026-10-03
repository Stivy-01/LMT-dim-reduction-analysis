# -*- coding: utf-8 -*-
"""Captions paper (placement e descrizioni)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.visualization.paper_style import set_paper_style

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "paper"


def _build(run_dir: Path, out_dir: Path) -> None:
    captions = f"""# Publication Figure Package

Output folder: `{out_dir}`

Significance notation: `* p < .005`. P-values are added only where a p-value is defined by the analysis output or by an explicit exact sign-flip test on paired/cage-level deltas. Absence of an asterisk means the comparison did not pass this threshold, not that the effect is biologically irrelevant.

## Figure 1. Individual identity-axis model effects
Model estimates and 95% CIs for post-stress phase effects and phase-by-treatment interactions. Faint points show mouse-level observed deltas aligned to each model term. The primary dataset does not show `p < .005` effects for the main individual axes; the sensitivity analysis excluding `wt_10132` shows strong phase effects for ID1 and ID2.

## Figure 2. Group-level behavioral response
Cage-level post-stress response summaries with bootstrap 95% CIs. Faint points show individual cage values. Signed metrics use exact sign-flip tests against zero; non-negative distance metrics are descriptive and are not assigned p-stars.

## Figure 3. Stress-aligned group trajectories
Mean group trajectories by experimental line, with individual cages shown faintly and mean +/- SEM overlaid. This figure is useful for Results because it shows the temporal shape of the group-level response.

## Figure 4. Genotype differences in identity/projection axes
Paired TG-WT deltas by source and phase. Raw paired values are shown as faint points; mean and bootstrap 95% CI are overlaid. Asterisks mark exact sign-flip `p < .005`.

## Figure 5. Genotype differences in behavioral composites
Composite TG-WT deltas grouped by behavioral domain. This is the clearest genotype-level figure for the thesis because it avoids overloading the reader with hundreds of individual features.

## Figure 6. Key genotype-linked behavioral features
Top individual features with exact sign-flip `p < .005`. Use this as a supporting/supplementary figure unless the Results section needs feature-level biological interpretation.

## Figures 7-8. Group centroids in identity-related space
Centroid trajectories in the two identity-related axes, shown with and without the WT outlier cage `wt_10132`. These figures are exploratory and should not be described as demonstrating Forkosh-style identity domains.
"""
    (out_dir / "paper_figure_captions_and_placement.md").write_text(captions, encoding="utf-8")



def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build captions (standalone or via build_all)."""
    set_paper_style()
    input_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _build(input_dir, out_dir)
    print(f"wrote captions to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build captions.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

