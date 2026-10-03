# -*- coding: utf-8 -*-
"""Main di gruppo: ricostruisce tutte le figure committate della tesi.

- Methods (Fig 1-3): src/visualization/methods/
- Results (Fig 4-17): src/visualization/results/
- Supplementary (S1-S10): src/visualization/supplementary/

Uso:
    python -m src.visualization.build_all
    python -m src.visualization.build_all --input-dir <csv-dir> --output-dir <fig-dir>

Tutti gli input di default sono in src/visualization/data/; gli output in
src/visualization/output/. Gli strumenti bulk non committati
(supplementary/pages.py, pdfs.py, window report) restano standalone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization.methods import fig_methods_01_setup, fig_methods_03_manipulation
from src.visualization.results import (
    fig_01_model_effects,
    fig_02_group_response,
    fig_03_group_trajectories,
    fig_04_genotype_axes,
    fig_05_genotype_composites,
    fig_06_genotype_key_features,
    fig_07_08_identity_space,
    fig_09_prepost_profiles,
    fig_11_within_cage,
    fig_17_longitudinal,
    fig_captions,
    fig_genotype,
    fig_group_dynamics,
    fig_per_feature_changes,
)
from src.visualization.supplementary import (
    fig_s01,
    fig_s02,
    fig_s03,
    fig_s04,
    fig_s05,
    fig_s06,
    fig_s07,
    fig_s08,
    fig_s09,
    fig_s10,
    s03_treatment_split,
    treatment_profiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output"

METHODS_MODULES = (fig_methods_01_setup, fig_methods_03_manipulation)
RESULTS_MODULES = (
    fig_01_model_effects,
    fig_02_group_response,
    fig_03_group_trajectories,
    fig_04_genotype_axes,
    fig_05_genotype_composites,
    fig_06_genotype_key_features,
    fig_07_08_identity_space,
    fig_09_prepost_profiles,
    fig_per_feature_changes,
    fig_11_within_cage,
    fig_17_longitudinal,
    fig_captions,
)
SUPP_MODULES = (
    fig_s01,
    fig_s02,
    fig_s03,
    fig_s04,
    fig_s05,
    fig_s06,
    fig_s07,
    fig_s08,
    fig_s09,
    fig_s10,
    s03_treatment_split,
)


def build_all(input_dir: Path | None = None, output_dir: Path | None = None) -> Path:
    input_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    methods_dir = output_dir / "methods"
    paper_dir = output_dir / "paper"
    supp_dir = output_dir / "supplementary"
    style_dir = output_dir / "paper_style"
    for module in METHODS_MODULES:
        module.build(input_dir, methods_dir)
    for module in RESULTS_MODULES:
        module.build(input_dir, paper_dir)
    fig_genotype.build_genotype_figures(input_dir, output_dir / "genotype")
    fig_group_dynamics.build_group_dynamics_figures(input_dir, output_dir / "group_dynamics")
    for module in SUPP_MODULES:
        module.build(input_dir, supp_dir)
    treatment_profiles.build(input_dir, style_dir)
    print(f"all figures written under {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all committed thesis figures as a group.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build_all(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
