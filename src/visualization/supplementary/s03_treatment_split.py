"""Candidate Supplementary Figure S3: identity-composite trajectories split by
treatment, with and without the influential WT cage wt_10132.

Each panel shows the cage-level mean identity-composite change from the same
cage-treatment baseline (blue: control mice, orange: stressed mice), with faint
lines for individual cages and SEM ribbons across cages.
"""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.paper_style import save_bundle, set_paper_style

RUN_DIR = Path(__file__).resolve().parents[3] / "src" / "visualization" / "data"
OUTLIER = "wt_10132"
TREATMENT_COLORS = {"control": "#0072B2", "stressed": "#D55E00"}
TREATMENT_LABELS = {"control": "Control", "stressed": "Stressed"}
NIGHT_TICKS = [-3, -2, -1, 1, 2, 3]
STEM = "Figure_S03_identity_trajectories_by_treatment"
CLIP_STEM = "Figure_S03_identity_trajectories_by_treatment_clipped"
CLIP_LIMIT = 3.2


def _cage_treatment_tracks(mouse: pd.DataFrame) -> pd.DataFrame:
    """Mean identity composite per cage-night-treatment, referenced to the same
    cage-treatment baseline mean."""
    data = mouse.loc[mouse["effect_eligible"].fillna(False).astype(bool)].copy()
    baseline = (
        data.loc[data["phase"].eq("baseline")]
        .groupby(["cage_id", "treatment"])["identity_composite"]
        .mean()
        .rename("baseline_mean")
    )
    grouped = (
        data.groupby(["cage_id", "treatment", "stress_aligned_night"])
        ["identity_composite"]
        .mean()
        .reset_index()
        .merge(baseline, on=["cage_id", "treatment"], how="left")
    )
    grouped["delta"] = grouped["identity_composite"] - grouped["baseline_mean"]
    return grouped


def _draw(out_dir: Path, tracks: pd.DataFrame, clip: bool) -> None:
    panels = [
        ("All cages (n = 13)", False),
        ("Excluding wt_10132 (n = 12)", True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, (title, drop_outlier) in zip(axes, panels):
        frame = tracks.loc[~tracks["cage_id"].eq(OUTLIER)] if drop_outlier \
            else tracks
        ax.axvspan(0.5, 1.5, color="#F2C94C", alpha=0.18, linewidth=0)
        ax.axhline(0, color="#111111", linewidth=0.8)
        for treatment in ("control", "stressed"):
            sub = frame.loc[frame["treatment"].eq(treatment)]
            color = TREATMENT_COLORS[treatment]
            for _, cage in sub.groupby("cage_id", sort=False):
                ordered = cage.sort_values("stress_aligned_night")
                ax.plot(ordered["stress_aligned_night"], ordered["delta"],
                        color=color, alpha=0.16, linewidth=0.8, zorder=1)
            mean = sub.groupby("stress_aligned_night")["delta"].mean()
            sem = sub.groupby("stress_aligned_night")["delta"].sem()
            x = mean.index.to_numpy(dtype=float)
            y = mean.to_numpy(dtype=float)
            err = sem.reindex(mean.index).fillna(0).to_numpy(dtype=float)
            ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=4,
                    label=TREATMENT_LABELS[treatment], zorder=3)
            ax.fill_between(x, y - err, y + err, color=color, alpha=0.15,
                            linewidth=0, zorder=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Night relative to the stress event")
        ax.set_xticks(NIGHT_TICKS)
        if clip:
            ax.set_ylim(-CLIP_LIMIT, CLIP_LIMIT)
    axes[0].set_ylabel("Identity-composite change\n(from own baseline,"
                       " LDA units)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    if clip:
        biggest = float(tracks.loc[tracks["cage_id"].eq(OUTLIER), "delta"].max())
        fig.text(0.01, -0.06, "One control track from wt_10132 leaves the axis"
                 f" range and reaches +{biggest:.0f} LDA units at +1 to +3.",
                 fontsize=7.5)
    fig.suptitle("Identity-composite trajectories by treatment", y=1.03,
                 fontsize=11, fontweight="bold")
    fig.subplots_adjust(wspace=0.30, top=0.82, bottom=0.20)
    save_bundle(fig, out_dir, CLIP_STEM if clip else STEM)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "supplementary"


def parse_args() -> argparse.Namespace:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory. Defaults to src/visualization/output/supplementary.")
    return parser.parse_args()


def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build S03 treatment-split variants (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else RUN_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mouse = pd.read_csv(run_dir / "mouse_night_analysis.csv")
    tracks = _cage_treatment_tracks(mouse)
    tracks = tracks.loc[tracks["stress_aligned_night"].isin(NIGHT_TICKS)]
    _draw(out_dir, tracks, clip=False)
    _draw(out_dir, tracks, clip=True)

    summary = (
        tracks.groupby(["treatment", "stress_aligned_night"])["delta"]
        .mean().unstack("stress_aligned_night").round(2)
    )
    print(summary.to_string())
    no_out = tracks.loc[~tracks["cage_id"].eq(OUTLIER)]
    print("\nwithout outlier:")
    print(no_out.groupby(["treatment", "stress_aligned_night"])["delta"]
          .mean().unstack("stress_aligned_night").round(2).to_string())
    print("\nsaved", out_dir / f"{STEM}.png")
    return out_dir


def main() -> int:
    args = parse_args()
    build(args.run_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
