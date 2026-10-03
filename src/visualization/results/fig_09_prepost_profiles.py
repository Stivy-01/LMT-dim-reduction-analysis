# -*- coding: utf-8 -*-
"""Figure 9: profili pre/post con pittogrammi (A+B)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from src.visualization.supplementary.pages import (
    CATEGORY_COLORS,
    CATEGORY_ORDER,
    METRIC_SPECS,
    behavior_category,
    label_feature,
    metric_features,
    normalized_summary,
)
from src.visualization.paper_style import save_bundle, set_paper_style
from src.visualization.results.icons import (
    CATEGORY_ICONS,
    FEATURE_ICONS,
    build_composite_icons,
    icon_image,
    icon_zoom,
    place_icon,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "paper"


CONTROL_COLOR = "#0072B2"
STRESSED_COLOR = "#D55E00"
PHASE_NIGHTS = (-1, 1)

REPRESENTATIVE_FEATURES = [
    ("Contact_active_count", "Contact (count)"),
    ("Stop_count", "Stop (count)"),
    ("Group_4_make_count", "Group 4 make (count)"),
    ("Group_4_break_count", "Group 4 break (count)"),
    ("Move_isolated_count", "Move isolated (count)"),
    ("Rear_isolated_count", "Rear isolated (count)"),
    ("Center_Zone_count", "Center zone (count)"),
    ("Huddling_count", "Huddling (count)"),
    ("Contact_mean_duration", "Contact (mean dur.)"),
    ("Huddling_mean_duration", "Huddling (mean dur.)"),
]

PANEL_METRICS = [
    ("count", "Event count change (%)"),
    ("mean_duration", "Mean event duration change (%)"),
    ("std_duration", "Duration variability change (%)"),
]

REPRESENTATIVE_FEATURES = [
    ("Contact_active_count", "Contact (count)"),
    ("Stop_count", "Stop (count)"),
    ("Group_4_make_count", "Group 4 make (count)"),
    ("Group_4_break_count", "Group 4 break (count)"),
    ("Move_isolated_count", "Move isolated (count)"),
    ("Rear_isolated_count", "Rear isolated (count)"),
    ("Center_Zone_count", "Center zone (count)"),
    ("Huddling_count", "Huddling (count)"),
    ("Contact_mean_duration", "Contact (mean dur.)"),
    ("Huddling_mean_duration", "Huddling (mean dur.)"),
]




def load_selection(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "mouse_night_analysis.csv")
    df = df.loc[df["effect_eligible"].fillna(False).astype(bool)].copy()
    return df.loc[df["stress_aligned_night"].isin(PHASE_NIGHTS)].copy()


def pct_change(sel: pd.DataFrame, features: list[str], treatment: str) -> pd.Series:
    d = sel.loc[sel["treatment"].eq(treatment)]
    pre = d.loc[d["stress_aligned_night"].eq(-1), features].apply(pd.to_numeric, errors="coerce").mean(axis=0)
    post = d.loc[d["stress_aligned_night"].eq(1), features].apply(pd.to_numeric, errors="coerce").mean(axis=0)
    return (post / pre.replace(0, np.nan) - 1.0) * 100.0


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_main(sel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    # the bottom-right panel carries one schematic per feature row, so the
    # bottom row gets extra height: that row pitch is what sets icon size
    fig, axes = plt.subplots(
        2, 2, figsize=(8.4, 7.4),
        gridspec_kw={"height_ratios": [1.0, 1.45]})
    rng = np.random.default_rng(20260723)
    rows: list[dict[str, object]] = []

    for ax, (metric, title) in zip(axes.ravel()[:3], PANEL_METRICS):
        features = metric_features(sel, metric)
        for x, (treatment, color) in enumerate(((("control", CONTROL_COLOR)), (("stressed", STRESSED_COLOR)))):
            series = pct_change(sel, features, treatment).dropna()
            for feature, value in series.items():
                rows.append({"panel": metric, "treatment": treatment, "feature": feature, "pct_change": value})
            jitter = rng.uniform(-0.16, 0.16, size=len(series))
            ax.scatter(
                np.full(len(series), x) + jitter,
                series.to_numpy(),
                color=color,
                alpha=0.55,
                s=13,
                linewidth=0,
                zorder=2,
            )
            ax.hlines(series.median(), x - 0.28, x + 0.28, color=color, linewidth=2.0, zorder=3)
        ax.axhline(0, color="#111111", linewidth=0.8, zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "Stressed"])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylabel("% change (post / pre \u2212 1)")
        ax.set_title(title, fontsize=10)
        style_axis(ax)

    ax = axes[1, 1]
    y_pos = np.arange(len(REPRESENTATIVE_FEATURES))[::-1]
    for y, (feature, label) in zip(y_pos, REPRESENTATIVE_FEATURES):
        control = pct_change(sel, [feature], "control").iloc[0]
        stressed = pct_change(sel, [feature], "stressed").iloc[0]
        rows.append({"panel": "representative", "treatment": "control", "feature": feature, "pct_change": control})
        rows.append({"panel": "representative", "treatment": "stressed", "feature": feature, "pct_change": stressed})
        ax.plot([control, stressed], [y, y], color="#B8BCC4", linewidth=1.0, zorder=1)
        ax.scatter(control, y, color=CONTROL_COLOR, s=20, zorder=3)
        ax.scatter(stressed, y, color=STRESSED_COLOR, s=20, zorder=3)
    ax.axvline(0, color="#111111", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label for _, label in REPRESENTATIVE_FEATURES], fontsize=8)
    ax.set_xlabel("% change (post / pre \u2212 1)")
    ax.set_title("Representative features", fontsize=10)
    style_axis(ax)
    fig.subplots_adjust(hspace=0.52, wspace=0.55, top=0.90, bottom=0.11)
    # schematics inside the panel, left of the plotted values: one row each,
    # as large as the row pitch allows
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_w_in = ax.get_window_extent(renderer=renderer).width / fig.dpi
    panel_h_in = ax.get_window_extent(renderer=renderer).height / fig.dpi
    row_pitch_in = panel_h_in / float(len(REPRESENTATIVE_FEATURES))
    icon_col_in = 0.95
    panel_icons = [FEATURE_ICONS.get(f) for f, _l in REPRESENTATIVE_FEATURES]
    x0, x1 = ax.get_xlim()
    span = x1 - x0
    reserve = (icon_col_in + 0.10) / panel_w_in
    ax.set_xlim(x0 - reserve * span, x1)
    icon_x = x0 - reserve * span + (icon_col_in / 2.0) / panel_w_in * span
    for y, icon in zip(y_pos, panel_icons):
        if icon:
            zoom = icon_zoom(icon, icon_col_in * 72.0,
                             row_pitch_in * 72.0, fill=0.98)
            place_icon(ax, icon, (icon_x, y), ax.transData, zoom=zoom,
                       box_alignment=(0.5, 0.5), pad=0.0)

    handles = [
        plt.Line2D([0], [0], marker="o", color=CONTROL_COLOR, linestyle="", label="Control"),
        plt.Line2D([0], [0], marker="o", color=STRESSED_COLOR, linestyle="", label="Stressed"),
    ]
    axes[0, 0].legend(handles=handles, frameon=False, loc="lower left")
    fig.suptitle("Pre- to post-stress changes in the behavioral feature repertoire", y=0.995, fontsize=11, fontweight="bold")
    fig.text(
        0.01,
        0.005,
        "Each point is one feature (A\u2013C) or one representative feature (D); horizontal bars (A\u2013C) mark the median across features. "
        "Positive values indicate an increase after the manipulation.",
        fontsize=7.5,
    )
    save_bundle(fig, out_dir, "paper_figure_09_prepost_feature_profiles")

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "paper_figure_09_prepost_feature_profiles_values.csv", index=False)
    return out




def _build(input_dir: Path, out_dir: Path) -> None:
    build_composite_icons()
    sel = load_selection(input_dir)
    values = figure_main(sel, out_dir)
    print(f"wrote main figures to {out_dir} ({len(values)} summary values)")

def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build figure 9 (standalone or via build_all)."""
    set_paper_style()
    input_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _build(input_dir, out_dir)
    print(f"wrote figure 9 to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build figure 9.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

