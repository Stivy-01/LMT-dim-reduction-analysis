from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from PIL import Image

from src.visualization.paper_style import save_bundle, set_paper_style
from src.visualization.supplementary.pages import (
    AXIS_SCALE_SPECS,
    CATEGORY_COLORS,
    CATEGORY_ORDER,
    METRIC_SPECS,
    PHASE_NIGHTS,
    SCALE_SPECS,
    TREATMENT_SPECS,
    add_category_bands,
    axis_ylabel,
    behavior_category,
    figure_title,
    label_feature,
    metric_features,
    normalized_summary,
    transform_values,
    transformed_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "paper_style"
NIGHTS = (-1, 1)
TREATMENTS = [("control", "Control mice", "control_mice_"), ("stressed", "Stressed mice", "stressed_mice_")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-style supplementary pre/post profile pages and PDFs.")
    parser.add_argument("run_dir", nargs="?", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with figure input CSVs. Defaults to src/visualization/data.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to src/visualization/output/paper_style.")
    return parser.parse_args()


def load_selection(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "mouse_night_analysis.csv")
    df = df.loc[df["effect_eligible"].fillna(False).astype(bool)].copy()
    return df.loc[df["stress_aligned_night"].isin(NIGHTS)].copy()


def render_page(summary, long_values, features, night, metric, scale, axis_scale, treatment, out_dir) -> None:
    spec = METRIC_SPECS[metric]
    scale_spec = SCALE_SPECS[scale]
    axis_spec = AXIS_SCALE_SPECS[axis_scale]
    treat_spec = TREATMENT_SPECS[treatment]

    phase_summary = summary.loc[summary["stress_aligned_night"].eq(night)].set_index("feature").loc[features]
    phase_values = long_values.loc[long_values["stress_aligned_night"].eq(night)]
    x = np.arange(len(features))
    colors = [CATEGORY_COLORS[behavior_category(feature)] for feature in features]
    means = transform_values(phase_summary[scale_spec["mean"]], axis_scale)

    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.25, width=0.76, zorder=2)
    ax.errorbar(
        x,
        means,
        yerr=transformed_error(phase_summary[scale_spec["mean"]], phase_summary[scale_spec["sem"]], axis_scale),
        fmt="none",
        ecolor="black",
        elinewidth=0.7,
        capsize=1.6,
        capthick=0.7,
        zorder=3,
    )
    rng = np.random.default_rng(20260723 + night)
    for index, feature in enumerate(features):
        values = transform_values(
            phase_values.loc[phase_values["feature"].eq(feature), scale_spec["value"]].dropna().to_numpy(),
            axis_scale,
        )
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=4, color="black", alpha=0.16, linewidth=0, zorder=4)

    upper = float(
        np.nanmax(
            transform_values(
                phase_summary[scale_spec["mean"]].to_numpy(dtype=float)
                + phase_summary[scale_spec["sem"]].fillna(0).to_numpy(dtype=float),
                axis_scale,
            )
        )
    )
    ax.set_ylim(0, max(1.35 if scale == "normalized" and axis_scale == "linear" else 0.01, upper * 1.18))
    if scale == "normalized":
        ax.axhline(transform_values([1], axis_scale)[0], color="#444444", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlim(-0.8, len(features) - 0.2)
    ax.set_ylabel(axis_ylabel(metric, scale, axis_scale))
    ax.set_xticks(x)
    ax.set_xticklabels([label_feature(feature) for feature in features], rotation=58, ha="right", fontsize=6)
    add_category_bands(ax, features)
    ax.set_title(figure_title(PHASE_NIGHTS[night] + " behavioral profile", metric, scale, axis_scale, treatment), fontsize=9, loc="center")

    legend_handles = [
        Patch(facecolor=CATEGORY_COLORS[category], edgecolor="black", linewidth=0.25, label=category)
        for category in CATEGORY_ORDER
        if category in set(phase_summary["category"])
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False, fontsize=7)
    fig.subplots_adjust(top=0.80, bottom=0.40, left=0.05, right=0.99)

    value_prefix = f"{scale_spec['stem_prefix']}{axis_spec['stem_prefix']}"
    phase_stem = "pre_stress" if night == -1 else "post_stress"
    stem = f"behavior_profile_{treat_spec['stem_prefix']}{phase_stem}_{value_prefix}{spec['stem']}"
    save_bundle(fig, out_dir, stem)


def build_pages(data: pd.DataFrame, treatment: str, out_dir: Path) -> None:
    for metric in ["count", "mean_duration", "std_duration"]:
        features = metric_features(data, metric)
        summary, long_values = normalized_summary(data, features)
        for night in NIGHTS:
            render_page(summary, long_values, features, night, metric, "normalized", "linear", treatment, out_dir)
    for metric in ["count", "mean_duration"]:
        features = metric_features(data, metric)
        summary, long_values = normalized_summary(data, features)
        for scale, axis_scale in [("raw", "linear"), ("raw", "log10p")]:
            for night in NIGHTS:
                render_page(summary, long_values, features, night, metric, scale, axis_scale, treatment, out_dir)


def page_order(prefix: str, label: str, log_raw: bool) -> list[tuple[str, str]]:
    raw_pref = "raw_log10p_" if log_raw else "raw_"
    raw_count = "raw event counts log10(1+x)" if log_raw else "raw event counts"
    raw_duration = "raw mean duration log10(1+x)" if log_raw else "raw mean duration"
    base = [
        ("normalized count profile", "pre-stress", "pre_stress_counts.png"),
        ("normalized count profile", "post-stress", "post_stress_counts.png"),
        ("normalized mean duration", "pre-stress", "pre_stress_mean_duration.png"),
        ("normalized mean duration", "post-stress", "post_stress_mean_duration.png"),
        ("normalized duration variability", "pre-stress", "pre_stress_std_duration.png"),
        ("normalized duration variability", "post-stress", "post_stress_std_duration.png"),
        (raw_count, "pre-stress", f"pre_stress_{raw_pref}counts.png"),
        (raw_count, "post-stress", f"post_stress_{raw_pref}counts.png"),
        (raw_duration, "pre-stress", f"pre_stress_{raw_pref}mean_duration.png"),
        (raw_duration, "post-stress", f"post_stress_{raw_pref}mean_duration.png"),
    ]
    return [(f"{label}: {lbl}: {phase}", f"behavior_profile_{prefix}{suffix}") for lbl, phase, suffix in base]


def assemble(pdf_path: Path, pages: list[tuple[str, str]], image_dir: Path) -> None:
    with PdfPages(pdf_path) as pdf:
        for title, stem in pages:
            image_path = image_dir / (stem if stem.endswith(".png") else f"{stem}.png")
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            image = Image.open(image_path)
            width, height = image.size
            fig_width = 11.69
            fig_height = max(3.5, fig_width * height / width + 0.5)
            fig = plt.figure(figsize=(fig_width, fig_height))
            ax = fig.add_axes((0.01, 0.02, 0.98, 0.9))
            ax.imshow(image)
            ax.axis("off")
            fig.suptitle(title, y=0.985, fontsize=9, fontweight="normal")
            pdf.savefig(fig, dpi=200)
            plt.close(fig)


def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build treatment-split pages + the 2 paper-style PDFs (standalone or via build_all)."""
    set_paper_style()
    run_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = load_selection(run_dir)
    for treatment, _label, _prefix in TREATMENTS:
        build_pages(sel.loc[sel["treatment"].eq(treatment)].copy(), treatment, out_dir)

    for log_raw, filename in [
        (False, "pre_post_behavior_profiles_paper_by_treatment.pdf"),
        (True, "pre_post_behavior_profiles_paper_log_raw_by_treatment.pdf"),
    ]:
        control_pages = page_order("control_mice_", "Control mice", log_raw)
        stressed_pages = page_order("stressed_mice_", "Stressed mice", log_raw)
        pages: list[tuple[str, str]] = []
        for control_page, stressed_page in zip(control_pages, stressed_pages):
            pages.append(control_page)
            pages.append(stressed_page)
        assemble(out_dir / filename, pages, out_dir)
        print(f"wrote {out_dir / filename}")

    print(f"paper-style supplementary written to {out_dir}")
    return out_dir


def main() -> int:
    args = parse_args()
    build(args.run_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
