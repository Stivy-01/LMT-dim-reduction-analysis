from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
PHASE_NIGHTS = {-1: "Pre-stress day", 1: "Post-stress day"}
METRIC_SPECS = {
    "count": {
        "suffix": "_count",
        "stem": "counts",
        "title": "event counts",
        "ylabel": "Value / pre-stress mean",
        "raw_title": "raw event counts",
        "raw_ylabel": "Events per mouse per 12 h",
    },
    "mean_duration": {
        "suffix": "_mean_duration",
        "stem": "mean_duration",
        "title": "mean duration",
        "ylabel": "Duration / pre-stress mean",
        "raw_title": "raw mean duration",
        "raw_ylabel": "Mean duration (s)",
    },
    "std_duration": {
        "suffix": "_std_duration",
        "stem": "std_duration",
        "title": "duration variability",
        "ylabel": "Std. duration / pre-stress mean",
        "raw_title": "raw duration variability",
        "raw_ylabel": "Std. duration (s)",
    },
}
SCALE_SPECS = {
    "normalized": {
        "mean": "normalized_mean",
        "sem": "normalized_sem",
        "value": "normalized_value",
        "stem_prefix": "",
    },
    "raw": {
        "mean": "raw_mean",
        "sem": "raw_sem",
        "value": "raw_value",
        "stem_prefix": "raw_",
    },
}
AXIS_SCALE_SPECS = {
    "linear": {
        "stem_prefix": "",
        "title_suffix": "",
    },
    "log10p": {
        "stem_prefix": "log10p_",
        "title_suffix": " (log10[1+x])",
    },
}
TREATMENT_SPECS = {
    "all": {
        "stem_prefix": "",
        "title_prefix": "",
    },
    "control": {
        "stem_prefix": "control_mice_",
        "title_prefix": "Control mice - ",
    },
    "stressed": {
        "stem_prefix": "stressed_mice_",
        "title_prefix": "Stressed mice - ",
    },
}
CATEGORY_ORDER = [
    "Body configuration",
    "Isolated behavior",
    "Position/context",
    "Contact type",
    "Social configuration",
    "Social approach",
    "Social escape",
    "Contact sequence",
    "Other",
]
CATEGORY_COLORS = {
    "Body configuration": "#D55E00",
    "Isolated behavior": "#E69F00",
    "Position/context": "#F0E442",
    "Contact type": "#009E73",
    "Social configuration": "#56B4E9",
    "Social approach": "#0072B2",
    "Social escape": "#CC79A7",
    "Contact sequence": "#6A3D9A",
    "Other": "#7F7F7F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Nature-style pre/post-stress behavior count profiles."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory with figure input CSVs (mouse_night_analysis.csv, ...). Defaults to src/visualization/data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/figures/pre_post_behavior_profiles.",
    )
    parser.add_argument(
        "--eligibility-column",
        default="sensitivity_effect_eligible",
        choices=["effect_eligible", "sensitivity_effect_eligible", "projection_eligible"],
        help="Boolean column used to select rows.",
    )
    parser.add_argument(
        "--metric",
        default="all",
        choices=["all", *METRIC_SPECS.keys()],
        help="Metric family to plot. Default builds count, mean_duration, and std_duration.",
    )
    parser.add_argument(
        "--scale",
        default="normalized",
        choices=["normalized", "raw", "all"],
        help="Plot normalized values, raw values, or both. Default: normalized.",
    )
    parser.add_argument(
        "--axis-scale",
        default="linear",
        choices=AXIS_SCALE_SPECS.keys(),
        help="Axis transform for plotted values. Use log10p for log10(1+x).",
    )
    parser.add_argument(
        "--treatment",
        default="all",
        choices=TREATMENT_SPECS.keys(),
        help="Treatment subgroup to plot. Default: all eligible mice.",
    )
    return parser.parse_args()


def set_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.9,
            "axes.axisbelow": True,
        }
    )


def behavior_category(feature: str) -> str:
    name = feature.lower()
    if name.startswith("seq_"):
        return "Contact sequence"
    if any(token in name for token in ["approach", "train2"]):
        return "Social approach"
    if any(token in name for token in ["get_away", "social_escape", "break_contact"]):
        return "Social escape"
    if any(token in name for token in ["group2", "group_3", "group_4", "huddling"]):
        return "Social configuration"
    if any(
        token in name
        for token in [
            "oral_genital",
            "oral_oral",
            "side_by_side",
            "contact",
        ]
    ):
        return "Contact type"
    if any(token in name for token in ["isolated", "isolated_count"]):
        return "Isolated behavior"
    if any(token in name for token in ["center_zone", "periphery_zone", "rear_at", "rear_in"]):
        return "Position/context"
    if any(token in name for token in ["rear", "rearing", "move_", "stop", "walljump", "sap"]):
        return "Body configuration"
    return "Other"


def label_feature(feature: str) -> str:
    label = feature
    for spec in METRIC_SPECS.values():
        label = label.removesuffix(spec["suffix"])
    label = label.replace("seq_oral_geni___oral_oral", "oral-genital -> oral-oral")
    label = label.replace("seq_oral_oral___oral_genital", "oral-oral -> oral-genital")
    label = label.replace("_active", " active")
    label = label.replace("_passive", " passive")
    label = label.replace("_", " ")
    label = label.replace("Contact", "contact")
    label = label.replace("Zone", "zone")
    return label


def metric_features(df: pd.DataFrame, metric: str) -> list[str]:
    suffix = METRIC_SPECS[metric]["suffix"]
    features = [
        column
        for column in df.columns
        if column.endswith(suffix) and column != "complete_feature_count"
    ]
    return sorted(
        features,
        key=lambda feature: (
            CATEGORY_ORDER.index(behavior_category(feature)),
            label_feature(feature).lower(),
        ),
    )


def normalized_summary(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = df.loc[
        df["stress_aligned_night"].isin(PHASE_NIGHTS)
    ].copy()
    pre = selected.loc[selected["stress_aligned_night"].eq(-1), features].apply(
        pd.to_numeric, errors="coerce"
    )
    denominator = pre.mean(axis=0).replace(0, np.nan)

    long_frames = []
    summary_rows = []
    for night, label in PHASE_NIGHTS.items():
        raw = selected.loc[selected["stress_aligned_night"].eq(night), features].apply(
            pd.to_numeric, errors="coerce"
        )
        normalized = raw.div(denominator, axis=1)
        for feature in features:
            values = normalized[feature].dropna()
            raw_values = raw[feature].dropna()
            category = behavior_category(feature)
            long_frames.append(
                pd.DataFrame(
                    {
                        "stress_aligned_night": night,
                        "phase_label": label,
                        "feature": feature,
                        "feature_label": label_feature(feature),
                        "category": category,
                        "normalized_value": normalized[feature],
                        "raw_value": raw[feature],
                    }
                )
            )
            summary_rows.append(
                {
                    "stress_aligned_night": night,
                    "phase_label": label,
                    "feature": feature,
                    "feature_label": label_feature(feature),
                    "category": category,
                    "baseline_denominator": denominator[feature],
                    "n_mice": int(values.count()),
                    "raw_mean": float(raw_values.mean()),
                    "raw_sd": float(raw_values.std(ddof=1)),
                    "raw_sem": float(raw_values.sem(ddof=1)),
                    "normalized_mean": float(values.mean()),
                    "normalized_sd": float(values.std(ddof=1)),
                    "normalized_sem": float(values.sem(ddof=1)),
                }
            )
    return pd.DataFrame(summary_rows), pd.concat(long_frames, ignore_index=True)


def add_category_bands(ax: plt.Axes, features: list[str]) -> None:
    y0, y1 = ax.get_ylim()
    span_y = y0 - (y1 - y0) * 0.05
    start = 0
    current = behavior_category(features[0])
    for index, feature in enumerate(features + ["__sentinel__"]):
        category = behavior_category(feature) if feature != "__sentinel__" else None
        if category == current:
            continue
        end = index - 1
        color = CATEGORY_COLORS[current]
        ax.hlines(span_y, start - 0.4, end + 0.4, color=color, linewidth=2.2, clip_on=False)
        start = index
        current = category


def transform_values(values: pd.Series | np.ndarray, axis_scale: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if axis_scale == "linear":
        return array
    if axis_scale == "log10p":
        return np.log10(np.clip(array, 0, None) + 1.0)
    raise ValueError(f"Unsupported axis scale: {axis_scale}")


def transformed_error(
    mean_values: pd.Series,
    sem_values: pd.Series,
    axis_scale: str,
) -> pd.DataFrame | np.ndarray:
    mean = mean_values.to_numpy(dtype=float)
    sem = sem_values.fillna(0).to_numpy(dtype=float)
    if axis_scale == "linear":
        return sem
    if axis_scale == "log10p":
        lower_raw = np.clip(mean - sem, 0, None)
        upper_raw = np.clip(mean + sem, 0, None)
        center = transform_values(mean, axis_scale)
        lower = center - transform_values(lower_raw, axis_scale)
        upper = transform_values(upper_raw, axis_scale) - center
        return np.vstack([lower, upper])
    raise ValueError(f"Unsupported axis scale: {axis_scale}")


def axis_ylabel(metric: str, scale: str, axis_scale: str) -> str:
    spec = METRIC_SPECS[metric]
    base = spec["ylabel"] if scale == "normalized" else spec["raw_ylabel"]
    if axis_scale == "log10p":
        return f"log10(1 + {base})"
    return base


def axis_title(metric: str, scale: str, axis_scale: str) -> str:
    spec = METRIC_SPECS[metric]
    title = spec["title"] if scale == "normalized" else spec["raw_title"]
    return f"{title}{AXIS_SCALE_SPECS[axis_scale]['title_suffix']}"


def figure_title(phase_label: str, metric: str, scale: str, axis_scale: str, treatment: str) -> str:
    return f"{TREATMENT_SPECS[treatment]['title_prefix']}{phase_label}: {axis_title(metric, scale, axis_scale)}"


def plot_profile(
    summary: pd.DataFrame,
    long_values: pd.DataFrame,
    features: list[str],
    night: int,
    metric: str,
    scale: str,
    axis_scale: str,
    treatment: str,
    output_dir: Path,
) -> None:
    spec = METRIC_SPECS[metric]
    scale_spec = SCALE_SPECS[scale]
    axis_spec = AXIS_SCALE_SPECS[axis_scale]
    treatment_spec = TREATMENT_SPECS[treatment]
    phase_summary = summary.loc[summary["stress_aligned_night"].eq(night)].set_index("feature").loc[features]
    phase_values = long_values.loc[long_values["stress_aligned_night"].eq(night)]
    x = np.arange(len(features))
    colors = [CATEGORY_COLORS[behavior_category(feature)] for feature in features]

    fig, ax = plt.subplots(figsize=(18, 5.4))
    ax.bar(
        x,
        transform_values(phase_summary[scale_spec["mean"]], axis_scale),
        color=colors,
        edgecolor="black",
        linewidth=0.25,
        width=0.76,
        zorder=2,
    )
    ax.errorbar(
        x,
        transform_values(phase_summary[scale_spec["mean"]], axis_scale),
        yerr=transformed_error(phase_summary[scale_spec["mean"]], phase_summary[scale_spec["sem"]], axis_scale),
        fmt="none",
        ecolor="black",
        elinewidth=0.7,
        capsize=1.8,
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
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=5,
            color="black",
            alpha=0.18,
            linewidth=0,
            zorder=4,
        )

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
    ax.set_title(figure_title(f"{PHASE_NIGHTS[night]} behavioral profile", metric, scale, axis_scale, treatment))
    ax.set_xticks(x)
    ax.set_xticklabels([label_feature(feature) for feature in features], rotation=58, ha="right")
    add_category_bands(ax, features)
    legend_handles = [
        Patch(facecolor=CATEGORY_COLORS[category], edgecolor="black", linewidth=0.25, label=category)
        for category in CATEGORY_ORDER
        if category in set(phase_summary["category"])
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=5,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.38, top=0.82)
    value_prefix = f"{scale_spec['stem_prefix']}{axis_spec['stem_prefix']}"
    stem = (
        f"behavior_profile_{treatment_spec['stem_prefix']}pre_stress_{value_prefix}{spec['stem']}"
        if night == -1
        else f"behavior_profile_{treatment_spec['stem_prefix']}post_stress_{value_prefix}{spec['stem']}"
    )
    save_bundle(fig, output_dir, stem)


def plot_combined(
    summary: pd.DataFrame,
    long_values: pd.DataFrame,
    features: list[str],
    metric: str,
    scale: str,
    axis_scale: str,
    treatment: str,
    output_dir: Path,
) -> None:
    spec = METRIC_SPECS[metric]
    scale_spec = SCALE_SPECS[scale]
    axis_spec = AXIS_SCALE_SPECS[axis_scale]
    treatment_spec = TREATMENT_SPECS[treatment]
    fig, axes = plt.subplots(2, 1, figsize=(18, 9.2), sharex=True, sharey=True)
    max_upper = 0.0
    for ax, night in zip(axes, [-1, 1]):
        phase_summary = summary.loc[summary["stress_aligned_night"].eq(night)].set_index("feature").loc[features]
        phase_values = long_values.loc[long_values["stress_aligned_night"].eq(night)]
        x = np.arange(len(features))
        colors = [CATEGORY_COLORS[behavior_category(feature)] for feature in features]
        ax.bar(
            x,
            transform_values(phase_summary[scale_spec["mean"]], axis_scale),
            color=colors,
            edgecolor="black",
            linewidth=0.25,
            width=0.76,
            zorder=2,
        )
        ax.errorbar(
            x,
            transform_values(phase_summary[scale_spec["mean"]], axis_scale),
            yerr=transformed_error(phase_summary[scale_spec["mean"]], phase_summary[scale_spec["sem"]], axis_scale),
            fmt="none",
            ecolor="black",
            elinewidth=0.7,
            capsize=1.8,
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
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=4.5,
                color="black",
                alpha=0.16,
                linewidth=0,
                zorder=4,
            )
        upper = float(
            np.nanmax(
                transform_values(
                    phase_summary[scale_spec["mean"]].to_numpy(dtype=float)
                    + phase_summary[scale_spec["sem"]].fillna(0).to_numpy(dtype=float),
                    axis_scale,
                )
            )
        )
        max_upper = max(max_upper, upper)
        if scale == "normalized":
            ax.axhline(transform_values([1], axis_scale)[0], color="#444444", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_ylabel(axis_ylabel(metric, scale, axis_scale))
        ax.set_title(figure_title(PHASE_NIGHTS[night], metric, scale, axis_scale, treatment), loc="left", fontweight="normal")
    axes[-1].set_xticks(np.arange(len(features)))
    axes[-1].set_xticklabels([label_feature(feature) for feature in features], rotation=58, ha="right")
    for ax in axes:
        ax.set_xlim(-0.8, len(features) - 0.2)
        ax.set_ylim(0, max(1.35 if scale == "normalized" and axis_scale == "linear" else 0.01, max_upper * 1.18))
    add_category_bands(axes[-1], features)
    legend_handles = [
        Patch(facecolor=CATEGORY_COLORS[category], edgecolor="black", linewidth=0.25, label=category)
        for category in CATEGORY_ORDER
        if category in {behavior_category(feature) for feature in features}
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=5,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.23, top=0.89, hspace=0.25)
    save_bundle(
        fig,
        output_dir,
        f"behavior_profile_{treatment_spec['stem_prefix']}pre_post_{scale_spec['stem_prefix']}{axis_spec['stem_prefix']}{spec['stem']}",
    )


def save_bundle(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_style()
    run_dir = args.run_dir
    output_dir = args.output_dir or PROJECT_ROOT / "src" / "visualization" / "output" / "pre_post_behavior_profiles"
    mouse_night_path = run_dir / "mouse_night_analysis.csv"
    df = pd.read_csv(mouse_night_path)
    if args.eligibility_column not in df.columns:
        raise ValueError(f"Missing eligibility column: {args.eligibility_column}")
    df = df.loc[df[args.eligibility_column].fillna(False).astype(bool)].copy()
    if args.treatment != "all":
        df = df.loc[df["treatment"].eq(args.treatment)].copy()
        if df.empty:
            raise ValueError(f"No selected rows for treatment: {args.treatment}")
    selected = df.loc[df["stress_aligned_night"].isin(PHASE_NIGHTS)]
    phase_counts = selected.groupby("stress_aligned_night")["mouse_id"].count()
    if any(phase_counts.get(night, 0) == 0 for night in PHASE_NIGHTS):
        raise ValueError("Both stress_aligned_night -1 and +1 must contain selected rows.")

    metrics = list(METRIC_SPECS) if args.metric == "all" else [args.metric]
    scales = list(SCALE_SPECS) if args.scale == "all" else [args.scale]
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        features = metric_features(df, metric)
        if not features:
            raise ValueError(f"No features found for metric: {metric}")
        spec = METRIC_SPECS[metric]
        summary, long_values = normalized_summary(selected, features)
        treatment_prefix = TREATMENT_SPECS[args.treatment]["stem_prefix"]
        summary.to_csv(
            output_dir / f"behavior_profile_{treatment_prefix}pre_post_{spec['stem']}_summary.csv",
            index=False,
        )
        long_values.to_csv(
            output_dir / f"behavior_profile_{treatment_prefix}pre_post_{spec['stem']}_mouse_values.csv",
            index=False,
        )
        for scale in scales:
            for night in PHASE_NIGHTS:
                plot_profile(summary, long_values, features, night, metric, scale, args.axis_scale, args.treatment, output_dir)
            plot_combined(summary, long_values, features, metric, scale, args.axis_scale, args.treatment, output_dir)
        print(f"{metric}: {len(features)} features ({', '.join(scales)}, {args.axis_scale}, {args.treatment})")
    print(f"Wrote {output_dir}")
    print(phase_counts.to_string())


if __name__ == "__main__":
    main()
