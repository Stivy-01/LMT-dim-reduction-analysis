from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GENOTYPE_SOURCES = ("16p", "CD del")
PHASE_ORDER = ("baseline", "post_stress")
SOURCE_LABELS = {"16p": "16p TG - WT", "CD del": "CD del TG - WT"}
PHASE_LABELS = {"baseline": "Baseline", "post_stress": "Post-stress"}

KEY_FEATURES = [
    "Huddling_count",
    "Huddling_mean_duration",
    "Contact_active_count",
    "Contact_mean_duration",
    "Side_by_side_Contact_active_count",
    "Oral_oral_Contact_active_count",
    "Oral_genital_Contact_active_count",
    "Group2_active_count",
    "Group_3_make_count",
    "Group_4_make_count",
    "isolated_count",
    "isolated_mean_duration",
    "Move_isolated_count",
    "Move_isolated_mean_duration",
    "Stop_isolated_count",
    "Stop_isolated_mean_duration",
    "Rear_isolated_count",
    "Rearing_count",
    "Rear_at_periphery_count",
    "Rear_in_centerWindow_count",
    "Center_Zone_count",
    "Periphery_Zone_count",
    "WallJump_count",
    "Train2_active_count",
    "Stop_count",
    "Stop_mean_duration",
    "SAP_count",
    "SAP_mean_duration",
]

COMPOSITES = {
    "Social proximity counts": [
        "Huddling_count",
        "Contact_active_count",
        "Contact_passive_count",
        "Side_by_side_Contact_active_count",
        "Side_by_side_Contact_passive_count",
        "Oral_oral_Contact_active_count",
        "Oral_genital_Contact_active_count",
        "Group2_active_count",
        "Group2_passive_count",
    ],
    "Social bout duration": [
        "Huddling_mean_duration",
        "Contact_mean_duration",
        "Side_by_side_Contact_mean_duration",
        "Oral_oral_Contact_mean_duration",
        "Oral_genital_Contact_mean_duration",
        "Group2_mean_duration",
    ],
    "Isolation counts": [
        "isolated_count",
        "Move_isolated_count",
        "Stop_isolated_count",
        "Rear_isolated_count",
    ],
    "Isolation duration": [
        "isolated_mean_duration",
        "Move_isolated_mean_duration",
        "Stop_isolated_mean_duration",
        "Rear_isolated_mean_duration",
    ],
    "Exploration / rearing": [
        "Rearing_count",
        "Rear_at_periphery_count",
        "Rear_in_centerWindow_count",
        "Center_Zone_count",
        "Periphery_Zone_count",
        "WallJump_count",
        "Train2_active_count",
        "Train2_passive_count",
    ],
    "Stop / fragmentation": [
        "Stop_count",
        "SAP_count",
        "Center_Zone_count",
        "Periphery_Zone_count",
    ],
}


@dataclass(frozen=True)
class BootstrapSummary:
    n: int
    mean: float
    ci_low: float
    ci_high: float


def _load_complete_features(run_dir: Path) -> list[str]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(manifest["complete_features"])


def _clean_label(feature: str) -> str:
    return (
        feature.replace("_active", "")
        .replace("_passive", "")
        .replace("_mean_duration", " duration")
        .replace("_std_duration", " variability")
        .replace("_count", "")
        .replace("_", " ")
    )


def _figure_bundle(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(figures_dir / f"{stem}.{suffix}", bbox_inches="tight", dpi=220)


def _bootstrap_mean(values: pd.Series, seed: int = 20240612, samples: int = 5000) -> BootstrapSummary:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return BootstrapSummary(0, np.nan, np.nan, np.nan)
    if len(clean) == 1:
        return BootstrapSummary(1, float(clean[0]), float(clean[0]), float(clean[0]))
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(samples, len(clean)), replace=True).mean(axis=1)
    return BootstrapSummary(
        n=int(len(clean)),
        mean=float(clean.mean()),
        ci_low=float(np.percentile(draws, 2.5)),
        ci_high=float(np.percentile(draws, 97.5)),
    )


def _prepare_effect_frame(run_dir: Path, complete_features: list[str]) -> pd.DataFrame:
    qc_path = run_dir / "quality_control.csv"
    if not qc_path.exists():
        raise FileNotFoundError(f"Missing quality_control.csv: {qc_path}")
    frame = pd.read_csv(qc_path)
    frame = frame.loc[
        frame["effect_eligible"].astype(bool)
        & frame["source_sheet"].isin(GENOTYPE_SOURCES)
        & frame["phase"].isin(PHASE_ORDER)
    ].copy()
    for feature in complete_features:
        frame[feature] = np.log1p(pd.to_numeric(frame[feature], errors="coerce"))
    return frame


def build_paired_feature_deltas(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["source_sheet", "cage_id", "interval_start", "phase", "treatment"]
    for keys, group in frame.groupby(group_cols, dropna=False, sort=False):
        if not {"TG", "WT"}.issubset(set(group["genotype"].dropna())):
            continue
        tg = group.loc[group["genotype"].eq("TG")].iloc[0]
        wt = group.loc[group["genotype"].eq("WT")].iloc[0]
        base = dict(zip(group_cols, keys))
        base.update(
            {
                "tg_mouse_id": int(tg["mouse_id"]),
                "wt_mouse_id": int(wt["mouse_id"]),
            }
        )
        for feature in features:
            rows.append(
                {
                    **base,
                    "feature": feature,
                    "tg_minus_wt": float(tg[feature] - wt[feature]),
                }
            )
    return pd.DataFrame(rows)


def build_axis_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    responses = [col for col in ["id_1", "id_2", "identity_composite", "pca_1", "pca_2", "pca_3"] if col in frame.columns]
    rows: list[dict[str, Any]] = []
    group_cols = ["source_sheet", "cage_id", "interval_start", "phase", "treatment"]
    for keys, group in frame.groupby(group_cols, dropna=False, sort=False):
        if not {"TG", "WT"}.issubset(set(group["genotype"].dropna())):
            continue
        tg = group.loc[group["genotype"].eq("TG")].iloc[0]
        wt = group.loc[group["genotype"].eq("WT")].iloc[0]
        base = dict(zip(group_cols, keys))
        for response in responses:
            rows.append(
                {
                    **base,
                    "response": response,
                    "tg_minus_wt": float(tg[response] - wt[response]),
                }
            )
    return pd.DataFrame(rows)


def build_composite_deltas(feature_deltas: pd.DataFrame, complete_features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    id_cols = ["source_sheet", "cage_id", "interval_start", "phase", "treatment", "tg_mouse_id", "wt_mouse_id"]
    for composite, raw_features in COMPOSITES.items():
        features = [feature for feature in raw_features if feature in complete_features]
        if not features:
            continue
        subset = feature_deltas.loc[feature_deltas["feature"].isin(features)]
        for keys, group in subset.groupby(id_cols, dropna=False, sort=False):
            rows.append(
                {
                    **dict(zip(id_cols, keys)),
                    "composite": composite,
                    "n_features": len(features),
                    "tg_minus_wt": float(group["tg_minus_wt"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_mean_delta(frame: pd.DataFrame, group_cols: list[str], value_col: str = "tg_minus_wt") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=False):
        summary = _bootstrap_mean(group[value_col], seed=20240612 + len(rows))
        rows.append(
            {
                **dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))),
                "n": summary.n,
                "mean": summary.mean,
                "ci_low": summary.ci_low,
                "ci_high": summary.ci_high,
            }
        )
    return pd.DataFrame(rows)


def figure_axis_deltas(axis_summary: pd.DataFrame, axis_deltas: pd.DataFrame, figures_dir: Path) -> None:
    responses = ["identity_composite", "id_1", "id_2"]
    fig, axes = plt.subplots(1, len(responses), figsize=(13.5, 4.6), sharey=False)
    colors = {"baseline": "#64748B", "post_stress": "#D1495B"}
    rng = np.random.default_rng(12)
    for ax, response in zip(axes, responses):
        sub_summary = axis_summary.loc[axis_summary["response"].eq(response)]
        sub_points = axis_deltas.loc[axis_deltas["response"].eq(response)]
        x_lookup = {"16p": 0, "CD del": 1}
        for phase in PHASE_ORDER:
            offset = -0.14 if phase == "baseline" else 0.14
            phase_summary = sub_summary.loc[sub_summary["phase"].eq(phase)]
            for _, row in phase_summary.iterrows():
                x = x_lookup[row["source_sheet"]] + offset
                ax.errorbar(
                    x,
                    row["mean"],
                    yerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
                    fmt="o",
                    color=colors[phase],
                    capsize=3,
                    markersize=6,
                    label=PHASE_LABELS[phase],
                )
            phase_points = sub_points.loc[sub_points["phase"].eq(phase)]
            for source, group in phase_points.groupby("source_sheet"):
                x = x_lookup[source] + offset + rng.normal(0, 0.025, len(group))
                ax.scatter(x, group["tg_minus_wt"], s=14, alpha=0.25, color=colors[phase], linewidth=0)
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["16p", "CD del"])
        ax.set_title(response)
        ax.set_ylabel("TG - WT\n(same cage/night/treatment)")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)
    fig.suptitle("Opposite genotype directions in identity/projection space", y=1.02)
    _figure_bundle(fig, figures_dir, "figure_08_genotype_axis_deltas")
    plt.close(fig)


def figure_feature_heatmap(feature_summary: pd.DataFrame, figures_dir: Path) -> None:
    columns = [(source, phase) for source in GENOTYPE_SOURCES for phase in PHASE_ORDER]
    available_features = [feature for feature in KEY_FEATURES if feature in set(feature_summary["feature"])]
    matrix = []
    for feature in available_features:
        row = []
        for source, phase in columns:
            value = feature_summary.loc[
                feature_summary["source_sheet"].eq(source)
                & feature_summary["phase"].eq(phase)
                & feature_summary["feature"].eq(feature),
                "mean",
            ]
            row.append(float(value.iloc[0]) if len(value) else np.nan)
        matrix.append(row)
    values = np.asarray(matrix, dtype=float)
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    limit = max(limit, 0.25)
    fig, ax = plt.subplots(figsize=(9.5, max(8, len(available_features) * 0.32)))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(np.arange(len(available_features)))
    ax.set_yticklabels([_clean_label(feature) for feature in available_features], fontsize=8)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(
        [f"{source}\n{PHASE_LABELS[phase]}" for source, phase in columns],
        fontsize=9,
    )
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isfinite(value):
                ax.text(
                    col_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if abs(value) > limit * 0.55 else "#111827",
                )
    ax.set_title("Behavioral signature: TG - WT paired log1p deltas")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Red: TG > WT     Blue: TG < WT", rotation=90)
    _figure_bundle(fig, figures_dir, "figure_09_genotype_behavior_heatmap")
    plt.close(fig)


def figure_composite_deltas(composite_summary: pd.DataFrame, composite_deltas: pd.DataFrame, figures_dir: Path) -> None:
    composites = list(COMPOSITES.keys())
    colors = {"baseline": "#64748B", "post_stress": "#D1495B"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True, sharex=True)
    y_lookup = {name: index for index, name in enumerate(composites)}
    rng = np.random.default_rng(55)
    for ax, source in zip(axes, GENOTYPE_SOURCES):
        ax.axvline(0, color="#111827", linewidth=0.8)
        for phase in PHASE_ORDER:
            offset = -0.13 if phase == "baseline" else 0.13
            summary = composite_summary.loc[
                composite_summary["source_sheet"].eq(source)
                & composite_summary["phase"].eq(phase)
            ]
            points = composite_deltas.loc[
                composite_deltas["source_sheet"].eq(source)
                & composite_deltas["phase"].eq(phase)
            ]
            for _, row in summary.iterrows():
                y = y_lookup[row["composite"]] + offset
                ax.errorbar(
                    row["mean"],
                    y,
                    xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
                    fmt="o",
                    color=colors[phase],
                    capsize=3,
                    label=PHASE_LABELS[phase],
                )
            for composite, group in points.groupby("composite"):
                y = y_lookup[composite] + offset + rng.normal(0, 0.025, len(group))
                ax.scatter(group["tg_minus_wt"], y, s=12, alpha=0.18, color=colors[phase], linewidth=0)
        ax.set_title(SOURCE_LABELS[source])
        ax.set_xlabel("TG - WT mean log1p delta")
        ax.set_yticks(list(y_lookup.values()))
        ax.set_yticklabels(composites)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)
    fig.suptitle("Genotype-specific behavioral composites", y=1.02)
    _figure_bundle(fig, figures_dir, "figure_10_genotype_behavior_composites")
    plt.close(fig)


def figure_key_feature_bars(feature_summary: pd.DataFrame, figures_dir: Path) -> None:
    features = [
        "Huddling_count",
        "Contact_mean_duration",
        "Move_isolated_count",
        "isolated_mean_duration",
        "Rear_in_centerWindow_count",
        "Center_Zone_count",
        "Stop_count",
        "Stop_mean_duration",
    ]
    features = [feature for feature in features if feature in set(feature_summary["feature"])]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5), sharey=False)
    axes = axes.ravel()
    colors = {"baseline": "#64748B", "post_stress": "#D1495B"}
    for ax, feature in zip(axes, features):
        subset = feature_summary.loc[feature_summary["feature"].eq(feature)]
        x_positions = {("16p", "baseline"): -0.18, ("16p", "post_stress"): 0.18, ("CD del", "baseline"): 0.82, ("CD del", "post_stress"): 1.18}
        for _, row in subset.iterrows():
            x = x_positions[(row["source_sheet"], row["phase"])]
            ax.bar(
                x,
                row["mean"],
                width=0.28,
                color=colors[row["phase"]],
                alpha=0.82,
            )
            ax.errorbar(
                x,
                row["mean"],
                yerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
                fmt="none",
                color="#111827",
                linewidth=0.8,
                capsize=2,
            )
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["16p", "CD del"])
        ax.set_title(_clean_label(feature), fontsize=10)
    for ax in axes[len(features) :]:
        ax.set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=colors[phase], lw=7, label=PHASE_LABELS[phase])
        for phase in PHASE_ORDER
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Key genotype features: paired TG - WT deltas", y=0.995)
    fig.text(0.01, 0.5, "TG - WT log1p delta", rotation=90, va="center")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.91))
    _figure_bundle(fig, figures_dir, "figure_11_genotype_key_features")
    plt.close(fig)


def build_genotype_figures(run_dir: Path, output_dir: Path | None = None) -> Path:
    complete_features = _load_complete_features(run_dir)
    frame = _prepare_effect_frame(run_dir, complete_features)
    figures_dir = Path(output_dir) if output_dir else (
        Path(__file__).resolve().parents[3] / "src" / "visualization" / "output" / "genotype"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    feature_deltas = build_paired_feature_deltas(frame, complete_features)
    axis_deltas = build_axis_deltas(frame)
    composite_deltas = build_composite_deltas(feature_deltas, complete_features)

    feature_summary = summarize_mean_delta(feature_deltas, ["source_sheet", "phase", "feature"])
    axis_summary = summarize_mean_delta(axis_deltas, ["source_sheet", "phase", "response"])
    composite_summary = summarize_mean_delta(composite_deltas, ["source_sheet", "phase", "composite"])

    feature_deltas.to_csv(run_dir / "genotype_feature_paired_deltas.csv", index=False)
    feature_summary.to_csv(run_dir / "genotype_feature_delta_summary.csv", index=False)
    axis_deltas.to_csv(run_dir / "genotype_axis_paired_deltas.csv", index=False)
    axis_summary.to_csv(run_dir / "genotype_axis_delta_summary.csv", index=False)
    composite_deltas.to_csv(run_dir / "genotype_composite_paired_deltas.csv", index=False)
    composite_summary.to_csv(run_dir / "genotype_composite_delta_summary.csv", index=False)

    figure_axis_deltas(axis_summary, axis_deltas, figures_dir)
    figure_feature_heatmap(feature_summary, figures_dir)
    figure_composite_deltas(composite_summary, composite_deltas, figures_dir)
    figure_key_feature_bars(feature_summary, figures_dir)
    return figures_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exploratory genotype figures from a thesis analysis run."
    )
    parser.add_argument("run_dir", type=Path, help="Path to an existing thesis analysis run directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional figure output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    figures_dir = build_genotype_figures(args.run_dir, args.output_dir)
    print(f"Wrote genotype figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
