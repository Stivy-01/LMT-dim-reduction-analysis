from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_ORDER = ["16p", "CD del", "wt"]
SOURCE_LABELS = {"16p": "16p", "CD del": "CD del", "wt": "WT"}
SOURCE_COLORS = {"16p": "#7B2CBF", "CD del": "#2A9D8F", "wt": "#4C78A8"}
PHASE_MARKERS = {"baseline": "o", "post_stress": "s"}
NIGHT_X = "stress_aligned_night"

BASELINE_DELTA_METRICS = [
    "mean_identity_composite",
    "mean_id_1",
    "mean_id_2",
    "dispersion",
    "pairwise_distance",
    "stressed_control_separation",
    "active_passive_imbalance",
]

RESPONSE_METRICS = {
    "whole_group_shift": {
        "label": "Whole-group shift",
        "description": "Mean post-stress distance from cage baseline centroid",
        "higher": "larger displacement",
    },
    "identity_shift": {
        "label": "Identity-domain shift",
        "description": "Post minus baseline mean identity composite",
        "higher": "more positive identity shift",
    },
    "dispersion_change": {
        "label": "Dispersion change",
        "description": "Post minus baseline within-cage dispersion",
        "higher": "less cohesive / more spread",
    },
    "pairwise_change": {
        "label": "Pairwise distance change",
        "description": "Post minus baseline mean pairwise mouse distance",
        "higher": "individuals farther apart in behavior space",
    },
    "stressed_control_separation_change": {
        "label": "Stress-control separation change",
        "description": "Post minus baseline distance between stressed and control centroids",
        "higher": "more separation between treatments",
    },
    "control_response": {
        "label": "Control response",
        "description": "Post-stress control centroid distance from control baseline",
        "higher": "larger control-mouse shift",
    },
    "synchronization_similarity": {
        "label": "Trajectory synchronization",
        "description": "Mean cosine similarity of mouse night-to-night displacement",
        "higher": "more parallel individual shifts",
    },
}


@dataclass(frozen=True)
class BootstrapSummary:
    n: int
    mean: float
    ci_low: float
    ci_high: float


def _figure_bundle(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(figures_dir / f"{stem}.{suffix}", bbox_inches="tight", dpi=220)


def _bootstrap_mean(values: pd.Series, seed: int = 20240612, samples: int = 5000) -> BootstrapSummary:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return BootstrapSummary(0, np.nan, np.nan, np.nan)
    if len(clean) == 1:
        value = float(clean[0])
        return BootstrapSummary(1, value, value, value)
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(samples, len(clean)), replace=True).mean(axis=1)
    return BootstrapSummary(
        n=int(len(clean)),
        mean=float(clean.mean()),
        ci_low=float(np.percentile(draws, 2.5)),
        ci_high=float(np.percentile(draws, 97.5)),
    )


def _load_group_data(run_dir: Path) -> pd.DataFrame:
    group_path = run_dir / "group_night_analysis.csv"
    metadata_path = run_dir / "metadata_enriched.csv"
    if not group_path.exists():
        raise FileNotFoundError(f"Missing group_night_analysis.csv: {group_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata_enriched.csv: {metadata_path}")

    group = pd.read_csv(group_path)
    metadata = pd.read_csv(metadata_path)
    cage_source = (
        metadata[["cage_id", "source_sheet"]]
        .dropna()
        .drop_duplicates(subset=["cage_id"])
    )
    out = group.merge(cage_source, on="cage_id", how="left", validate="many_to_one")
    out = out.loc[out["source_sheet"].isin(SOURCE_ORDER)].copy()
    out["source_label"] = out["source_sheet"].map(SOURCE_LABELS)
    out["night_date"] = pd.to_datetime(out["night_date"], errors="coerce")
    return out.sort_values(["source_sheet", "cage_id", "night_date"]).reset_index(drop=True)


def _paired_stress_cages(group: pd.DataFrame) -> list[str]:
    keep = []
    for cage_id, cage in group.groupby("cage_id", sort=False):
        phases = set(cage["phase"].dropna())
        if {"baseline", "post_stress"}.issubset(phases):
            keep.append(cage_id)
    return keep


def add_baseline_deltas(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    for metric in BASELINE_DELTA_METRICS:
        if metric not in out.columns:
            continue
        baseline = (
            out.loc[out["phase"].eq("baseline")]
            .groupby("cage_id")[metric]
            .mean()
        )
        out[f"{metric}_baseline_mean"] = out["cage_id"].map(baseline)
        out[f"{metric}_delta"] = out[metric] - out[f"{metric}_baseline_mean"]
    return out


def build_response_summary(group: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paired_cages = set(_paired_stress_cages(group))
    for cage_id, cage in group.loc[group["cage_id"].isin(paired_cages)].groupby("cage_id", sort=False):
        source_sheet = str(cage["source_sheet"].iloc[0])
        baseline = cage.loc[cage["phase"].eq("baseline")]
        post = cage.loc[cage["phase"].eq("post_stress")]
        if baseline.empty or post.empty:
            continue
        rows.append(
            {
                "source_sheet": source_sheet,
                "source_label": SOURCE_LABELS.get(source_sheet, source_sheet),
                "cage_id": cage_id,
                "n_baseline_nights": int(len(baseline)),
                "n_post_stress_nights": int(len(post)),
                "whole_group_shift": float(post["baseline_centroid_distance"].mean()),
                "identity_shift": float(post["mean_identity_composite"].mean() - baseline["mean_identity_composite"].mean()),
                "dispersion_change": float(post["dispersion"].mean() - baseline["dispersion"].mean()),
                "pairwise_change": float(post["pairwise_distance"].mean() - baseline["pairwise_distance"].mean()),
                "stressed_control_separation_change": float(
                    post["stressed_control_separation"].mean()
                    - baseline["stressed_control_separation"].mean()
                ),
                "control_response": float(post["control_response"].mean()),
                "synchronization_similarity": float(post["synchronization_similarity"].mean()),
                "active_passive_imbalance_change": float(
                    post["active_passive_imbalance"].mean()
                    - baseline["active_passive_imbalance"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_sheet", "cage_id"]).reset_index(drop=True)


def summarize_response_by_source(response: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_sheet in SOURCE_ORDER:
        subset = response.loc[response["source_sheet"].eq(source_sheet)]
        for metric in RESPONSE_METRICS:
            summary = _bootstrap_mean(subset[metric], seed=20240612 + len(rows))
            rows.append(
                {
                    "source_sheet": source_sheet,
                    "source_label": SOURCE_LABELS[source_sheet],
                    "metric": metric,
                    "metric_label": RESPONSE_METRICS[metric]["label"],
                    "n_cages": summary.n,
                    "mean": summary.mean,
                    "ci_low": summary.ci_low,
                    "ci_high": summary.ci_high,
                    "interpretation_higher": RESPONSE_METRICS[metric]["higher"],
                }
            )
    return pd.DataFrame(rows)


def figure_group_trajectories(group: pd.DataFrame, figures_dir: Path) -> None:
    paired = group.loc[group["cage_id"].isin(_paired_stress_cages(group))].copy()
    x_col = NIGHT_X if NIGHT_X in paired.columns else "relative_night"
    panels = [
        ("baseline_centroid_distance", "Whole-group displacement"),
        ("dispersion_delta", "Within-cage dispersion change"),
        ("stressed_control_separation_delta", "Stressed-control separation change"),
        ("control_response", "Control response from baseline"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, panels):
        ax.axhline(0, color="#111827", linewidth=0.7)
        for source_sheet in SOURCE_ORDER:
            source = paired.loc[paired["source_sheet"].eq(source_sheet)]
            if source.empty:
                continue
            color = SOURCE_COLORS[source_sheet]
            for _, cage in source.groupby("cage_id", sort=False):
                ordered = cage.sort_values(x_col)
                cage_id = str(ordered["cage_id"].iloc[0])
                is_outlier = cage_id == "wt_10132"
                ax.plot(
                    ordered[x_col],
                    ordered[metric],
                    color=color,
                    alpha=0.85 if is_outlier else 0.28,
                    linewidth=2.2 if is_outlier else 1.0,
                )
                if is_outlier and len(ordered):
                    last = ordered.iloc[-1]
                    ax.annotate(
                        "wt_10132",
                        xy=(last[x_col], last[metric]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color="#111827",
                    )
            summary = (
                source.groupby(x_col)[metric]
                .agg(["mean", "sem", "count"])
                .reset_index()
            )
            ax.errorbar(
                summary[x_col],
                summary["mean"],
                yerr=summary["sem"].fillna(0.0),
                marker="o",
                color=color,
                linewidth=2.2,
                capsize=2,
                label=f"{SOURCE_LABELS[source_sheet]} (n={source['cage_id'].nunique()})",
            )
        ax.set_title(title)
        if x_col == NIGHT_X:
            ax.axvspan(-0.5, 0.5, color="#FDE68A", alpha=0.22)
            ax.set_xticks([-3, -2, -1, 1, 2, 3])
            ax.set_xticklabels(["-3", "-2", "-1", "+1", "+2", "+3"])
            ax.set_xlabel("Night relative to stress event")
        else:
            ax.set_xlabel("Relative night")
        ax.set_ylabel(metric)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Group-level stress dynamics by experimental line", y=1.01)
    fig.tight_layout()
    _figure_bundle(fig, figures_dir, "figure_12_group_dynamics_trajectories")
    plt.close(fig)


def figure_response_summary(summary: pd.DataFrame, response: pd.DataFrame, figures_dir: Path) -> None:
    metrics = list(RESPONSE_METRICS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), sharey=True)
    y_lookup = {metric: index for index, metric in enumerate(metrics[::-1])}
    finite_limits = []
    for _, row in summary.iterrows():
        finite_limits.extend([row["ci_low"], row["ci_high"], row["mean"]])
    for metric in metrics:
        finite_limits.extend(response[metric].dropna().tolist() if metric in response else [])
    finite = np.asarray([value for value in finite_limits if np.isfinite(value)], dtype=float)
    if finite.size:
        span = float(np.nanmax(finite) - np.nanmin(finite)) or 1.0
        shared_xlim = (float(np.nanmin(finite) - 0.08 * span), float(np.nanmax(finite) + 0.08 * span))
    else:
        shared_xlim = None
    for ax, source_sheet in zip(axes, SOURCE_ORDER):
        color = SOURCE_COLORS[source_sheet]
        source_summary = summary.loc[summary["source_sheet"].eq(source_sheet)]
        source_points = response.loc[response["source_sheet"].eq(source_sheet)]
        ax.axvline(0, color="#111827", linewidth=0.8)
        for _, row in source_summary.iterrows():
            y = y_lookup[row["metric"]]
            ax.errorbar(
                row["mean"],
                y,
                xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=6,
            )
            points = source_points[row["metric"]].dropna().to_numpy(dtype=float)
            if len(points):
                jitter = np.linspace(-0.12, 0.12, len(points)) if len(points) > 1 else np.array([0.0])
                ax.scatter(points, y + jitter, color=color, alpha=0.25, s=24, linewidth=0)
        n_cages = int(source_points["cage_id"].nunique()) if not source_points.empty else 0
        ax.set_title(f"{SOURCE_LABELS[source_sheet]} (n={n_cages} cages)")
        ax.set_xlabel("Cage-level post-stress response")
        if shared_xlim is not None:
            ax.set_xlim(shared_xlim)
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
        ax.set_yticks(list(y_lookup.values()))
        ax.set_yticklabels([RESPONSE_METRICS[metric]["label"] for metric in metrics[::-1]])
    fig.suptitle("Stress response of the quartetto as experimental unit\n95% bootstrap CI across cages; n=2 estimates are descriptive", y=1.03)
    fig.tight_layout()
    _figure_bundle(fig, figures_dir, "figure_13_group_response_summary")
    plt.close(fig)


def figure_response_heatmap(summary: pd.DataFrame, figures_dir: Path) -> None:
    metrics = list(RESPONSE_METRICS.keys())
    matrix = []
    annotations = []
    for metric in metrics:
        row = []
        ann_row = []
        for source_sheet in SOURCE_ORDER:
            match = summary.loc[
                summary["source_sheet"].eq(source_sheet)
                & summary["metric"].eq(metric)
            ]
            value = float(match["mean"].iloc[0]) if len(match) else np.nan
            n = int(match["n_cages"].iloc[0]) if len(match) else 0
            row.append(value)
            ann_row.append(f"{value:+.2f}\nn={n}" if np.isfinite(value) else f"NA\nn={n}")
        matrix.append(row)
        annotations.append(ann_row)
    values = np.asarray(matrix, dtype=float)
    scaled = values.copy()
    for row_index in range(scaled.shape[0]):
        row = scaled[row_index]
        max_abs = np.nanmax(np.abs(row)) if np.isfinite(row).any() else 1.0
        if max_abs > 0:
            scaled[row_index] = row / max_abs
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(scaled, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SOURCE_ORDER)))
    ax.set_xticklabels([SOURCE_LABELS[source] for source in SOURCE_ORDER])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([RESPONSE_METRICS[metric]["label"] for metric in metrics])
    for y in range(len(metrics)):
        for x in range(len(SOURCE_ORDER)):
            ax.text(x, y, annotations[y][x], ha="center", va="center", fontsize=8)
    ax.set_title("Relative group-response signatures by line\nrow-scaled pattern only, not inferential; annotations are raw means")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Row-scaled response direction")
    _figure_bundle(fig, figures_dir, "figure_14_group_response_heatmap")
    plt.close(fig)


def figure_group_identity_space(
    group: pd.DataFrame,
    figures_dir: Path,
    *,
    excluded_cages: set[str] | None = None,
    filename: str = "figure_15_group_identity_space",
) -> None:
    paired = group.loc[group["cage_id"].isin(_paired_stress_cages(group))].copy()
    excluded_cages = excluded_cages or set()
    if excluded_cages:
        paired = paired.loc[~paired["cage_id"].isin(excluded_cages)].copy()
    fig, ax = plt.subplots(figsize=(8, 7))
    labeled_cages = {"wt_10132", "wt_9314"}
    for source_sheet in SOURCE_ORDER:
        source = paired.loc[paired["source_sheet"].eq(source_sheet)]
        color = SOURCE_COLORS[source_sheet]
        for cage_id, cage in source.groupby("cage_id", sort=False):
            centers = cage.groupby("phase")[["mean_id_1", "mean_id_2"]].mean()
            for phase, row in centers.iterrows():
                ax.scatter(
                    row["mean_id_1"],
                    row["mean_id_2"],
                    color=color,
                    marker=PHASE_MARKERS.get(str(phase), "o"),
                    s=60,
                    alpha=0.75,
                    edgecolor="white",
                    linewidth=0.5,
                    label=f"{SOURCE_LABELS[source_sheet]} {phase}",
                )
            if {"baseline", "post_stress"}.issubset(centers.index):
                start = centers.loc["baseline"]
                end = centers.loc["post_stress"]
                ax.annotate(
                    "",
                    xy=(end["mean_id_1"], end["mean_id_2"]),
                    xytext=(start["mean_id_1"], start["mean_id_2"]),
                    arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.55, "lw": 1.4},
                )
                if str(cage_id) in labeled_cages:
                    ax.text(
                        end["mean_id_1"],
                        end["mean_id_2"],
                        cage_id,
                        fontsize=8,
                        color=color,
                        alpha=0.9,
                    )
    ax.axhline(0, color="#CBD5E1", linewidth=0.8)
    ax.axvline(0, color="#CBD5E1", linewidth=0.8)
    ax.set_xlabel("Group centroid ID_1")
    ax.set_ylabel("Group centroid ID_2")
    suffix = (
        f"\nexcluding {', '.join(sorted(excluded_cages))}"
        if excluded_cages
        else "\nlabels shown only for highlighted WT cages"
    )
    ax.set_title(f"Quartetto centroid shifts in identity-domain space{suffix}")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=7, ncol=2)
    _figure_bundle(fig, figures_dir, filename)
    plt.close(fig)


def build_group_dynamics_figures(run_dir: Path, output_dir: Path | None = None) -> Path:
    group = _load_group_data(run_dir)
    group = add_baseline_deltas(group)
    paired_cages = _paired_stress_cages(group)
    response = build_response_summary(group)
    source_summary = summarize_response_by_source(response)

    figures_dir = Path(output_dir) if output_dir else (
        Path(__file__).resolve().parents[3] / "src" / "visualization" / "output" / "group_dynamics"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    group.to_csv(run_dir / "group_lineage_night_metrics.csv", index=False)
    response.to_csv(run_dir / "group_lineage_response_summary.csv", index=False)
    source_summary.to_csv(run_dir / "group_lineage_response_by_source.csv", index=False)

    figure_group_trajectories(group, figures_dir)
    figure_response_summary(source_summary, response, figures_dir)
    figure_response_heatmap(source_summary, figures_dir)
    figure_group_identity_space(group, figures_dir)
    figure_group_identity_space(
        group,
        figures_dir,
        excluded_cages={"wt_10132"},
        filename="figure_16_group_identity_space_without_wt10132",
    )

    counts = (
        response.groupby("source_sheet")["cage_id"]
        .nunique()
        .reindex(SOURCE_ORDER)
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    print(f"Paired stress-response cages: {counts}")
    print(f"Paired cage IDs: {paired_cages}")
    return figures_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exploratory group-level dynamics figures from a thesis analysis run."
    )
    parser.add_argument("run_dir", type=Path, help="Path to an existing thesis analysis run directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    figures_dir = build_group_dynamics_figures(args.run_dir, args.output_dir)
    print(f"Wrote group-dynamics figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
