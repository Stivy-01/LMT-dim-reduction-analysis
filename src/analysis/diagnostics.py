# -*- coding: utf-8 -*-
"""Figure diagnostiche interne (ex BLOCCO 9 di thesis_analysis)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


# ==============================================================================
# BLOCCO 9 — Figure diagnostiche interne (righe ~1399-1853)
# Timeline QC, heatmap missingness, traiettorie, frecce di proiezione,
# forest plot, dispersione gruppi, heatmap feature. (Le figure paper
# stanno in src/visualization/.)
# -> futuro modulo: src/analysis/diagnostics.py
# ==============================================================================
def _figure_bundle(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(figures_dir / f"{stem}.{suffix}", bbox_inches="tight")


def _figure_timeline_qc(frame: pd.DataFrame, figures_dir: Path) -> None:
    summary = (
        frame.assign(cage_label=frame["cage_id"].fillna("unmapped"))
        .groupby(["cage_label", "night_date", "relative_night", "stress_aligned_night"], dropna=False)
        .agg(
            projection_eligible=("projection_eligible", "sum"),
            effect_eligible=("effect_eligible", "sum"),
            sensitivity_eligible=("sensitivity_effect_eligible", "sum"),
            partial=("flag_partial_final_night", "sum"),
            out_of_window=("flag_out_of_window", "sum"),
            date_anomaly=("flag_date_anomaly", "sum"),
            out_of_protocol=("flag_out_of_protocol_window", "sum"),
            first_recorded=("flag_first_recorded_night", "sum"),
        )
        .reset_index()
    )
    summary = summary.loc[summary["stress_aligned_night"].isin([-3, -2, -1, 1, 2, 3])].copy()
    summary = summary.sort_values(["cage_label", "night_date"])
    summary["status"] = "excluded"
    summary.loc[summary["projection_eligible"].gt(0), "status"] = "projection only"
    summary.loc[summary["sensitivity_eligible"].gt(0), "status"] = "sensitivity"
    summary.loc[summary["effect_eligible"].gt(0), "status"] = "primary"
    summary.loc[
        summary[["partial", "out_of_window", "date_anomaly", "out_of_protocol", "first_recorded"]]
        .sum(axis=1)
        .gt(0),
        "status",
    ] = "excluded / flagged"
    cage_order = list(dict.fromkeys(summary["cage_label"].astype(str)))
    status_colors = {
        "primary": "#2A9D8F",
        "sensitivity": "#E9C46A",
        "projection only": "#4C78A8",
        "excluded": "#B8C2CC",
        "excluded / flagged": "#E45756",
    }
    fig, ax = plt.subplots(figsize=(11, max(6, len(cage_order) * 0.48)))
    y_lookup = {cage: index for index, cage in enumerate(cage_order)}
    for status, group in summary.groupby("status", sort=False):
        ax.scatter(
            group["stress_aligned_night"],
            group["cage_label"].map(y_lookup),
            s=85,
            marker="s",
            color=status_colors[status],
            label=status,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axvspan(-0.5, 0.5, color="#FDE68A", alpha=0.35)
    ax.axvline(0, color="#92400E", linewidth=1.0, linestyle="--")
    ax.set_yticks(range(len(cage_order)))
    ax.set_yticklabels(cage_order)
    ax.set_xticks([-3, -2, -1, 1, 2, 3])
    ax.set_xticklabels(["-3", "-2", "-1", "+1", "+2", "+3"])
    ax.set_xlim(-3.6, 3.6)
    ax.set_title("Stress-aligned night availability and quality by cage")
    ax.set_xlabel("Night relative to stress event")
    ax.set_ylabel("Cage / project")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _figure_bundle(fig, figures_dir, "figure_01_timeline_qc")
    plt.close(fig)


def _clean_feature_label(feature: str) -> str:
    replacements = {
        "_active": "",
        "_passive": "",
        "_mean_duration": " mean duration",
        "_median_duration": " median duration",
        "_total_duration": " total duration",
        "_std_duration": " variability",
        "_count": "",
        "_": " ",
    }
    label = feature
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label[:55] + "..." if len(label) > 58 else label


def _figure_missingness_heatmap(frame: pd.DataFrame, candidate_features: list[str], figures_dir: Path) -> None:
    scopes = [
        ("projection_eligible", "Usable\nanalysis rows"),
        ("not_projection_eligible", "Excluded /\nflagged rows"),
        ("all", "All\nrows"),
    ]
    temp = frame.copy()
    temp["not_projection_eligible"] = ~temp["projection_eligible"].astype(bool)
    temp["all"] = True
    rows = []
    for feature in candidate_features:
        row = {"feature": feature}
        for mask_col, _ in scopes:
            subset = temp.loc[temp[mask_col].astype(bool), feature]
            row[mask_col] = float(subset.isna().mean()) if len(subset) else np.nan
        row["max_missing"] = max(
            value for key, value in row.items() if key != "feature" and np.isfinite(value)
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    complete_features = summary.loc[summary["all"].eq(0), "feature"].tolist()
    incomplete = summary.loc[summary["max_missing"].gt(0)].sort_values(
        ["projection_eligible", "all"],
        ascending=False,
    )
    top = incomplete.head(28).copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, max(7, len(top) * 0.26)),
        gridspec_kw={"width_ratios": [0.9, 2.2]},
    )
    ax_bar, ax = axes
    counts = [
        len(complete_features),
        len(candidate_features) - len(complete_features),
    ]
    ax_bar.barh(["Used\ncomplete", "Excluded\nmissing"], counts, color=["#2A9D8F", "#E9C46A"])
    for y, value in enumerate(counts):
        ax_bar.text(value + max(counts) * 0.02, y, str(value), va="center", fontsize=11)
    ax_bar.set_title("Feature set")
    ax_bar.set_xlabel("Number of features")
    ax_bar.spines[["top", "right"]].set_visible(False)

    if top.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No missing feature values detected.", ha="center", va="center")
    else:
        matrix = top[[scope[0] for scope in scopes]].to_numpy(dtype=float)
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([_clean_feature_label(feature) for feature in top["feature"]], fontsize=8)
        ax.set_xticks(range(len(scopes)))
        ax.set_xticklabels([label for _, label in scopes], fontsize=9)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if np.isfinite(value):
                    ax.text(
                        col_index,
                        row_index,
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#111827" if value < 0.65 else "white",
                    )
        ax.set_title("Top missing features (not used in primary feature set)")
        fig.colorbar(im, ax=ax, label="Missing fraction", fraction=0.046, pad=0.02)
    fig.suptitle("Feature completeness summary", y=1.01)
    fig.tight_layout()
    _figure_bundle(fig, figures_dir, "figure_02_missingness_heatmap")
    plt.close(fig)


def _figure_trajectories(frame: pd.DataFrame, figures_dir: Path) -> None:
    data = frame.loc[frame["effect_eligible"]].copy()
    baseline_mean = (
        data.loc[data["phase"].eq("baseline")]
        .groupby("mouse_id")["identity_composite"]
        .mean()
    )
    data["identity_change"] = data["identity_composite"] - data["mouse_id"].map(baseline_mean)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"control": "#4C78A8", "stressed": "#E45756"}
    cage_summary = (
        data.groupby(["cage_id", "treatment", "stress_aligned_night"], as_index=False)[
            "identity_change"
        ].mean()
    )
    for (_, treatment), group in cage_summary.groupby(["cage_id", "treatment"], sort=False):
        ax.plot(
            group["stress_aligned_night"],
            group["identity_change"],
            color=colors.get(str(treatment), "#999999"),
            alpha=0.25,
            linewidth=1.2,
        )
    for treatment, group in cage_summary.groupby("treatment", dropna=False):
        summary = (
            group.groupby("stress_aligned_night")["identity_change"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        color = colors.get(str(treatment), "#999999")
        ax.plot(
            summary["stress_aligned_night"],
            summary["mean"],
            marker="o",
            color=color,
            linewidth=2.2,
            label=str(treatment),
        )
        ax.fill_between(
            summary["stress_aligned_night"],
            summary["mean"] - summary["sem"].fillna(0),
            summary["mean"] + summary["sem"].fillna(0),
            color=color,
            alpha=0.12,
        )
    n_cages = int(cage_summary["cage_id"].nunique())
    n_mice = int(data["mouse_id"].nunique())
    ax.set_title(f"Stress-aligned trajectories ({n_cages} cages; {n_mice} mice)")
    ax.set_xlabel("Night relative to stress event")
    ax.set_xticks([-3, -2, -1, 1, 2, 3])
    ax.set_xticklabels(["-3", "-2", "-1", "+1", "+2", "+3"])
    ax.axvspan(-0.5, 0.5, color="#FDE68A", alpha=0.28, label="stress event")
    ax.axhline(0, color="#64748B", linewidth=0.8)
    ax.set_ylabel("Identity composite change from each mouse's baseline mean")
    ax.text(
        0.01,
        0.02,
        "Thin lines: cage-treatment means; thick lines/SEM are across cages, not mouse-night rows",
        transform=ax.transAxes,
        fontsize=8,
        color="#475569",
        ha="left",
        va="bottom",
    )
    ax.legend(frameon=False)
    _figure_bundle(fig, figures_dir, "figure_03_trajectories")
    plt.close(fig)


def _figure_projection_arrows(
    frame: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plot = frame.loc[
        frame["projection_eligible"]
        & frame["phase"].isin(["baseline", "post_stress"])
    ].copy()
    colors = {"control": "#4C78A8", "stressed": "#E45756", "unknown": "#9AA5B1"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, x_col, y_col, title in [
        (axes[0], "pca_1", "pca_2", "PCA baseline to post-stress shift"),
        (axes[1], "id_1", "id_2", "Identity-domain shift"),
    ]:
        if x_col not in plot or y_col not in plot:
            ax.set_visible(False)
            continue
        baseline_rows = plot["phase"].eq("baseline")
        x_mean = float(plot.loc[baseline_rows, x_col].mean())
        y_mean = float(plot.loc[baseline_rows, y_col].mean())
        x_scale = float(plot.loc[baseline_rows, x_col].std(ddof=0)) or 1.0
        y_scale = float(plot.loc[baseline_rows, y_col].std(ddof=0)) or 1.0
        plot[f"{x_col}_display"] = (plot[x_col] - x_mean) / x_scale
        plot[f"{y_col}_display"] = (plot[y_col] - y_mean) / y_scale
        for _, group in plot.groupby("mouse_id", sort=False):
            centers = group.groupby("phase")[
                [f"{x_col}_display", f"{y_col}_display"]
            ].mean()
            treatment = str(group["treatment"].iloc[0])
            color = colors.get(treatment, colors["unknown"])
            if {"baseline", "post_stress"}.issubset(centers.index):
                start = centers.loc["baseline"]
                end = centers.loc["post_stress"]
                ax.annotate(
                    "",
                    xy=(end[f"{x_col}_display"], end[f"{y_col}_display"]),
                    xytext=(start[f"{x_col}_display"], start[f"{y_col}_display"]),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "alpha": 0.65,
                        "lw": 1.2,
                    },
                )
                ax.scatter(
                    start[f"{x_col}_display"],
                    start[f"{y_col}_display"],
                    s=24,
                    facecolor="white",
                    edgecolor=color,
                )
                ax.scatter(
                    end[f"{x_col}_display"],
                    end[f"{y_col}_display"],
                    s=28,
                    color=color,
                )
        ax.axhline(0, color="#CBD5E1", linewidth=0.7)
        ax.axvline(0, color="#CBD5E1", linewidth=0.7)
        ax.set_xlabel(f"{x_col.upper()} (baseline z-score)")
        ax.set_ylabel(f"{y_col.upper()} (baseline z-score)")
        ax.set_title(title)
    handles = [
        plt.Line2D([0], [0], color=color, marker="o", linestyle="-", label=label)
        for label, color in colors.items()
    ]
    axes[1].legend(handles=handles, frameon=False, loc="best")
    _figure_bundle(fig, figures_dir, "figure_04_projection_arrows")
    plt.close(fig)


def _figure_forest_plot(feature_effects: pd.DataFrame, figures_dir: Path) -> None:
    source = feature_effects
    if "analysis_scope" in source.columns:
        primary = source[source["analysis_scope"].eq("primary")]
        if not primary.empty:
            source = primary
    top = (
        source.assign(abs_interaction=source["interaction_estimate"].abs())
        .sort_values(["interaction_q_value", "abs_interaction"], ascending=[True, False])
        .head(15)
        .copy()
    )
    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.4)))
    y = np.arange(len(top))
    ax.errorbar(
        top["interaction_estimate"],
        y,
        xerr=np.vstack(
            [
                top["interaction_estimate"] - top["interaction_ci_low"],
                top["interaction_ci_high"] - top["interaction_estimate"],
            ]
        ),
        fmt="o",
        color="#4C78A8",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"])
    xmin, xmax = ax.get_xlim()
    x_text = xmax - 0.02 * (xmax - xmin)
    for row_index, (_, row) in enumerate(top.iterrows()):
        q_value = float(row["interaction_q_value"])
        label = "q<.001" if np.isfinite(q_value) and q_value < 0.001 else f"q={q_value:.3f}"
        ax.text(x_text, row_index, label, va="center", ha="right", fontsize=8, color="#475569")
    significant = int((source["interaction_q_value"] < 0.05).sum())
    min_q = float(source["interaction_q_value"].min()) if len(source) else np.nan
    min_q_label = "<.001" if np.isfinite(min_q) and min_q < 0.001 else f"{min_q:.3f}"
    ax.set_xlabel("Phase-by-treatment estimate (95% CI)")
    ax.set_title(
        f"Exploratory feature interactions; BH-significant features: {significant}/"
        f"{len(source)} (min q={min_q_label})"
    )
    _figure_bundle(fig, figures_dir, "figure_05_forest_plot")
    plt.close(fig)


def _figure_group_dispersion(group_night: pd.DataFrame, figures_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharex=True)
    x_col = "stress_aligned_night" if "stress_aligned_night" in group_night.columns else "relative_night"
    for cage_id, group in group_night.groupby("cage_id", sort=False):
        ordered = group.sort_values(x_col)
        baseline = ordered.loc[ordered["phase"].eq("baseline"), "dispersion"]
        reference = float(baseline.mean()) if len(baseline) else float(ordered["dispersion"].iloc[0])
        if np.isclose(reference, 0.0):
            reference = 1.0
        alpha = 0.9
        linewidth = 1.2
        if str(cage_id) == "wt_10132":
            alpha = 1.0
            linewidth = 2.4
        ax1.plot(
            ordered[x_col],
            ordered["dispersion"] / reference - 1.0,
            marker="o",
            linewidth=linewidth,
            alpha=alpha,
            label=str(cage_id),
        )
        ax2.plot(
            ordered[x_col],
            ordered["baseline_centroid_distance"],
            marker="o",
            linewidth=linewidth,
            alpha=alpha,
            label=str(cage_id),
        )
        if str(cage_id) == "wt_10132" and len(ordered):
            last = ordered.iloc[-1]
            ax2.annotate(
                "wt_10132",
                xy=(last[x_col], last["baseline_centroid_distance"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="#111827",
            )
    ax1.axhline(0, color="#64748B", linewidth=0.8)
    xlabel = "Night relative to stress event" if x_col == "stress_aligned_night" else "Relative night"
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Relative dispersion change from cage baseline")
    ax1.set_title("Within-cage dispersion")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Distance from cage baseline centroid")
    ax2.set_title("Whole-group displacement")
    if x_col == "stress_aligned_night":
        for ax in [ax1, ax2]:
            ax.set_xticks([-3, -2, -1, 1, 2, 3])
            ax.set_xticklabels(["-3", "-2", "-1", "+1", "+2", "+3"])
            ax.axvspan(-0.5, 0.5, color="#FDE68A", alpha=0.18)
    ax2.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    _figure_bundle(fig, figures_dir, "figure_06_group_dispersion_centroid")
    plt.close(fig)


def _figure_feature_heatmap(pca_loadings: pd.DataFrame, identity_loadings: pd.DataFrame, figures_dir: Path) -> None:
    loadings = (
        pca_loadings[pca_loadings["component"].isin(["pca_1", "pca_2"])]
        .pivot(index="feature", columns="component", values="loading")
        .fillna(0.0)
    )
    if not identity_loadings.empty:
        identity = (
            identity_loadings[identity_loadings["component"].isin(["id_1", "id_2"])]
            .pivot(index="feature", columns="component", values="loading")
            .fillna(0.0)
        )
        loadings = loadings.join(identity, how="outer").fillna(0.0)
    raw = loadings.to_numpy(dtype=float)
    column_scale = np.nanmax(np.abs(raw), axis=0)
    column_scale[~np.isfinite(column_scale) | (column_scale == 0)] = 1.0
    normalized = pd.DataFrame(raw / column_scale, index=loadings.index, columns=loadings.columns)
    top = (
        normalized.assign(rank=lambda d: d.abs().max(axis=1))
        .sort_values("rank", ascending=False)
        .head(30)
    )
    displayed = top.drop(columns=["rank"])
    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.22)))
    loading_cmap = LinearSegmentedColormap.from_list(
        "blue_yellow_red_loadings",
        ["#1f4eae", "#fff176", "#b2182b"],
    )
    im = ax.imshow(
        displayed.to_numpy(dtype=float),
        aspect="auto",
        cmap=loading_cmap,
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
    )
    ax.set_yticks(range(len(top.index)))
    ax.set_yticklabels(top.index, fontsize=7)
    ax.set_xticks(range(len(displayed.columns)))
    ax.set_xticklabels(displayed.columns, rotation=20)
    ax.set_title("Feature loading heatmap (column-normalized)")
    ax.set_xlabel("Each axis is scaled by its own maximum absolute loading")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative loading: blue low, yellow zero, red high")
    _figure_bundle(fig, figures_dir, "figure_07_feature_heatmap")
    plt.close(fig)

