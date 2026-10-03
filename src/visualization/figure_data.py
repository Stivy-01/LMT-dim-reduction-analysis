# -*- coding: utf-8 -*-
"""Caricatori dati condivisi delle figure paper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization.paper_style import (
    SOURCE_LABELS,
    SOURCE_ORDER,
    bool_series,
    bootstrap_mean_ci,
    sign_flip_p,
    star,
)


def model_observed_points(run_dir: Path, responses: list[str], scopes: list[tuple[str, str]]) -> pd.DataFrame:
    mouse = pd.read_csv(run_dir / "mouse_night_analysis.csv")
    rows: list[dict[str, object]] = []
    for scope, _ in scopes:
        data = mouse.loc[
            bool_series(mouse["effect_eligible"])
            & mouse["phase"].isin(["baseline", "post_stress"])
            & mouse["mouse_id"].notna()
            & mouse["cage_id"].notna()
        ].copy()
        if scope == "exclude_wt_10132":
            data = data.loc[~data["cage_id"].eq("wt_10132")].copy()

        for response in responses:
            if response not in data.columns:
                continue
            per_mouse = (
                data.groupby(["cage_id", "mouse_id", "treatment", "phase"], dropna=False)[response]
                .mean()
                .unstack("phase")
                .dropna(subset=["baseline", "post_stress"])
                .reset_index()
            )
            if per_mouse.empty:
                continue
            per_mouse["delta"] = per_mouse["post_stress"] - per_mouse["baseline"]

            control = per_mouse.loc[per_mouse["treatment"].eq("control")].copy()
            for row in control.itertuples(index=False):
                rows.append(
                    {
                        "analysis_scope": scope,
                        "response": response,
                        "term": "phase_post_stress",
                        "value": float(row.delta),
                        "mouse_id": row.mouse_id,
                        "cage_id": row.cage_id,
                    }
                )

            control_by_cage = control.groupby("cage_id")["delta"].mean()
            stressed = per_mouse.loc[per_mouse["treatment"].eq("stressed")].copy()
            stressed["control_cage_delta"] = stressed["cage_id"].map(control_by_cage)
            stressed = stressed.dropna(subset=["control_cage_delta"])
            stressed["interaction_delta"] = stressed["delta"] - stressed["control_cage_delta"]
            for row in stressed.itertuples(index=False):
                rows.append(
                    {
                        "analysis_scope": scope,
                        "response": response,
                        "term": "phase_post_stress:treatment_stressed",
                        "value": float(row.interaction_delta),
                        "mouse_id": row.mouse_id,
                        "cage_id": row.cage_id,
                    }
                )
    return pd.DataFrame(rows)


def summarize_group_response(response: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("whole_group_shift", "Whole-group displacement", False),
        ("control_response", "Control displacement", False),
        ("identity_shift", "Identity composite shift", True),
        ("dispersion_change", "Dispersion change", True),
        ("pairwise_change", "Pairwise distance change", True),
        ("stressed_control_separation_change", "Stressed-control separation", True),
        ("synchronization_similarity", "Trajectory synchronization", True),
    ]
    rows: list[dict[str, object]] = []
    groups = [("ALL", response)] + [(src, response.loc[response["source_sheet"].eq(src)]) for src in SOURCE_ORDER]
    for src, subset in groups:
        for idx, (metric, label, signed) in enumerate(metrics):
            n, mean, lo, hi = bootstrap_mean_ci(subset[metric], seed=20240612 + idx)
            p = sign_flip_p(subset[metric]) if signed else np.nan
            rows.append(
                {
                    "source_sheet": src,
                    "source_label": SOURCE_LABELS[src],
                    "metric": metric,
                    "metric_label": label,
                    "signed_tested": signed,
                    "n": n,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "signflip_p": p,
                    "sig": star(p),
                }
            )
    return pd.DataFrame(rows)


def genotype_pvalues(frame: pd.DataFrame, group_cols: list[str], value_col: str = "tg_minus_wt") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=False):
        rows.append({**dict(zip(group_cols, keys)), "p_value": sign_flip_p(group[value_col])})
    return pd.DataFrame(rows)

