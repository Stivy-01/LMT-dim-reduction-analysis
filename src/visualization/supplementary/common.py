# -*- coding: utf-8 -*-
"""Costanti e helper condivisi delle supplementary."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization.paper_style import SOURCE_ORDER, clean_label

PRIMARY_RESPONSES = ["id_1", "id_2", "identity_composite"]
STRESS_BAND = dict(color="#F2C94C", alpha=0.18, linewidth=0)
GROUP_METRICS = [
    ("whole_group_shift", "Whole-group displacement"),
    ("control_response", "Control displacement"),
    ("identity_shift", "Identity composite shift"),
    ("dispersion_change", "Dispersion change"),
    ("pairwise_change", "Pairwise distance change"),
    ("stressed_control_separation_change", "Stressed-control separation"),
    ("synchronization_similarity", "Trajectory synchronization"),
]
FAMILY_COLORS = {"count": "#0072B2", "mean_duration": "#D55E00",
                 "std_duration": "#009E73"}


OUTLIER = "wt_10132"
NIGHT_TICKS = [-3, -2, -1, 1, 2, 3]


def _pretty(feature: str) -> str:
    """Readable feature label without underscores or double spaces."""
    return " ".join(clean_label(feature).split())


PRIMARY_RESPONSES = ["id_1", "id_2", "identity_composite"]
STRESS_BAND = dict(color="#F2C94C", alpha=0.18, linewidth=0)
GROUP_METRICS = [
    ("whole_group_shift", "Whole-group displacement"),
    ("control_response", "Control displacement"),
    ("identity_shift", "Identity composite shift"),
    ("dispersion_change", "Dispersion change"),
    ("pairwise_change", "Pairwise distance change"),
    ("stressed_control_separation_change", "Stressed-control separation"),
    ("synchronization_similarity", "Trajectory synchronization"),
]


def _cage_order(qc: pd.DataFrame) -> list[str]:
    order: list[str] = []
    for source in SOURCE_ORDER:
        cages = sorted(qc.loc[qc["source_sheet"].eq(source), "cage_id"].unique())
        order.extend(cages)
    return order


def _cage_labels(qc: pd.DataFrame, cages: list[str]) -> list[str]:
    source_of = qc.groupby("cage_id")["source_sheet"].first().to_dict()
    labels = []
    for cage in cages:
        suffix = " *" if cage == OUTLIER else ""
        labels.append(f"{cage} ({source_of.get(cage, '?')}){suffix}")
    return labels

