# -*- coding: utf-8 -*-
"""Stile paper, salvataggio bundle e statistiche condivise delle figure."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd


SOURCE_ORDER = ["16p", "CD del", "wt"]
SOURCE_LABELS = {"16p": "16p", "CD del": "CD-del", "wt": "WT", "ALL": "All cages"}
SOURCE_COLORS = {
    "16p": "#0072B2",
    "CD del": "#009E73",
    "wt": "#D55E00",
    "ALL": "#111111",
}
PHASE_LABELS = {"baseline": "Baseline", "post_stress": "Post-stress"}
PHASE_MARKERS = {"baseline": "o", "post_stress": "s"}
SIGNIFICANCE_ALPHA = 0.005


def set_paper_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "axes.axisbelow": True,
        }
    )


def save_bundle(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def bootstrap_mean_ci(values: Iterable[float], seed: int = 20240612, samples: int = 5000) -> tuple[int, float, float, float]:
    clean = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return 0, np.nan, np.nan, np.nan
    if len(clean) == 1:
        value = float(clean[0])
        return 1, value, value, value
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(samples, len(clean)), replace=True).mean(axis=1)
    return int(len(clean)), float(clean.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def sign_flip_p(values: Iterable[float]) -> float:
    clean = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    clean = clean[clean != 0]
    n = len(clean)
    if n == 0:
        return np.nan
    observed = abs(clean.mean())
    if n <= 16:
        hits = 0
        total = 2**n
        for signs in itertools.product((-1, 1), repeat=n):
            if abs((clean * np.array(signs)).mean()) >= observed - 1e-12:
                hits += 1
        return hits / total
    rng = np.random.default_rng(20240612)
    signs = rng.choice((-1, 1), size=(100_000, n))
    return float(((np.abs((signs * clean).mean(axis=1)) >= observed).sum() + 1) / 100_001)


def star(p_value: float) -> str:
    return "*" if pd.notna(p_value) and float(p_value) < SIGNIFICANCE_ALPHA else ""


def p_text(p_value: float) -> str:
    if pd.isna(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}".replace("0.", ".")


def clean_label(label: str) -> str:
    return (
        str(label)
        .replace("identity_composite", "Identity composite")
        .replace("id_1", "ID1")
        .replace("id_2", "ID2")
        .replace("pca_1", "PCA1")
        .replace("pca_2", "PCA2")
        .replace("pca_3", "PCA3")
        .replace("_", " ")
        .replace("CD del", "CD-del")
    )


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])

