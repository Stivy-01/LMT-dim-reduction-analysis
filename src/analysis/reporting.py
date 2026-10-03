# -*- coding: utf-8 -*-
"""Report HTML e I/O della run (ex BLOCCO 10 di thesis_analysis)."""
from __future__ import annotations

import stat
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.analysis.config import ThesisAnalysisResult


def _render_report(result: ThesisAnalysisResult) -> Path:
    report = result.output_dir / "thesis_analysis_report.html"
    model_html = (
        result.model_effects.to_html(index=False, border=0)
        if not result.model_effects.empty
        else "<p>No dimension-level model could be fitted.</p>"
    )
    bootstrap_html = (
        result.bootstrap_permutation.to_html(index=False, border=0)
        if not result.bootstrap_permutation.empty
        else "<p>No cage-level change-in-change estimate was available.</p>"
    )
    top_features = result.feature_effects
    if "analysis_scope" in top_features.columns:
        top_features = top_features[top_features["analysis_scope"].eq("primary")]
    top_features = top_features.sort_values("interaction_q_value").head(20)
    feature_html = (
        top_features.to_html(index=False, border=0)
        if not top_features.empty
        else "<p>No feature-level effect table was available.</p>"
    )
    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          table {{ border-collapse: collapse; }}
          th, td {{ border: 1px solid #ccc; padding: 6px 8px; }}
          img {{ max-width: 100%; height: auto; }}
          code {{ background: #f2f2f2; padding: 1px 4px; }}
        </style>
      </head>
      <body>
        <h1>Scientific thesis analysis pipeline</h1>
        <p>Run directory: <code>{result.output_dir}</code></p>
        <p>Projection eligible rows: {result.projection_rows}</p>
        <p>Effect eligible rows: {result.effect_rows}</p>
        <p>Selected complete features: {len(result.feature_columns)}</p>
        <p><strong>Interpretation:</strong> primary results use explicit phase
        and treatment metadata. Additional scopes are exported only when they
        change the analytical sample, including the explicit exclusion of
        outlier cage <code>wt_10132</code>. Treatment is never inferred from
        behavior. With few cages, confidence intervals and permutation results
        are more informative than isolated p-values.</p>
        <h2>Figures</h2>
        <ul>
          <li><img src="figures/figure_01_timeline_qc.png" alt="timeline qc"/></li>
          <li><img src="figures/figure_02_missingness_heatmap.png" alt="missingness"/></li>
          <li><img src="figures/figure_03_trajectories.png" alt="trajectories"/></li>
          <li><img src="figures/figure_04_projection_arrows.png" alt="projection arrows"/></li>
          <li><img src="figures/figure_05_forest_plot.png" alt="forest plot"/></li>
          <li><img src="figures/figure_06_group_dispersion_centroid.png" alt="group dispersion"/></li>
          <li><img src="figures/figure_07_feature_heatmap.png" alt="feature heatmap"/></li>
        </ul>
        <h2>Dimension-level models</h2>
        {model_html}
        <h2>Paired cage change-in-change</h2>
        {bootstrap_html}
        <h2>Top primary feature effects</h2>
        {feature_html}
      </body>
    </html>
    """
    report.write_text(textwrap.dedent(html), encoding="utf-8")
    return report


def _make_run_directory(output_root: Path, run_name: str | None) -> Path:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / (run_name or f"thesis_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    candidate = base
    index = 1
    while candidate.exists():
        index += 1
        candidate = base.with_name(f"{base.name}_{index}")
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _set_readonly(path: Path) -> None:
    try:
        path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except Exception:
        pass


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _set_readonly(path)
