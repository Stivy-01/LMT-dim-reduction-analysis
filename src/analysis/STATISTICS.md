# Statistical Methods — Final Thesis Analysis

This document summarizes the statistical methods used for the final analysis of the
Live Mouse Tracker (LMT) thesis. It describes **what** was used, **where** it belongs in
the pipeline, and **why**. Formulas and line-level references are intentionally omitted.

## Scope and where the statistics live

The statistics of the final analysis live in three places:

- **`src/analysis/`** — the analysis pipeline, split into one module per stage:
  `projection.py` (dimensionality reduction), `effects.py` (individual-level
  models, feature-level tests), `group_stats.py` (cage-level resampling),
  `fit_models.py`, with `thesis_analysis.py` as the orchestrator.
- **`src/visualization/figures/fig_*.py`** (group entry: `build_all.py`) —
  figure-level statistics: bootstrap confidence intervals and exact sign-flip
  tests used for the thesis figures.
- **`src/visualization/figures/fig_09_prepost_profiles.py`**,
  **`fig_per_feature_changes.py`** and **`prepost_supplementary.py`** —
  descriptive statistics (group means and SEM) for the pre/post-stress
  behavioral profiles.

The legacy statistics module and the old identity-domain/LDA/PCA cluster were
removed from this branch and are **not** used by the final analysis.

## Analysis units and design

- Unit of individual analysis: one **mouse-night** (one mouse during the 12 h active
  interval). Unit of group analysis: one **cage-night**.
- The primary inferential sample is the effect-eligible set of paired cages, with
  **baseline** and **post-stress** observations and a **treatment** factor
  (control vs stressed).
- Only **metadata-explicit** treatment labels are used; stress/control is never inferred
  from behavior.
- The 126 complete behavioral features are used consistently across all analyses; count
  and duration measures are `log1p`-transformed for dimensionality reduction and modelling.

## 1. Dimensionality reduction (identity-related axes)

- **Principal Component Analysis (PCA)**, fitted only on baseline nights, to summarize the
  high-dimensional behavioral space.
- **Regularized Linear Discriminant Analysis (LDA)** with mouse identity as the class label,
  to derive two identity-related axes and a composite score.

**Why:** to reduce a large, correlated feature space to a small number of interpretable
dimensions and to capture individual-level structure. Regularization is used because the
sample is small relative to the number of features. These axes are treated as
*identity-related behavioral axes*, not as validated identity domains.

## 2. Individual-level models

- **Linear mixed-effects models** with fixed effects for phase, treatment, the
  phase-by-treatment interaction, and relative night, a random intercept per mouse, and a
  cage-level variance component when supported.
- When mixed models are unstable, a **heteroskedasticity-robust ordinary least squares**
  fallback is used.

**Why:** repeated mouse-night observations are not independent, and animals share a cage
environment, so grouping must be modeled. The **phase-by-treatment interaction** is the key
term for stress-associated change; results are interpreted through effect sizes and
uncertainty, not as a binary significance outcome.

## 3. Multiple-testing control

- **Benjamini–Hochberg false-discovery-rate (FDR)** correction, applied within analysis
  families (feature-level and genotype tests).
- A conservative display threshold (**p < .005**) is used in figures.

**Why:** many behavioral features are correlated and many tests are performed; FDR control
limits false positives, and the stricter display threshold reduces over-interpretation of
isolated nominal p-values.

## 4. Feature-level tests

- Phase-by-treatment interaction tested for each complete behavioral feature, with
  Benjamini–Hochberg adjusted q-values.

**Why:** to ask whether any single behavioral feature changes *selectively* in stressed
mice. The analysis returned no significant feature-level interaction, which supports the
interpretation of a broad, shared post-manipulation change rather than a stressed-specific
one.

## 5. Cage-level (group) statistics

- **Bootstrap confidence intervals**, resampling at the cage level, for cage-level response
  metrics (e.g., centroid displacement, dispersion, pairwise distance, synchronization,
  control response).
- **Sign-flip / permutation tests** against zero for signed group metrics.

**Why:** the cage is the biologically meaningful unit and the number of complete cages is
small, so resampling-based inference is more appropriate than parametric tests at the cage
level. Non-negative distance metrics are treated as descriptive (no p-values).

## 6. Genotype analyses (exploratory)

- **Paired TG−WT differences** computed within the same cage, night, and treatment to reduce
  environmental confounds.
- **Exact sign-flip permutation tests** on the paired deltas.

**Why:** pairing controls for cage/day/phase structure; exact permutation testing is used
because the number of complete cages per genotype is small. These comparisons are framed as
exploratory, not confirmatory.

## 7. Descriptive pre/post-stress profiles

- Values normalized to the **pre-stress group mean** (expressed as ratios); group summaries
  as the **mean across mice with the standard error of the mean**; a `log10(1+x)` scale is
  used only for display.

**Why:** normalization makes behaviors of very different magnitudes comparable, and the SEM
shows between-mouse uncertainty. This section is purely descriptive and applies no
inferential test.

## 8. Sensitivity and robustness checks

- Re-running the models after **excluding the influential cage `wt_10132`**.
- **Temporal validation** (leave-last-baseline-night) to check the stability of the
  identity-related structure.

**Why:** to quantify how much the results depend on single influential cages and to check
temporal stability of the derived axes.

## 9. Software

Analyses use `pandas`, `numpy`, `scipy`, `scikit-learn` and `statsmodels` (with
`matplotlib`/`seaborn` for figures). Randomization-based procedures use fixed seeds for
reproducibility.

## What is not used

- The legacy statistics module and the old identity-domain / LDA / PCA cluster were
  removed from this branch and are **not** part of the final analysis.
