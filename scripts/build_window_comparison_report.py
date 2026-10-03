"""Build the PDF report that compares the approved -1 vs +1 analysis with an
extended window (-2/-1 baseline versus +1/+2/+3 post-stress).

Figures produced (all under outputs/<run>/longitudinal_window/figures/):
    fig_<family>_window_bars.png     approved layout, window formula
    fig_<family>_night_tracks.png    per-feature trajectories per night
    fig_acute_vs_window_scatter.png  agreement between the two definitions
    fig_peak_night.png               which night carries the largest effect
    fig_axes_night_tracks.png        composite axes, SD units
    fig_model_terms_acute_vs_window.png  phase / phase x treatment estimates

The PDF report embeds the approved figures next to the recreated ones.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from scipy import stats

from src.visualization.paper_style import save_bundle, set_paper_style
from src.visualization.supplementary.pages import (
    CATEGORY_COLORS,
    behavior_category,
    label_feature,
    metric_features,
)
try:
    from scripts.longitudinal_window_comparison import (
        POST_NIGHTS,
        PRE_NIGHTS,
        per_mouse_ratios,
        window_values,
    )
except ImportError:  # direct script execution
    from longitudinal_window_comparison import (  # noqa: E402
        POST_NIGHTS,
        PRE_NIGHTS,
        per_mouse_ratios,
        window_values,
    )

RUN_DIR = Path(
    "outputs/thesis_full_runs/"
    "thesis_analysis_final_with_last2_wt_metadata_fixed_20260701"
)
FAMILIES = {
    "count": ("Event count change (%)", "counts"),
    "mean_duration": ("Mean event duration change (%)", "mean duration"),
    "std_duration": ("Duration variability change (%)", "duration variability"),
}
NIGHTS = [-2, -1, 1, 2, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=RUN_DIR)
    return parser.parse_args()


def load_inputs(run_dir: Path):
    mouse = pd.read_csv(run_dir / "mouse_night_analysis.csv")
    mouse = mouse.loc[mouse["effect_eligible"].fillna(False).astype(bool)]
    mouse = mouse.loc[mouse["stress_aligned_night"].isin(NIGHTS)].copy()
    full = pd.read_csv(run_dir / "mouse_night_analysis.csv")
    full = full.loc[full["effect_eligible"].fillna(False).astype(bool)]
    families = {metric: metric_features(full, metric)
                for metric in FAMILIES}
    return mouse, families


def per_mouse_pct(mouse: pd.DataFrame, features: list[str], treatment: str,
                  night: int | None = None,
                  window: bool = False) -> pd.DataFrame:
    sub = mouse.loc[mouse["treatment"].eq(treatment)]
    ratios = per_mouse_ratios(sub, features)
    if window:
        windows = window_values(ratios, features)
        out = pd.DataFrame({
            f: 100.0 * (windows[f"post_{f}"] - windows[f"pre_{f}"])
            for f in features})
        return out
    if night is None:
        raise ValueError("night or window required")
    rows = ratios.loc[ratios["stress_aligned_night"].eq(night), features]
    return 100.0 * (rows - 1.0)


def figure_window_bars(mouse: pd.DataFrame, features: list[str],
                       metric: str, out_dir: Path) -> Path:
    """Same layout as the approved per-feature figure, window formula."""
    title, _ = FAMILIES[metric]
    # use the same y range as the approved figure (computed from the acute
    # contrast) so the two panels are visually comparable
    acute_bars = []
    for treatment in ("control", "stressed"):
        acute = per_mouse_pct(mouse, features, treatment, night=1)
        acute_bars.append(acute.mean(axis=0).to_numpy())
    limit = float(np.nanmax(np.abs(np.concatenate(acute_bars)))) * 1.15
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 8.0), sharex=True)
    x = np.arange(len(features))
    colors = [CATEGORY_COLORS[behavior_category(f)] for f in features]
    for ax, treatment in zip(axes, ("control", "stressed")):
        values = per_mouse_pct(mouse, features, treatment, window=True)
        bar = values.mean(axis=0).to_numpy()
        sem = values.sem(axis=0).to_numpy()
        ax.bar(x, bar, color=colors, edgecolor="black", linewidth=0.25,
               width=0.76, zorder=2)
        ax.errorbar(x, bar, yerr=sem, fmt="none", ecolor="black",
                    elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
        for index, feature in enumerate(features):
            vals = values[feature].dropna().to_numpy()
            jitter = np.random.default_rng(20260723).uniform(
                -0.18, 0.18, size=len(vals))
            ax.scatter(np.full(len(vals), index) + jitter, vals, s=4,
                       color="black", alpha=0.18, linewidth=0, zorder=4)
        ax.axhline(0, color="#111111", linewidth=0.8, zorder=1)
        ax.set_ylim(-limit, limit)
        n_clipped = int((np.abs(bar) > limit).sum())
        ax.set_ylabel(title, fontsize=9)
        ax.text(0.012, 0.97, treatment.capitalize(), transform=ax.transAxes,
                ha="left", va="top", fontsize=10)
        if n_clipped:
            ax.text(0.99, 0.03, "%d feature fuori scala" % n_clipped,
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#666666")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if treatment == "stressed":
            ax.set_xticks(x)
            ax.set_xticklabels([label_feature(f) for f in features],
                               rotation=45, ha="right", fontsize=6)
        else:
            ax.set_xticklabels([])
    fig.suptitle("Extended window: mean(+1..+3) vs mean(-2..-1) — %s"
                 % title.split(" (")[0].lower(), y=0.997, fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_bundle(fig, out_dir, "fig_%s_window_bars" % metric)
    plt.close(fig)
    return out_dir / ("fig_%s_window_bars.png" % metric)


def figure_night_tracks(mouse: pd.DataFrame, features: list[str],
                        metric: str, out_dir: Path) -> Path:
    """Per-feature effect per night, with the median and IQR band."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, treatment in zip(axes, ("control", "stressed")):
        curves = []
        for night in NIGHTS:
            values = per_mouse_pct(mouse, features, treatment, night=night)
            curves.append(values.mean(axis=0))
        table = pd.concat(curves, axis=1)
        table.columns = NIGHTS
        for _, row in table.iterrows():
            ax.plot(NIGHTS, row.to_numpy(), color="#9aa0a6", linewidth=0.4,
                    alpha=0.45, zorder=1)
        median = table.median(axis=0).to_numpy()
        q1 = table.quantile(0.25, axis=0).to_numpy()
        q3 = table.quantile(0.75, axis=0).to_numpy()
        ax.fill_between(NIGHTS, q1, q3, color="#0072B2", alpha=0.25, zorder=2)
        ax.plot(NIGHTS, median, color="#0072B2", linewidth=1.8, zorder=3)
        ax.axhline(0, color="#111111", linewidth=0.8)
        ax.axvline(-0.5, color="#bbbbbb", linewidth=0.8, linestyle="--")
        ax.set_xticks(NIGHTS)
        ax.set_xlabel("night relative to stress")
        ax.set_title(treatment.capitalize(), fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("change vs night -1 (%)")
    fig.suptitle("Per-night effects, %s" % FAMILIES[metric][1], y=0.99,
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_bundle(fig, out_dir, "fig_%s_night_tracks" % metric)
    plt.close(fig)
    return out_dir / ("fig_%s_night_tracks.png" % metric)


def figure_agreement(mouse: pd.DataFrame, families: dict[str, list[str]],
                     out_dir: Path) -> tuple[Path, float]:
    colours = {"count": "#0072B2", "mean_duration": "#D55E00",
               "std_duration": "#009E73"}
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    all_x, all_y = [], []
    for metric, features in families.items():
        for treatment in ("control", "stressed"):
            acute = per_mouse_pct(mouse, features, treatment, night=1) \
                .mean(axis=0).to_numpy()
            window = per_mouse_pct(mouse, features, treatment, window=True) \
                .mean(axis=0).to_numpy()
            all_x.extend(acute)
            all_y.extend(window)
            ax.scatter(acute, window, s=12,
                       color=colours[metric], alpha=0.75,
                       label=FAMILIES[metric][1] if treatment == "control"
                       else None, edgecolor="none")
    stacked = np.concatenate([np.abs(all_x), np.abs(all_y)])
    lim = float(np.nanpercentile(stacked, 98)) * 1.1
    n_out = int((stacked > lim).sum())
    ax.plot([-lim, lim], [-lim, lim], color="#999999", linewidth=0.8,
            linestyle="--")
    ax.axhline(0, color="#dddddd", linewidth=0.8)
    ax.axvline(0, color="#dddddd", linewidth=0.8)
    r = float(stats.pearsonr(all_x, all_y)[0])
    slope = float(np.polyfit(all_x, all_y, 1)[0])
    ax.set_xlabel("effect at night +1 (%)")
    ax.set_ylabel("effect over the window (%)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    note = "r = %.3f, slope = %.2f" % (r, slope)
    if n_out:
        note += "\n%d punti fuori scala" % n_out
    ax.set_title("Acute vs extended window\n" + note, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_bundle(fig, out_dir, "fig_acute_vs_window_scatter")
    plt.close(fig)
    return out_dir / "fig_acute_vs_window_scatter.png", r


def figure_peak_night(mouse: pd.DataFrame, families: dict[str, list[str]],
                      out_dir: Path) -> Path:
    rows = []
    for metric, features in families.items():
        for treatment in ("control", "stressed"):
            columns = {}
            for night in (1, 2, 3):
                columns[night] = per_mouse_pct(
                    mouse, features, treatment, night=night).mean(axis=0)
            table = pd.DataFrame(columns)
            peak = table.abs().idxmax(axis=1).value_counts(normalize=True) * 100
            for night in (1, 2, 3):
                rows.append({"metric": FAMILIES[metric][1],
                             "treatment": treatment,
                             "night": "+%d" % night,
                             "share": float(peak.get(night, 0.0))})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    metrics = list(dict.fromkeys(data["metric"]))
    width = 0.13
    for k, (treatment, marker) in enumerate((("control", None),
                                             ("stressed", "//"))):
        for j, night in enumerate(("+1", "+2", "+3")):
            heights = []
            for metric in metrics:
                sel = data.loc[data["metric"].eq(metric)
                               & data["treatment"].eq(treatment)
                               & data["night"].eq(night)]
                heights.append(float(sel["share"].iloc[0]) if len(sel) else 0.0)
            xs = np.arange(len(metrics)) + (k * 3 + j - 2.5) * width
            colour = {"+1": "#0072B2", "+2": "#56B4E9", "+3": "#999999"}[night]
            ax.bar(xs, heights, width=width, color=colour, hatch=marker,
                   edgecolor="white", linewidth=0.4,
                   label="%s, night %s" % (treatment, night))
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel("features with largest |effect| (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_bundle(fig, out_dir, "fig_peak_night")
    plt.close(fig)
    return out_dir / "fig_peak_night.png"


def figure_axes_tracks(mouse: pd.DataFrame, out_dir: Path) -> Path:
    axes_names = ["identity_composite", "pca_1", "pca_2", "id_1", "id_2"]
    available = [a for a in axes_names if a in mouse.columns]
    fig, panels = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for panel, treatment in zip(panels, ("control", "stressed")):
        sub = mouse.loc[mouse["treatment"].eq(treatment)]
        pre = sub.loc[sub["stress_aligned_night"].isin(PRE_NIGHTS)]
        for axis in available:
            sd = float(pre[axis].std(ddof=1))
            baseline = float(pre[axis].mean())
            means, los, his = [], [], []
            for night in NIGHTS:
                values = sub.loc[sub["stress_aligned_night"].eq(night), axis] \
                    .dropna()
                means.append((values.mean() - baseline) / sd)
                sem = values.sem(ddof=1) / sd
                los.append(means[-1] - 1.96 * sem)
                his.append(means[-1] + 1.96 * sem)
            panel.errorbar(NIGHTS, means, yerr=[np.array(means) - los,
                                                np.array(his) - np.array(means)],
                           marker="o", markersize=3, linewidth=1.2, capsize=2,
                           label=axis)
        panel.axhline(0, color="#111111", linewidth=0.8)
        panel.axvline(-0.5, color="#bbbbbb", linewidth=0.8, linestyle="--")
        panel.set_xticks(NIGHTS)
        panel.set_xlabel("night relative to stress")
        panel.set_title(treatment.capitalize(), fontsize=10)
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)
    panels[0].set_ylabel("change vs baseline window (SD)")
    panels[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Composite axes across nights", y=0.99, fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_bundle(fig, out_dir, "fig_axes_night_tracks")
    plt.close(fig)
    return out_dir / "fig_axes_night_tracks.png"


def model_comparison(mouse: pd.DataFrame, responses: list[str]) -> pd.DataFrame:
    """phase and phase x treatment estimates for the acute and window designs."""
    import statsmodels.formula.api as smf

    rows = []
    for design, nights in (("acute (-1/+1)", {-1: "baseline", 1: "post"}),
                           ("window (-2..-1 / +1..+3)",
                            {-2: "baseline", -1: "baseline", 1: "post",
                             2: "post", 3: "post"})):
        sub = mouse.loc[mouse["stress_aligned_night"].isin(nights)].copy()
        sub["phase"] = sub["stress_aligned_night"].map(nights)
        for response in responses:
            if response not in sub.columns:
                continue
            data = sub[["mouse_id", "cage_id", response, "phase",
                        "treatment"]].dropna()
            if data["phase"].nunique() < 2:
                continue
            model = smf.mixedlm("%s ~ C(phase) * C(treatment)" % response,
                                data, groups=data["mouse_id"])
            fit = model.fit(reml=False)
            for term, label in (("C(phase)[T.post]", "phase (post vs baseline)"),
                                ("C(phase)[T.post]:C(treatment)[T.stressed]",
                                 "phase x stressed")):
                if term in fit.params:
                    rows.append({
                        "design": design,
                        "response": response,
                        "term": label,
                        "estimate": float(fit.params[term]),
                        "ci_low": float(fit.conf_int().loc[term, 0]),
                        "ci_high": float(fit.conf_int().loc[term, 1]),
                        "p": float(fit.pvalues[term]),
                        "n": int(data["mouse_id"].nunique()),
                    })
    return pd.DataFrame(rows)


def figure_model_terms(table: pd.DataFrame, out_dir: Path) -> Path:
    responses = list(dict.fromkeys(table["response"]))
    terms = ["phase (post vs baseline)", "phase x stressed"]
    fig, panels = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for panel, term in zip(panels, terms):
        sub = table.loc[table["term"].eq(term)]
        for k, design in enumerate(sub["design"].unique()):
            sel = sub.loc[sub["design"].eq(design)]
            y = np.arange(len(responses)) + (k - 0.5) * 0.28
            est = [float(sel.loc[sel["response"].eq(r), "estimate"].iloc[0])
                   if len(sel.loc[sel["response"].eq(r)]) else np.nan
                   for r in responses]
            low = [float(sel.loc[sel["response"].eq(r), "ci_low"].iloc[0])
                   if len(sel.loc[sel["response"].eq(r)]) else np.nan
                   for r in responses]
            high = [float(sel.loc[sel["response"].eq(r), "ci_high"].iloc[0])
                    if len(sel.loc[sel["response"].eq(r)]) else np.nan
                    for r in responses]
            panel.errorbar(est, y, xerr=[np.array(est) - np.array(low),
                                         np.array(high) - np.array(est)],
                           fmt="o", markersize=3, capsize=2,
                           label=design)
        panel.axvline(0, color="#111111", linewidth=0.8)
        panel.set_yticks(np.arange(len(responses)))
        panel.set_yticklabels([r.replace("_", " ") for r in responses],
                              fontsize=8)
        panel.set_title(term, fontsize=9)
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)
    panels[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Model estimates: acute design vs extended window", y=0.99,
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_bundle(fig, out_dir, "fig_model_terms_acute_vs_window")
    plt.close(fig)
    return out_dir / "fig_model_terms_acute_vs_window.png"


def genotype_comparison(run_dir: Path, responses: list[str]) -> pd.DataFrame:
    """TG - WT paired deltas per model, for the acute design and the window."""
    qc = pd.read_csv(run_dir / "quality_control.csv")
    qc = qc.loc[qc["effect_eligible"].fillna(False).astype(bool)]
    qc = qc.loc[qc["source_sheet"].isin(["16p", "CD del"])]
    designs = {
        "acute (-1/+1)": {-1: "baseline", 1: "post"},
        "window (-2..-1 / +1..+3)": {-2: "baseline", -1: "baseline",
                                     1: "post", 2: "post", 3: "post"},
    }
    rows = []
    for design, mapping in designs.items():
        sub = qc.loc[qc["stress_aligned_night"].isin(mapping)].copy()
        sub["phase"] = sub["stress_aligned_night"].map(mapping)
        for response in responses:
            if response not in sub.columns:
                continue
            per_mouse = (sub.groupby(
                ["source_sheet", "cage_id", "interval_start", "phase",
                 "genotype"], dropna=False)[response].mean().reset_index())
            for (source, phase), block in per_mouse.groupby(
                    ["source_sheet", "phase"]):
                wide = block.pivot_table(
                    index=["cage_id", "interval_start"], columns="genotype",
                    values=response)
                if not {"TG", "WT"}.issubset(wide.columns):
                    continue
                delta = (wide["TG"] - wide["WT"]).dropna()
                if not len(delta):
                    continue
                sem = float(delta.sem(ddof=1)) if len(delta) > 1 else np.nan
                rows.append({
                    "design": design, "model": source, "phase": phase,
                    "response": response, "n_pairs": int(len(delta)),
                    "delta": float(delta.mean()),
                    "ci_low": float(delta.mean() - 1.96 * sem)
                    if np.isfinite(sem) else np.nan,
                    "ci_high": float(delta.mean() + 1.96 * sem)
                    if np.isfinite(sem) else np.nan,
                })
    return pd.DataFrame(rows)


def figure_genotype(table: pd.DataFrame, out_dir: Path) -> Path:
    models = list(dict.fromkeys(table["model"]))
    responses = list(dict.fromkeys(table["response"]))
    fig, panels = plt.subplots(1, len(models), figsize=(7.4, 3.2), sharey=True)
    if len(models) == 1:
        panels = [panels]
    for panel, model in zip(panels, models):
        for k, design in enumerate(table["design"].unique()):
            sel = table.loc[table["model"].eq(model)
                            & table["design"].eq(design)
                            & table["phase"].eq("post")]
            est, low, high = [], [], []
            for response in responses:
                hit = sel.loc[sel["response"].eq(response)]
                est.append(float(hit["delta"].iloc[0]) if len(hit) else np.nan)
                low.append(float(hit["ci_low"].iloc[0]) if len(hit) else np.nan)
                high.append(float(hit["ci_high"].iloc[0]) if len(hit) else np.nan)
            y = np.arange(len(responses)) + (k - 0.5) * 0.3
            panel.errorbar(est, y, xerr=[np.array(est) - np.array(low),
                                         np.array(high) - np.array(est)],
                           fmt="o", markersize=3, capsize=2, label=design)
        panel.axvline(0, color="#111111", linewidth=0.8)
        panel.set_yticks(np.arange(len(responses)))
        panel.set_yticklabels([r.replace("_", " ") for r in responses],
                              fontsize=8)
        panel.set_title(model, fontsize=10)
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)
    panels[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Genotype TG - WT deltas in the post phase: acute vs window",
                 y=0.99, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_bundle(fig, out_dir, "fig_genotype_acute_vs_window")
    plt.close(fig)
    return out_dir / "fig_genotype_acute_vs_window.png"


def text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
    fig.text(0.06, 0.95, title, fontsize=15, fontweight="bold", va="top")
    y = 0.90
    wrote = False
    for line in lines:
        weight = "bold" if line.startswith("#") else "normal"
        size = 11 if line.startswith("#") else 9.5
        text = line.lstrip("# ").strip()
        fig.text(0.06, y, text, fontsize=size, va="top", fontweight=weight)
        wrote = True
        y -= 0.023 if not line.startswith("#") else 0.030
        if y < 0.06:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.94
            wrote = False
    if wrote or y < 0.90:
        pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf: PdfPages, title: str, paths: list[Path],
               captions: list[str] | None = None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.965, title, fontsize=13, fontweight="bold", ha="center",
             va="top")
    n = len(paths)
    top, bottom = 0.93, 0.05
    slot = (top - bottom) / n
    for i, path in enumerate(paths):
        image = plt.imread(path)
        # keep the image aspect ratio and centre it inside its slot
        fig_w, fig_h = 8.27, 11.69
        img_h, img_w = image.shape[:2]
        width = 0.86
        height = width * fig_w * (img_h / img_w) / fig_h
        height = min(height, slot * 0.9)
        width = min(0.86, height * fig_h * (img_w / img_h) / fig_w)
        x0 = (1 - width) / 2
        y0 = top - (i + 1) * slot + (slot - height) / 2
        ax = fig.add_axes([x0, y0, width, height])
        ax.imshow(image)
        ax.axis("off")
        if captions and i < len(captions):
            ax.set_title(captions[i], fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_paper_style()
    run_dir = args.run_dir
    window_dir = run_dir / "longitudinal_window"
    fig_dir = window_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = run_dir / "figures" / "paper"

    mouse, families = load_inputs(run_dir)
    print("nights:", sorted(mouse["stress_aligned_night"].unique()),
          "| mice:", mouse["mouse_id"].nunique())

    produced = {}
    for metric, features in families.items():
        produced[("bars", metric)] = figure_window_bars(
            mouse, features, metric, fig_dir)
        produced[("tracks", metric)] = figure_night_tracks(
            mouse, features, metric, fig_dir)
    scatter, r = figure_agreement(mouse, families, fig_dir)
    peak = figure_peak_night(mouse, families, fig_dir)
    axes_track = figure_axes_tracks(mouse, fig_dir)

    responses = [r for r in ["identity_composite", "id_1", "id_2", "pca_1",
                             "pca_2"] if r in mouse.columns]
    model_table = model_comparison(mouse, responses)
    model_table.to_csv(window_dir / "model_terms_acute_vs_window.csv",
                       index=False)
    model_fig = figure_model_terms(model_table, fig_dir)

    geno = genotype_comparison(run_dir, responses)
    geno.to_csv(window_dir / "genotype_acute_vs_window.csv", index=False)
    geno_fig = figure_genotype(geno, fig_dir) if len(geno) else None

    numbers = pd.read_csv(window_dir / "features_current_vs_extended.csv")
    did = pd.read_csv(window_dir / "stress_specific_did.csv")
    summary_lines = ["# Sintesi operativa"]
    for treatment in ("control", "stressed"):
        sub = numbers.loc[numbers["set"].eq(treatment)]
        summary_lines.append("## %s" % treatment)
        summary_lines.append(
            "mediana |effetto|: +1 %.1f%%, +2 %.1f%%, +3 %.1f%%, finestra "
            "%.1f%%" % (sub["pct_night_+1"].abs().median(),
                        sub["pct_night_+2"].abs().median(),
                        sub["pct_night_+3"].abs().median(),
                        sub["pct_window_change"].abs().median()))
        peak_share = sub[["pct_night_+1", "pct_night_+2", "pct_night_+3"]] \
            .abs().idxmax(axis=1).value_counts(normalize=True) * 100
        summary_lines.append("effetto massimo a: " + ", ".join(
            "%s %.0f%%" % (k.replace("pct_night_", "notte "), v)
            for k, v in peak_share.items()))
        summary_lines.append("corr(+1, finestra) = %.3f"
                             % sub[["pct_night_+1", "pct_window_change"]]
                             .corr().iloc[0, 1])
        summary_lines.append("rapporto mediano effetto(+3)/effetto(+1) = %.2f"
                             % (sub["pct_night_+3"] / sub["pct_night_+1"])
                             .replace([np.inf, -np.inf], np.nan).median())
        summary_lines.append(
            "feature con |effetto|>=10%%: +1 %d, finestra %d (su %d)"
            % (int((sub["pct_night_+1"].abs() >= 10).sum()),
               int((sub["pct_window_change"].abs() >= 10).sum()), len(sub)))
    (window_dir / "report_summary.md").write_text("\n".join(summary_lines),
                                                  encoding="utf-8")

    pdf_path = window_dir / "window_comparison_report.pdf"
    with PdfPages(pdf_path) as pdf:
        text_page(pdf, "Scelta della finestra temporale", [
            "Confronto fra l'analisi approvata (notte -1 vs +1) e una",
            "finestra estesa (media -2/-1 vs media +1/+2/+3).",
            "",
            "Scopo: capire se aggiungere le altre notti cambia le conclusioni",
            "oppure se il contrasto -1 vs +1 è sufficiente.",
            "",
            "# Dati",
            "24-26 topi per gruppo con notte +1/+2/+3 eleggibile;",
            "22 per gruppo con -2; 10 per gruppo con -3 (esclusa).",
            "",
            "# Cosa contiene il report",
            "1. Effetti per feature: figura approvata vs stessa figura con la",
            "   formula estesa, traiettorie notte per notte e accordo fra le",
            "   due definizioni.",
            "2. Assi composite: traiettorie in unità SD.",
            "3. Modelli: stime di fase e fase x trattamento con disegno",
            "   acuto e con finestra estesa.",
            "4. Sintesi e raccomandazione per la tesi.",
        ])
        for metric, (title, human) in FAMILIES.items():
            approved = paper_dir / (
                "paper_figure_prepost_per_feature_%s_changes.png" % metric)
            image_page(pdf, "Per-feature: %s" % human,
                       [p for p in [approved, produced[("bars", metric)]]
                        if p.exists()],
                       ["analisi approvata: notte -1 vs +1",
                        "stessa figura con la finestra estesa"])
            image_page(pdf, "Traiettorie notte per notte: %s" % human,
                       [produced[("tracks", metric)]])
        image_page(pdf, "Accordo fra le due definizioni", [scatter],
                   ["ogni punto è una feature; r = %.3f" % r])
        image_page(pdf, "Dove si trova l'effetto massimo", [peak])
        image_page(pdf, "Assi composite", [axes_track])
        image_page(pdf, "Modelli: acuto vs finestra", [model_fig])
        if geno_fig is not None:
            image_page(pdf, "Genotipi: acuto vs finestra", [geno_fig])
        text_page(pdf, "Sintesi dei numeri", summary_lines)
        text_page(pdf, "Verdetto e raccomandazione", [
            "# Risposta breve",
            "L'analisi -1 vs +1 resta la scelta migliore per le figure",
            "descrittive per-feature: la notte +1 è quella con l'effetto più",
            "grande (65-85%% delle feature) e la finestra estesa diluisce",
            "l'effetto di circa il 20%% (slope 0.80) pur mantenendo lo stesso",
            "ordine di feature (r = 0.94).",
            "",
            "# Perché non serve sostituire -1 vs +1",
            "1. Sensibilità: le feature con |effetto| >= 10%% sono 100/+1",
            "   contro 89/finestra nei controlli, 105 contro 96 negli stressed.",
            "2. La finestra media notti con dinamiche diverse: l'effetto al +3",
            "   vale in mediana il 70%% di quello al +1, quindi mediare attenua",
            "   il segnale acuto che la tesi vuole mostrare.",
            "3. Le analisi inferenziali della tesi (modelli mixed con fase e",
            "   fase x trattamento, e le figure genotipiche) usano GIÀ la",
            "   media di tutte le notti eleggibili di ciascuna fase: la",
            "   critica 'solo 2 notti su 7' riguarda quindi solo le figure",
            "   descrittive, non l'inferenza.",
            "",
            "# Cosa aggiungono davvero le altre notti",
            "- persistenza: al +3 l'effetto resta ~70%% di quello acuto",
            "  (informazione che -1 vs +1 non può dare);",
            "- precisione dei modelli: con la finestra i CI si stringono",
            "  (es. identità x stressed: CI acuto [-6.9, 2.3] vs finestra",
            "  [-4.8, 0.1]);",
            "- distinzione fra effetto acuto e deriva comune: entrambi i",
            "  gruppi scendono dopo la manipolazione, quindi il confronto",
            "  stressed-control resta necessario.",
            "",
            "# Proposta per la tesi",
            "Mantenere -1 vs +1 come analisi primaria e aggiungere, in",
            "Discussione o come figura supplementare, la traiettoria",
            "notte-per-notte con la nota che gli effetti sono massimi al +1 e",
            "parzialmente persistenti fino al +3. Così il claim sul vantaggio",
            "del long-term recording è sostenuto da un'analisi e non solo",
            "affermato.",
        ])
        top = did.reindex(did["did_night_+1"].abs()
                          .sort_values(ascending=False).index).head(12)
        lines = ["# Feature con la differenza stressed-control più grande a +1",
                 "feature, differenza a +1 (%), differenza finestra (%), q"]
        for _, row in top.iterrows():
            lines.append("%s   %+.1f   %+.1f   %.3f"
                         % (row["feature"], row["did_night_+1"],
                            row["did_window"],
                            float(row.get("q_did_night_+1", np.nan))))
        text_page(pdf, "Dettaglio stress-specifico", lines)
    print("report:", pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
