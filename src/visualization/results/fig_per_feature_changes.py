# -*- coding: utf-8 -*-
"""Figure 2-4: variazioni per-feature pre/post (count, mean duration, variability)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from src.visualization.supplementary.pages import (
    CATEGORY_COLORS,
    CATEGORY_ORDER,
    METRIC_SPECS,
    behavior_category,
    label_feature,
    metric_features,
    normalized_summary,
)
from src.visualization.paper_style import save_bundle, set_paper_style
from src.visualization.results.icons import (
    BAND_IN,
    CATEGORY_DISPLAY,
    CATEGORY_ICON_LAYOUT,
    CATEGORY_ICONS,
    SHOW_LEGEND,
    build_composite_icons,
    category_icon_list,
    icon_aspect,
    icon_image,
    icon_zoom,
    place_icon,
)
from src.visualization.results.fig_09_prepost_profiles import (
    PHASE_NIGHTS,
    load_selection,
    style_axis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "paper"


def per_feature_change(sel: pd.DataFrame, features: list[str], treatment: str):
    d = sel.loc[sel["treatment"].eq(treatment)]
    summary, long_values = normalized_summary(d, features)
    phase = summary.loc[summary["stress_aligned_night"].eq(1)].set_index("feature").loc[features]
    bar = (phase["normalized_mean"].to_numpy(dtype=float) - 1.0) * 100.0
    sem = phase["normalized_sem"].to_numpy(dtype=float) * 100.0
    lv = long_values.loc[long_values["stress_aligned_night"].eq(1)]
    dots = [
        (lv.loc[lv["feature"].eq(feature), "normalized_value"].dropna().to_numpy(dtype=float) - 1.0) * 100.0
        for feature in features
    ]
    return bar, sem, dots


def figure_per_feature_split(sel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260723)
    for metric, row_label, stem in [
        ("count", "Event count change (%)", "prepost_per_feature_count_changes"),
        ("mean_duration", "Mean event duration change (%)", "prepost_per_feature_mean_duration_changes"),
        ("std_duration", "Duration variability change (%)", "prepost_per_feature_duration_variability_changes"),
    ]:
        features = metric_features(sel, metric)
        x = np.arange(len(features))
        colors = [CATEGORY_COLORS[behavior_category(feature)] for feature in features]
        used_categories = [c for c in CATEGORY_ORDER if c in {behavior_category(f) for f in features}]
        data = {treatment: per_feature_change(sel, features, treatment) for treatment in ("control", "stressed")}

        lo, hi = 0.0, 0.0
        for bar, sem, _ in data.values():
            lo = min(lo, float(np.nanmin(bar - sem)))
            hi = max(hi, float(np.nanmax(bar + sem)))
        pad = 0.10 * (hi - lo) if hi > lo else 1.0

        fig, axes = plt.subplots(2, 1, figsize=(7.4, 8.2), sharex=True, sharey=True)
        for r, (treatment, title) in enumerate((("control", "Control"), ("stressed", "Stressed"))):
            ax = axes[r]
            bar, sem, dots = data[treatment]
            ax.bar(x, bar, color=colors, edgecolor="black", linewidth=0.25, width=0.76, zorder=2)
            ax.errorbar(
                x, bar, yerr=sem, fmt="none", ecolor="black", elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3
            )
            for index, values in enumerate(dots):
                jitter = rng.uniform(-0.18, 0.18, size=len(values))
                ax.scatter(
                    np.full(len(values), index) + jitter,
                    values,
                    s=4,
                    color="black",
                    alpha=0.18,
                    linewidth=0,
                    zorder=4,
                )
            ax.axhline(0, color="#111111", linewidth=0.8, zorder=1)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-0.8, len(features) - 0.2)
            ax.set_xticks(x)
            if r == 1:
                ax.set_xticklabels([label_feature(feature) for feature in features], rotation=45, ha="right", fontsize=6)
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(row_label, fontsize=9)
            ax.text(0.012, 0.97, title, transform=ax.transAxes, ha="left",
                    va="top", fontsize=10)
            style_axis(ax)
            for feature, value, err in zip(features, bar, sem):
                rows.append({"panel": metric, "treatment": treatment, "feature": feature, "pct_change": value, "sem": err})

        if SHOW_LEGEND:
            handles = [
                Patch(facecolor=CATEGORY_COLORS[category], edgecolor="black",
                      linewidth=0.25, label=category)
                for category in used_categories
            ]
            fig.legend(handles=handles, loc="upper center",
                       bbox_to_anchor=(0.5, 0.985), ncol=5, frameon=False,
                       fontsize=8)

        # one schematic per behavioural category, centred over its group of
        # bars and followed by the category name and its colour: schematic ->
        # name -> colour swatch -> bars of that category
        fig.subplots_adjust(top=0.755, bottom=0.26, hspace=0.16)
        categories = [behavior_category(feature) for feature in features]
        bands = []
        for category in used_categories:
            idx = [i for i, c in enumerate(categories) if c == category]
            if idx:
                bands.append((category, category_icon_list(category), idx))
        if bands:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax_in = axes[0].get_window_extent(renderer=renderer)
            axes_w_in = ax_in.width / fig.dpi
            axes_h_in = ax_in.height / fig.dpi
            # the panels keep a margin around the features, so strip elements
            # are placed in data coordinates (blended transform) rather than
            # assuming that column i sits at i / n of the axes width
            x0_data, x1_data = axes[0].get_xlim()
            per_data_in = axes_w_in / (x1_data - x0_data)
            strip_tr = axes[0].get_xaxis_transform()

            def y_above(inches: float) -> float:
                """Axes fraction for a point `inches` above the panel top."""
                return 1.005 + inches / axes_h_in

            swatch_in = 0.085
            name_rows_in = (0.235, 0.375)      # two rows, used only if needed
            icon_h_in = BAND_IN - 0.55
            icon_cy = y_above(0.55 + icon_h_in / 2.0)

            # one group of schematics per category: work out the widest
            # layout that still keeps neighbouring groups apart
            groups = []
            for category, icons, idx in bands:
                lo_i, hi_i = min(idx) - 0.5, max(idx) + 0.5
                block_w_in = (hi_i - lo_i) * per_data_in
                centre_i = (lo_i + hi_i) / 2.0
                layout = CATEGORY_ICON_LAYOUT.get(category, "row")
                if icons and layout == "stack" and len(icons) > 1:
                    slot_h = icon_h_in / len(icons)
                    sizes = [(slot_h * icon_aspect(name), slot_h)
                             for name in icons]
                    want_w = max(w for w, _h in sizes)
                elif icons:
                    sizes = [(icon_h_in * icon_aspect(name), icon_h_in)
                             for name in icons]
                    want_w = sum(w for w, _h in sizes) + 0.05 * (len(icons) - 1)
                else:
                    sizes, want_w = [], 0.0
                groups.append({"category": category, "icons": icons,
                               "idx": idx, "block_w": block_w_in,
                               "centre_i": centre_i, "layout": layout,
                               "sizes": sizes, "want_w": want_w,
                               "w": want_w})
            # shrink groups that would collide (their centres are fixed)
            for _ in range(6):
                for a, b in zip(groups, groups[1:]):
                    if a["w"] <= 0 or b["w"] <= 0:
                        continue
                    gap = (b["centre_i"] - a["centre_i"]) * per_data_in
                    need = a["w"] / 2.0 + b["w"] / 2.0
                    if need > gap - 0.03:
                        scale = (gap - 0.03) / need
                        a["w"] = max(a["block_w"] * 0.55, a["w"] * scale)
                        b["w"] = max(b["block_w"] * 0.55, b["w"] * scale)
            # never let a group cross the separator of its own column
            for group in groups:
                if group["w"] > 0:
                    gaps = 0.05 * max(len(group["icons"]) - 1, 0)
                    room = max(group["block_w"] * 0.88 - gaps,
                               group["block_w"] * 0.45)
                    group["w"] = min(group["w"], room)

            for k, group in enumerate(groups):
                category = group["category"]
                lo_x = group["idx"][0] - 0.5
                hi_x = group["idx"][-1] + 0.5
                # colour key of this category, directly under its name
                axes[0].add_patch(Rectangle(
                    (lo_x + 0.06, y_above(0.02)),
                    max(hi_x - lo_x - 0.12, 0.05), swatch_in / axes_h_in,
                    transform=strip_tr,
                    facecolor=CATEGORY_COLORS[category], edgecolor="none",
                    clip_on=False, zorder=3))
                if k:
                    axes[0].plot([lo_x, lo_x],
                                 [y_above(0.02), y_above(BAND_IN)],
                                 transform=strip_tr, color="#9e9e9e",
                                 linewidth=0.6, clip_on=False, zorder=4)
                group["name"] = axes[0].text(
                    group["centre_i"], y_above(name_rows_in[0]),
                    CATEGORY_DISPLAY.get(category, category),
                    transform=strip_tr, ha="center", va="center",
                    fontsize=7, color="#2b2b2b", linespacing=1.1,
                    clip_on=False)
            # category names: first nudge neighbours apart (a few tenths of a
            # column, so a long name can use the free space next to it), then
            # move a name to a second row if it still would not fit
            texts = [g["name"] for g in groups]
            px_per_data = per_data_in * fig.dpi
            nudge_px = 0.11 * fig.dpi            # ~2.8 mm of freedom per side
            for _ in range(5):
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                boxes = [t.get_window_extent(renderer=renderer) for t in texts]
                moved = False
                for i in range(len(texts) - 1):
                    a, b = boxes[i], boxes[i + 1]
                    if abs((a.y0 + a.y1) / 2 - (b.y0 + b.y1) / 2) > \
                            0.5 * (a.height + b.height) / 2:
                        continue                 # already on different rows
                    overlap = a.x1 + 2 - b.x0
                    if overlap <= 0:
                        continue
                    left = min(overlap / 2.0, nudge_px)
                    right = min(overlap - left, nudge_px)
                    texts[i].set_x(texts[i].get_position()[0]
                                   - left / px_per_data)
                    texts[i + 1].set_x(texts[i + 1].get_position()[0]
                                       + right / px_per_data)
                    moved = True
                if not moved:
                    break
            # the outermost names may slide away from their column if a long
            # word does not fit in it (eg. "configuration" in the count figure
            # or "sequence" in the other two): keep them on the free side so
            # they never sit on top of the neighbouring columns
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            edge_shift_px = 0.20 * fig.dpi
            for index, direction in ((0, -1.0), (len(texts) - 1, 1.0)):
                if len(texts) < 2:
                    break
                group = groups[index]
                lo_data = group["idx"][0] - 0.5
                hi_data = group["idx"][-1] + 0.5
                lo_px = axes[0].transData.transform((lo_data, 0))[0]
                hi_px = axes[0].transData.transform((hi_data, 0))[0]
                box = texts[index].get_window_extent(renderer=renderer)
                if direction < 0:
                    overflow = box.x1 - (hi_px - 2)
                else:
                    overflow = (lo_px + 2) - box.x0
                if overflow > 0:
                    texts[index].set_x(
                        texts[index].get_position()[0]
                        + direction * min(overflow, edge_shift_px) / px_per_data)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            row_right: list[float | None] = [None, None]
            for text in texts:
                box = text.get_window_extent(renderer=renderer)
                for row in (0, 1):
                    if row_right[row] is None or box.x0 > row_right[row] + 3:
                        text.set_y(y_above(name_rows_in[row]))
                        row_right[row] = box.x1
                        break
                else:
                    text.set_y(y_above(name_rows_in[1]))
                    row_right[1] = box.x1
            # NOTE: strip_frac was undefined in the original script (latent
            # NameError on the "stack" branch). 1.0 keeps offsets in data units.
            strip_frac = 1.0
            for group in groups:
                icons = group["icons"]
                if not icons:
                    continue
                sizes = group["sizes"]
                total_w = sum(w for w, _h in sizes)
                scale = group["w"] / total_w if total_w else 1.0
                if group["layout"] == "stack" and len(icons) > 1:
                    heights = [h * scale for _w, h in sizes]
                    slot = group["w"]
                    ys = [icon_cy + (0.30 - 0.60 * j) * strip_frac
                          for j in range(len(icons))]
                    for icon, h, y in zip(icons, heights, ys):
                        zoom = icon_zoom(icon, slot * 72.0, h * 72.0, fill=0.98)
                        place_icon(axes[0], icon, (group["centre_i"], y),
                                   strip_tr, zoom=zoom,
                                   box_alignment=(0.5, 0.5))
                    continue
                widths = [w * scale for w, _h in sizes]
                total = sum(widths) + 0.05 * (len(icons) - 1)
                x = group["centre_i"] - (total / 2.0) / per_data_in
                n = len(icons)
                for icon, w, h in zip(icons, widths, [s[1] * scale for s in sizes]):
                    zoom = icon_zoom(icon, w * 72.0, h * 72.0, fill=0.98)
                    place_icon(axes[0], icon,
                               (x + (w / 2.0) / per_data_in, icon_cy),
                               strip_tr, zoom=zoom,
                               box_alignment=(0.5, 0.5))
                    x += (w + 0.05) / per_data_in
        fig.suptitle(f"Per-feature pre- to post-stress changes: {METRIC_SPECS[metric]['title']}", y=0.997, fontsize=11, fontweight="bold")
        save_bundle(fig, out_dir, f"paper_figure_{stem}")

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "paper_figure_prepost_per_feature_values.csv", index=False)
    return out




def _build(input_dir: Path, out_dir: Path) -> None:
    sel = load_selection(input_dir)
    per_feature = figure_per_feature_split(sel, out_dir)
    print(f"wrote per-feature figures to {out_dir} ({len(per_feature)} values)")

def build(input_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build per-feature figures (standalone or via build_all)."""
    set_paper_style()
    input_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _build(input_dir, out_dir)
    print(f"wrote per-feature figures to {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-feature figures.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

