"""Stress-effect window comparison: night -1 vs +1 versus an extended window.

Question answered here: is the approved contrast (night -1 versus night +1)
sufficient, or does adding nights -2, +2 and +3 change the picture?

The "current" numbers are reproduced with exactly the formula used by the
approved figures (value of each mouse divided by the *group mean* of night -1,
then averaged across mice), so the +1 column here equals the published bars.
The extended version adds the other nights and a window contrast:

    pre  window : mean(night -2, night -1)
    post window : mean(night +1, night +2, night +3)

Outputs (all under outputs/<run>/longitudinal_window/):
    features_current_vs_extended.csv   per feature, per treatment, per night
    axes_current_vs_extended.csv       same for the identity / PCA axes
    conclusions.md                     numbers behind the report text
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

RUN_DIR = Path(
    "outputs/thesis_full_runs/"
    "thesis_analysis_final_with_last2_wt_metadata_fixed_20260701"
)
PRE_NIGHTS = {-2, -1}
POST_NIGHTS = {1, 2, 3}
ALL_NIGHTS = sorted(PRE_NIGHTS | POST_NIGHTS)

AXES = ["identity_composite", "pca_1", "pca_2", "pca_3", "id_1", "id_2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=RUN_DIR)
    return parser.parse_args()


def load_frame(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "mouse_night_analysis.csv")
    df = df.loc[df["effect_eligible"].fillna(False).astype(bool)].copy()
    return df.loc[df["stress_aligned_night"].isin(ALL_NIGHTS)].copy()


def per_mouse_ratios(group: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Ratios to the group mean of night -1 (the approved normalisation)."""
    denominator = group.loc[
        group["stress_aligned_night"].eq(-1), features
    ].apply(pd.to_numeric, errors="coerce").mean(axis=0).replace(0, np.nan)
    out = group[["mouse_id", "stress_aligned_night"]].copy()
    values = group[features].apply(pd.to_numeric, errors="coerce")
    out[features] = values.div(denominator, axis=1)
    return out


def window_values(ratios: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """One row per mouse with the pre- and post-window means."""
    rows = []
    for mouse, sub in ratios.groupby("mouse_id", sort=False):
        row = {"mouse_id": mouse}
        pre = sub.loc[sub["stress_aligned_night"].isin(PRE_NIGHTS), features]
        post = sub.loc[sub["stress_aligned_night"].isin(POST_NIGHTS), features]
        for feature in features:
            row[f"pre_{feature}"] = pre[feature].mean(skipna=True)
            row[f"post_{feature}"] = post[feature].mean(skipna=True)
            row[f"n_pre_{feature}"] = int(pre[feature].notna().sum())
            row[f"n_post_{feature}"] = int(post[feature].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(
    ratios: pd.DataFrame, windows: pd.DataFrame, features: list[str],
    label: str,
) -> pd.DataFrame:
    """Per-feature effect sizes plus paired tests against the -1 level."""
    rows = []
    for feature in features:
        rec = {"set": label, "feature": feature}
        for night in ALL_NIGHTS:
            values = ratios.loc[
                ratios["stress_aligned_night"].eq(night), feature
            ].dropna()
            rec["n_night_%+d" % night] = int(values.count())
            rec["pct_night_%+d" % night] = 100.0 * (values.mean() - 1.0) \
                if values.count() else np.nan
            if night > 0 and values.count() > 4:
                rec["p_night_%+d" % night] = float(
                    stats.wilcoxon(values - 1.0).pvalue)
            else:
                rec["p_night_%+d" % night] = np.nan
        for arm in ("pre", "post"):
            values = windows[f"{arm}_{feature}"].dropna()
            rec["n_%s_window" % arm] = int(values.count())
            rec["pct_%s_window" % arm] = 100.0 * (values.mean() - 1.0) \
                if values.count() else np.nan
        rec["pct_window_effect"] = rec["pct_post_window"]
        paired = (windows[f"post_{feature}"] - windows[f"pre_{feature}"]).dropna()
        rec["n_paired"] = int(paired.count())
        rec["pct_window_change"] = 100.0 * paired.mean() if paired.count() \
            else np.nan
        if paired.count() > 4:
            rec["p_window_change"] = float(stats.wilcoxon(paired).pvalue)
        else:
            rec["p_window_change"] = np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    for night in (1, 2, 3):
        column = "p_night_%+d" % night
        if column in out:
            ok = out[column].notna()
            out.loc[ok, "q_night_%+d" % night] = multipletests(
                out.loc[ok, column], method="fdr_bh")[1]
    ok = out["p_window_change"].notna()
    out.loc[ok, "q_window_change"] = multipletests(
        out.loc[ok, "p_window_change"], method="fdr_bh")[1]
    return out


def summarise_axes(frame: pd.DataFrame, axes: list[str],
                   treatment: str) -> pd.DataFrame:
    """Axis changes in data units and in pre-window SD units (no percentages:
    the identity/PCA axes are centred near zero)."""
    sub = frame.loc[frame["treatment"].eq(treatment)]
    pre = sub.loc[sub["stress_aligned_night"].isin(PRE_NIGHTS)]
    sd = pre[axes].apply(pd.to_numeric, errors="coerce").std(ddof=1)
    rows = []
    for axis in axes:
        rec = {"set": treatment, "axis": axis, "pre_sd": float(sd[axis])}
        pre_values = pre[axis].dropna()
        rec["n_pre_window"] = int(pre_values.count())
        rec["raw_pre_window"] = float(pre_values.mean())
        rec["sd_pre_window"] = float(pre_values.std(ddof=1))
        post = sub.loc[sub["stress_aligned_night"].isin(POST_NIGHTS), axis] \
            .dropna()
        rec["n_post_window"] = int(post.count())
        rec["raw_post_window"] = float(post.mean())
        rec["sd_post_window"] = float(post.std(ddof=1))
        rec["delta_window"] = rec["raw_post_window"] - rec["raw_pre_window"]
        rec["delta_window_sd"] = rec["delta_window"] / rec["pre_sd"]
        if pre_values.count() > 4 and post.count() > 4:
            rec["p_window"] = float(stats.mannwhitneyu(
                post, pre_values).pvalue)
        for night in (-2, -1, 1, 2, 3):
            values = sub.loc[sub["stress_aligned_night"].eq(night), axis] \
                .dropna()
            rec["n_night_%+d" % night] = int(values.count())
            rec["raw_night_%+d" % night] = float(values.mean()) \
                if values.count() else np.nan
            rec["delta_night_%+d" % night] = \
                rec["raw_night_%+d" % night] - rec["raw_pre_window"] \
                if values.count() else np.nan
            rec["delta_night_%+d_sd" % night] = \
                rec["delta_night_%+d" % night] / rec["pre_sd"] \
                if values.count() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def metric_feature_lists(run_dir: Path, frame: pd.DataFrame) -> dict[str, list[str]]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_pre_post_behavior_profiles import metric_features

    full = pd.read_csv(run_dir / "mouse_night_analysis.csv")
    full = full.loc[full["effect_eligible"].fillna(False).astype(bool)]
    return {metric: metric_features(full, metric)
            for metric in ("count", "mean_duration", "std_duration")}


def main() -> int:
    args = parse_args()
    frame = load_frame(args.run_dir)
    out_dir = args.run_dir / "longitudinal_window"
    out_dir.mkdir(parents=True, exist_ok=True)

    families = metric_feature_lists(args.run_dir, frame)
    all_features = sorted({f for feats in families.values() for f in feats})
    axes = [a for a in AXES if a in frame.columns]

    feature_rows = []
    axis_rows = []
    ratios_by_arm = {}
    windows_by_arm = {}
    for treatment in ("control", "stressed"):
        sub = frame.loc[frame["treatment"].eq(treatment)]
        ratios = per_mouse_ratios(sub, all_features + axes)
        windows = window_values(ratios, all_features + axes)
        ratios_by_arm[treatment] = ratios
        windows_by_arm[treatment] = windows
        feature_rows.append(
            summarise(ratios, windows, all_features, treatment))
        axis_rows.append(summarise_axes(frame, axes, treatment))

    features = pd.concat(feature_rows, ignore_index=True)
    axes_df = pd.concat(axis_rows, ignore_index=True)
    features.to_csv(out_dir / "features_current_vs_extended.csv", index=False)
    axes_df.to_csv(out_dir / "axes_current_vs_extended.csv", index=False)

    # stressed minus control (difference in differences), per feature
    did_rows = []
    for feature in all_features:
        rec = {"feature": feature}
        for night in (1, 2, 3):
            arms = {
                arm: ratios_by_arm[arm].loc[
                    ratios_by_arm[arm]["stress_aligned_night"].eq(night),
                    feature].dropna()
                for arm in ("control", "stressed")
            }
            rec["did_night_%+d" % night] = \
                float(arms["stressed"].mean() - arms["control"].mean())
            if arms["stressed"].count() > 4 and arms["control"].count() > 4:
                rec["p_did_night_%+d" % night] = float(stats.mannwhitneyu(
                    arms["stressed"], arms["control"]).pvalue)
            else:
                rec["p_did_night_%+d" % night] = np.nan
        for arm in ("control", "stressed"):
            w = windows_by_arm[arm]
            change = (w.loc[:, ["post_" + feature]].to_numpy()
                      - w.loc[:, ["pre_" + feature]].to_numpy()).ravel()
            rec["did_window_%s" % arm] = float(np.nanmean(change))
        rec["did_window"] = rec["did_window_stressed"] - rec["did_window_control"]
        did_rows.append(rec)
    did = pd.DataFrame(did_rows)
    ok = did["p_did_night_+1"].notna()
    did.loc[ok, "q_did_night_+1"] = multipletests(
        did.loc[ok, "p_did_night_+1"], method="fdr_bh")[1]
    did.to_csv(out_dir / "stress_specific_did.csv", index=False)

    lines = ["# Window comparison: summary numbers", ""]
    lines.append("Mice per treatment: %s" % (
        frame.groupby("treatment")["mouse_id"].nunique().to_dict()))
    lines.append("Features analysed: %d (counts, mean durations, duration "
                 "variability)" % len(all_features))
    lines.append("")
    lines.append("## Per-feature effects (per cent change vs night -1)")
    for treatment in ("control", "stressed"):
        sub = features.loc[features["set"].eq(treatment)]
        lines.append("")
        lines.append("### %s" % treatment)
        for night in (1, 2, 3):
            col = "pct_night_%+d" % night
            lines.append(
                "  night %+d : median |effect| %5.1f%% | |effect|>=10%% in "
                "%3d/%d features | significant (q<0.05) %3d"
                % (night, float(sub[col].abs().median()),
                   int((sub[col].abs() >= 10).sum()), len(sub),
                   int((sub["q_night_%+d" % night] < 0.05).sum())))
        lines.append(
            "  window (+1..+3 vs -2..-1): median |effect| %5.1f%% | "
            "significant %3d | corr with +1 = %.3f"
            % (float(sub["pct_window_change"].abs().median()),
               int((sub["q_window_change"] < 0.05).sum()),
               float(sub[["pct_night_+1", "pct_window_change"]]
                     .corr().iloc[0, 1])))
        peak = sub[["pct_night_+1", "pct_night_+2", "pct_night_+3"]] \
            .abs().idxmax(axis=1).value_counts(normalize=True) * 100
        lines.append("  largest effect at: %s"
                     % {k.replace("pct_night_", "night "): round(v, 1)
                        for k, v in peak.items()})
        ratio = (sub["pct_night_+3"] / sub["pct_night_+1"]).replace(
            [np.inf, -np.inf], np.nan)
        lines.append("  median ratio effect(+3)/effect(+1) = %.2f"
                     % float(ratio.median()))
    lines.append("")
    lines.append("## Stress-specific contrast (stressed minus control)")
    for night in (1, 2, 3):
        col = "did_night_%+d" % night
        sig = int((did["q_did_night_+1"] < 0.05).sum()) if night == 1 else \
            int((did["p_did_night_%+d" % night] < 0.05).sum())
        lines.append("  night %+d : median |difference| %5.1f%% | raw p<0.05 "
                     "in %3d/%d features"
                     % (night, float(did[col].abs().median()), sig, len(did)))
    lines.append("  window  : median |difference| %5.1f%%"
                 % float(did["did_window"].abs().median()))
    lines.append("")
    lines.append("## Axes (data units, no percentages)")
    for treatment in ("control", "stressed"):
        sub = axes_df.loc[axes_df["set"].eq(treatment)]
        lines.append("### %s" % treatment)
        for _, rec in sub.iterrows():
            lines.append(
                "  %-18s pre %+7.2f  post %+7.2f  delta %+7.2f  "
                "(%+5.2f SD)  | nights vs pre: -2 %+6.2f, -1 %+6.2f, "
                "+1 %+6.2f, +2 %+6.2f, +3 %+6.2f"
                % (rec["axis"], rec["raw_pre_window"], rec["raw_post_window"],
                   rec["delta_window"], rec["delta_window_sd"],
                   rec["delta_night_-2_sd"], rec["delta_night_-1_sd"],
                   rec["delta_night_+1_sd"], rec["delta_night_+2_sd"],
                   rec["delta_night_+3_sd"]))
        lines.append("")
    (out_dir / "conclusions.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out_dir / "features_current_vs_extended.csv")
    print("wrote", out_dir / "axes_current_vs_extended.csv")
    print("wrote", out_dir / "conclusions.md")
    print("\n".join(lines[:24]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
