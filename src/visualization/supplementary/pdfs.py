from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = PROJECT_ROOT / "src" / "visualization" / "output" / "pre_post_behavior_profiles"

REQUESTED_8_PAGE_ORDER = [
    ("Normalized count profile: pre-stress", "behavior_profile_pre_stress_counts.png"),
    ("Normalized count profile: post-stress", "behavior_profile_post_stress_counts.png"),
    ("Raw event counts: pre-stress", "behavior_profile_pre_stress_raw_counts.png"),
    ("Raw event counts: post-stress", "behavior_profile_post_stress_raw_counts.png"),
    ("Raw mean duration: pre-stress", "behavior_profile_pre_stress_raw_mean_duration.png"),
    ("Raw mean duration: post-stress", "behavior_profile_post_stress_raw_mean_duration.png"),
    ("Normalized duration variability: pre-stress", "behavior_profile_pre_stress_std_duration.png"),
    ("Normalized duration variability: post-stress", "behavior_profile_post_stress_std_duration.png"),
]

REQUESTED_8_PAGE_LOG_RAW_ORDER = [
    ("Normalized count profile: pre-stress", "behavior_profile_pre_stress_counts.png"),
    ("Normalized count profile: post-stress", "behavior_profile_post_stress_counts.png"),
    ("Raw event counts log10(1+x): pre-stress", "behavior_profile_pre_stress_raw_log10p_counts.png"),
    ("Raw event counts log10(1+x): post-stress", "behavior_profile_post_stress_raw_log10p_counts.png"),
    ("Raw mean duration log10(1+x): pre-stress", "behavior_profile_pre_stress_raw_log10p_mean_duration.png"),
    ("Raw mean duration log10(1+x): post-stress", "behavior_profile_post_stress_raw_log10p_mean_duration.png"),
    ("Normalized duration variability: pre-stress", "behavior_profile_pre_stress_std_duration.png"),
    ("Normalized duration variability: post-stress", "behavior_profile_post_stress_std_duration.png"),
]

RAW_LOG_4_PAGE_ORDER = [
    ("Raw event counts log10(1+x): pre-stress", "behavior_profile_pre_stress_raw_log10p_counts.png"),
    ("Raw event counts log10(1+x): post-stress", "behavior_profile_post_stress_raw_log10p_counts.png"),
    ("Raw mean duration log10(1+x): pre-stress", "behavior_profile_pre_stress_raw_log10p_mean_duration.png"),
    ("Raw mean duration log10(1+x): post-stress", "behavior_profile_post_stress_raw_log10p_mean_duration.png"),
]

ALL_GENERATED_10_PAGE_ORDER = [
    ("Normalized count profile: pre-stress", "behavior_profile_pre_stress_counts.png"),
    ("Normalized count profile: post-stress", "behavior_profile_post_stress_counts.png"),
    ("Normalized mean duration: pre-stress", "behavior_profile_pre_stress_mean_duration.png"),
    ("Normalized mean duration: post-stress", "behavior_profile_post_stress_mean_duration.png"),
    ("Normalized duration variability: pre-stress", "behavior_profile_pre_stress_std_duration.png"),
    ("Normalized duration variability: post-stress", "behavior_profile_post_stress_std_duration.png"),
    ("Raw event counts: pre-stress", "behavior_profile_pre_stress_raw_counts.png"),
    ("Raw event counts: post-stress", "behavior_profile_post_stress_raw_counts.png"),
    ("Raw mean duration: pre-stress", "behavior_profile_pre_stress_raw_mean_duration.png"),
    ("Raw mean duration: post-stress", "behavior_profile_post_stress_raw_mean_duration.png"),
]

ALL_GENERATED_10_PAGE_LOG_RAW_ORDER = [
    ("Normalized count profile: pre-stress", "behavior_profile_pre_stress_counts.png"),
    ("Normalized count profile: post-stress", "behavior_profile_post_stress_counts.png"),
    ("Normalized mean duration: pre-stress", "behavior_profile_pre_stress_mean_duration.png"),
    ("Normalized mean duration: post-stress", "behavior_profile_post_stress_mean_duration.png"),
    ("Normalized duration variability: pre-stress", "behavior_profile_pre_stress_std_duration.png"),
    ("Normalized duration variability: post-stress", "behavior_profile_post_stress_std_duration.png"),
    ("Raw event counts log10(1+x): pre-stress", "behavior_profile_pre_stress_raw_log10p_counts.png"),
    ("Raw event counts log10(1+x): post-stress", "behavior_profile_post_stress_raw_log10p_counts.png"),
    ("Raw mean duration log10(1+x): pre-stress", "behavior_profile_pre_stress_raw_log10p_mean_duration.png"),
    ("Raw mean duration log10(1+x): post-stress", "behavior_profile_post_stress_raw_log10p_mean_duration.png"),
]


def treatment_10_page_log_raw_order(treatment_prefix: str, treatment_label: str) -> list[tuple[str, str]]:
    return [
        (f"{treatment_label}: normalized count profile: pre-stress", f"behavior_profile_{treatment_prefix}pre_stress_counts.png"),
        (f"{treatment_label}: normalized count profile: post-stress", f"behavior_profile_{treatment_prefix}post_stress_counts.png"),
        (f"{treatment_label}: normalized mean duration: pre-stress", f"behavior_profile_{treatment_prefix}pre_stress_mean_duration.png"),
        (f"{treatment_label}: normalized mean duration: post-stress", f"behavior_profile_{treatment_prefix}post_stress_mean_duration.png"),
        (f"{treatment_label}: normalized duration variability: pre-stress", f"behavior_profile_{treatment_prefix}pre_stress_std_duration.png"),
        (f"{treatment_label}: normalized duration variability: post-stress", f"behavior_profile_{treatment_prefix}post_stress_std_duration.png"),
        (f"{treatment_label}: raw event counts log10(1+x): pre-stress", f"behavior_profile_{treatment_prefix}pre_stress_raw_log10p_counts.png"),
        (f"{treatment_label}: raw event counts log10(1+x): post-stress", f"behavior_profile_{treatment_prefix}post_stress_raw_log10p_counts.png"),
        (f"{treatment_label}: raw mean duration log10(1+x): pre-stress", f"behavior_profile_{treatment_prefix}pre_stress_raw_log10p_mean_duration.png"),
        (f"{treatment_label}: raw mean duration log10(1+x): post-stress", f"behavior_profile_{treatment_prefix}post_stress_raw_log10p_mean_duration.png"),
    ]


def treatment_pair_20_page_order(log_raw: bool) -> list[tuple[str, str]]:
    raw_count_prefix = "raw_log10p_" if log_raw else "raw_"
    raw_duration_prefix = "raw_log10p_" if log_raw else "raw_"
    raw_count_label = "raw event counts log10(1+x)" if log_raw else "raw event counts"
    raw_duration_label = "raw mean duration log10(1+x)" if log_raw else "raw mean duration"
    base_pages = [
        ("normalized count profile", "pre-stress", "pre_stress_counts.png"),
        ("normalized count profile", "post-stress", "post_stress_counts.png"),
        ("normalized mean duration", "pre-stress", "pre_stress_mean_duration.png"),
        ("normalized mean duration", "post-stress", "post_stress_mean_duration.png"),
        ("normalized duration variability", "pre-stress", "pre_stress_std_duration.png"),
        ("normalized duration variability", "post-stress", "post_stress_std_duration.png"),
        (raw_count_label, "pre-stress", f"pre_stress_{raw_count_prefix}counts.png"),
        (raw_count_label, "post-stress", f"post_stress_{raw_count_prefix}counts.png"),
        (raw_duration_label, "pre-stress", f"pre_stress_{raw_duration_prefix}mean_duration.png"),
        (raw_duration_label, "post-stress", f"post_stress_{raw_duration_prefix}mean_duration.png"),
    ]
    pages: list[tuple[str, str]] = []
    for label, phase, filename_suffix in base_pages:
        pages.append((f"Control mice: {label}: {phase}", f"behavior_profile_control_mice_{filename_suffix}"))
        pages.append((f"Stressed mice: {label}: {phase}", f"behavior_profile_stressed_mice_{filename_suffix}"))
    return pages


def add_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image = Image.open(image_path)
    width, height = image.size
    fig_width = 11.69
    fig_height = max(4.0, fig_width * height / width + 0.55)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes((0.015, 0.02, 0.97, 0.9))
    ax.imshow(image)
    ax.axis("off")
    fig.suptitle(title, y=0.985, fontsize=11, fontweight="normal")
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def write_pdf(output_path: Path, pages: list[tuple[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for title, filename in pages:
            add_image_page(pdf, title, FIGURE_DIR / filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble pre/post profile PDFs from generated per-feature pages.")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR, help="Directory with per-feature pages. Defaults to the run outputs.")
    return parser.parse_args()


def main() -> None:
    global FIGURE_DIR
    FIGURE_DIR = parse_args().figure_dir
    requested_path = FIGURE_DIR / "pre_post_behavior_profiles_requested_8_pages.pdf"
    requested_log_raw_path = FIGURE_DIR / "pre_post_behavior_profiles_requested_8_pages_log_raw.pdf"
    raw_log_path = FIGURE_DIR / "pre_post_behavior_profiles_raw_log10p_4_pages.pdf"
    all_path = FIGURE_DIR / "pre_post_behavior_profiles_all_generated_10_pages.pdf"
    all_log_raw_path = FIGURE_DIR / "pre_post_behavior_profiles_all_generated_10_pages_log_raw.pdf"
    control_log_raw_path = FIGURE_DIR / "pre_post_behavior_profiles_control_mice_10_pages_log_raw.pdf"
    stressed_log_raw_path = FIGURE_DIR / "pre_post_behavior_profiles_stressed_mice_10_pages_log_raw.pdf"
    all_by_treatment_path = FIGURE_DIR / "pre_post_behavior_profiles_all_generated_20_pages_by_treatment.pdf"
    all_log_raw_by_treatment_path = FIGURE_DIR / "pre_post_behavior_profiles_all_generated_20_pages_log_raw_by_treatment.pdf"
    write_pdf(requested_path, REQUESTED_8_PAGE_ORDER)
    write_pdf(requested_log_raw_path, REQUESTED_8_PAGE_LOG_RAW_ORDER)
    write_pdf(raw_log_path, RAW_LOG_4_PAGE_ORDER)
    write_pdf(all_path, ALL_GENERATED_10_PAGE_ORDER)
    write_pdf(all_log_raw_path, ALL_GENERATED_10_PAGE_LOG_RAW_ORDER)
    write_pdf(control_log_raw_path, treatment_10_page_log_raw_order("control_mice_", "Control mice"))
    write_pdf(stressed_log_raw_path, treatment_10_page_log_raw_order("stressed_mice_", "Stressed mice"))
    write_pdf(all_by_treatment_path, treatment_pair_20_page_order(log_raw=False))
    write_pdf(all_log_raw_by_treatment_path, treatment_pair_20_page_order(log_raw=True))
    print(f"Wrote {requested_path}")
    print(f"Wrote {requested_log_raw_path}")
    print(f"Wrote {raw_log_path}")
    print(f"Wrote {all_path}")
    print(f"Wrote {all_log_raw_path}")
    print(f"Wrote {control_log_raw_path}")
    print(f"Wrote {stressed_log_raw_path}")
    print(f"Wrote {all_by_treatment_path}")
    print(f"Wrote {all_log_raw_by_treatment_path}")


if __name__ == "__main__":
    main()
