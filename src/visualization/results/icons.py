# -*- coding: utf-8 -*-
"""Pittogrammi compositi delle categorie comportamentali."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "src" / "visualization" / "configurations"

# every configuration was rasterised at this common scale, so one zoom value
# keeps the mice the same size in every pictogram
try:
    _SCALE = json.loads((CONFIG_DIR / "scale.json").read_text(encoding="utf-8"))
except FileNotFoundError:                                   # pragma: no cover
    _SCALE = {"px_per_unit": 6.0, "icons": {}}
PX_PER_UNIT = float(_SCALE.get("px_per_unit", 6.0))

# schematic used for each behavioural category
CATEGORY_ICONS = {
    "Body configuration": "03_individual_rearing",
    "Isolated behavior": "01_individual_moving_alone",
    # the "contact type" band is wide enough for one schematic per contact
    # type: side by side, oral-oral (= nose-nose in the repertoire figure)
    # and oral-genital (= nose-anogenital)
    "Contact type": [
        "10_dyadic_state_contact_side_by_side_same_way",
        "12_dyadic_state_contact_nose_to_nose",
        "13_dyadic_state_contact_nose_to_anogenital",
    ],
    "Social configuration": "22_social_group_of_4_mice",
    "Social approach": "16_dyadic_dynamic_making_contact",
    "Social escape": "18_dyadic_dynamic_breaking_contact",
    # drawn schematics (see build_composite_icons): the arena view for the
    # place-related category and a contact-type sequence for the last one
    "Position/context": "_icon_position_context",
    "Contact sequence": "_icon_contact_sequence",
}

# how several schematics share one category band: "row" spreads them
# horizontally, "stack" puts them one above the other (for narrow bands)
CATEGORY_ICON_LAYOUT: dict[str, str] = {}

BAND_IN = 1.25          # height of the schematic strip above the panels
SHOW_LEGEND = False     # the colour key is drawn under each category name
INLINE_ICON_COLOR = "#111111"

# names are short in the strip; the long ones are broken by hand so they stay
# inside their own column
CATEGORY_DISPLAY = {
    "Body configuration": "Body\nconfiguration",
    "Isolated behavior": "Isolated\nbehavior",
    "Position/context": "Position/\ncontext",
    "Social configuration": "Social\nconfiguration",
    "Social approach": "Social\napproach",
    "Social escape": "Social\nescape",
    "Contact sequence": "Contact\nsequence",
}

# schematic shown next to each representative feature in panel D
FEATURE_ICONS = {
    "Contact_active_count": "10_dyadic_state_contact_side_by_side_same_way",
    "Stop_count": "02_individual_stopped_alone",
    "Group_4_make_count": "29_group_making_4_mice",
    "Group_4_break_count": "31_group_breaking_4_mice",
    "Move_isolated_count": "01_individual_moving_alone",
    "Rear_isolated_count": "03_individual_rearing",
    "Huddling_count": "04_individual_stopped_and_huddled",
    "Contact_mean_duration": "10_dyadic_state_contact_side_by_side_same_way",
    "Huddling_mean_duration": "04_individual_stopped_and_huddled",
}

_ICON_CACHE: dict[str, np.ndarray] = {}


def icon_image(name: str) -> np.ndarray | None:
    """Load one behaviour configuration (black, transparent background)."""
    if name in _ICON_CACHE:
        return _ICON_CACHE[name]
    path = CONFIG_DIR / f"{name}.png"
    if not path.exists():
        return None
    arr = np.asarray(Image.open(path).convert("RGBA"))
    _ICON_CACHE[name] = arr
    return arr


def icon_units(name: str) -> tuple[float, float]:
    info = _SCALE.get("icons", {}).get(name)
    if not info:
        arr = icon_image(name)
        if arr is None:
            return 1.0, 1.0
        return float(arr.shape[1]) / PX_PER_UNIT, float(arr.shape[0]) / PX_PER_UNIT
    return float(info["w_units"]), float(info["h_units"])


def icon_zoom(name: str, box_w_pt: float, box_h_pt: float,
              fill: float = 0.85) -> float:
    """Zoom that fits one configuration into the box, as large as it allows."""
    if icon_image(name) is None:
        return 0.0
    w_units, h_units = icon_units(name)
    per_unit = min(box_w_pt * fill / w_units, box_h_pt * fill / h_units)
    return per_unit / PX_PER_UNIT


def icon_aspect(name: str) -> float:
    """width / height of one configuration, in SVG units."""
    w_units, h_units = icon_units(name)
    return w_units / h_units if h_units else 1.0


def category_icon_list(category: str) -> list[str]:
    """Schematic(s) shown in a category band: one, or one per contact type."""
    icons = CATEGORY_ICONS.get(category)
    if not icons:
        return []
    return [icons] if isinstance(icons, str) else list(icons)


def _config_image(name: str):
    """Load one extracted configuration as RGBA."""
    return Image.open(CONFIG_DIR / f"{name}.png").convert("RGBA")


def _scale_to_height(img, height_px: int):
    ratio = height_px / float(img.height)
    return img.resize((max(1, int(round(img.width * ratio))), height_px),
                      Image.LANCZOS)


def _draw_arrow(draw, x0, y0, x1, y1, width=6):
    """Small solid arrow, used to join schematics in a sequence."""
    draw.line([(x0, y0), (x1, y1)], fill=INLINE_ICON_COLOR, width=width)
    head = max(width * 3, 14)
    if abs(x1 - x0) >= abs(y1 - y0):
        sign = 1 if x1 >= x0 else -1
        draw.polygon([(x1, y1), (x1 - sign * head, y1 - head * 0.7),
                      (x1 - sign * head, y1 + head * 0.7)],
                     fill=INLINE_ICON_COLOR)
    else:
        sign = 1 if y1 >= y0 else -1
        draw.polygon([(x1, y1), (x1 - head * 0.7, y1 - sign * head),
                      (x1 + head * 0.7, y1 - sign * head)],
                     fill=INLINE_ICON_COLOR)


def build_composite_icons() -> None:
    """Draw the two schematics that are not single repertoire configurations.

    Position/context is an arena seen from above with the centre zone shaded
    and mice placed in the centre and at the periphery; the contact-sequence
    schematic is two contact configurations joined by an arrow.
    """
    if _ICON_CACHE.get("_icon_position_context") is not None:
        return

    # --- arena view -------------------------------------------------------
    size = 900
    arena = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(arena)
    m = 40
    box = [m, m, size - m, size - m]
    draw.rounded_rectangle(box, radius=26, fill=(245, 245, 245, 255),
                           outline="#3a3a3a", width=7)
    inner = size * 0.30
    draw.rectangle([size / 2 - inner / 2, size / 2 - inner / 2,
                    size / 2 + inner / 2, size / 2 + inner / 2],
                   fill=(214, 214, 214, 255), outline="#8c8c8c", width=4)
    mouse = _scale_to_height(_config_image("02_individual_stopped_alone"), 150)
    arena.alpha_composite(mouse, (int(size / 2 - mouse.width / 2),
                                  int(size / 2 - mouse.height / 2)))
    rear = _scale_to_height(_config_image("03_individual_rearing"), 170)
    rear = rear.transpose(Image.FLIP_LEFT_RIGHT)
    arena.alpha_composite(rear, (int(size * 0.72 - rear.width / 2),
                                 int(size * 0.80 - rear.height / 2)))
    _ICON_CACHE["_icon_position_context"] = np.asarray(arena)

    # --- contact sequence -------------------------------------------------
    height = 380
    left = _scale_to_height(_config_image("13_dyadic_state_contact_nose_to_anogenital"),
                            height)
    right = _scale_to_height(_config_image("12_dyadic_state_contact_nose_to_nose"),
                             height)
    gap = 110
    canvas = Image.new("RGBA", (left.width + gap + right.width,
                                height + 40), (255, 255, 255, 0))
    canvas.alpha_composite(left, (0, 20))
    canvas.alpha_composite(right, (left.width + gap, 20))
    draw = ImageDraw.Draw(canvas)
    y = int(canvas.height / 2)
    _draw_arrow(draw, left.width + 24, y, left.width + gap - 24, y, width=16)
    _ICON_CACHE["_icon_contact_sequence"] = np.asarray(canvas)


def place_icon(ax: plt.Axes, name: str, xy, xycoords, zoom,
               box_alignment=(0.5, 0.5), pad=0.0):
    arr = icon_image(name)
    if arr is None or zoom <= 0:
        return
    box = AnnotationBbox(OffsetImage(arr, zoom=zoom), xy,
                         xycoords=xycoords, frameon=False,
                         box_alignment=box_alignment, pad=pad,
                         annotation_clip=False)
    ax.add_artist(box)

CONTROL_COLOR = "#0072B2"
STRESSED_COLOR = "#D55E00"

