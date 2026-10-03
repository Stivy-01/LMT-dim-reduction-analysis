"""Rasterise the extracted behaviour configurations for the figures.

All configurations come from the same source figure, so they are rendered at
one common scale (pixels per SVG user unit).  That keeps the mice the same
size across pictograms when they are placed in the figures.

Usage:
    python scripts/build_configuration_assets.py <source_dir> <out_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

PX_PER_UNIT = 6.0          # rasterisation scale, common to every configuration
INK = "#111111"

NODE = (r"C:\Users\andre\.cache\codex-runtimes\codex-primary-runtime"
        r"\dependencies\node\bin\node.exe")
NODE_MODULES = (r"C:\Users\andre\.cache\codex-runtimes\codex-primary-runtime"
                r"\dependencies\node\node_modules")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dir", type=Path,
        nargs="?", default=Path("svg_configurations"))
    parser.add_argument(
        "out_dir", type=Path, nargs="?",
        default=Path(__file__).resolve().parents[1]
        / "src" / "visualization" / "configurations")
    return parser.parse_args()


def svg_units(path: Path) -> tuple[float, float]:
    head = path.read_text(encoding="utf-8")[:600]
    m = re.search(r'viewBox="([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+'
                  r'([-\d.eE]+)"', head)
    if m:
        return float(m.group(3)), float(m.group(4))
    m = re.search(r'width="([\d.]+)"\s+height="([\d.]+)"', head)
    if m:
        return float(m.group(1)), float(m.group(2))
    raise ValueError("cannot read size of %s" % path)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    svgs = sorted(args.source_dir.glob("*.svg"))
    if not svgs:
        raise SystemExit("no SVG found in %s" % args.source_dir)

    manifest: dict[str, dict[str, float]] = {}
    rendered: list[str] = []
    for src in svgs:
        w_units, h_units = svg_units(src)
        text = src.read_text(encoding="utf-8")
        text = text.replace("currentColor", INK)
        local = out_dir / src.name
        local.write_text(text, encoding="utf-8")
        px_w = max(int(round(w_units * PX_PER_UNIT)), 1)
        px_h = max(int(round(h_units * PX_PER_UNIT)), 1)
        stem = src.stem
        rendered.append(stem)
        manifest[stem] = {"w_units": w_units, "h_units": h_units,
                          "px_w": px_w, "px_h": px_h}

    # one render call per configuration, absolute paths so cwd does not matter
    calls = []
    abs_out = out_dir.resolve().as_posix()
    for stem in rendered:
        info = manifest[stem]
        calls.append(
            "await sharp('%s/%s.svg', {density: 400})"
            ".resize({width: %d, height: %d, fit: 'contain',"
            " background: {r:0,g:0,b:0,alpha:0}})"
            ".png().toFile('%s/%s.png');"
            % (abs_out, stem, info["px_w"], info["px_h"], abs_out, stem))
    script = ("const sharp = require('sharp');(async () => {%s"
              "console.log('ok');})();" % "".join(calls))
    env = dict(os.environ, NODE_PATH=NODE_MODULES)
    subprocess.run([NODE, "-e", script], check=True, env=env, cwd=str(out_dir))

    # trim the transparent margin so every pictogram fills the box it is
    # placed in (the common pixel-per-unit scale is preserved)
    from PIL import Image

    for stem in rendered:
        png = out_dir / f"{stem}.png"
        im = Image.open(png).convert("RGBA")
        bbox = im.getbbox()
        if bbox:
            im = im.crop(bbox)
            im.save(png)
        manifest[stem]["px_w"] = im.width
        manifest[stem]["px_h"] = im.height
        manifest[stem]["w_units"] = im.width / PX_PER_UNIT
        manifest[stem]["h_units"] = im.height / PX_PER_UNIT

    (out_dir / "scale.json").write_text(
        json.dumps({"px_per_unit": PX_PER_UNIT, "icons": manifest}, indent=1),
        encoding="utf-8")
    print("configurations:", len(rendered), "->", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
