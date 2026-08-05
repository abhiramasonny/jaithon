#!/usr/bin/env python3
"""Render every shipped raster from assets/logo/jaithon.svg.

    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
      uv run --with cairosvg --with pillow python scripts/render_logo.py

The glyph fills only part of the SVG's 1000x1000 viewBox, so rendering the
canvas as-is leaves lopsided padding and wastes pixels at icon sizes. Measure
the inked bounds, pad to a square with a uniform margin so the mark is
optically centred, and downsample every size from one high-resolution
rasterisation — re-rendering per size would let cairo's hinting shift the
curve slightly between them.
"""

import io
import sys
from pathlib import Path

import cairosvg
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "assets/logo/jaithon.svg"
OUT = ROOT / "assets/logo"

RENDER = 4096          # well above the largest output
MARGIN = 0.10          # breathing room, as a fraction of the glyph's long side
SIZES = (512, 256, 192, 128, 96, 64, 48, 32, 16)

# Consumers that need their own copy of a particular size.
COPIES = {
    128: [ROOT / "editors/vscode/icon.png", ROOT / "editors/vscode/file-icon.png"],
}


def main() -> int:
    if not SVG.exists():
        print(f"error: {SVG} not found", file=sys.stderr)
        return 1

    full = Image.open(io.BytesIO(cairosvg.svg2png(
        url=str(SVG), output_width=RENDER, output_height=RENDER))).convert("RGBA")

    bounds = full.getbbox()
    if bounds is None:
        print("error: the SVG rendered empty", file=sys.stderr)
        return 1
    glyph = full.crop(bounds)

    margin = round(max(glyph.size) * MARGIN)
    side = max(glyph.size) + margin * 2
    master = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    master.paste(glyph, ((side - glyph.width) // 2, (side - glyph.height) // 2), glyph)

    master.resize((1024, 1024), Image.LANCZOS).save(OUT / "jaithon.png")
    print(f"jaithon.png          1024  (from a {side}px square, glyph {glyph.size})")

    for n in SIZES:
        scaled = master.resize((n, n), Image.LANCZOS)
        path = OUT / f"jaithon-{n}.png"
        scaled.save(path)
        for copy in COPIES.get(n, []):
            copy.parent.mkdir(parents=True, exist_ok=True)
            scaled.save(copy)
            print(f"{path.name:20} {n:4}  -> {copy.relative_to(ROOT)}")
        else:
            if n not in COPIES:
                print(f"{path.name:20} {n:4}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
