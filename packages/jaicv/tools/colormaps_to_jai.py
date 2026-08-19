#!/usr/bin/env python3
"""Record OpenCV's colour maps as tables, so jaicv reproduces them exactly.

Every one of OpenCV's maps is a 256-entry lookup table, including the ones
that started life as formulas: `applyColorMap` builds them by interpolating
sixty-four control points, so a formula written from the same definition lands
a level or two away almost everywhere and nowhere near on `RAINBOW`, `PINK`
and `HOT`. Reading the tables straight out of OpenCV is the only way to agree
with it, so that is what this does -- it applies each map to a 0..255 ramp and
writes what comes back.

Run it against the OpenCV whose output you want to match:

    ~/.venvs/scratch/bin/python packages/jaicv/tools/colormaps_to_jai.py \
        packages/jaicv/src/jaicv/imgproc/colormap_data.jai
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

#: OpenCV's own numbering. The order is the file's index order too.
MAPS = [
    "AUTUMN", "BONE", "JET", "WINTER", "RAINBOW", "OCEAN", "SUMMER", "SPRING",
    "COOL", "HSV", "PINK", "HOT", "PARULA", "MAGMA", "INFERNO", "PLASMA",
    "VIRIDIS", "CIVIDIS", "TWILIGHT", "TWILIGHT_SHIFTED", "TURBO", "DEEPGREEN",
]

LEVELS = 256


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(sys.argv[1])

    ramp = np.arange(LEVELS, dtype=np.uint8).reshape(1, LEVELS)
    tables = []
    for index in range(len(MAPS)):
        tables.append(cv2.applyColorMap(ramp, index)[0])

    lines = [
        "#: OpenCV's colour maps, as the tables OpenCV actually applies.",
        "#:",
        "#: Each map is 256 entries of blue, green and red, in that order, laid",
        "#: end to end: map `id` starts at `id * 768`. Recorded from OpenCV"
        f" {cv2.__version__}",
        "#: by `tools/colormaps_to_jai.py`, which is how they come out exact --",
        "#: OpenCV builds even the maps that have a formula behind them by",
        "#: interpolating sixty-four control points, so writing the formula",
        "#: instead lands a level away nearly everywhere.",
        "#:",
        "#: Do not edit by hand.",
        f"pub let COLORMAP_LEVELS = {LEVELS}",
        f"pub let COLORMAP_COUNT = {len(MAPS)}",
        "",
        "pub let COLORMAP_TABLE = [",
    ]
    for name, table in zip(MAPS, tables):
        lines.append(f"    #: {name}")
        flat = [int(v) for row in table for v in row]
        for start in range(0, len(flat), 24):
            lines.append("    " + ", ".join(str(v) for v in flat[start:start + 24]) + ",")
    lines.append("]")
    out.write_text("\n".join(lines) + "\n")
    print(f"{out}: {len(MAPS)} maps, {len(MAPS) * LEVELS * 3} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
