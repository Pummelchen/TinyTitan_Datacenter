#!/usr/bin/env python3
"""Generate the TinyTitan app icon from the project's brand image.

The Mac app's Dock icon is the same artwork the README leads with
(`assets/tinytitan-hero.png`), clipped to the rounded-square shape an app icon
needs and centred in a transparent gutter so it sits correctly in the Dock.

Deterministic and dependency-light: Pillow only, no SVG rasterizer. Re-run it to
regenerate the committed PNG:

    python3 tools/make_app_icon.py                     # writes the app resource
    python3 tools/make_app_icon.py /tmp/icon.png       # or anywhere

It is deliberately not wired into the build. The PNG is the committed asset
(`sources/TinyTitanApp/Mac/Resources/tinytitan-app-icon.png`, declared in
`Package.swift` and turned into a `.icns` by `tools/install_tinytitan.sh`); this
script exists so the asset can be reproduced rather than being an opaque blob.

Requires `PIL` (`python3 -m pip install pillow`). Note that this is a *separate*
dependency from the converters' numpy/ml_dtypes/safetensors stack that
`tools/lib/python.sh` resolves, so it is not part of that search: it is a
one-off asset generator, not part of an install or a release.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
HERO = ROOT / "assets" / "tinytitan-hero.png"
DEFAULT_OUT = ROOT / "sources" / "TinyTitanApp" / "Mac" / "Resources" / "tinytitan-app-icon.png"

SIZE = 1024          # the committed asset's size
SUPERSAMPLE = 4      # the mask is drawn 4x and reduced, so its edge is smooth
MARGIN = 48          # transparent gutter; the artwork occupies 928x928 of 1024
RADIUS = 207         # macOS-style corner radius for that content box


def squircle_mask() -> Image.Image:
    """An antialiased rounded-square mask via supersampled drawing."""
    big = SIZE * SUPERSAMPLE
    margin = MARGIN * SUPERSAMPLE
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (margin, margin, big - margin, big - margin),
        radius=RADIUS * SUPERSAMPLE, fill=255,
    )
    return mask.resize((SIZE, SIZE), Image.LANCZOS)


def artwork() -> Image.Image:
    """The brand image, scaled to cover the content box without distortion."""
    inner = SIZE - 2 * MARGIN
    source = Image.open(HERO).convert("RGBA")
    scale = max(inner / source.width, inner / source.height)
    scaled = source.resize(
        (round(source.width * scale), round(source.height * scale)), Image.LANCZOS
    )
    left = (scaled.width - inner) // 2
    top = (scaled.height - inner) // 2
    return scaled.crop((left, top, left + inner, top + inner))


def build() -> Image.Image:
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.paste(artwork(), (MARGIN, MARGIN))
    canvas.putalpha(squircle_mask())
    return canvas


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    icon = build()
    icon.save(out, format="PNG", optimize=True)
    print(f"wrote {out} ({out.stat().st_size} bytes, {icon.width}x{icon.height} {icon.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
