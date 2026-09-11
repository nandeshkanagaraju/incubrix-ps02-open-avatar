#!/usr/bin/env python
"""Compose a labelled side-by-side of the same job rendered on two routes.

Used by the demo to show reproducibility visually: the image just generated on
this laptop next to the committed Kaggle-GPU render of the same spec and seed.

    python scripts/compare_image.py <new.png> <reference.png> <out.png> \
        --left "label" --right "label" [--note "text"]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAD, BAR, GAP = 24, 46, 20
BG, FG, SUB = (18, 18, 20), (238, 238, 240), (150, 150, 158)


def _font(size: int):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left"); ap.add_argument("right"); ap.add_argument("out")
    ap.add_argument("--left-label", default="A"); ap.add_argument("--right-label", default="B")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    ims = [Image.open(a.left).convert("RGB"), Image.open(a.right).convert("RGB")]
    h = max(i.height for i in ims)
    ims = [i.resize((round(i.width * h / i.height), h), Image.LANCZOS) for i in ims]

    note_h = 40 if a.note else 0
    W = PAD * 2 + GAP + sum(i.width for i in ims)
    H = PAD * 2 + BAR + h + note_h
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    f_lab, f_note = _font(20), _font(17)

    x = PAD
    for im, lab in zip(ims, (a.left_label, a.right_label)):
        d.text((x, PAD), lab, font=f_lab, fill=FG)
        canvas.paste(im, (x, PAD + BAR))
        x += im.width + GAP

    if a.note:
        d.text((PAD, PAD + BAR + h + 12), a.note, font=f_note, fill=SUB)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.out)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
