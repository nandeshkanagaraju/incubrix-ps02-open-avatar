#!/usr/bin/env python
"""Labelled contact sheet of a batch, for the demo and the report.

    python scripts/contact_sheet.py evidence/batch_a_sd15 /tmp/sheet.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAD, CAP, GAP, COLS = 22, 58, 14, 3
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
    ap.add_argument("batch_dir"); ap.add_argument("out")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    bd = Path(a.batch_dir)
    manifest = json.loads((bd / "avatar_manifest.json").read_text())
    entries = [e for e in manifest["avatars"] if (bd / "images" / e["output"]["file"]).exists()]
    if not entries:
        print("no images", flush=True)
        return 1

    ims, caps = [], []
    for e in entries:
        ims.append(Image.open(bd / "images" / e["output"]["file"]).convert("RGB"))
        sp = e["spec"] or {}
        ap_ = sp.get("appearance", {})
        caps.append((
            f"{e['spec_id']}   seed {e['seed']}",
            f"{ap_.get('age_band','?')} · {ap_.get('presentation','?')} · {ap_.get('skin_tone','?')}",
            f"{ap_.get('hair_length','?')}/{ap_.get('hair_texture','?')} · {ap_.get('attire','?')}",
        ))

    w = h = max(max(i.width for i in ims), max(i.height for i in ims))
    ims = [i.resize((w, h), Image.LANCZOS) for i in ims]
    rows = (len(ims) + COLS - 1) // COLS
    title_h = 52
    W = PAD * 2 + COLS * w + (COLS - 1) * GAP
    H = PAD * 2 + title_h + rows * (h + CAP) + (rows - 1) * GAP

    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    f_t, f_c, f_s = _font(24), _font(16), _font(13)
    d.text((PAD, PAD), a.title or f"{bd.name} — {len(ims)} avatars", font=f_t, fill=FG)

    for idx, (im, cap) in enumerate(zip(ims, caps)):
        r, c = divmod(idx, COLS)
        x = PAD + c * (w + GAP)
        y = PAD + title_h + r * (h + CAP + GAP)
        canvas.paste(im, (x, y))
        d.text((x, y + h + 6), cap[0], font=f_c, fill=FG)
        d.text((x, y + h + 25), cap[1], font=f_s, fill=SUB)
        d.text((x, y + h + 41), cap[2], font=f_s, fill=SUB)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.out)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
