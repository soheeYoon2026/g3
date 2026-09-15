"""One card built around a real render, not a diagram.

    make_social_photo_card.py --image var/social/drivaer.png --out ~/다운로드/social

The four drawn cards read as tool output: even beige, hairline frame, evenly spaced
paragraphs. This one is the opposite: the picture bleeds to the edges, the type sits on a
dark band at the bottom, one number is allowed to be big. The geometry shown is DrivAer, the
public reference car, so nothing confidential leaves the machine.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--image", type=Path, required=True)
ap.add_argument("--out", type=Path, default=Path.home() / "다운로드" / "social")
ap.add_argument("--crop", default="0.508,0.060,0.990,0.515", help="left,top,right,bottom as fractions")
ap.add_argument("--headline", default="이 차는 아직\n해석할 수 없다")
ap.add_argument("--body", default="문 틈, 그릴, 휠하우스가 열려 있어서\n격자가 차 안으로 샌다.")
ap.add_argument("--stat", default="0.22 mm")
ap.add_argument("--stat_label", default="정리 후 원본과의 차이 (중앙값)")
ap.add_argument("--footer", default="DrivAer · 공개 기준 형상")
ap.add_argument("--name", default="photo_card.png")
ap.add_argument("--dark-bg", action="store_true", help="repaint the renderer's light ground dark")
args = ap.parse_args()

W, H = 1080, 1350
FONT = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"
head_f = ImageFont.truetype(BOLD, 56)
body_f = ImageFont.truetype(FONT, 27)
stat_f = ImageFont.truetype(BOLD, 44)
small_f = ImageFont.truetype(FONT, 20)

src = Image.open(args.image).convert("RGB")
l, t, r, b = (float(v) for v in args.crop.split(","))
src = src.crop((int(l * src.width), int(t * src.height), int(r * src.width), int(b * src.height)))
scale = max(W / src.width, (H * 0.62) / src.height)
src = src.resize((int(src.width * scale), int(src.height * scale)), Image.LANCZOS)
src = ImageEnhance.Contrast(src).enhance(1.12)
if args.dark_bg:
    # the renderer paints a light ground; swap it for the card's so the picture does not
    # end in a white slab above the text
    import numpy as np
    a = np.asarray(src).astype(np.int16)
    light = (a.min(axis=2) > 232)
    a[light] = np.array([16, 18, 22])
    edge = (~light) & (a.min(axis=2) > 205)
    a[edge] = (a[edge] * 0.45 + np.array([16, 18, 22]) * 0.55).astype(np.int16)
    src = Image.fromarray(a.astype("uint8"))

card = Image.new("RGB", (W, H), (16, 18, 22))
# with the ground repainted dark there is no edge to hide: sit the picture in the upper half
card.paste(src, ((W - src.width) // 2, int(H * 0.30) - src.height // 2))
draw = ImageDraw.Draw(card, "RGBA")
if not args.dark_bg:
    band = int(H * 0.50)
    for i in range(240):
        draw.line([(0, band + i), (W, band + i)], fill=(16, 18, 22, int(255 * (i / 240) ** 1.4)))
    draw.rectangle([(0, band + 240), (W, H)], fill=(16, 18, 22))

y = int(H * 0.60)
draw.multiline_text((64, y), args.headline, font=head_f, fill=(243, 241, 236), spacing=16)
y += 56 * (args.headline.count("\n") + 1) + 58
draw.multiline_text((64, y), args.body, font=body_f, fill=(168, 176, 186), spacing=12)
y += 27 * (args.body.count("\n") + 1) + 46
draw.line([(64, y), (W - 64, y)], fill=(58, 64, 72), width=2)
draw.text((64, y + 26), args.stat, font=stat_f, fill=(232, 78, 62))
draw.text((64 + draw.textlength(args.stat, font=stat_f) + 20, y + 44), args.stat_label,
          font=small_f, fill=(150, 158, 168))
draw.text((64, H - 62), args.footer, font=small_f, fill=(110, 118, 128))
draw.text((W - 64 - draw.textlength("AOX", font=small_f), H - 62), "AOX", font=small_f, fill=(200, 205, 212))

args.out.mkdir(parents=True, exist_ok=True)
card.save(args.out / args.name)
print(f"저장 {args.out / args.name}")
