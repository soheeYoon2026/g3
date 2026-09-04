"""One page for BAIC: what happened to CAS-A between their STEP and a CFD surface.

Four panels left to right, each with a real render and its numbers, and a strip
underneath saying what still needs their answer. Built locally with PIL - the
geometry is theirs, but it does not go through any hosting on the way. A PPTX
wrapper with the same image is written when python-pptx is available.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", type=Path, required=True, help="output PNG")
ap.add_argument("--raw", type=Path, required=True, help="render of the raw model with holes")
ap.add_argument("--healed", type=Path, required=True, help="render of the healed STEP")
ap.add_argument("--wrapped", type=Path, required=True, help="render of the watertight wrap")
args = ap.parse_args()


def font(size, bold=False):
    for path in ("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf" if bold else
                 "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                 "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# Tile geometry of render_geometry.py sheets: 2x2 tiles of 760x520, 10 px gaps,
# 34 px title band
def tile(sheet, row, col):
    x = 10 + col * 770
    y = 44 + row * 530
    # skip the tile's own label band at the top
    return sheet.crop((x + 1, y + 34, x + 759, y + 519))


W, H = 1920, 1080
page = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(page)
INK = (28, 30, 36)
MUTED = (96, 100, 110)
ACCENT = (24, 96, 176)
RULE = (214, 218, 224)

d.text((60, 42), "CAS-A: from styling surfaces to a CFD-ready closed surface",
       fill=INK, font=font(38, bold=True))
d.text((60, 92), "Automated where the geometry decides; explicit where an engineer must. "
                 "All numbers measured on the CAS-A STEP you provided.",
       fill=MUTED, font=font(20))

panels = [
    ("1  As received", args.raw, (0, 0), [
        "CATIA V5 export, 2,847 faces / 2,847 shells",
        "12,724 free edges; panels overlap",
        "(glass sits 4 mm above the body)",
        "46 openings: underbody 4.85 m, cabin,",
        "wheels, panel gaps",
    ], "coloured lines = open boundaries"),
    ("2  Automated healing (B-rep)", args.healed, (0, 0), [
        "Sewn in stages 1 → 5 → 10.5 mm → 19 shells",
        "40 of 46 openings closed",
        "Every patch verified: within 10 mm,",
        "nothing added outside the body",
        "Returned as STEP — real surfaces, 35 MB",
    ], "remaining lines = left for decisions"),
    ("3  Engineering decisions", args.wrapped, (1, 0), [
        "Underbody: flat floor at the sill (z = 150)",
        "Symmetry: mirrored about y = 0",
        "Wheels: closed rims (on request)",
        "Glass overlay: needs trimmed CAD,",
        "not repair — left as is",
    ], "underside after the decisions; each one declared"),
    ("4  CFD-ready surface", args.wrapped, (0, 0), [
        "Alpha wrap 15 mm → watertight",
        "550,656 triangles, volume 7.34 m³",
        "Frontal area 2.526 m² (yours: 2.52)",
        "Overlaps and seams resolved by the wrap",
        "Solver gets this STL; engineer gets the STEP",
    ], "one closed surface, no hole left"),
]

x0, y0 = 60, 140
pw, gap = 435, 20
ph = 300
for i, (title, path, (row, col), lines, caption) in enumerate(panels):
    x = x0 + i * (pw + gap)
    sheet = Image.open(path).convert("RGB")
    img = tile(sheet, row, col).resize((pw, ph), Image.LANCZOS)
    page.paste(img, (x, y0 + 40))
    d.rectangle([x, y0 + 40, x + pw - 1, y0 + 40 + ph - 1], outline=RULE)
    d.text((x, y0), title, fill=ACCENT, font=font(24, bold=True))
    d.text((x, y0 + 40 + ph + 8), caption, fill=MUTED, font=font(15))
    ty = y0 + 40 + ph + 38
    for line in lines:
        d.text((x, ty), "•  " + line, fill=INK, font=font(16))
        ty += 25
    if i < len(panels) - 1:
        ax = x + pw + 3
        ay = y0 + 40 + ph // 2
        d.polygon([(ax, ay - 8), (ax + 14, ay), (ax, ay + 8)], fill=ACCENT)

# bottom strip
sy = 700
d.line([(60, sy), (W - 60, sy)], fill=RULE, width=2)
d.text((60, sy + 18), "What we need from BAIC to remove the assumptions",
       fill=INK, font=font(24, bold=True))
asks = [
    ("Underbody", "Is there an underbody model, or is a flat floor acceptable? At what ground clearance?"),
    ("Trimmed surfaces", "The ANSA-cleaned counterpart of the same CAS — lets us resolve the overlapping panels and benchmark the automation."),
    ("Wheels", "Closed rims, or open spokes / rotating wheels? This moves Cd directly."),
    ("Openings to keep", "Smallest inlet or slot that must stay open, so the sealing size sits below it."),
]
ay = sy + 62
for label, text in asks:
    d.text((60, ay), label, fill=ACCENT, font=font(19, bold=True))
    d.text((300, ay), text, fill=INK, font=font(19))
    ay += 34

d.text((60, H - 70), "Frontal area from projected union of triangles, 0.5 mm raster. "
                     "Volume 49% of bounding box (a closed car measures 50-56%). "
                     "Flat floor at the sill moves absolute Cd; intended for trend work until the underbody is known.",
       fill=MUTED, font=font(15))
d.text((60, H - 44), "AOX · geometry preparation for G3 · 2026-09-04", fill=MUTED, font=font(15))

args.out.parent.mkdir(parents=True, exist_ok=True)
page.save(args.out)
print(f"PNG: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")

try:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_picture(str(args.out), 0, 0, width=prs.slide_width, height=prs.slide_height)
    pptx_path = args.out.with_suffix(".pptx")
    prs.save(pptx_path)
    print(f"PPTX: {pptx_path}  ({pptx_path.stat().st_size / 1e6:.1f} MB)")
except Exception as exc:
    print(f"PPTX 생략: {type(exc).__name__}: {exc}")
