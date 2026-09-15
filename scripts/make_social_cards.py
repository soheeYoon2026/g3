"""Instagram cards that explain the geometry work without showing anyone's car.

    make_social_cards.py --out ~/다운로드/social

The vehicle models we work on are confidential, so every drawing here is synthetic: two
plates with a gap, a probe ball, a wrapped outline. The numbers quoted are ours and carry no
geometry. 1080x1350 (4:5), the size Instagram shows largest in the feed.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", type=Path, default=Path.home() / "다운로드" / "social")
ap.add_argument("--dpi", type=int, default=135)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

FONT = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"
title_f = FontProperties(fname=BOLD if Path(BOLD).exists() else FONT, size=40)
body_f = FontProperties(fname=FONT, size=23)
small_f = FontProperties(fname=FONT, size=17)
tag_f = FontProperties(fname=BOLD if Path(BOLD).exists() else FONT, size=19)

INK = "#10161d"
PAPER = "#f4f1ea"
RED = "#d8392b"
BLUE = "#2f5f8f"
GREY = "#98a2ad"
W, H = 1080, 1350


def card(name):
    fig = plt.figure(figsize=(W / args.dpi, H / args.dpi), dpi=args.dpi)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.06, 0.055), 0.88, 0.89, boxstyle="round,pad=0.012",
                                linewidth=2.2, edgecolor=INK, facecolor="none"))
    ax.text(0.09, 0.915, "AOX", fontproperties=tag_f, color=INK)
    ax.text(0.91, 0.915, name, fontproperties=small_f, color=GREY, ha="right")
    return fig, ax


def save(fig, filename):
    path = args.out / filename
    fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  {path.name}")


def ball(ax, x, y, r, colour, lw=2.6):
    """A visual circle: the card is 4:5, so a circle in axes coordinates comes out an egg."""
    ax.add_patch(Ellipse((x, y), width=2 * r * (H / W), height=2 * r,
                         facecolor="none", edgecolor=colour, lw=lw))


def plate(ax, x, y, w, h, angle=0.0, color=INK):
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    c, s = np.cos(angle), np.sin(angle)
    pts = pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=color))


# 1 — the problem
fig, ax = card("1 / 4")
ax.text(0.10, 0.855, "자동차 CAD는\n밀폐돼 있지 않다", fontproperties=title_f, color=INK, linespacing=1.3, va="top")
ax.text(0.10, 0.705, "설계 파일은 만들기 위한 것이라\n판과 판 사이에 틈이 남는다.\n유동 해석은 닫힌 면을 요구한다.",
        fontproperties=body_f, color="#3c4750", linespacing=1.7, va="top")
plate(ax, 0.33, 0.47, 0.30, 0.035)
plate(ax, 0.70, 0.47, 0.30, 0.035)
ax.annotate("", xy=(0.485, 0.47), xytext=(0.555, 0.47),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=2.6))
ax.text(0.52, 0.515, "틈", fontproperties=body_f, color=RED, ha="center")
ax.plot([0.33, 0.45, 0.52, 0.60, 0.80], [0.555, 0.520, 0.470, 0.420, 0.385],
        color=BLUE, lw=2.4, linestyle=(0, (5, 4)))
ax.annotate("", xy=(0.83, 0.378), xytext=(0.79, 0.387),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.4))
ax.text(0.10, 0.335, "격자가 이 길로 차 안까지 샌다.", fontproperties=body_f, color=BLUE, va="top")
ax.text(0.10, 0.255, "한 대 정리하는 데 사람 손으로 며칠.",
        fontproperties=body_f, color="#3c4750", linespacing=1.7, va="top")
save(fig, "card1_problem.png")

# 2 — the probe
fig, ax = card("2 / 4")
ax.text(0.10, 0.855, "공을 굴려서 막는다", fontproperties=title_f, color=INK, va="top")
ax.text(0.10, 0.765, "반지름 알파인 공이 들어가지 못하는 틈은 덮고,\n들어가는 구멍은 그대로 둔다.",
        fontproperties=body_f, color="#3c4750", linespacing=1.7, va="top")
# left: a narrow gap the ball cannot enter; right: a wide one it can
plate(ax, 0.27, 0.50, 0.26, 0.030)
plate(ax, 0.47, 0.50, 0.10, 0.030)
plate(ax, 0.79, 0.50, 0.22, 0.030)
top = 0.515
ax.plot([0.14, 0.40], [top, top], color=BLUE, lw=3.0)
ax.plot([0.40, 0.42, 0.52], [top, top + 0.004, top], color=BLUE, lw=3.0)
ax.plot([0.52, 0.60], [top, top], color=BLUE, lw=3.0)
ax.plot([0.60, 0.625, 0.665, 0.66, 0.70], [top, 0.470, 0.462, 0.470, top], color=BLUE, lw=3.0)
ax.plot([0.70, 0.90], [top, top], color=BLUE, lw=3.0)
ball(ax, 0.41, top + 0.031, 0.030, BLUE)
ax.text(0.41, 0.605, "좁은 틈 → 못 들어감 → 덮는다", fontproperties=small_f, color=BLUE, ha="center")
ball(ax, 0.645, 0.487, 0.023, RED)
ax.text(0.70, 0.415, "넓은 구멍 → 들어감 → 남긴다", fontproperties=small_f, color=RED, ha="center")
ax.text(0.10, 0.305, "규칙 한 줄:", fontproperties=body_f, color="#3c4750", va="top")
ax.text(0.10, 0.255, "알파 = 살려야 할\n가장 작은 구멍의 절반", fontproperties=title_f, color=INK,
        linespacing=1.3, va="top")
save(fig, "card2_alpha.png")

# 3 — the trade-off
fig, ax = card("3 / 4")
ax.text(0.10, 0.855, "알파 하나가\n두 가지를 가른다", fontproperties=title_f, color=INK, linespacing=1.3, va="top")
for k, (label, radius, colour, note) in enumerate([
        ("작게 잡으면", 0.020, BLUE, "구멍은 살지만 이음매로 샌다"),
        ("크게 잡으면", 0.040, RED, "새지 않지만 그릴·덕트가 막힌다")]):
    y = 0.575 - k * 0.235
    ax.text(0.10, y + 0.075, label, fontproperties=body_f, color=INK)
    plate(ax, 0.40, y, 0.28, 0.028)
    plate(ax, 0.76, y, 0.24, 0.028)
    # small ball drops into the gap, big one rests on top of it
    ball(ax, 0.565, y if k == 0 else y + radius + 0.012, radius, colour)
    ax.text(0.10, y - 0.065, note, fontproperties=small_f, color=colour, va="top")
ax.text(0.10, 0.205, "사람이 정할 것은 하나다.", fontproperties=body_f, color="#3c4750", va="top")
ax.text(0.10, 0.155, "“유동이 지나야 하는\n가장 작은 구멍은 무엇입니까”", fontproperties=body_f, color=INK,
        linespacing=1.6, va="top")
save(fig, "card3_tradeoff.png")

# 4 — the result, numbers only
fig, ax = card("4 / 4")
ax.text(0.10, 0.855, "자동으로 돌린 결과", fontproperties=title_f, color=INK, va="top")
rows = [("원본과의 차이 (중앙값)", "0.22 mm"),
        ("덧댄 면적", "17 %"),
        ("한 대 정리 시간", "6 분"),
        ("사람이 답할 질문", "5 개")]
for i, (label, value) in enumerate(rows):
    y = 0.685 - i * 0.115
    ax.text(0.10, y, label, fontproperties=body_f, color="#3c4750")
    ax.text(0.90, y, value, fontproperties=title_f, color=INK, ha="right", fontsize=34)
    ax.plot([0.10, 0.90], [y - 0.028, y - 0.028], color="#d8d2c6", lw=1.4)
ax.text(0.10, 0.185, "규칙이 재고, 도구를 돌리고, 결과를 검증한다.\n사람은 뜻이 필요한 곳에서만 답한다.",
        fontproperties=body_f, color=INK, linespacing=1.6, va="top")
save(fig, "card4_result.png")

caption = """차를 해석하려면 먼저 '닫아야' 합니다.

설계 파일은 만들기 위한 것이라 판과 판 사이에 틈이 남습니다.
그대로 격자를 만들면 차 안으로 샙니다.

그래서 반지름 알파인 공을 굴립니다.
공이 못 들어가는 틈은 덮고, 들어가는 구멍은 남깁니다.
알파를 작게 잡으면 이음매로 새고, 크게 잡으면 그릴과 덕트가 막힙니다.

규칙은 한 줄입니다.
알파 = 살려야 할 가장 작은 구멍의 절반.

지금은 이 판단을 코드가 하고, 사람은 뜻이 필요한 다섯 가지에만 답합니다.
표면은 원본에서 중앙값 0.22 mm 안에 들어옵니다.

#CFD #자동차 #공력 #해석 #형상전처리 #메시 #엔지니어링 #AOX
"""
(args.out / "caption.txt").write_text(caption, encoding="utf-8")
print(f"  caption.txt")
print(f"\n저장 위치: {args.out}")
