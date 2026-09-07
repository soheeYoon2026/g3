"""Charts and the architecture diagram for the geometry-cleaning technical document.

Every number plotted here was measured on CAS-A during the work and is recorded
in VERSION2_PLAN.md; the charts put the measurements next to the thresholds they
justified. Written to a folder of PNGs that the document embeds.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

CJK = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
if Path(CJK).exists():
    font_manager.fontManager.addfont(CJK)
    # The family name inside the TTC is what matplotlib matches on; take it from
    # the file rather than guessing "Noto Sans CJK KR"
    family = font_manager.FontProperties(fname=CJK).get_name()
    plt.rcParams["font.family"] = [family, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
INK, GREY, BLUE, RED = "#1c1e24", "#8a8f99", "#1f5fa8", "#c8302a"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(args.out / name, dpi=150)
    plt.close(fig)
    print("  ", name)


# 1. sewing sweep: single pass vs cumulative -------------------------------
tol = [0.26, 0.53, 1.05, 2.63, 5.25, 10.5, 21.0]
single_free = [1779, 1709, 1591, 1621, 1014, 1203, 1455]
single_inv = [63, 67, 33, 65, 81, 96, 182]
cum_tol = [1.05, 2.63, 5.25, 10.5, 21.0]
cum_free = [1591, 1538, 746, 722, 582]
cum_inv = [33, 48, 55, 58, 78]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogx(tol, single_free, "o-", color=GREY, label="단일 통과")
ax[0].semilogx(cum_tol, cum_free, "s-", color=BLUE, label="누적 (1.05에서 시작)")
ax[0].set_xlabel("꿰맴 허용오차 (mm)"); ax[0].set_ylabel("자유모서리 수"); ax[0].legend()
ax[0].set_title("무엇을 닫는가")
ax[1].semilogx(tol, single_inv, "o-", color=GREY, label="단일 통과")
ax[1].semilogx(cum_tol, cum_inv, "s-", color=BLUE, label="누적")
ax[1].axvspan(1.0, 11, color=BLUE, alpha=0.08)
ax[1].text(1.15, 170, "채택한 사다리 1.05 → 5.25 → 10.5", color=BLUE, fontsize=9)
ax[1].set_xlabel("꿰맴 허용오차 (mm)"); ax[1].set_ylabel("무효면 수"); ax[1].legend()
ax[1].set_title("무엇을 망가뜨리는가")
for a in ax:
    a.grid(alpha=0.25)
save(fig, "fig_sewing_sweep.png")

# 2. patch acceptance: area ratio and reach populations ----------------------
good_ratio = [0.39] * 8 + [1.09, 0.08, 0.20, 0.26, 1.15, 0.29, 0.29, 0.29, 0.30, 0.33, 0.30, 0.33, 0.35, 0.35, 0.98, 1.05, 1.09, 1.09, 1.12, 0.45, 0.21, 0.12, 0.25, 0.14]
bad_ratio = [204.19, 21.66, 12.27, 38.61, 6.49, 1.90, 33.21, 26.31, 7.25]
good_reach = [0.00, 0.00, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.06, 0.06, 0.07, 0.40, 0.49, 0.52]
bad_reach = [1.18, 1.18, 1.20, 1.20, 1.21, 1.22, 1.23, 1.23, 1.23, 1.24, 2.98, 3.11, 3.12, 3.16, 3.23, 7.44, 17.90, 42.04]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, good, bad, thr, label in ((ax[0], good_ratio, bad_ratio, 1.5, "패치 면적 / 구멍 크기²"),
                                 (ax[1], good_reach, bad_reach, 0.75, "구멍 상자 밖 도달 / 구멍 크기")):
    a.scatter(good, np.random.default_rng(1).uniform(-0.3, 0.3, len(good)), color=BLUE, s=22, label="수락")
    a.scatter(bad, np.random.default_rng(2).uniform(-0.3, 0.3, len(bad)), color=RED, s=22, marker="x", label="거절")
    a.axvline(thr, color=INK, ls="--", lw=1)
    a.text(thr * 1.15, -0.45, f"기준 {thr}", fontsize=9)
    a.set_ylim(-0.55, 0.55)
    a.set_xscale("log"); a.set_yticks([]); a.set_xlabel(label); a.legend(loc="upper right", fontsize=8)
    a.grid(axis="x", alpha=0.25)
ax[0].set_title("면적으로 보이는 폭주")
ax[1].set_title("면적으로는 안 보이는 폭주 (가느다란 조각)")
save(fig, "fig_patch_populations.png")

# 3. hole sizes and the sealing-size gap --------------------------------------
sizes = [4851, 2537, 871, 799, 785, 762, 757, 696, 696, 696, 696, 471, 317, 238, 225, 221, 219, 217, 215, 213,
         210, 210, 208, 206, 206, 204, 204, 203, 201, 200, 200, 198, 197, 195, 155, 153, 150, 148, 145, 145, 68, 38, 7]
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.semilogy(range(len(sizes)), sizes, "o", color=INK, ms=4)
ax.axhline(1487, color=BLUE, ls="--", lw=1)
ax.text(len(sizes) - 12, 1700, "자동 제안 봉합 크기 1,487 (2537 ↔ 871 사이)", color=BLUE, fontsize=9)
ax.axhline(900, color=GREY, ls=":", lw=1)
ax.text(len(sizes) - 12, 1000, "손으로 쓴 값 900", color=GREY, fontsize=9)
ax.annotate("언더바디 + 대칭면", (0, 4851), (3, 4000), fontsize=9, arrowprops=dict(arrowstyle="-", color=GREY))
ax.annotate("캐빈 밴드", (1, 2537), (4, 2000), fontsize=9, arrowprops=dict(arrowstyle="-", color=GREY))
ax.annotate("유리·휠 림", (3, 799), (7, 1100), fontsize=9, arrowprops=dict(arrowstyle="-", color=GREY))
ax.annotate("스포크 틈", (20, 210), (24, 330), fontsize=9, arrowprops=dict(arrowstyle="-", color=GREY))
ax.set_xlabel("자유경계 (크기 순)"); ax.set_ylabel("크기 (mm, 상자 대각선)"); ax.grid(alpha=0.25)
ax.set_title("CAS-A 열린 경계 46곳의 크기 분포")
save(fig, "fig_hole_sizes.png")

# 4. floor height scan -------------------------------------------------------
scan_z = [-290, -230, -170, -110, -50, 10, 70, 130, 190, 250, 310, 370, 430]
scan_a = [0.5, 1.2, 1.3, 1.5, 1.5, 1.6, 73.3, 76.4, 77.0, 1.4, 1.3, 1.1, 81.3]
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(scan_z, scan_a, "o-", color=INK)
ax.axvline(150, color=BLUE, ls="--", lw=1); ax.text(160, 40, "채택: 첫 닫힘 + 한 단계 = 150", color=BLUE, fontsize=9)
ax.axvline(410, color=RED, ls=":", lw=1); ax.text(300, 55, "'최대 윤곽' 규칙이 고른 410 (도어 높이)", color=RED, fontsize=9)
ax.annotate("휠 아치가 열린 휠하우스로 뚫려 윤곽이 다시 열림", (310, 1.3), (200, 22), fontsize=9,
            arrowprops=dict(arrowstyle="-", color=GREY))
ax.set_xlabel("단면 높이 z (mm)"); ax.set_ylabel("닫힌 윤곽 면적 / 평면 상자 (%)"); ax.grid(alpha=0.25)
ax.set_title("평바닥 높이 결정: 수평 단면 훑기")
save(fig, "fig_floor_scan.png")

# 5. architecture diagram (PIL) ------------------------------------------------
W, H = 1600, 620
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)


def font(size, bold=False):
    for p in ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc" if bold
              else "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def box(x, y, w, h, title, lines, tint=(245, 247, 250)):
    d.rectangle([x, y, x + w, y + h], fill=tint, outline=(150, 155, 165), width=2)
    d.text((x + 16, y + 12), title, fill=INK, font=font(21, True))
    for k, line in enumerate(lines):
        d.text((x + 16, y + 50 + k * 26), line, fill=INK, font=font(16))


def arrow(x1, y, x2, label=""):
    d.line([(x1, y), (x2 - 12, y)], fill=(90, 95, 105), width=3)
    d.polygon([(x2 - 12, y - 8), (x2, y), (x2 - 12, y + 8)], fill=(90, 95, 105))
    if label:
        d.text(((x1 + x2) / 2 - 40, y - 30), label, fill=(90, 95, 105), font=font(14))


box(30, 200, 210, 200, "입력 STEP", ["CATIA 서피스 내보내기", "면 = 셸, 자유모서리 다수", "판 겹침·자기교차", "색·부품 구분 없음"])
arrow(240, 300, 300)
box(300, 120, 380, 360, "1  B-rep 층  (OpenCascade)", [
    "진단 → 단계 꿰맴 1→5→10.5",
    "자유경계 검출 · 크기 분류",
    "봉합 크기 이하: 공유 모서리 캡",
    "  (투영·다각형·GeomPlate 예비)",
    "수락 검사: 면적 · 도달 · 뒤덮임",
    "STEP 되쓰기 (pcurve 없이)",
    "→ 의도 목록 intent.md"])
arrow(680, 300, 740, "healed.stp")
box(740, 120, 340, 360, "2  메쉬 층", [
    "감김 맞춘 삼각화",
    "경계 정점 병합",
    "남은 고리: 최소가중 삼각화",
    "  / 귀 자르기 / 부채꼴",
    "세분 상한 · 주머니 규칙",
    "(페어링: 있음, 기본 꺼짐)",
    "→ mesh.stl"])
arrow(1080, 300, 1140, "+ 평바닥 · 미러")
box(1140, 120, 300, 360, "3  랩 층  (CGAL)", [
    "알파 랩 15~29 mm",
    "겹침·이음매 흡수",
    "수밀 검사 · 체적",
    "전면 면적 대조",
    "→ wrapped.stl (솔버)"])
d.text((1465, 250), "솔버", fill=INK, font=font(18, True))
d.text((1465, 280), "CFD 격자", fill=GREY, font=font(15))
# side outputs
d.rectangle([300, 520, 1080, 590], fill=(255, 250, 240), outline=(200, 180, 140), width=2)
d.text((316, 532), "의도 게이트 — 봉합 크기 · close-near(닫힌 림) · 평바닥 높이 · 대칭면:  도구가 정하지 않고 사람이 정한 것을 기록",
       fill=INK, font=font(16))
d.rectangle([30, 30, 680, 90], fill=(240, 246, 252), outline=(150, 170, 200), width=2)
d.text((46, 42), "자동 제안기 (autotune) — 단위 · 절반 모델 · 꿰맴 사다리 · 봉합 크기 · 휠 위치를 모델에서 측정해 제안",
       fill=INK, font=font(16))
img.save(args.out / "fig_architecture.png")
print("   fig_architecture.png")

# 6. AI helper diagram ----------------------------------------------------------
W2, H2 = 1600, 560
img = Image.new("RGB", (W2, H2), "white")
d = ImageDraw.Draw(img)


def box2(x, y, w, h, title, lines, tint):
    d.rectangle([x, y, x + w, y + h], fill=tint, outline=(150, 155, 165), width=2)
    d.text((x + 16, y + 12), title, fill=INK, font=font(20, True))
    for k, line in enumerate(lines):
        d.text((x + 16, y + 48 + k * 25), line, fill=INK, font=font(15))


def arrow2(x1, y1, x2, y2, label=""):
    d.line([(x1, y1), (x2, y2)], fill=(90, 95, 105), width=3)
    dx, dy = x2 - x1, y2 - y1
    L = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    d.polygon([(x2, y2), (x2 - 14 * ux + 7 * px, y2 - 14 * uy + 7 * py),
               (x2 - 14 * ux - 7 * px, y2 - 14 * uy - 7 * py)], fill=(90, 95, 105))
    if label:
        d.text(((x1 + x2) / 2 + 8, (y1 + y2) / 2 - 24), label, fill=(90, 95, 105), font=font(14))


d.text((30, 20), "AI helper — 무엇이 숫자를 정하고, 무엇이 뜻을 정하는가", fill=INK, font=font(22, True))
box2(30, 80, 330, 200, "모델 (STEP)", ["측정 대상", "대각선 · 정점 분포", "꿰맴 쓸기 반응", "구멍 크기 분포", "고리 원형도·위치"],
     (245, 247, 250))
box2(430, 80, 380, 200, "측정 기반 제안기  (지금)", ["단위 · 절반 모델 · 사다리", "봉합 크기 · 닫힌 림 후보",
                                                "→ params.json (값 + 근거)", "→ questions (뜻이 필요한 것)",
                                                "숫자는 여기서 끝난다"], (240, 246, 252))
box2(880, 80, 330, 200, "사람  (엔지니어 / BAIC)", ["intent.md 의 항목에 답", "언더바디 · 휠 · 통로 · 대칭",
                                                 "→ 명시 플래그", "  (--seal-below, --close-near, --floor-z)"],
     (255, 250, 240))
box2(1250, 80, 320, 200, "파이프라인 실행", ["B-rep → 메쉬 → 랩", "가정이 아니라 결정으로 기록",
                                          "→ STEP · STL · summary"], (245, 247, 250))
arrow2(360, 180, 430, 180)
arrow2(810, 180, 880, 180, "questions")
arrow2(1210, 180, 1250, 180, "플래그")
# planned layers
box2(430, 330, 380, 190, "조언자 (LLM)  — 계획", ["산출물·렌더를 읽고", "항목의 정체를 분류", "  휠 / 유리 / 그릴 / 틈",
                                               "질문 문안 작성", "판단은 사람이"], (250, 250, 250))
box2(880, 330, 330, 190, "의도 프로파일  — 계획", ["프로그램별 결정 저장", "같은 프로그램의", "  2·3번째 변형은 무개입"],
     (250, 250, 250))
box2(1250, 330, 320, 190, "학습 분류기  — 계획", ["사례에서 결정을 학습", "원본 + 트림본 + 결정 + Cd",
                                              "형상은 짓지 않는다"], (250, 250, 250))
arrow2(620, 280, 620, 330, "params · intent")
arrow2(1045, 280, 1045, 330, "답변")
arrow2(1410, 280, 1410, 330, "사례 축적")
d.text((30, 340), "AI가 하는 일: 뜻을 읽고 분류하고 묻는다\n", fill=INK, font=font(16, True))
d.text((30, 370), "AI가 하지 않는 일: 형상을 지어내지 않는다\n(오차 한계 없음 · STEP 불가 ·\n정보가 이미 경계에 있음)", fill=GREY, font=font(14))
img.save(args.out / "fig_ai_helper.png")
print("   fig_ai_helper.png")
