#!/usr/bin/env python3
"""체적 해(flow.vtu)에서 형상 경계상자 상대좌표로 지정한 창 안의 Cp 통계를 낸다.

옆 LES 팀의 `lvl_region_cp.py med()` 와 짝을 맞추기 위한 계기다. 그쪽은
"코 앞 작은 상자 안 유체 셀 밀도의 중앙값"을 Cp 로 환산한다. 표면 면적가중
평균도 아니고 최대값도 아니라서, 같은 정의로 재지 않으면 비교가 성립하지 않는다.

상대좌표는 형상 경계상자 기준이다: 0 = 최소면, 1 = 최대면, 음수면 그 앞.
형상 경계상자는 --surface (표면 해) 에서 잡는다. 체적 해의 경계상자가 아니다.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("volume", help="flow.vtu (유체 셀)")
    ap.add_argument("--surface", required=True, help="surface_flow.vtu (형상 경계상자를 여기서 잡는다)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--scale-json")
    ap.add_argument("--window", required=True,
                    help="상대좌표 창 'x0,x1,y0,y1,z0,z1' 예: -0.030,-0.004,0.30,0.70,0.20,0.70")
    ap.add_argument("--slabs", type=int, default=0, help="x 를 N 등분해 접근 단면도 같이 낸다")
    ap.add_argument("--vref", type=float, default=0.0,
                    help="기준 속도 (m/s). 주면 전압 계수 Cp_t = Cp + (|U|/vref)^2 도 낸다. "
                         "비점성 접근류에서 1.0 이어야 하고, 1.0 을 넘으면 어디선가 에너지가 생긴 것이다.")
    ap.add_argument("--span", action="store_true",
                    help="자유류 Cp 를 같이 재고 '정체 − 자유류' 진폭을 낸다. 이 진폭은 1.000 이어야 하고, "
                         "빼기라서 기준점 보정(offset)이 무엇이든 상쇄된다. 즉 척도만 시험한다.")
    ap.add_argument("--coarsen", default="",
                    help="쉼표로 준 칸 크기(mm)로 장을 일부러 굵게 만들어 통계가 어느 쪽으로 움직이는지 본다. "
                         "격자 굵기가 값을 올리는지 내리는지를 재는 대조군이다.")
    ap.add_argument("--out")
    a = ap.parse_args()

    import pyvista as pv
    scale = a.scale
    if a.scale_json:
        scale = json.load(open(a.scale_json))["coord_scale"] / 1000.0

    srf = pv.read(a.surface).extract_surface()
    SP = np.asarray(srf.points) * scale
    lo, hi = SP.min(0), SP.max(0)
    span = hi - lo

    vol = pv.read(a.volume)
    if "Pressure_Coefficient" not in vol.point_data:
        sys.exit("체적 해에 Pressure_Coefficient 가 없다")
    vol = vol.point_data_to_cell_data()
    vol = vol.compute_cell_sizes(length=False, area=False, volume=True)
    C = np.asarray(vol.cell_centers().points) * scale
    Cp = np.asarray(vol["Pressure_Coefficient"])
    V = np.abs(np.asarray(vol["Volume"])) * scale ** 3
    Cpt = None
    if a.vref > 0 and "Velocity" in vol.cell_data:
        Cpt = Cp + (np.linalg.norm(np.asarray(vol["Velocity"]), axis=1) / a.vref) ** 2

    w = [float(v) for v in a.window.split(",")]
    box_lo = lo + np.array(w[0::2]) * span
    box_hi = lo + np.array(w[1::2]) * span

    out = []
    p = out.append
    p(f"체적 해 {a.volume}")
    p(f"형상 경계상자(m)  x {lo[0]:+.3f}..{hi[0]:+.3f}  y {lo[1]:+.3f}..{hi[1]:+.3f}  z {lo[2]:+.3f}..{hi[2]:+.3f}")
    p(f"창 (상대)  x {w[0]:+.3f}..{w[1]:+.3f}  y {w[2]:.3f}..{w[3]:.3f}  z {w[4]:.3f}..{w[5]:.3f}")
    p(f"창 (실제 m) x {box_lo[0]:+.4f}..{box_hi[0]:+.4f}  y {box_lo[1]:+.4f}..{box_hi[1]:+.4f}  z {box_lo[2]:+.4f}..{box_hi[2]:+.4f}")

    def stats(label, m):
        if not m.any():
            p(f"  {label:<22s} (셀 없음)")
            return
        c, v = Cp[m], V[m]
        p(f"  {label:<22s} 셀 {int(m.sum()):6d}  중앙값 {np.median(c):+.4f}  "
          f"체적가중평균 {np.average(c, weights=v):+.4f}  p10 {np.percentile(c,10):+.4f}  "
          f"p90 {np.percentile(c,90):+.4f}  최대 {c.max():+.4f}")

    m = np.all((C >= box_lo) & (C <= box_hi), axis=1)
    p("")
    p("■ 지정한 창")
    stats("창 전체", m)

    if a.slabs:
        p("")
        p(f"  · x 를 {a.slabs} 등분 (코 쪽 → 몸체 쪽)")
        e = np.linspace(box_lo[0], box_hi[0], a.slabs + 1)
        yz = np.all((C[:, 1:] >= box_lo[1:]) & (C[:, 1:] <= box_hi[1:]), axis=1)
        for i in range(a.slabs):
            stats(f"x {e[i]:+.4f}..{e[i+1]:+.4f}", yz & (C[:, 0] >= e[i]) & (C[:, 0] < e[i+1]))

    if Cpt is not None:
        q = Cpt[m]
        p("")
        p(f"  · 전압 계수 Cp_t = Cp + (|U|/{a.vref:g})²  — 비점성 접근류면 1.0, 넘으면 에너지가 생긴 것")
        p(f"    창 안  중앙값 {np.median(q):+.4f}  체적가중평균 {np.average(q, weights=V[m]):+.4f}  "
          f"p5 {np.percentile(q,5):+.4f}  p95 {np.percentile(q,95):+.4f}  최대 {q.max():+.4f}")
        bad = np.abs(q - 1.0) > 0.05
        p(f"    |Cp_t − 1| > 0.05 인 셀 {int(bad.sum())} / {len(q)} ({bad.sum()/len(q)*100:.1f} %)")
        p("    읽는 법: 정체 Cp 중앙값이 1 을 넘는데 Cp_t 도 같이 넘으면 수치 결함이다.")
        p("            Cp 만 넘고 Cp_t 가 1 이면 속도장 쪽(기준속도·보정)을 먼저 의심한다.")

    if a.span:
        L = hi[0] - lo[0]
        far = C[:, 0] < lo[0] - 2 * L
        p("")
        p("  · 진폭 시험 — '정체 − 자유류' 는 규약과 무관하게 1.000 이어야 한다")
        p("    (빼기라서 기준점 보정이 무엇이든 상쇄된다. 척도만 시험하는 잣대다.)")
        if not far.any():
            p("    자유류 구역(코 앞 2 차체길이 밖)에 셀이 없다 — 계산영역이 짧다")
        else:
            cp_inf = float(np.median(Cp[far]))
            p(f"    자유류 Cp∞ (코 앞 2 차체길이 밖, 셀 {int(far.sum())})   {cp_inf:+.5f}")
            if Cpt is not None:
                sp_far = np.linalg.norm(np.asarray(vol["Velocity"]), axis=1)[far]
                p(f"    자유류 |U| 중앙값 {np.median(sp_far):.3f} (기준 {a.vref:g}, 어긋남 "
                  f"{(np.median(sp_far)/a.vref-1)*100:+.2f} %)")
            p(f"    정체창 Cp 최대 {Cp[m].max():+.5f}  →  진폭 {Cp[m].max()-cp_inf:+.5f}")
            p("    진폭이 1 보다 크면 기준동압이 그만큼 작게 잡힌 것이다(보정으로는 안 고쳐진다).")

    if a.coarsen:
        p("")
        p("  · 대조군 — 장을 일부러 굵게 만들면 어디로 움직이나")
        p("    (주의: 푼 장을 사후 평균낸 것이라 '표본 추출' 성분만 잰다. 굵은 격자로 다시 푸는")
        p("     것은 해 자체도 바꾸므로 여기서 재는 값이 전부가 아니다.)")
        cp_w, V_w, C_w = Cp[m], V[m], C[m]
        p(f"    {'원본 그대로':<26s} 중앙값 {np.median(cp_w):+.4f}  최대 {cp_w.max():+.4f}")
        for s in [float(x) for x in a.coarsen.split(",") if x.strip()]:
            g = np.floor((C_w - box_lo) / (s / 1000.0)).astype(np.int64)
            key = g[:, 0] * 1000000 + g[:, 1] * 1000 + g[:, 2]
            u, inv = np.unique(key, return_inverse=True)
            avg = np.bincount(inv, weights=cp_w * V_w) / np.bincount(inv, weights=V_w)
            p(f"    {str(int(s)) + ' mm 균일칸 체적평균':<26s} 중앙값 {np.median(avg):+.4f}  최대 {avg.max():+.4f}  칸 {len(u)}")

    txt = "\n".join(out)
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
        print(f"\n저장: {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
