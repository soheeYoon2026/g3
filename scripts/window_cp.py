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

    txt = "\n".join(out)
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
        print(f"\n저장: {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
