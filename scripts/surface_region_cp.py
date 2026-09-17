#!/usr/bin/env python3
"""SU2 표면 해(surface_flow.vtu)를 구간별로 쪼개 Cp 와 항력 기여를 보고한다.

정의를 전부 출력에 박아 둔다 (다른 계기와 대조할 때 정의가 어긋나면 값이 달라지므로):
  * 좌표는 --scale 로 실제 미터로 되돌린다 (파이프라인 v8 은 STL 을 coord_scale 로 정규화한다).
  * 면 법선은 바깥쪽, 항력 기여 Fx = -(Cp * n_x * A)  [압력 성분만]
  * 마찰은 Skin_Friction_Coefficient 의 x 성분을 면적가중으로 더한다.
  * 구간 비율의 분모는 압력 기여 총합이다 (기준면적 무관, 배율 무관).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np


def load(vtu: Path, scale: float):
    import pyvista as pv
    s = pv.read(str(vtu)).extract_surface()
    s = s.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
    s = s.point_data_to_cell_data().compute_cell_sizes(length=False, area=True, volume=False)
    C = np.asarray(s.cell_centers().points) * scale
    A = np.asarray(s["Area"]) * scale * scale
    N = np.asarray(s["Normals"])
    Cp = np.asarray(s["Pressure_Coefficient"])
    Cf = np.asarray(s["Skin_Friction_Coefficient"]) if "Skin_Friction_Coefficient" in s.cell_data else None
    return C, A, N, Cp, Cf


def row(label, m, C, A, N, Cp, tot):
    if not m.any():
        return f"  {label:<26s}  (셀 없음)"
    a = A[m].sum()
    return (f"  {label:<26s} 셀 {int(m.sum()):6d}  면적 {a:6.3f} m²  "
            f"Cp 평균 {np.average(Cp[m], weights=A[m]):+7.4f}  최대 {Cp[m].max():+7.4f}  "
            f"기여 {-(Cp[m]*N[m,0]*A[m]).sum()/tot*100:+7.2f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtu")
    ap.add_argument("--scale", type=float, default=1.0, help="해석단위 → m (geometry_scale.json 의 coord_scale/1000)")
    ap.add_argument("--scale-json", help="geometry_scale.json 경로 (있으면 --scale 을 덮어씀)")
    ap.add_argument("--front-frac", type=float, default=0.25)
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument("--ref-area", type=float, default=None, help="해석단위 기준면적 (cfg 의 REF_AREA)")
    ap.add_argument("--out")
    a = ap.parse_args()

    scale = a.scale
    if a.scale_json:
        scale = json.load(open(a.scale_json))["coord_scale"] / 1000.0

    C, A, N, Cp, Cf = load(Path(a.vtu), scale)
    lo, hi = C.min(0), C.max(0)
    L = hi[0] - lo[0]
    Fx = -(Cp * N[:, 0] * A)
    tot = Fx.sum()

    out = []
    p = out.append
    p(f"파일 {a.vtu}")
    p(f"배율 1 해석단위 = {scale*1000:.3f} mm   (좌표를 실제 m 로 되돌린 뒤 계산)")
    p(f"경계상자(m)  x {lo[0]:+.3f}..{hi[0]:+.3f}  y {lo[1]:+.3f}..{hi[1]:+.3f}  z {lo[2]:+.3f}..{hi[2]:+.3f}   전장 {L:.3f} m")
    p(f"표면적 {A.sum():.3f} m²   셀 {len(A)}")
    p(f"압력 항력 적분 ΣFx = {tot:.4f} m²  (기준면적 미적용)")
    if Cf is not None:
        fx_f = float((Cf[:, 0] * A).sum())
        p(f"마찰 항력 적분 = {fx_f:.4f} m²  (압력 대비 {fx_f/tot*100:+.1f} %)")
    if a.ref_area:
        ra = a.ref_area * scale * scale
        p(f"기준면적 {a.ref_area:.5f} 해석단위 = {ra:.4f} m²  →  Cd(압력) = {tot/ra:.4f}")

    p("")
    p(f"세로 3등분 (전장 기준)")
    e3 = [lo[0], lo[0]+L/4, lo[0]+3*L/4, hi[0]]
    for nm, x0, x1 in [("앞 25%", e3[0], e3[1]), ("가운데 50%", e3[1], e3[2]), ("뒤 25%", e3[2], e3[3])]:
        m = (C[:, 0] >= x0) & (C[:, 0] < x1) if x1 < hi[0] else (C[:, 0] >= x0)
        p(row(f"{nm}  {x0:+.2f}..{x1:+.2f}", m, C, A, N, Cp, tot))

    xf = lo[0] + a.front_frac * L
    front = C[:, 0] < xf
    p("")
    p(f"■ 앞 {a.front_frac*100:.0f} % 상세  (x < {xf:+.3f} m)")
    p(row("앞 구간 전체", front, C, A, N, Cp, tot))
    p("")
    p(f"  · x 를 {a.bands} 등분")
    e = np.linspace(lo[0], xf, a.bands + 1)
    for i in range(a.bands):
        m = front & (C[:, 0] >= e[i]) & (C[:, 0] < e[i+1])
        p(row(f"    {e[i]:+.3f} .. {e[i+1]:+.3f} m", m, C, A, N, Cp, tot))

    zg = lo[2]
    h = C[:, 2] - zg
    p("")
    p(f"  · 지면 위 높이 h 구간 — h=0 은 차 최저점 z={zg:+.3f} m (지면판은 그 5 mm 아래)")
    zs = [(0.0, 0.15, "h 0.00~0.15 접지·에어댐 밑"), (0.15, 0.35, "h 0.15~0.35 타이어 하부"),
          (0.35, 0.65, "h 0.35~0.65 범퍼"), (0.65, 0.95, "h 0.65~0.95 그릴"),
          (0.95, 1.25, "h 0.95~1.25 후드"), (1.25, 9.0, "h > 1.25 앞유리·루프")]
    for z0, z1, nm in zs:
        m = front & (h >= z0) & (h < z1)
        p(row(f"    {nm}", m, C, A, N, Cp, tot))

    p("")
    p("  · 면이 앞을 보는 정도 (n_x 문턱)")
    for t in (-0.3, -0.5, -0.7, -0.9):
        m = front & (N[:, 0] < t)
        p(row(f"    n_x < {t:+.1f}", m, C, A, N, Cp, tot))
    m = front & (N[:, 0] > 0.3)
    p(row("    n_x > +0.3 (뒤를 봄)", m, C, A, N, Cp, tot))

    p("")
    p("  · Cp 과잉(비물리 Cp>1) 점검")
    for thr in (1.0, 1.1, 1.3, 1.5):
        m = front & (Cp > thr)
        p(row(f"    Cp > {thr:.1f}", m, C, A, N, Cp, tot))
    contact = front & (h < 0.05) & (np.abs(C[:, 1]) > 0.55)
    p(row("    타이어 접지 근처 전체", contact, C, A, N, Cp, tot))
    keep = front & ~((h < 0.05) & (np.abs(C[:, 1]) > 0.55))
    p(f"    접지 근처를 빼면 앞 구간 기여 {-(Cp[keep]*N[keep,0]*A[keep]).sum()/tot*100:+.2f} % "
      f"(빼기 전 {-(Cp[front]*N[front,0]*A[front]).sum()/tot*100:+.2f} %)")

    i = np.argmax(np.where(front, Cp, -9))
    p("")
    p(f"  · 최대 Cp {Cp[i]:+.4f} 위치 x {C[i,0]:+.3f}  y {C[i,1]:+.3f}  z {C[i,2]:+.3f} m  (n_x {N[i,0]:+.2f})")
    hi_cp = front & (Cp > 1.0)
    p(row("    Cp > 1.0 인 면", hi_cp, C, A, N, Cp, tot))

    p("")
    p("  · 앞에서부터 누적 기여 (x 오름차순)")
    o = np.argsort(C[:, 0])
    cum = np.cumsum(Fx[o]) / tot * 100
    xs = C[o, 0]
    for f in (0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.90, 1.00):
        x = lo[0] + f * L
        j = np.searchsorted(xs, x) - 1
        j = max(j, 0)
        p(f"    x < {x:+.3f} m  (앞에서 {f*100:5.1f} %)   누적 {cum[j]:+7.2f} %")

    txt = "\n".join(out)
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
        print(f"\n저장: {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
