"""Contact prevention for the wrap: reopen the gaps a coarse wrap closed by
re-wrapping only those places at a finer alpha and splicing the pieces in.

    coarse wrap W (alpha_L, cheap everywhere)
    for each closed patch with gap >= --keep-above (minus --slack):
        P     = patch bbox grown by --margin                 # the splice region
    group the P boxes into merged wrap boxes B (overlapping ones together)
    for each B:
        S     = reference ∩ (B grown by --cut-margin)       # closed solid piece, cut faces capped
        W_S   = alpha wrap of S at --alpha (about keep/2)
        W     = (W − P_B) ∪ (W_S ∩ P_B)                     # splice only inside the patch boxes

Only the neighbourhood of a targeted opening takes the fine wrap; everywhere
else the coarse wrap's decisions stand (hollow tubes stay filled, gaps under the
threshold stay closed). That is what "contact prevention" means in the
commercial wrappers. The reference must be closed and consistently oriented
(the welded mesh is), so the cut piece is a solid and the fine wrap cannot carve
into the body through the cut. Reports every box and what is left closed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aox_g3.openings import closed_patches  # noqa: E402
from aox_g3.seal import alpha_wrap  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--reference", type=Path, required=True, help="the welded mesh that was wrapped")
ap.add_argument("--wrap", type=Path, required=True, help="the coarse wrap to fix")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--keep-above", type=float, default=13.0, help="reopen closed gaps at least this wide (mm)")
ap.add_argument("--alpha", type=float, help="fine alpha (default keep-above / 2)")
ap.add_argument("--margin", type=float, help="splice box margin around a patch (default 2 × alpha + 10)")
ap.add_argument("--slack", type=float, default=1.5, help="mm below --keep-above still targeted (the gap estimate is ±1 mm)")
ap.add_argument("--cut-margin", type=float, help="extra margin for the cut solid (default 6 × alpha)")
ap.add_argument("--offset", type=float, help="fine wrap offset (default alpha/30; pass the coarse wrap's offset "
                     "so the two surfaces coincide where nothing changed)")
ap.add_argument("--clean-tolerance", type=float, default=0.01,
                help="manifold-preserving simplification tolerance after the splice (mm; 0 = off)")
ap.add_argument("--web-distance", type=float, default=1.5)
ap.add_argument("--min-area", type=float, default=2000.0)
ap.add_argument("--report", type=Path)
args = ap.parse_args()

t0 = time.time()
alpha = args.alpha if args.alpha else args.keep_above / 2.0
margin = args.margin if args.margin is not None else 2.0 * alpha + 10.0
cut_margin = args.cut_margin if args.cut_margin is not None else 6.0 * alpha

ref = trimesh.load(args.reference, force="mesh")
ref.merge_vertices()
W = trimesh.load(args.wrap, force="mesh")
W.merge_vertices()
print(f"기준 {args.reference.name}: 삼각형 {len(ref.faces):,} 수밀 {ref.is_watertight}  |  "
      f"거친 랩 {args.wrap.name}: 삼각형 {len(W.faces):,} 수밀 {W.is_watertight} 몸체 {W.body_count}")
print(f"지킬 틈 ≥ {args.keep_above:.0f} mm → 국소 알파 {alpha:.1f} mm, 상자 여유 {margin:.0f} mm, 절단 여유 {cut_margin:.0f} mm")

patches, _ = closed_patches(ref, W, args.web_distance, args.min_area)
targets = [p for p in patches if p.gap >= args.keep_above - args.slack]
print(f"닫힌 자리 {len(patches)}곳 중 다시 열 곳 {len(targets)}곳 (틈 ≥ {args.keep_above - args.slack:.1f} mm)")

# splice boxes, one per patch; wrap boxes = groups of overlapping splice boxes
splice = [(p.bbox_min - margin, p.bbox_max + margin) for p in targets]
groups = [[i] for i in range(len(splice))]
bounds = [list(b) for b in splice]
merged = True
while merged and len(groups) > 1:
    merged = False
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a0, a1 = bounds[i]; b0, b1 = bounds[j]
            if np.all(a0 <= b1) and np.all(b0 <= a1):
                bounds[i] = [np.minimum(a0, b0), np.maximum(a1, b1)]
                groups[i] += groups[j]
                del groups[j]; del bounds[j]
                merged = True
                break
        if merged:
            break
boxes = [(np.asarray(b[0]), np.asarray(b[1])) for b in bounds]
print(f"이음 상자 {len(splice)}개 → 랩 상자 {len(boxes)}개 (겹치는 것은 한 조각으로 랩)")


def box_mesh(lo, hi):
    return trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))


report = {"alpha": alpha, "keep_above": args.keep_above, "boxes": []}
for k, (lo, hi) in enumerate(boxes):
    tb = time.time()
    outer = box_mesh(lo - cut_margin, hi + cut_margin)
    piece = trimesh.boolean.intersection([ref, outer], engine="manifold")
    if piece.is_empty or len(piece.faces) == 0:
        print(f"  상자 {k}: 기준 형상이 없음, 건너뜀")
        continue
    fine = alpha_wrap(piece, alpha, args.offset)
    fine.merge_vertices()
    if fine.volume < 0:
        fine.invert()
    P = trimesh.boolean.union([box_mesh(*splice[i]) for i in groups[k]], engine="manifold") \
        if len(groups[k]) > 1 else box_mesh(*splice[groups[k][0]])
    # Measured 2026-09-08 (front-wing box): clipping the fine wrap to the coarse one as
    # well, matching the offsets, or merging with a manifold tolerance all produced
    # self-touching sheets (2-30 non-manifold edges after welding). Plain fine ∩ P with
    # the fine offset below the coarse one is the only variant that stays clean
    # (0 non-manifold, 75 slivers); the slivers are removed by the simplify() below.
    fine_in = trimesh.boolean.intersection([fine, P], engine="manifold")
    W_out = trimesh.boolean.difference([W, P], engine="manifold")
    W_new = trimesh.boolean.union([W_out, fine_in], engine="manifold")
    W_new.merge_vertices()
    entry = {"box_min": lo.round(1).tolist(), "box_max": hi.round(1).tolist(), "splice_boxes": len(groups[k]),
             "piece_faces": int(len(piece.faces)), "fine_faces": int(len(fine.faces)),
             "faces_before": int(len(W.faces)), "faces_after": int(len(W_new.faces)),
             "watertight": bool(W_new.is_watertight), "seconds": round(time.time() - tb, 1)}
    report["boxes"].append(entry)
    e = hi - lo
    print(f"  상자 {k}: {e[0]:.0f}×{e[1]:.0f}×{e[2]:.0f} mm, 이음 상자 {len(groups[k])}개  조각 {len(piece.faces):,}면 → 국소 랩 {len(fine.faces):,}면  "
          f"합친 뒤 {len(W_new.faces):,}면 수밀 {W_new.is_watertight}  {time.time()-tb:.0f}s")
    if not W_new.is_watertight:
        print("     ! 합친 결과가 수밀이 아님 — 이 상자는 건너뜀")
        continue
    W = W_new

# The boolean seams leave slivers and micron edges along the splice boxes; a
# manifold-preserving simplification removes them (and the coplanar splits the
# booleans introduced) without touching the shape beyond --clean-tolerance.
if args.clean_tolerance > 0:
    from manifold3d import Manifold, Mesh
    mm = Manifold(Mesh(vert_properties=np.asarray(W.vertices, np.float32), tri_verts=np.asarray(W.faces, np.uint32)))
    ms = mm.simplify(float(args.clean_tolerance)).to_mesh()
    cleaned = trimesh.Trimesh(np.asarray(ms.vert_properties)[:, :3], np.asarray(ms.tri_verts), process=False)
    cleaned.merge_vertices()
    if cleaned.is_watertight:
        print(f"정리(simplify {args.clean_tolerance} mm): 삼각형 {len(W.faces):,} → {len(cleaned.faces):,}, "
              f"최단 변 {W.edges_unique_length.min():.1e} → {cleaned.edges_unique_length.min():.1e} mm")
        W = cleaned
    else:
        print("정리 결과가 수밀이 아니라 정리 전 것을 씀")
W.merge_vertices()
patches_after, _ = closed_patches(ref, W, args.web_distance, args.min_area)
left = [p for p in patches_after if p.gap >= args.keep_above]
print(f"\n결과: 삼각형 {len(W.faces):,}  수밀 {W.is_watertight}  몸체 {W.body_count}  체적 {abs(W.volume)/1e9:.4f} m³  {time.time()-t0:.0f}s")
print(f"남은 닫힌 자리 {len(patches_after)}곳, 그중 틈 ≥ {args.keep_above:.0f} mm: {len(left)}곳")
for p in patches_after[:8]:
    e = p.extent
    print(f"    {p.area/100:6.0f} cm²  ({p.center[0]:6.0f},{p.center[1]:5.0f},{p.center[2]:4.0f})  {e[0]:4.0f}×{e[1]:4.0f}×{e[2]:3.0f} mm  틈 ≈ {p.gap:4.1f} mm")
W.export(args.out)
report.update({"faces": int(len(W.faces)), "watertight": bool(W.is_watertight), "bodies": int(W.body_count),
               "volume_m3": float(abs(W.volume)) / 1e9, "left_closed": [p.as_dict() for p in patches_after]})
if args.report:
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(f"저장 {args.out}")
