"""변형된 형상이 아직 "그 차"인지 보는 기하 검사 — 추론 전에 1 ms 로 거른다.

BO_자료(body_bo.py)의 geometry_ok 은 LES 6분을 태우기 전에 망가진 메시를 걸렀다.
같은 자리에 같은 이유로 둔다. 다만 검사 내용은 다르다: 저쪽은 CAD 를 다시 만들어
면이 사라지거나 NaN 이 나올 수 있었지만, 여기는 기존 메시의 정점만 밀기 때문에
위상은 그대로다. 그래서 정점 변형에서 실제로 나는 고장을 본다.

왜 필요했나(2026-09-04): δ 10% 로 넓혔더니 지붕과 펜더에 뿔이 솟은 형상이 나왔는데
서로게이트는 40개 전부 "학습 분포 안"이라고 답했다. 실현가능성 GP(cEI)는 실패
라벨을 먹고 사는데 실패가 0건이라 놀고 있었다. OOD 판정을 못 믿으니 기하로 막는다.

검사는 전부 원본 대비 비율이다 — 차 크기나 단위(m/mm)에 안 걸리게.
"""

from __future__ import annotations

import numpy as np

from .recommend import LENGTH_AXIS, WIDTH_AXIS

# 면 뒤집힘 허용치. 슬리버 삼각형은 부동소수 잡음만으로도 뒤집혀서 0 은 못 쓴다.
FLIP_FRACTION = 0.002
# 부피·전면적 허용 범위. 벗어나면 Cd 를 원본과 비교하는 게 의미가 없다.
VOLUME_RANGE = (0.75, 1.25)
FRONTAL_RANGE = (0.85, 1.15)
# 바닥 뚫기. 차는 지면 위에 있다 — 전장의 1% 넘게 내려가면 노면 아래다.
FLOOR_DROP_FRACTION = 0.01

HEIGHT_AXIS = 3 - LENGTH_AXIS - WIDTH_AXIS


# 사유 문자열의 앞머리 = 고장의 종류. 뒤에 붙는 수치는 설계마다 달라서, 요약할 때는
# 이 앞머리로 묶는다("volume changed +73%" 와 "+41%" 를 한 줄로).
KINDS = (
    "surface folded",
    "volume changed",
    "frontal area changed",
    "body sinks",
    "collapsed extent",
    "vertex NaN/Inf",
    "zero length",
)


def kind_of(reason: str) -> str:
    for kind in KINDS:
        if reason.startswith(kind):
            return kind
    return reason or "infeasible"


def summarise(reasons) -> dict[str, int]:
    """실패 사유를 종류별 개수로. 카드 밑에 네 줄로 보이라고 만든 것."""
    counts: dict[str, int] = {}
    for reason in reasons:
        key = kind_of(reason)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return np.cross(b - a, c - a)


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """발산정리로 닫힌 메시의 부피. 열린 메시면 값이 흔들리지만 비율로만 쓴다."""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _frontal_area(vertices: np.ndarray, faces: np.ndarray, axis: int) -> float:
    """흐름축으로 투영한 면적. 삼각형 투영면적의 절반 합 — 볼록 형상에서 정확하다."""
    normals = _face_normals(vertices, faces)
    return float(np.abs(normals[:, axis]).sum() / 4.0)


def geometry_ok(
    original: np.ndarray,
    deformed: np.ndarray,
    faces: np.ndarray,
    *,
    length_axis: int = LENGTH_AXIS,
) -> tuple[bool, str]:
    """(통과 여부, 이유). 이유는 실현가능성 GP 의 라벨이자 사용자에게 보일 문구다."""
    if not np.isfinite(deformed).all():
        return False, "vertex NaN/Inf"

    extents = deformed.max(axis=0) - deformed.min(axis=0)
    if not np.isfinite(extents).all() or float(extents.min()) <= 0:
        return False, "collapsed extent"

    length = float(original[:, length_axis].max() - original[:, length_axis].min())
    if length <= 0:
        return False, "zero length"

    # 표면이 자기를 뚫고 접혔는가. 자기교차 전수검사는 O(n²) 이라 못 하고,
    # 접힘은 면 법선이 뒤집히는 것으로 드러난다 — 정점 밀기의 대표 고장이다.
    before = _face_normals(original, faces)
    after = _face_normals(deformed, faces)
    scale = np.linalg.norm(before, axis=1)
    live = scale > scale.max() * 1e-6  # 원래부터 퇴화한 면은 판단에서 뺀다
    if live.any():
        flipped = np.einsum("ij,ij->i", before[live], after[live]) < 0
        share = float(flipped.mean())
        if share > FLIP_FRACTION:
            return False, f"surface folded ({share:.1%} of faces)"

    volume = abs(_signed_volume(original, faces))
    if volume > 0:
        ratio = abs(_signed_volume(deformed, faces)) / volume
        if not VOLUME_RANGE[0] <= ratio <= VOLUME_RANGE[1]:
            return False, f"volume changed {ratio - 1:+.0%}"

    frontal = _frontal_area(original, faces, length_axis)
    if frontal > 0:
        ratio = _frontal_area(deformed, faces, length_axis) / frontal
        if not FRONTAL_RANGE[0] <= ratio <= FRONTAL_RANGE[1]:
            return False, f"frontal area changed {ratio - 1:+.0%}"

    drop = float(original[:, HEIGHT_AXIS].min() - deformed[:, HEIGHT_AXIS].min())
    if drop > FLOOR_DROP_FRACTION * length:
        return False, f"body sinks {drop / length:.1%} of length below the floor"

    return True, ""
