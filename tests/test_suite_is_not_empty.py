"""The suite must notice when it stops running.

On 2026-09-17 this repository had sixteen test files and collected none of them: three
experiment scripts under scripts/ are named test_*.py and call parse_args() at import, so
collection died with SystemExit and pytest said "no tests collected". `unittest discover`
said "Ran 0 tests ... OK". Both looked like success.

These two tests are the alarm: they fail if collection shrinks, or if a test function stops
asserting anything.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINIMUM = 80          # 2026-09-17: 85 collected. Raise this when the suite grows.


def test_files_hold_enough_test_functions():
    found = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text())
        found += [n.name for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")]
    assert len(found) >= MINIMUM, (
        f"시험 함수가 {len(found)}개뿐입니다 (최소 {MINIMUM}). 수집이 줄었거나 파일이 사라졌습니다.")


def test_every_test_asserts_something():
    empty = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test"):
                continue
            body = ast.dump(node)
            if any(isinstance(n, ast.Assert) for n in ast.walk(node)) or "raises" in body:
                continue
            # 정당한 예외가 있다: "아무 예외도 안 나는 것" 자체가 단언인 양성 대조 같은 것.
            # 무조건 면제하면 규칙이 죽고 무조건 실패로 두면 늑대가 되니, 본문에 이유를 적게 한다.
            lines = path.read_text().splitlines()[node.lineno - 1: getattr(node, "end_lineno", node.lineno)]
            if any("NO_ASSERT_OK:" in line for line in lines):
                continue
            empty.append(f"{path.name}::{node.name}")
    assert not empty, ("단언이 없는 시험: " + ", ".join(empty)
                       + "  — 의도한 것이면 함수 본문에 '# NO_ASSERT_OK: <이유>' 를 적으세요")
